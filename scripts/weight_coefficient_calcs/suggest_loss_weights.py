#!/usr/bin/env python3

# suggest_loss_weights.py
# -----------------------------------------------------------------------------
# Purpose
# -------
# Given train/val datasets (EXTXYZ), estimate label scales for
#   * per-atom energy [eV/atom]
#   * force components [eV/Å]
#   * stress (either full tensor MSE or config-aware pressure/deviatoric) [eV/Å^3 or GPa]
#
# and propose relative loss weights for E/F/S that (approximately) equalize
# their contributions under your chosen loss types (MSE, MAE, Huber).
#
# Rationale
# ---------
# 1) Different tasks (E, F, S) have different physical units and typical scales.
#    We normalize by *label scale* as a proxy for early-training residual scale.
# 2) For config-aware stresses, only some sections contribute pressure and/or
#    deviatoric parts. We account for this by computing an "effective" stress
#    scale using the dataset's section distribution.
# 3) Optionally, you can skew the target contributions (e.g., prioritize forces).
#
# References (for dynamic alternatives you may want to implement in training):
#   - Kendall & Gal, "Multi-Task Learning Using Uncertainty to Weigh Losses",
#     CVPR 2018 (homoscedastic uncertainty weighting).
#   - Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing",
#     ICML 2018.
#
# Usage
# -----
#   python suggest_loss_weights.py \
#       --train TRAIN.extxyz [--train TRAIN2.extxyz ...] \
#       --val VAL.extxyz \
#       --stress-units auto \
#       --use-config-aware \
#       --weight-p 0.30 --weight-s 0.70 \
#       --allow-pressure-for-defects \
#       --include-intermetallics-in-full false \
#       --target E=1 F=1 S=1 \
#       --loss-types energy=mse force=huber stress=huber \
#       --normalize-to energy
#
# Output: prints a human-readable table and emits a JSON with suggested weights.
#
# Notes
# -----
# - The script requires ASE (pip install ase).
# - It does not require your training code, and it does NOT look at predictions.
# - It infers section labels from `atoms.info['config_type']` using a robust
#   mapping consistent with your `config_aware_stress.py`. If your dataset uses
#   different keys, use `--config-type-key` to override.
# -----------------------------------------------------------------------------

from __future__ import annotations
import argparse, json, math, os, sys, statistics
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

try:
    import numpy as np
    import ase.io
except Exception as e:
    print("This script requires numpy and ase. Install with:", file=sys.stderr)
    print("  pip install numpy ase", file=sys.stderr)
    raise

# --- Unit conversion ---
GPA_PER_EVA3 = 160.21766208

def eva3_to_gpa(x: float | np.ndarray) -> float | np.ndarray:
    return x * GPA_PER_EVA3

def gpa_to_eva3(x: float | np.ndarray) -> float | np.ndarray:
    return x / GPA_PER_EVA3

# --- Section mapping (kept consistent with your repo's config_aware_stress.py) ---
def normalize_config_type(raw: Optional[str]) -> str:
    if raw is None:
        return "unknown"
    ct = str(raw).strip().replace(" ", "_")
    import re
    ct = re.sub(r"_aa(-mc)?$", "", ct)
    ct = ct.replace("vac_", "vacancy_")
    if ct in {"vac", "vacancy-alloy"}:
        ct = "vacancy"
    return ct

def section_of(ct: str) -> str:
    if ct.startswith("surface_") or ct.startswith("gamma_surface"):
        return "Surfaces & γ"
    if (ct.startswith("vac") or ct in {"di-vacancy", "tri-vacancy", "sia", "di-sia"}):
        return "Point defects"
    if ct in {"A15", "C15"}:
        return "Intermetallics"
    if ct in {"bcc_distorted", "fcc", "hcp", "dia"}:
        return "Bulk crystals"
    if (ct.startswith("liquid") or ct.startswith("comp-explore") or ct.startswith("surf_liquid")):
        return "Liquids & explore"
    if ct.startswith("phonon"):
        return "Phonon"
    if ct.startswith("elastic"):
        return "Elastic"
    if ct.startswith("neb"):
        return "NEB"
    return "Other"

SECTIONS_FULL_SIGMA = {"Bulk crystals", "Elastic"}
SECTIONS_PRESSURE_ONLY = {"Liquids & explore"}
SECTIONS_EXCLUDE = {"Surfaces & γ", "NEB", "Intermetallics"}

# --- Stress helpers ---
def voigt6_to_mat(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6).reshape(-1, 6)
    xx, yy, zz, yz, xz, xy = [v6[:, i] for i in range(6)]
    S = np.stack([
        np.stack([xx, xy, xz], axis=-1),
        np.stack([xy, yy, yz], axis=-1),
        np.stack([xz, yz, zz], axis=-1)
    ], axis=-2)  # (N,3,3)
    return S

def mat33(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2 and x.shape == (3,3):
        return x[None, ...]
    if x.ndim == 3 and x.shape[-2:] == (3,3):
        return x
    if x.ndim == 1 and x.shape[0] == 6:
        return voigt6_to_mat(x)
    if x.ndim == 1 and x.shape[0] == 9:
        # Handle flattened 3x3 matrices
        return x.reshape(1, 3, 3)
    raise ValueError(f"Cannot interpret as (N,3,3): shape {x.shape}")

def decompose_stress(S_GPa: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # S_GPa: (N,3,3) in GPa. Pressure positive in compression: p = -tr(S)/3.
    tr = S_GPa[...,0,0] + S_GPa[...,1,1] + S_GPa[...,2,2]
    p = -tr / 3.0  # (N,)
    I = np.eye(3)[None, ...]
    dev = S_GPa + p[..., None, None] * I
    return p, dev

def von_mises(dev_GPa: np.ndarray) -> np.ndarray:
    return np.sqrt(1.5 * (dev_GPa * dev_GPa).sum(axis=(-2, -1)))

# --- Data reading ---
def detect_stress_and_units(atoms) -> Tuple[Optional[np.ndarray], str]:
    """
    Return (S, units) where units in {"eva3","gpa"}; S has shape (3,3).
    Heuristics: prefer 'stress' if present; else compute from 'virial'/volume if present.
    """
    # Try info dict keys for stress/virial
    cand = None
    units = "eva3"
    info = atoms.info or {}
    arrays = atoms.arrays or {}

    # Common locations / names
    keys = [
        ("stress", info.get("REF_stress", None)),
        ("stress", info.get("stress", None)),
        ("virial", info.get("virial", None)),
        ("virials", info.get("virials", None)),
        ("virial", arrays.get("virial", None)),
        ("virials", arrays.get("virials", None)),
        ("sigma", info.get("sigma", None)),
    ]

    for name, val in keys:
        if val is None:
            continue
        arr = np.array(val, dtype=float)
        if name == "stress":
            # Assume eV/Å^3 unless magnitudes clearly look like GPa
            S = mat33(arr)[0]
            # crude unit inference
            m = np.nanmedian(np.abs(S))
            units = "gpa" if m > 100.0 else "eva3"
            return S, units
        if name in {"virial", "virials"}:
            # virial in eV; stress ≈ -virial / volume (sign conventions vary)
            V = atoms.get_volume()
            S = -mat33(arr)[0] / max(V, 1e-8)
            units = "eva3"
            return S, units
        if name == "sigma":
            # Some codes store stress in GPa under 'sigma'
            S = mat33(arr)[0]
            units = "gpa"
            return S, units

    return None, "eva3"

@dataclass
class SplitStats:
    n_structs: int
    n_atoms_total: int
    per_atom_energy: List[float]
    force_components: List[float]
    stress_tensors_eva3: List[np.ndarray]  # each (3,3) in eV/Å^3
    sections: List[str]

def read_extxyz(paths: List[str], config_type_key: str = "config_type", stress_units_override: Optional[str] = None) -> SplitStats:
    per_atom_E = []
    force_comps = []
    stress_list_eva3 = []
    sections = []
    n_structs = 0
    n_atoms_total = 0

    for p in paths:
        frames = ase.io.read(p, index=":")
        for atoms in frames:
            n_structs += 1
            n = len(atoms)
            n_atoms_total += n

            # Energy
            E = atoms.info.get("REF_energy", atoms.info.get("total_energy", None))
            if E is None:
                # try free_energy
                E = atoms.info.get("free_energy", None)
            if E is not None:
                per_atom_E.append(float(E) / max(n, 1))

            # Forces
            F = atoms.arrays.get("REF_force", atoms.arrays.get("force", None))
            if F is None:
                F = atoms.arrays.get("dft_force", None)
            if F is not None:
                F = np.asarray(F, dtype=float)
                force_comps.extend(F.reshape(-1))

            # Stress
            S, units = detect_stress_and_units(atoms)
            if S is not None:
                if stress_units_override in {"eva3", "gpa"}:
                    units = stress_units_override
                S_eva3 = S if units == "eva3" else gpa_to_eva3(S)
                stress_list_eva3.append(S_eva3)

            # Section
            raw = atoms.info.get(config_type_key, None)
            sec = section_of(normalize_config_type(raw))
            sections.append(sec)

    return SplitStats(
        n_structs=n_structs,
        n_atoms_total=n_atoms_total,
        per_atom_energy=per_atom_E,
        force_components=force_comps,
        stress_tensors_eva3=stress_list_eva3,
        sections=sections,
    )

# --- Scale estimators ---
def robust_rms(x: List[float]) -> float:
    if not x:
        return float("nan")
    arr = np.asarray(x, dtype=float)
    return float(np.sqrt(np.nanmean(arr * arr)))

def robust_std(x: List[float]) -> float:
    if not x:
        return float("nan")
    arr = np.asarray(x, dtype=float)
    return float(np.nanstd(arr))

def stress_component_rms_eva3(stress_list_eva3: List[np.ndarray]) -> float:
    """RMS over the six independent components (per-structure)."""
    if not stress_list_eva3:
        return float("nan")
    v6 = []
    for S in stress_list_eva3:
        v6.append(np.array([S[0,0], S[1,1], S[2,2], S[1,2], S[0,2], S[0,1]], dtype=float))
    v6 = np.stack(v6, axis=0)  # (N,6)
    return float(np.sqrt(np.nanmean(v6**2)))

def pressure_and_dev_scales_GPa(stress_list_eva3: List[np.ndarray]) -> Tuple[float, float]:
    """Return (rms_pressure_GPa, rms_dev_frob_GPa)."""
    if not stress_list_eva3:
        return float("nan"), float("nan")
    S_eva3 = np.stack(stress_list_eva3, axis=0)  # (N,3,3)
    S_gpa = eva3_to_gpa(S_eva3)
    p, dev = decompose_stress(S_gpa)
    p_rms = float(np.sqrt(np.nanmean(p**2)))
    dev_frob = np.sqrt((dev * dev).sum(axis=(-2,-1)))  # Frobenius norm per-sample
    dev_rms = float(np.sqrt(np.nanmean(dev_frob**2)))
    return p_rms, dev_rms

def section_fractions(sections: List[str], allow_pressure_for_defects: bool, include_intermetallics_in_full: bool) -> Dict[str, float]:
    N = max(len(sections), 1)
    counts = {"full_sigma": 0, "pressure": 0, "exclude": 0}
    for s in sections:
        in_full = (s in SECTIONS_FULL_SIGMA) or (include_intermetallics_in_full and s == "Intermetallics")
        in_press = (s in SECTIONS_PRESSURE_ONLY) or in_full or (allow_pressure_for_defects and s == "Point defects")
        in_excl  = (s in SECTIONS_EXCLUDE)
        counts["full_sigma"] += int(in_full)
        counts["pressure"]   += int(in_press)
        counts["exclude"]    += int(in_excl)
    return {k: v / float(N) for k, v in counts.items()}

# --- Weight suggestions ---
def suggest_weights(train: SplitStats,
                    loss_types: Dict[str, str],
                    use_config_aware: bool,
                    weight_p: float,
                    weight_s: float,
                    allow_pressure_for_defects: bool,
                    include_intermetallics_in_full: bool,
                    stress_units_out: str,
                    target_contrib: Dict[str, float],
                    normalize_to: str = "energy") -> Dict[str, object]:
    """
    loss_types: {'energy': 'mse|mae|huber', 'force': 'mse|mae|huber', 'stress': 'mse|huber'}
    stress_units_out: 'eva3' or 'gpa' -- what units your *loss term* uses
    target_contrib: desired relative contributions per task, e.g. {'energy':1,'force':1,'stress':1}
    """
    # Exponents approximate how loss scales with label scale for different loss types
    exp_map = {"mse": 2.0, "mae": 1.0, "huber": 1.5}
    e_exp = exp_map.get(loss_types.get("energy","mse"), 2.0)
    f_exp = exp_map.get(loss_types.get("force","huber"), 1.5)
    s_exp = exp_map.get(loss_types.get("stress","huber"), 1.5)

    # Label scales
    e_scale = robust_std(train.per_atom_energy)  # eV/atom
    f_scale = robust_rms(train.force_components) # eV/Å
    s_comp_rms_eva3 = stress_component_rms_eva3(train.stress_tensors_eva3)  # eV/Å^3

    # Config-aware effective stress scale
    p_rms_GPa, dev_rms_GPa = pressure_and_dev_scales_GPa(train.stress_tensors_eva3)
    fracs = section_fractions(train.sections, allow_pressure_for_defects, include_intermetallics_in_full)
    frac_full = fracs["full_sigma"]
    frac_press = fracs["pressure"]
    # In your ConfigAwareStressHuber, the per-sample combined loss is:
    # L = (w_p * m_press * |Δp| + w_s * m_full * |Δdev|) / (w_p*m_press + w_s*m_full)
    # where |Δdev| is averaged over matrix entries. As a proxy scale, we combine p and dev
    # with the same normalization:
    if math.isnan(p_rms_GPa) or math.isnan(dev_rms_GPa):
        s_eff_eva3 = s_comp_rms_eva3
    else:
        denom = max(weight_p * frac_press + weight_s * frac_full, 1e-8)
        # Convert the numerator to the *loss units* domain
        p_term = weight_p * frac_press * (gpa_to_eva3(p_rms_GPa) if stress_units_out == "eva3" else p_rms_GPa)
        d_term = weight_s * frac_full  * (gpa_to_eva3(dev_rms_GPa) if stress_units_out == "eva3" else dev_rms_GPa)
        s_eff = (p_term + d_term) / denom  # proxy magnitude for per-sample loss
        s_eff_eva3 = s_eff if stress_units_out == "eva3" else eva3_to_gpa(s_eff)  # for reporting both

    # Choose stress scale according to mode
    if use_config_aware:
        s_scale = s_eff_eva3 if stress_units_out == "eva3" else eva3_to_gpa(s_eff_eva3)
    else:
        # Baseline: full-tensor MSE proxy is RMS over 6 comps
        s_scale = s_comp_rms_eva3 if stress_units_out == "eva3" else eva3_to_gpa(s_comp_rms_eva3)

    # Compute unnormalized weights ~ target_contrib / (scale^exp)
    def w_from(scale, exp, target):
        if scale is None or math.isnan(scale) or scale <= 0:
            return 0.0
        return float(target) / float(scale ** max(exp, 1e-6))

    wE = w_from(max(e_scale, 1e-12), e_exp, target_contrib.get("energy",1.0))
    wF = w_from(max(f_scale, 1e-12), f_exp, target_contrib.get("force",1.0))
    wS = w_from(max(s_scale, 1e-12), s_exp, target_contrib.get("stress",1.0))

    # Normalize relative to `normalize_to`
    base = {"energy": wE, "force": wF, "stress": wS}.get(normalize_to, wE) or 1.0
    wE_n = wE / base
    wF_n = wF / base
    wS_n = wS / base

    # Also estimate expected relative contributions with a proposed ratio r = (wE_n, wF_n, wS_n)
    # using scale^exp proxies
    def contribution_ratio(w, scale, exp):
        return w * (scale ** exp)

    cE = contribution_ratio(wE_n, max(e_scale,1e-12), e_exp)
    cF = contribution_ratio(wF_n, max(f_scale,1e-12), f_exp)
    cS = contribution_ratio(wS_n, max(s_scale,1e-12), s_exp)
    c_sum = max(cE + cF + cS, 1e-12)
    c_rel = {"energy": cE / c_sum, "force": cF / c_sum, "stress": cS / c_sum}

    # Pack result
    out = {
        "scales": {
            "energy_per_atom_std_eV": e_scale,
            "force_component_rms_eV_per_A": f_scale,
            "stress_component_rms_eVa3": s_comp_rms_eva3,
            "pressure_rms_GPa": p_rms_GPa,
            "dev_frobenius_rms_GPa": dev_rms_GPa,
            "section_fractions": fracs,
            "effective_stress_scale_in_loss_units": s_scale,
            "stress_loss_units": stress_units_out,
        },
        "loss_types": loss_types,
        "use_config_aware": use_config_aware,
        "config_aware_params": {
            "weight_p": weight_p,
            "weight_s": weight_s,
            "allow_pressure_for_defects": allow_pressure_for_defects,
            "include_intermetallics_in_full": include_intermetallics_in_full,
        },
        "targets": target_contrib,
        "weights_normalized_to_%s" % normalize_to: {
            "energy": wE_n,
            "force": wF_n,
            "stress": wS_n,
        },
        "approx_expected_contribution_fraction": c_rel,
    }
    return out

def pretty_print_suggestion(title: str, sugg: Dict[str, object]):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))
    sc = sugg["scales"]
    print(f"Energy per-atom std [eV]:       {sc['energy_per_atom_std_eV']:.6g}")
    print(f"Force component RMS [eV/Å]:     {sc['force_component_rms_eV_per_A']:.6g}")
    if not math.isnan(sc["stress_component_rms_eVa3"]):
        print(f"Stress comp. RMS [eV/Å^3]:      {sc['stress_component_rms_eVa3']:.6g}")
    if not math.isnan(sc["pressure_rms_GPa"]):
        print(f"Pressure RMS [GPa]:             {sc['pressure_rms_GPa']:.6g}")
    if not math.isnan(sc["dev_frobenius_rms_GPa"]):
        print(f"Deviatoric Frobenius RMS [GPa]: {sc['dev_frobenius_rms_GPa']:.6g}")
    print(f"Section fractions:               {sc['section_fractions']}")
    print(f"Effective stress scale [{sc['stress_loss_units']}]: {sc['effective_stress_scale_in_loss_units']:.6g}")
    w = sugg[[k for k in sugg.keys() if k.startswith("weights_normalized_to_")][0]]
    print("\nSuggested TRAIN loss weights (normalized):")
    print(f"  energy : {w['energy']:.3g}")
    print(f"  force  : {w['force']:.3g}")
    print(f"  stress : {w['stress']:.3g}")
    cr = sugg["approx_expected_contribution_fraction"]
    print("Approx. expected contribution fractions under these weights:")
    print(f"  energy : {100*cr['energy']:.1f}%")
    print(f"  force  : {100*cr['force']:.1f}%")
    print(f"  stress : {100*cr['stress']:.1f}%")

def main():
    ap = argparse.ArgumentParser(description="Suggest E/F/S loss weights from dataset statistics.")
    ap.add_argument("--train", nargs="+", required=True, help="Path(s) to train EXTXYZ files")
    ap.add_argument("--val", nargs="*", default=[], help="Optional path(s) to val EXTXYZ files (only for reporting)")
    ap.add_argument("--config-type-key", default="config_type",
                    help="EXTXYZ info key holding config_type (default: config_type)")
    ap.add_argument("--stress-units", default="auto", choices=["auto","eva3","gpa"],
                    help="Force units for stress input detection override (auto/eva3/gpa).")
    ap.add_argument("--loss-types", default="energy=mse,force=huber,stress=huber",
                    help="Comma list: energy=<mse|mae|huber>,force=<...>,stress=<...>")
    ap.add_argument("--use-config-aware", action="store_true", help="Use config-aware stress policy for scale estimation")
    ap.add_argument("--weight-p", type=float, default=0.30, help="Config-aware pressure term weight (numerator)")
    ap.add_argument("--weight-s", type=float, default=0.70, help="Config-aware deviatoric term weight (numerator)")
    ap.add_argument("--allow-pressure-for-defects", action="store_true", help="Treat Point defects as pressure-only")
    ap.add_argument("--include-intermetallics-in-full", action="store_true", help="Include Intermetallics in full-sigma")
    ap.add_argument("--target", default="E=1,F=1,S=1", help="Desired relative contributions, e.g. E=1,F=2,S=1")
    ap.add_argument("--normalize-to", default="energy", choices=["energy","force","stress"],
                    help="Which task's weight should be normalized to 1.0 in the output")
    ap.add_argument("--json-out", default=None, help="Optional path to write JSON summary")
    args = ap.parse_args()

    # Parse loss types
    lt = {}
    for kv in args.loss_types.split(","):
        k,v = kv.split("=")
        lt[k.strip()] = v.strip().lower()

    # Parse targets
    targ = {}
    for kv in args.target.split(","):
        k,v = kv.split("=")
        key = {"E":"energy","F":"force","S":"stress"}.get(k.strip(), k.strip()).lower()
        targ[key] = float(v)

    # Read train (and val) stats
    su_override = None if args.stress_units == "auto" else args.stress_units
    train_stats = read_extxyz(args.train, config_type_key=args.config_type_key, stress_units_override=su_override)
    if args.val:
        val_stats = read_extxyz(args.val, config_type_key=args.config_type_key, stress_units_override=su_override)
    else:
        val_stats = None

    # Suggest weights for: (1) Baseline stress, (2) Config-aware stress (if enabled)
    out = {}
    base_sugg = suggest_weights(train_stats, lt, use_config_aware=False,
                                weight_p=args.weight_p, weight_s=args.weight_s,
                                allow_pressure_for_defects=args.allow_pressure_for_defects,
                                include_intermetallics_in_full=args.include_intermetallics_in_full,
                                stress_units_out="eva3",  # typical training unit
                                target_contrib=targ, normalize_to=args.normalize_to)
    pretty_print_suggestion("Baseline (full-tensor) stress", base_sugg)
    out["baseline"] = base_sugg

    if args.use_config_aware:
        ca_sugg = suggest_weights(train_stats, lt, use_config_aware=True,
                                  weight_p=args.weight_p, weight_s=args.weight_s,
                                  allow_pressure_for_defects=args.allow_pressure_for_defects,
                                  include_intermetallics_in_full=args.include_intermetallics_in_full,
                                  stress_units_out="eva3",
                                  target_contrib=targ, normalize_to=args.normalize_to)
        pretty_print_suggestion("Config-aware stress", ca_sugg)
        out["config_aware"] = ca_sugg

    # Simple VAL guidance: recommend standardized (z-scored) composite for early stopping
    if val_stats is not None:
        # Compute scales on TRAIN and echo that VAL will be standardized by TRAIN scales
        print("\nValidation weighting tip:")
        print("  For model selection, use a composite 'val score' that sums standardized metrics:")
        print("      zE = E_val / std_train(E_per_atom)")
        print("      zF = F_val / rms_train(F_component)")
        print("      zS = S_val / (effective stress scale in train)")
        print("  Then: val_score = zE + zF + zS (1:1:1).")

    # Write JSON if requested
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote JSON summary to: {args.json_out}")

if __name__ == "__main__":
    main()

