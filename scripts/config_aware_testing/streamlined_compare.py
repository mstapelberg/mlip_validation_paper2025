
#!/usr/bin/env python3
import re
import tarfile
import argparse
import pickle
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# Global plotting color palette
PLOT_COLOR_PALETTE = [
    "#2A33C3",  # blue
    "#A35D00",  # brown/orange
    "#0B7285",  # teal
    "#8F2D56",  # magenta
    "#6E8B00",  # olive
]

# Set matplotlib options using the unified matplotlib.rcParams interface
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

try:
    from cycler import cycler as _cycler  # type: ignore
    matplotlib.rcParams["axes.prop_cycle"] = _cycler(color=PLOT_COLOR_PALETTE)
except Exception:
    # If cycler isn't available, silently keep matplotlib defaults
    pass

# Optional imports for evaluation mode (XYZ + models)
_ASE_AVAILABLE = True
try:
    from ase import Atoms  # type: ignore
    from ase.io import read as ase_read  # type: ignore
except Exception:
    _ASE_AVAILABLE = False

_NEQUIP_AVAILABLE = True
try:
    from nequip.ase import NequIPCalculator  # type: ignore
except Exception:
    _NEQUIP_AVAILABLE = False

SECTION_ORDER = [
    "Bulk crystals", "Intermetallics", "Surfaces & γ", "Point defects",
    "NEB", "Phonon", "Elastic", "Liquids & explore", "Composition explore (comp-explore)", "Other",
]
SECTION_METRIC_MASK = {
    # Config-aware rules:
    # - Use full stress (s_mean) for Bulk, Intermetallics, Elastic; ignore pressure there
    # - Use hydrostatic pressure (p_mean) for Liquids & explore, Point defects, Other
    # - Ignore stress/pressure for NEB, Surfaces & γ, and Composition explore (comp-explore)
    "Bulk crystals":        {"f_mean": True, "e_mean": True, "p_mean": False, "s_mean": True},
    "Intermetallics":       {"f_mean": True, "e_mean": True, "p_mean": False, "s_mean": True},
    "Surfaces & γ":         {"f_mean": True, "e_mean": True, "p_mean": False, "s_mean": False},
    "Point defects":        {"f_mean": True, "e_mean": True, "p_mean": True,  "s_mean": False},
    "NEB":                  {"f_mean": True, "e_mean": True, "p_mean": False, "s_mean": False},
    "Phonon":               {"f_mean": True, "e_mean": True, "p_mean": False, "s_mean": False},
    "Elastic":              {"f_mean": True, "e_mean": True, "p_mean": False, "s_mean": True},
    "Liquids & explore":    {"f_mean": True, "e_mean": True, "p_mean": True,  "s_mean": False},
    "Composition explore (comp-explore)": {"f_mean": True, "e_mean": True, "p_mean": True, "s_mean": False},
    "Other":                {"f_mean": True, "e_mean": True, "p_mean": False,  "s_mean": False},
}

# Unit conversion
GPA_PER_EVA3 = 160.21766208

# -----------------------------------
# User-editable model configuration
# -----------------------------------
# Specify models explicitly here to avoid CLI for potential paths.
#
# Example:
MODELS_BY_GENERATION = {
     0: [
         "../../data/potentials/allegro/per_generation/compiled_models/gen0_seed0.nequip.pt2",
         "../../data/potentials/allegro/per_generation/compiled_models/gen0_seed1.nequip.pt2",
         "../../data/potentials/allegro/per_generation/compiled_models/gen0_seed2.nequip.pt2"
     ],
     7: [
        "../../data/potentials/allegro/per_generation/compiled_models/gen7_seed0.nequip.pt2",
        "../../data/potentials/allegro/per_generation/compiled_models/gen7_seed1.nequip.pt2",
        "../../data/potentials/allegro/per_generation/compiled_models/gen7_seed2.nequip.pt2"
        ],
     10: [
        "../../data/potentials/allegro/per_generation/compiled_models/gen10_seed0.nequip.pt2", 
        "../../data/potentials/allegro/per_generation/compiled_models/gen10_seed1.nequip.pt2",
        "../../data/potentials/allegro/per_generation/compiled_models/gen10_seed2.nequip.pt2"
        ],
 }

MODELS_BY_LOSS = {
     "MSE":   ["../../data/potentials/allegro/config_aware_test/compiled_models/mse_lmax1_nlayers2_mlp512_zbl_epoch140.nequip.pt2"],
     "MSETW": ["../../data/potentials/allegro/config_aware_test/compiled_models/msetw_lmax1_nlayers2_mlp512_nlh.nequip.pt2"],
     "CA":    ["../../data/potentials/allegro/config_aware_test/compiled_models/ca_lmax1_nlayers2_mlp512_nlh_epoch150.nequip.pt2"],
     "CATW":  ["../../data/potentials/allegro/config_aware_test/compiled_models/catw_lmax1_nlayers2_mlp512_nlh_epoch146.nequip.pt2"],
}

# Optional: Provide multiple hyperparameter groups of loss-variant models.
# Example structure:
MODELS_BY_LOSS_GROUPS = {
     "lmax1_nlayers2": {
        "MSE":   ["../../data/potentials/allegro/config_aware_test/compiled_models/mse_lmax1_nlayers2_mlp512_zbl_epoch140.nequip.pt2"],
        "MSETW": ["../../data/potentials/allegro/config_aware_test/compiled_models/msetw_lmax1_nlayers2_mlp512_nlh.nequip.pt2"],
        "CA":    ["../../data/potentials/allegro/config_aware_test/compiled_models/ca_lmax1_nlayers2_mlp512_nlh_epoch150.nequip.pt2"],
        "CATW":  ["../../data/potentials/allegro/config_aware_test/compiled_models/catw_lmax1_nlayers2_mlp512_nlh_epoch146.nequip.pt2"],
     },
     "lmax2_nlayers2": {
         "MSE":   ["../../data/potentials/allegro/config_aware_test/compiled_models/mse_lmax2_nlayers2_mlp512_zbl_epoch107.nequip.pt2"],
         "MSETW": ["../../data/potentials/allegro/config_aware_test/compiled_models/msetw_lmax2_nlayers2_mlp512_nlh.nequip.pt2"],
         "CA":    ["../../data/potentials/allegro/config_aware_test/compiled_models/ca_lmax2_nlayers2_mlp512_zbl_epoch205.nequip.pt2"],
         "CATW":  ["../../data/potentials/allegro/config_aware_test/compiled_models/catw_lmax2_nlayers2_mlp512_zbl.nequip.pt2"],
     },
}

def parse_generation(text: str):
    text = text.lower()
    for pat in [r"generation[_\-]?(\d+)", r"\bgen[_\-]?(\d+)\b", r"\bg(\d+)\b"]:
        m = re.search(pat, text)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None

def parse_seed(text: str):
    text = text.lower()
    for pat in [r"\br(\d+)\b", r"\bseed[_\-]?(\d+)\b", r"\bs(\d+)\b"]:
        m = re.search(pat, text)
        if m:
            return int(m.group(1))
    return None

def parse_shielding(text: str):
    t = text.lower()
    if "nlh" in t: return "NLH"
    if "zbl" in t: return "ZBL"
    return "unknown"


def normalize_config_type(raw: str | None, atoms: 'Atoms') -> str:
    """Map raw config_type to benchmark sections with special groups used here.

    Returns one of the known section buckets; see SECTION_ORDER.
    """
    if raw is None:
        symbols = set(atoms.get_chemical_symbols()) if hasattr(atoms, "get_chemical_symbols") else set()
        return "Intermetallics" if len(symbols) > 1 else "Bulk crystals"
    s = str(raw).strip().lower()
    if "aa" in s or "adversarial" in s:
        return "Adversarial attacks (aa)"
    if any(k in s for k in ["comp-explore", "composition_explore", "comp_explore"]):
        return "Composition explore (comp-explore)"
    if any(k in s for k in ["liquid", "liq", "melt"]):
        return "Liquids & explore"
    if any(k in s for k in ["vacancy", "di-vacancy", "tri-vacancy", "divac", "trivac", "sia", "di-sia", "interstitial", "intst", "int"]):
        return "Point defects"
    if "neb" in s:
        return "NEB"
    if any(k in s for k in ["surface", "surf", "slab", "gamma", "γ"]):
        return "Surfaces & γ"
    if "phonon" in s:
        return "Phonon"
    if any(k in s for k in ["elastic", "elasticity", "strain", "stress_test"]):
        return "Elastic"
    if any(k in s for k in ["intermetal", "alloy", "mix", "im_", "intermetallic"]):
        return "Intermetallics"
    if any(k in s for k in ["bcc", "fcc", "hcp", "dia", "bulk", "crystal", "bcc_distorted"]):
        return "Bulk crystals"
    return "Other"


def is_pure_structure(at: 'Atoms') -> bool:
    try:
        return len(set(at.get_atomic_numbers())) == 1
    except Exception:
        return False


def voigt6_to_mat(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6, float).ravel()
    return np.array([[v6[0], v6[5], v6[4]],
                     [v6[5], v6[1], v6[3]],
                     [v6[4], v6[3], v6[2]]], float)


def mat_to_voigt6(M: np.ndarray) -> np.ndarray:
    return np.array([M[0,0], M[1,1], M[2,2], M[1,2], M[0,2], M[0,1]], float)


def read_structures(xyz_path: Path, max_structures: int | None = None) -> list['Atoms']:
    if not _ASE_AVAILABLE:
        raise RuntimeError("ASE is required to read XYZ files. Install with `pip install ase`.")
    selection = ":" if max_structures is None else f":{max_structures}"
    try:
        atoms_list = ase_read(str(xyz_path), index=selection, format="extxyz")
    except Exception:
        try:
            atoms_list = ase_read(str(xyz_path), index=selection)
        except Exception:
            atoms_list = ase_read(str(xyz_path), index=selection, format="xyz")
    if isinstance(atoms_list, Atoms):
        atoms_list = [atoms_list]
    return atoms_list


def extract_truth(at: 'Atoms') -> tuple[float | None, np.ndarray | None, np.ndarray | None, str, str]:
    """Extract reference energy (eV), forces (eV/Å), and stress (GPa as 3x3) from an Atoms object.

    Returns: (E_true_eV, F_true, S_true_GPa, section_name, purity_label)
    """
    info = getattr(at, "info", {}) or {}
    arrays = getattr(at, "arrays", {}) or {}

    # Energy
    e_keys = ["REF_energy", "energy", "total_energy", "E", "dft_energy", "ref_energy", "reference_energy", "energy_total", "Etot", "E_ref", "E0"]
    e_true = None
    for k in e_keys:
        if k in info and info[k] is not None:
            try:
                e_true = float(info[k])
                break
            except Exception:
                pass

    # Forces
    f_keys = ["REF_force", "forces", "F", "dft_forces", "ref_forces", "reference_forces", "forces_ref", "force"]
    f_true = None
    for k in f_keys:
        if k in arrays and arrays[k] is not None:
            arr = np.asarray(arrays[k])
            if arr.ndim == 2 and arr.shape[1] == 3:
                f_true = arr.astype(float)
                break

    # Stress (input often in eV/Å³ as 6-voigt or 3x3). Convert to GPa 3x3.
    s_keys = ["REF_stress", "stress", "dft_stress", "ref_stress", "reference_stress"]
    S_true_GPa = None
    # Try info first
    for k in s_keys:
        if k in info and info[k] is not None:
            val = np.asarray(info[k])
            try:
                if val.shape == (6,):
                    S_true_GPa = voigt6_to_mat(val) * GPA_PER_EVA3
                    break
                if val.shape == (3,3):
                    S_true_GPa = np.array(val, float) * GPA_PER_EVA3
                    break
                flat = val.reshape(-1)
                if flat.size == 9:
                    S_true_GPa = flat.reshape(3,3).astype(float) * GPA_PER_EVA3
                    break
            except Exception:
                pass
    if S_true_GPa is None:
        for k in s_keys:
            if k in arrays and arrays[k] is not None:
                arr = np.asarray(arrays[k])
                try:
                    if arr.ndim == 2 and arr.shape[1] in (6, 9):
                        flat = arr[0].reshape(-1)
                    elif arr.ndim == 1 and arr.size in (6, 9):
                        flat = arr.reshape(-1)
                    else:
                        flat = arr.reshape(-1)
                    if flat.size == 6:
                        S_true_GPa = voigt6_to_mat(flat) * GPA_PER_EVA3
                        break
                    if flat.size == 9:
                        S_true_GPa = flat.reshape(3,3).astype(float) * GPA_PER_EVA3
                        break
                except Exception:
                    pass

    section = normalize_config_type(info.get("config_type"), at)
    purity = "pure" if is_pure_structure(at) else "alloy"
    return e_true, f_true, S_true_GPa, section, purity


def _require_nequip() -> None:
    if not _NEQUIP_AVAILABLE:
        raise RuntimeError(
            "NequIP is required for model evaluation. Install with `pip install nequip`."
        )


def _load_calculator(model_path: str, device: str):
    _require_nequip()
    if model_path.endswith(".pt2"):
        return NequIPCalculator.from_compiled_model(compile_path=str(model_path), device=device)
    else:   
        return NequIPCalculator._from_packaged_model(package_path=str(model_path), device=device)


def _predict_single(calc, at: 'Atoms') -> tuple[float | None, np.ndarray | None, np.ndarray | None]:
    a = at.copy()
    a.calc = calc
    try:
        e = float(a.get_potential_energy())
    except Exception:
        e = None
    try:
        f = a.get_forces()
        f = np.asarray(f, float) if f is not None else None
    except Exception:
        f = None
    S = None
    try:
        s_voigt = a.get_stress(voigt=True)
        if s_voigt is not None:
            S = voigt6_to_mat(np.asarray(s_voigt, float)) * GPA_PER_EVA3
    except Exception:
        S = None
    return e, f, S


def _get_cache_key(models: list[str], atoms_list: list['Atoms'], device: str) -> str:
    """Generate a cache key based on model paths, structure count, and device."""
    # Use model paths, number of structures, and device for cache key
    model_str = "|".join(sorted(models))
    struct_count = len(atoms_list)
    return hashlib.md5(f"{model_str}_{struct_count}_{device}".encode()).hexdigest()

def _load_from_cache(cache_file: Path) -> tuple[list[float | None], list[np.ndarray | None], list[np.ndarray | None]] | None:
    """Load evaluation results from cache file."""
    try:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

def _save_to_cache(results: tuple, cache_file: Path) -> None:
    """Save evaluation results to cache file."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(results, f)

def save_precomputed_results(gen_df: pd.DataFrame, loss_df: pd.DataFrame, loss_groups_df: pd.DataFrame, outfile: Path) -> None:
    """Save precomputed results to JSON for distribution."""
    data = {
        "gen_eval_df": gen_df.to_dict(orient="records") if len(gen_df) else [],
        "loss_eval_df": loss_df.to_dict(orient="records") if len(loss_df) else [],
        "loss_groups_eval_df": loss_groups_df.to_dict(orient="records") if len(loss_groups_df) else [],
    }
    with open(outfile, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Saved precomputed results to {outfile}")

def load_precomputed_results(infile: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load precomputed results from JSON."""
    with open(infile, 'r') as f:
        data = json.load(f)
    gen_df = pd.DataFrame(data["gen_eval_df"]) if data["gen_eval_df"] else pd.DataFrame()
    loss_df = pd.DataFrame(data["loss_eval_df"]) if data["loss_eval_df"] else pd.DataFrame()
    loss_groups_df = pd.DataFrame(data["loss_groups_eval_df"]) if data["loss_groups_eval_df"] else pd.DataFrame()
    
    # Ensure proper data types
    for df in [gen_df, loss_df, loss_groups_df]:
        if len(df) and 'section' in df.columns:
            df['section'] = pd.Categorical(df['section'], categories=SECTION_ORDER, ordered=True)
    
    print(f"[INFO] Loaded precomputed results from {infile}")
    return gen_df, loss_df, loss_groups_df

def evaluate_ensemble(models: list[str], atoms_list: list['Atoms'], device: str, cache_dir: Path | None = None) -> tuple[list[float | None], list[np.ndarray | None], list[np.ndarray | None]]:
    if len(models) == 0:
        raise ValueError("No models provided for evaluation.")
    
    # Try to load from cache first
    if cache_dir is not None:
        cache_key = _get_cache_key(models, atoms_list, device)
        cache_file = cache_dir / f"eval_cache_{cache_key}.pkl"
        cached_results = _load_from_cache(cache_file)
        if cached_results is not None:
            print(f"[INFO] Loaded evaluation results from cache: {cache_file.name}")
            return cached_results
    
    # Compute results if not cached
    calcs = [_load_calculator(m, device=device) for m in models]
    energies = [None] * len(atoms_list)
    forces = [None] * len(atoms_list)
    stresses = [None] * len(atoms_list)
    # Collect per seed then average
    per_seed_E: list[list[float | None]] = []
    per_seed_F: list[list[np.ndarray | None]] = []
    per_seed_S: list[list[np.ndarray | None]] = []
    for calc in calcs:
        E_seed: list[float | None] = []
        F_seed: list[np.ndarray | None] = []
        S_seed: list[np.ndarray | None] = []
        for at in atoms_list:
            e, f, S = _predict_single(calc, at)
            E_seed.append(e)
            F_seed.append(f)
            S_seed.append(S)
        per_seed_E.append(E_seed)
        per_seed_F.append(F_seed)
        per_seed_S.append(S_seed)
    # Mean across seeds when possible
    for i in range(len(atoms_list)):
        e_vals = [e_list[i] for e_list in per_seed_E if e_list[i] is not None]
        f_vals = [f_list[i] for f_list in per_seed_F if f_list[i] is not None]
        s_vals = [s_list[i] for s_list in per_seed_S if s_list[i] is not None]
        energies[i] = float(np.mean(np.array(e_vals, float))) if len(e_vals) else None
        if len(f_vals):
            try:
                f_stack = np.stack(f_vals, axis=0)
                forces[i] = np.mean(f_stack, axis=0)
            except Exception:
                forces[i] = None
        else:
            forces[i] = None
        if len(s_vals):
            try:
                s_stack = np.stack(s_vals, axis=0)
                stresses[i] = np.mean(s_stack, axis=0)
            except Exception:
                stresses[i] = None
        else:
            stresses[i] = None
    
    # Save to cache if cache_dir provided
    if cache_dir is not None:
        cache_key = _get_cache_key(models, atoms_list, device)
        cache_file = cache_dir / f"eval_cache_{cache_key}.pkl"
        _save_to_cache((energies, forces, stresses), cache_file)
        print(f"[INFO] Saved evaluation results to cache: {cache_file.name}")
    
    return energies, forces, stresses


def compute_metrics_for_dataset(
    atoms_list: list['Atoms'],
    truths: list[tuple[float | None, np.ndarray | None, np.ndarray | None, str, str]],
    E_pred: list[float | None],
    F_pred: list[np.ndarray | None],
    S_pred_GPa: list[np.ndarray | None],
) -> pd.DataFrame:
    rows: list[dict] = []
    for idx, at in enumerate(atoms_list):
        e_true, f_true, S_true, section, purity = truths[idx]
        e_pred = E_pred[idx]
        f_pred = F_pred[idx]
        S_pred = S_pred_GPa[idx]
        nat = int(len(at))
        e_pa_abs_meV = np.nan
        if (e_true is not None) and (e_pred is not None) and nat > 0:
            e_pa_abs_meV = abs((e_pred - e_true) / nat) * 1000.0
        f_rmse = np.nan
        if (f_true is not None) and (f_pred is not None) and f_true.shape == f_pred.shape:
            diff = f_pred - f_true
            f_rmse = float(np.sqrt(np.mean(diff**2)))
        p_abs_GPa = np.nan
        sigma_rmse_GPa = np.nan
        von_mises_abs_GPa = np.nan
        if (S_true is not None) and (S_pred is not None) and S_true.shape == (3,3) and S_pred.shape == (3,3):
            # pressure (compression positive); ASE conv: pressure = -trace(S)/3
            p_pred = -np.trace(S_pred) / 3.0
            p_true = -np.trace(S_true) / 3.0
            p_abs_GPa = float(abs(p_pred - p_true))
            # RMSE over 6 Voigt components
            dv = mat_to_voigt6(S_pred) - mat_to_voigt6(S_true)
            sigma_rmse_GPa = float(np.sqrt(np.mean(dv**2)))
            # von Mises magnitudes
            identity_matrix = np.eye(3)
            s_pred = S_pred + p_pred * identity_matrix
            s_true = S_true + p_true * identity_matrix
            vm_pred = float(np.sqrt(1.5 * np.sum(s_pred * s_pred)))
            vm_true = float(np.sqrt(1.5 * np.sum(s_true * s_true)))
            von_mises_abs_GPa = float(abs(vm_pred - vm_true))
        rows.append({
            "structure_index": idx,
            "n": nat,
            "section": section,
            "purity": purity,
            "e_pa_abs_meV": e_pa_abs_meV,
            "f_rmse": f_rmse,
            "p_abs_GPa": p_abs_GPa,
            "sigma_rmse_GPa": sigma_rmse_GPa,
            "von_mises_abs_GPa": von_mises_abs_GPa,
        })
    return pd.DataFrame(rows)


def add_config_stress_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add two derived columns to df:

    - stress_all_GPa: alias for sigma_rmse_GPa (full stress tensor RMSE)
    - config_stress_metric_GPa: config-aware stress metric per SECTION_METRIC_MASK
        Uses stress RMSE where mask['s_mean'] is True; else pressure |dP| where
        mask['p_mean'] is True; otherwise NaN (ignored sections like NEB, Surfaces & γ).
    """
    if df is None or len(df) == 0:
        return df
    d = df.copy()
    # Ensure columns exist; if missing, just fill with NaN so downstream still works
    if "sigma_rmse_GPa" not in d.columns:
        d["sigma_rmse_GPa"] = np.nan
    if "p_abs_GPa" not in d.columns:
        d["p_abs_GPa"] = np.nan
    # Direct alias for clarity
    d["stress_all_GPa"] = d["sigma_rmse_GPa"].astype(float)
    # Build config-aware value per section
    def _choose_config_value(sec, s_val, p_val):
        sec_key = str(sec) if pd.notna(sec) else ""
        mask = SECTION_METRIC_MASK.get(sec_key, {})
        if bool(mask.get("s_mean", False)):
            return float(s_val) if np.isfinite(s_val) else np.nan
        if bool(mask.get("p_mean", False)):
            return float(p_val) if np.isfinite(p_val) else np.nan
        return np.nan
    d["config_stress_metric_GPa"] = [
        _choose_config_value(sec, s, p)
        for sec, s, p in zip(d.get("section", pd.Series([None]*len(d))), d["sigma_rmse_GPa"], d["p_abs_GPa"])  # type: ignore
    ]
    return d


def evaluate_by_generation(atoms_list: list['Atoms'], truths, device: str, cache_dir: Path | None = None) -> pd.DataFrame:
    if not MODELS_BY_GENERATION:
        return pd.DataFrame()
    all_rows: list[pd.DataFrame] = []
    for gen, models in sorted(MODELS_BY_GENERATION.items()):
        E, F, S = evaluate_ensemble(models, atoms_list, device=device, cache_dir=cache_dir)
        df = compute_metrics_for_dataset(atoms_list, truths, E, F, S)
        df["generation"] = int(gen)
        all_rows.append(df)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def evaluate_by_loss_variant(atoms_list: list['Atoms'], truths, device: str, cache_dir: Path | None = None) -> pd.DataFrame:
    if not MODELS_BY_LOSS:
        return pd.DataFrame()
    all_rows: list[pd.DataFrame] = []
    for loss, models in MODELS_BY_LOSS.items():
        E, F, S = evaluate_ensemble(models, atoms_list, device=device, cache_dir=cache_dir)
        df = compute_metrics_for_dataset(atoms_list, truths, E, F, S)
        df["loss_variant"] = str(loss)
        all_rows.append(df)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def evaluate_by_loss_groups(atoms_list: list['Atoms'], truths, device: str, cache_dir: Path | None = None) -> pd.DataFrame:
    """Evaluate multiple hyperparameter groups of loss variants.

    Returns a long-form DataFrame with columns including loss_group and loss_variant.
    """
    if not MODELS_BY_LOSS_GROUPS:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for group_name, loss_map in MODELS_BY_LOSS_GROUPS.items():
        for loss, models in loss_map.items():
            E, F, S = evaluate_ensemble(models, atoms_list, device=device, cache_dir=cache_dir)
            df = compute_metrics_for_dataset(atoms_list, truths, E, F, S)
            df["loss_group"] = str(group_name)
            df["loss_variant"] = str(loss)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _paired_permutation_pvalue(a: np.ndarray, b: np.ndarray, n_resamples: int = 10000, alternative: str = "two-sided", rng: np.random.Generator | None = None) -> float:
    """Paired permutation test via random sign flips on differences.

    Tests H0: mean(a - b) == 0; supports alternatives: 'two-sided', 'less', 'greater'.
    Returns an approximate p-value.
    """
    if rng is None:
        rng = np.random.default_rng()
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.nan
    obs = float(np.mean(d))
    # generate sign-flipped means
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, d.size))
    sims = (signs * d).mean(axis=1)
    if alternative == "less":
        p = (np.sum(sims <= obs) + 1.0) / (n_resamples + 1.0)
    elif alternative == "greater":
        p = (np.sum(sims >= obs) + 1.0) / (n_resamples + 1.0)
    else:
        p = (np.sum(np.abs(sims) >= abs(obs)) + 1.0) / (n_resamples + 1.0)
    return float(p)


def significance_tests_loss(df: pd.DataFrame, outdir: Path, scope_label: str, group_value: str | None = None, n_resamples: int = 10000) -> pd.DataFrame:
    """Run paired permutation tests comparing CATW vs other loss variants for each metric.

    df: per-structure records including 'structure_index' and 'loss_variant'. Optionally includes 'loss_group'.
    scope_label: label describing the scope (e.g., 'overall' or 'group').
    group_value: concrete group name when scope is per-group.
    Returns the results table and writes CSV once when called last; caller can concatenate and write.
    """
    metrics = [
        ("e_pa_abs_meV", "|dE|/atom (meV)"),
        ("f_rmse", "Force RMSE (eV/Å)"),
        ("p_abs_GPa", "|dP| (GPa)"),
        ("sigma_rmse_GPa", "Stress RMSE (GPa)"),
    ]
    targets = ["MSE", "MSETW", "CA"]
    rows: list[dict] = []
    # Pivot by loss_variant for quick selection
    for metric_key, _ in metrics:
        sub = df[["structure_index", "loss_variant", metric_key]].dropna(subset=[metric_key]).copy()
        # build series per variant
        try:
            catw = sub[sub["loss_variant"] == "CATW"][["structure_index", metric_key]].set_index("structure_index")[metric_key]
        except Exception:
            catw = None
        if catw is None or len(catw) == 0:
            continue
        for other in targets:
            other_series = sub[sub["loss_variant"] == other][["structure_index", metric_key]].set_index("structure_index")[metric_key]
            joined = pd.concat([catw, other_series], axis=1, keys=["catw", "other"]).dropna()
            if len(joined) == 0:
                continue
            a = joined["catw"].values
            b = joined["other"].values
            diff = a - b
            p_less = _paired_permutation_pvalue(a, b, n_resamples=n_resamples, alternative="less")
            rows.append({
                "scope": scope_label,
                "group": group_value,
                "metric": metric_key,
                "compare": f"CATW vs {other}",
                "n_pairs": int(len(diff)),
                "mean_catw": float(np.mean(a)),
                "mean_other": float(np.mean(b)),
                "mean_diff": float(np.mean(diff)),
                "median_diff": float(np.median(diff)),
                "p_perm_less": p_less,
            })
    res = pd.DataFrame(rows)
    if len(res):
        # simple Benjamini-Hochberg FDR across rows
        k = len(res)
        res = res.sort_values("p_perm_less").reset_index(drop=True)
        ranks = np.arange(1, k + 1)
        res["p_fdr"] = np.minimum.accumulate((res["p_perm_less"].values * k / ranks)[::-1])[::-1]
        res.to_csv(outdir / "loss_significance.csv", index=False)
    return res

def save_metric_panel(df: pd.DataFrame, group_col: str, outfile: Path, order: list, title_prefix: str) -> None:
    """Create a 2x2 panel for E, F, P, S metrics aggregated by group_col.

    df: DataFrame with columns [group_col, e_pa_abs_meV, f_rmse, p_abs_GPa, sigma_rmse_GPa]
    order: order of groups on x-axis
    """
    import matplotlib.pyplot as plt
    import numpy as np
    metrics = [
        ("e_pa_abs_meV", "|ΔE| per atom (meV)"),
        ("f_rmse", "Force RMSE (eV/Å)"),
        ("p_abs_GPa", "|ΔP| (GPa)"),
        ("sigma_rmse_GPa", "Stress RMSE (GPa)"),
    ]
    means = {}
    sems = {}
    grp = df.groupby(group_col, dropna=False)
    for key, _ in metrics:
        agg = grp[key].agg(["mean", "std", "count"]).reset_index()
        agg = agg.set_index(group_col)
        mean_vals = [float(agg.loc[g, "mean"]) if g in agg.index else np.nan for g in order]
        sem_vals = []
        for g in order:
            if g in agg.index:
                n = max(int(agg.loc[g, "count"]), 1)
                s = float(agg.loc[g, "std"]) if np.isfinite(agg.loc[g, "std"]) else 0.0
                sem_vals.append(s / np.sqrt(n))
            else:
                sem_vals.append(0.0)
        means[key] = mean_vals
        sems[key] = sem_vals
    plt.figure(figsize=(12, 6.5))
    for i, (key, ylabel) in enumerate(metrics, start=1):
        ax = plt.subplot(2, 2, i)
        x = np.arange(len(order))
        # Distinct colors per bar
        if len(PLOT_COLOR_PALETTE) > 0:
            colors = [PLOT_COLOR_PALETTE[j % len(PLOT_COLOR_PALETTE)] for j in range(len(order))]
        else:
            colors = None
        ax.bar(x, means[key], yerr=sems[key], capsize=3, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels([str(xv) for xv in order], rotation=0)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix} — {ylabel}", pad=2)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout(pad=0.25, w_pad=0.4, h_pad=0.45)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=200)
    plt.close()


def save_stress_compare_bars(
    df: pd.DataFrame,
    group_col: str,
    outfile: Path | str,
    order: list,
    title_prefix: str,
    metric_all: str = "stress_all_GPa",
    metric_cfg: str = "config_stress_metric_GPa",
    ylabel: str = "Error (GPa)",
):
    """Grouped bars comparing two stress metrics (all-stress vs config-aware) by group_col.

    df must include columns: group_col, metric_all, metric_cfg.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    if df is None or len(df) == 0:
        return
    d = df.copy()
    # aggregate means and sems
    grp = d.groupby(group_col, dropna=False)
    def _agg_for(col: str):
        a = grp[col].agg(["mean", "std", "count"]).reset_index().set_index(group_col)
        means = [float(a.loc[g, "mean"]) if g in a.index else np.nan for g in order]
        sems = []
        for g in order:
            if g in a.index:
                n = max(int(a.loc[g, "count"]), 1)
                s = float(a.loc[g, "std"]) if np.isfinite(a.loc[g, "std"]) else 0.0
                sems.append(s / np.sqrt(n))
            else:
                sems.append(0.0)
        return means, sems
    means_all, sem_all = _agg_for(metric_all)
    means_cfg, sem_cfg = _agg_for(metric_cfg)
    x = np.arange(len(order))
    width = 0.36
    plt.figure(figsize=(12, 4.5))
    color_all = PLOT_COLOR_PALETTE[0] if len(PLOT_COLOR_PALETTE) else None
    color_cfg = PLOT_COLOR_PALETTE[3] if len(PLOT_COLOR_PALETTE) > 3 else None
    plt.bar(x - width/2, means_all, yerr=sem_all, width=width, capsize=3, label="All-stress", color=color_all)
    plt.bar(x + width/2, means_cfg, yerr=sem_cfg, width=width, capsize=3, label="Config-aware", color=color_cfg)
    plt.xticks(x, [str(v) for v in order], rotation=0)
    plt.ylabel(ylabel)
    plt.title(f"{title_prefix} — stress prediction comparison")
    plt.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.legend(frameon=False)
    plt.tight_layout()
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=200)
    plt.close()


def save_four_metric_horizontal_by_loss_variant(
    df: pd.DataFrame,
    outfile: Path | str,
    variants_order: list[str] | None = None,
    elasticity_json_path: Path | str | None = None,
):
    """Horizontal grouped bar chart of four metrics by loss_variant.

    Metrics: e_pa_abs_meV, f_rmse, stress_all_GPa, config_stress_metric_GPa.
    Adds C44 and stability labels if elasticity_json_path is provided.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import json
    from pathlib import Path

    if df is None or len(df) == 0 or "loss_variant" not in df.columns:
        return

    d = df.copy()

    # Load elasticity data if provided
    elasticity_data = {}
    c44_dft = None
    c44_exp = None
    if elasticity_json_path is not None:
        try:
            with open(elasticity_json_path, "r") as f:
                elast_json = json.load(f)

            variant_mapping = [
                ("CATW", ["catw"]),
                ("MSETW", ["msetw"]),
                ("CA", ["ca"]),
                ("MSE", ["mse"]),
            ]

            matched_keys = set()

            for variant, search_terms in variant_mapping:
                for key, data in elast_json.items():
                    if key in matched_keys:
                        continue
                    backend = data.get("backend", "").lower()
                    if any(term in backend for term in search_terms):
                        c44 = data.get("cubic_constants_gpa", {}).get("C44")
                        c11 = data.get("cubic_constants_gpa", {}).get("C11")
                        c12 = data.get("cubic_constants_gpa", {}).get("C12")

                        # --- FIXED: nested replicate extraction ---
                        c44_replicates = []
                        replicate_details = data.get("replicates", {}).get("details", [])
                        for rep in replicate_details:
                            rep_c44 = None
                            if isinstance(rep, dict):
                                rep_c44 = rep.get("cubic_constants_gpa", {}).get("C44", rep.get("C44"))
                            if rep_c44 is not None:
                                c44_replicates.append(float(rep_c44))

                        c44_std = np.std(c44_replicates, ddof=1) if len(c44_replicates) > 1 else 0.0
                        c44_sem = c44_std / np.sqrt(len(c44_replicates)) if len(c44_replicates) > 0 else 0.0

                        if c44 is not None and c11 is not None and c12 is not None:
                            born_stable = (c44 > 0 and (c11 - c12) > 0 and c11 > 0)
                        else:
                            born_stable = False

                        elasticity_data[variant] = {
                            "C44": c44,
                            "C44_std": c44_std,
                            "C44_sem": c44_sem,
                            "C44_replicates": c44_replicates,
                            "C11": c11,
                            "C12": c12,
                            "born_stable": born_stable,
                            "kept_replicates": data.get("replicates", {}).get("kept", 0),
                            "dropped_replicates": data.get("replicates", {}).get("dropped", 0),
                        }
                        matched_keys.add(key)
                        break

            if "DFT" in elast_json:
                c44_dft = elast_json["DFT"].get("cubic_constants_gpa", {}).get("C44")
            if "PREDEXP" in elast_json:
                c44_exp = elast_json["PREDEXP"].get("cubic_constants_gpa", {}).get("C44")

        except Exception as e:
            print(f"[WARN] Could not load elasticity data: {e}")
            elasticity_data = {}

    # Aggregate metrics
    metrics = [
        ("e_pa_abs_meV", "|ΔE|/atom RMSE (meV)"),
        ("f_rmse", "Force RMSE (eV/Å)"),
        ("stress_all_GPa", "Stress RMSE (GPa)"),
        ("config_stress_metric_GPa", "Config-aware stress RMSE (GPa)"),
    ]
    canonical = ["MSE", "MSETW", "CA", "CATW"]
    if variants_order is None:
        variants_order = [v for v in canonical if v in d["loss_variant"].astype(str).unique().tolist()]
    if not variants_order:
        return

    grp = d.groupby("loss_variant", dropna=False)
    means, sems = {}, {}
    for key, _ in metrics:
        agg = grp[key].agg(["mean", "std", "count"]).reset_index().set_index("loss_variant")
        mvals, evals = [], []
        for v in variants_order:
            if v in agg.index:
                n = max(int(agg.loc[v, "count"]), 1)
                s = float(agg.loc[v, "std"]) if np.isfinite(agg.loc[v, "std"]) else 0.0
                mvals.append(float(agg.loc[v, "mean"]))
                evals.append(s / np.sqrt(n))
            else:
                mvals.append(np.nan)
                evals.append(0.0)
        means[key] = mvals
        sems[key] = evals

    # Prepare plotting data
    means_top = {
        "e_pa_abs_meV": means.get("e_pa_abs_meV", []),
        "f_rmse_meV_A": [m * 1000.0 for m in means.get("f_rmse", [])],
    }
    sems_top = {
        "e_pa_abs_meV": sems.get("e_pa_abs_meV", []),
        "f_rmse_meV_A": [s * 1000.0 for s in sems.get("f_rmse", [])],
    }

    means_bottom = {
        "stress_all_GPa": means.get("stress_all_GPa", []),
        "config_stress_metric_GPa": means.get("config_stress_metric_GPa", []),
    }
    sems_bottom = {
        "stress_all_GPa": sems.get("stress_all_GPa", []),
        "config_stress_metric_GPa": sems.get("config_stress_metric_GPa", []),
    }

    # Add C44
    if elasticity_data:
        c44_vals, c44_errs = [], []
        for v in variants_order:
            if v in elasticity_data and elasticity_data[v]["C44"] is not None:
                c44_vals.append(abs(elasticity_data[v]["C44"]))
                c44_errs.append(elasticity_data[v].get("C44_sem", 0.0))
            else:
                c44_vals.append(np.nan)
                c44_errs.append(0.0)
        means_bottom["c44_GPa"] = c44_vals
        sems_bottom["c44_GPa"] = c44_errs

    # Plot
    import matplotlib.pyplot as plt
    fig, ax_bottom = plt.subplots(figsize=(14, 6))
    ax_top = ax_bottom.twiny()

    y = np.arange(len(variants_order))
    k = 5 if elasticity_data else 4
    height_total = 0.8
    height_each = height_total / max(1, k)
    base_offsets = (np.arange(k) - (k - 1) / 2.0) * height_each

    metric_order = [
        ("e_pa_abs_meV", "|ΔE|/atom RMSE (meV)", "top"),
        ("f_rmse_meV_A", "Force RMSE (meV/A)", "top"),
        ("stress_all_GPa", "Stress RMSE (GPa)", "bottom"),
        ("config_stress_metric_GPa", "Config-aware stress RMSE (GPa)", "bottom"),
    ]
    colors = ["#2A33C3", "#A35D00", "#0B7285", "#8F2D56"]
    if elasticity_data:
        metric_order.append(("c44_GPa", "C₄₄ (GPa)", "bottom"))
        colors.append("#6E8B00")

    errkw_top = dict(elinewidth=1.25, ecolor="black", capsize=3, capthick=1.25, zorder=6)
    errkw_bottom = dict(elinewidth=1.25, ecolor="black", capsize=3, capthick=1.25, zorder=6)

    for i, (key, label, which) in enumerate(metric_order):
        ypos = y + base_offsets[i]
        if which == "top":
            vals, errs = means_top[key], sems_top[key]
            ax_top.barh(ypos, vals, xerr=errs, height=height_each * 0.95,
                        label=label, color=colors[i], error_kw=errkw_top)
            for val, err, ycoord in zip(vals, errs, ypos):
                if np.isfinite(val) and val > 0:
                    ax_top.text(val + err + 2, ycoord, f"{val:.1f}",
                                va="center", ha="left", fontsize=8, fontweight="bold")
        else:
            vals, errs = means_bottom[key], sems_bottom[key]
            ax_bottom.barh(ypos, vals, xerr=errs, height=height_each * 0.95,
                           label=label, color=colors[i], error_kw=errkw_bottom)
            if key == "c44_GPa":  # Explicit overlay for visibility
                ax_bottom.errorbar(vals, ypos, xerr=errs, fmt="none", **errkw_bottom)
            for val, err, ycoord in zip(vals, errs, ypos):
                if np.isfinite(val) and val > 0:
                    # Adjust C44 labels specifically to be closer to bars
                    if key == "c44_GPa" and elasticity_data:
                        x_pos = val * 1.08  # Closer to bar for C44 on log scale
                    elif elasticity_data:
                        x_pos = val * 1.3
                    else:
                        x_pos = val + err + 0.05
                    ax_bottom.text(x_pos, ycoord, f"{val:.2f}",
                                   va="center", ha="left", fontsize=8, fontweight="bold")

    # Y labels and colors
    y_labels = []
    for v in variants_order:
        label = v + "*" if elasticity_data and not elasticity_data.get(v, {}).get("born_stable", True) else v
        y_labels.append(label)
    ax_bottom.set_yticks(y)
    ax_bottom.set_yticklabels(y_labels, fontsize=12, fontweight="bold")
    if elasticity_data:
        for v, tick in zip(variants_order, ax_bottom.get_yticklabels()):
            if not elasticity_data.get(v, {}).get("born_stable", True):
                tick.set_color("red")

    ax_bottom.invert_yaxis()
    ax_top.set_ylim(ax_bottom.get_ylim())

    # Axis scales
    if elasticity_data:
        ax_bottom.set_xscale("log")
        ax_bottom.set_xlim(1e-2, 1e2)
        ax_top.set_xlim(0, 200)
        ylim = ax_bottom.get_ylim()
        ymid = (ylim[0] + ylim[1]) / 2
        if c44_dft:
            ax_bottom.axvline(c44_dft, color="black", ls="--", lw=2, alpha=0.8)
            ax_bottom.text(c44_dft * 0.89, ymid, f"DFT C₄₄\n{c44_dft:.1f} GPa",
                           rotation=90, va="center", ha="right", fontsize=10,
                           fontweight="bold", color="black",
                           bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.9))
        if c44_exp:
            ax_bottom.axvline(c44_exp, color="dimgray", ls="--", lw=2, alpha=0.8)
            ax_bottom.text(c44_exp * 1.15, ymid, f"Exp C₄₄\n{c44_exp:.1f} GPa",
                           rotation=90, va="center", ha="left", fontsize=10,
                           fontweight="bold", color="dimgray",
                           bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="dimgray", alpha=0.9))
    else:
        ax_top.set_xlim(0, 200)
        ax_bottom.set_xlim(0, 2.5)

    ax_top.set_xlabel("|ΔE|/atom RMSE (meV), F RMSE (meV/A)", fontsize=12, fontweight="bold")
    ax_bottom.set_xlabel("RMSE (GPa)", fontsize=12, fontweight="bold")
    ax_bottom.grid(True, axis="x", linestyle=":", alpha=0.6)

    # --- FIXED: Figure-level legend (no clipping) ---
    handles_bottom, labels_bottom = ax_bottom.get_legend_handles_labels()
    handles_top, labels_top = ax_top.get_legend_handles_labels()
    handles, labels = handles_top + handles_bottom, labels_top + labels_bottom

    legend_title = None
    if elasticity_data and any(
        not elasticity_data.get(v, {}).get("born_stable", True) for v in variants_order
    ):
        legend_title = "* = Born unstable"

    legend = fig.legend(
         handles, labels, title=legend_title, loc="upper right",
         bbox_to_anchor=(1.0, 0.98),
         frameon=True, fancybox=False, edgecolor="black",
         facecolor="white", framealpha=1.0,
         handlelength=2.5, handleheight=1.2, labelspacing=0.6,
     )
    legend.set_zorder(1000)
    legend.set_in_layout(False)

    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_four_metric_vertical_by_loss_variant(
    df: pd.DataFrame,
    outfile: Path | str,
    variants_order: list[str] | None = None,
    elasticity_json_path: Path | str | None = None,
):
    """Vertical grouped bar chart of four metrics by loss_variant.

    Metrics: e_pa_abs_meV, f_rmse, stress_all_GPa, config_stress_metric_GPa, C44.
    Adds C44 and stability labels if elasticity_json_path is provided.
    Variants on X-axis, two Y-axes (left for E/F, right for stress/C44 with log scale).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import json
    from pathlib import Path

    if df is None or len(df) == 0 or "loss_variant" not in df.columns:
        return

    d = df.copy()

    # Load elasticity data if provided
    elasticity_data = {}
    c44_dft = None
    c44_exp = None
    if elasticity_json_path is not None:
        try:
            with open(elasticity_json_path, "r") as f:
                elast_json = json.load(f)

            variant_mapping = [
                ("CATW", ["catw"]),
                ("MSETW", ["msetw"]),
                ("CA", ["ca"]),
                ("MSE", ["mse"]),
            ]

            matched_keys = set()

            for variant, search_terms in variant_mapping:
                for key, data in elast_json.items():
                    if key in matched_keys:
                        continue
                    backend = data.get("backend", "").lower()
                    if any(term in backend for term in search_terms):
                        c44 = data.get("cubic_constants_gpa", {}).get("C44")
                        c11 = data.get("cubic_constants_gpa", {}).get("C11")
                        c12 = data.get("cubic_constants_gpa", {}).get("C12")

                        c44_replicates = []
                        replicate_details = data.get("replicates", {}).get("details", [])
                        for rep in replicate_details:
                            rep_c44 = None
                            if isinstance(rep, dict):
                                rep_c44 = rep.get("cubic_constants_gpa", {}).get("C44", rep.get("C44"))
                            if rep_c44 is not None:
                                c44_replicates.append(float(rep_c44))

                        c44_std = np.std(c44_replicates, ddof=1) if len(c44_replicates) > 1 else 0.0
                        c44_sem = c44_std / np.sqrt(len(c44_replicates)) if len(c44_replicates) > 0 else 0.0

                        if c44 is not None and c11 is not None and c12 is not None:
                            born_stable = (c44 > 0 and (c11 - c12) > 0 and c11 > 0)
                        else:
                            born_stable = False

                        elasticity_data[variant] = {
                            "C44": c44,
                            "C44_std": c44_std,
                            "C44_sem": c44_sem,
                            "C44_replicates": c44_replicates,
                            "C11": c11,
                            "C12": c12,
                            "born_stable": born_stable,
                            "kept_replicates": data.get("replicates", {}).get("kept", 0),
                            "dropped_replicates": data.get("replicates", {}).get("dropped", 0),
                        }
                        matched_keys.add(key)
                        break

            if "DFT" in elast_json:
                c44_dft = elast_json["DFT"].get("cubic_constants_gpa", {}).get("C44")
            if "PREDEXP" in elast_json:
                c44_exp = elast_json["PREDEXP"].get("cubic_constants_gpa", {}).get("C44")

        except Exception as e:
            print(f"[WARN] Could not load elasticity data: {e}")
            elasticity_data = {}

    # Aggregate metrics
    metrics = [
        ("e_pa_abs_meV", "|ΔE|/atom (meV)"),
        ("f_rmse", "Force RMSE (eV/Å)"),
        ("stress_all_GPa", "Stress RMSE (GPa)"),
        ("config_stress_metric_GPa", "Config-aware stress (GPa)"),
    ]
    canonical = ["MSE", "MSETW", "CA", "CATW"]
    if variants_order is None:
        variants_order = [v for v in canonical if v in d["loss_variant"].astype(str).unique().tolist()]
    if not variants_order:
        return

    grp = d.groupby("loss_variant", dropna=False)
    means, sems = {}, {}
    for key, _ in metrics:
        agg = grp[key].agg(["mean", "std", "count"]).reset_index().set_index("loss_variant")
        mvals, evals = [], []
        for v in variants_order:
            if v in agg.index:
                n = max(int(agg.loc[v, "count"]), 1)
                s = float(agg.loc[v, "std"]) if np.isfinite(agg.loc[v, "std"]) else 0.0
                mvals.append(float(agg.loc[v, "mean"]))
                evals.append(s / np.sqrt(n))
            else:
                mvals.append(np.nan)
                evals.append(0.0)
        means[key] = mvals
        sems[key] = evals

    # Prepare plotting data - convert force to meV/Å for consistency
    means_left = {
        "e_pa_abs_meV": means.get("e_pa_abs_meV", []),  # Corrected variable name
        "f_rmse_meV_A": [m * 1000.0 for m in means.get("f_rmse", [])],
    }
    sems_left = {
        "e_pa_abs_meV": sems.get("e_pa_abs_meV", []),
        "f_rmse_meV_A": [s * 1000.0 for s in sems.get("f_rmse", [])],
    }

    means_right = {
        "stress_all_GPa": means.get("stress_all_GPa", []),
        "config_stress_metric_GPa": means.get("config_stress_metric_GPa", []),
    }
    sems_right = {
        "stress_all_GPa": sems.get("stress_all_GPa", []),
        "config_stress_metric_GPa": sems.get("config_stress_metric_GPa", []),
    }

    # Add C44
    if elasticity_data:
        c44_vals, c44_errs = [], []
        for v in variants_order:
            if v in elasticity_data and elasticity_data[v]["C44"] is not None:
                c44_vals.append(abs(elasticity_data[v]["C44"]))
                c44_errs.append(elasticity_data[v].get("C44_sem", 0.0))
            else:
                c44_vals.append(np.nan)
                c44_errs.append(0.0)
        means_right["c44_GPa"] = c44_vals
        sems_right["c44_GPa"] = c44_errs

    # Plot setup
    fig, ax_left = plt.subplots(figsize=(14, 10))  # Wider figure
    ax_right = ax_left.twinx()

    x = np.arange(len(variants_order))
    
    # === BAR WIDTH ADJUSTMENT ===
    # Adjust these values to change bar spacing:
    # - total_width: overall width of all bars per variant (0.0 to 1.0, where 1.0 = no gap between variants)
    # - Increase total_width to make bars wider, decrease to make them narrower
    total_width = 0.85  # ADJUST THIS: 0.6=narrow, 0.85=wide, 1.0=touching next variant
    k = 5 if elasticity_data else 4  # number of metrics
    width_each = total_width / max(1, k)
    base_offsets = (np.arange(k) - (k - 1) / 2.0) * width_each
    # ============================

    # Metric order and colors (using global PLOT_COLOR_PALETTE)
    colors = ["#2A33C3", "#A35D00", "#0B7285", "#8F2D56"]  # Energy, Force, Stress_all, Config_stress
    if elasticity_data:
        colors.append("#6E8B00")  # C44

    metric_order = [
        ("e_pa_abs_meV", "|ΔE|/atom RMSE (meV)", "left", colors[0]),
        ("f_rmse_meV_A", "Force RMSE (meV/Å)", "left", colors[1]),
        ("stress_all_GPa", "Stress RMSE (GPa)", "right", colors[2]),
        ("config_stress_metric_GPa", "Config-aware stress RMSE (GPa)", "right", colors[3]),
    ]
    if elasticity_data:
        metric_order.append(("c44_GPa", "C₄₄ (GPa)", "right", colors[4]))

    # Error bar styling (all text sizes +2 points)
    errkw = dict(elinewidth=1.25, ecolor="black", capsize=5, capthick=1.25, zorder=6)

    # Draw bars
    for i, (key, label, which_ax, color) in enumerate(metric_order):
        xpos = x + base_offsets[i]
        if which_ax == "left":
            vals, errs = means_left[key], sems_left[key]
            ax = ax_left
        else:
            vals, errs = means_right[key], sems_right[key]
            ax = ax_right

        ax.bar(xpos, vals, yerr=errs, width=width_each * 0.95,
               label=label, color=color, error_kw=errkw, zorder=5)

        # Value labels
        for val, err, xcoord in zip(vals, errs, xpos):
            if np.isfinite(val) and val > 0:
                # === VALUE LABEL POSITION ADJUSTMENT ===
                # Change label_offset to adjust vertical position of value labels
                # Positive = above bar, negative = inside bar
                label_offset = err + (val * 0.05)  # ADJUST THIS: on top of bar
                # For inside bars, use: label_offset = -val * 0.15
                # ======================================
                
                y_pos = val + label_offset
                
                # Format based on magnitude
                if key == "c44_GPa":
                    label_text = f"{val:.2f}"
                elif val < 10:
                    label_text = f"{val:.2f}"
                else:
                    label_text = f"{val:.1f}"
                
                ax.text(xcoord, y_pos, label_text,
                       ha="center", va="bottom", fontsize=12, fontweight="bold")  # +4 from 8

    # X-axis labels with stability markers
    x_labels = []
    for v in variants_order:
        label = v + "*" if elasticity_data and not elasticity_data.get(v, {}).get("born_stable", True) else v
        x_labels.append(label)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(x_labels, fontsize=14, fontweight="bold")  # +2 from 12
    
    # Color unstable variants in red
    if elasticity_data:
        for v, tick in zip(variants_order, ax_left.get_xticklabels()):
            if not elasticity_data.get(v, {}).get("born_stable", True):
                tick.set_color("red")

    # Y-axis scales and labels
    # Left axis (Energy and Force) - linear scale
    ax_left.set_ylim(0, 200)
    ax_left.set_ylabel("Energy & Force error", fontsize=14, fontweight="bold",
                       color="black")  # +2 from 12
    ax_left.tick_params(axis='y', labelsize=12, labelcolor="black")  # +2 from 10
    
    # Right axis (Stress and C44) - log scale
    if elasticity_data:
        ax_right.set_yscale("log")
        ax_right.set_ylim(1e-2, 1e2)
    else:
        ax_right.set_ylim(0, 2.5)
    ax_right.set_ylabel("Stress & elasticity (GPa)", fontsize=14, fontweight="bold",
                        color="black")  # +2 from 12
    ax_right.tick_params(axis='y', labelsize=12, labelcolor="black")  # +2 from 10

    # Horizontal reference lines for DFT and Exp C44
    if elasticity_data and c44_dft:
        ax_right.axhline(c44_dft, color="black", ls="--", lw=2, alpha=0.8, zorder=1)
        ax_right.text(len(variants_order) - 1.75, c44_dft * 0.7, 
                     f"DFT C₄₄: {c44_dft:.1f} GPa",
                     ha="right", va="bottom", fontsize=12, fontweight="bold",  # +2 from 10
                     color="black",
                     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.9))
    
    if elasticity_data and c44_exp:
        ax_right.axhline(c44_exp, color="dimgray", ls="--", lw=2, alpha=0.8, zorder=1)
        ax_right.text(len(variants_order) - 1.75, c44_exp * 1.45,
                     f"Exp C₄₄: {c44_exp:.1f} GPa",
                     ha="right", va="top", fontsize=12, fontweight="bold",  # +2 from 10
                     color="dimgray",
                     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="dimgray", alpha=0.9))

    # Grid
    ax_left.grid(True, axis="y", linestyle=":", alpha=0.6, zorder=0)

    # Legend
    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    handles, labels = handles_left + handles_right, labels_left + labels_right

    legend_title = None
    if elasticity_data and any(
        not elasticity_data.get(v, {}).get("born_stable", True) for v in variants_order
    ):
        legend_title = "* = Born unstable (red)"

    legend = fig.legend(
        handles, labels, title=legend_title, loc="upper left",
        bbox_to_anchor=(0.055, 0.98),  # Moved left from 0.12 to 0.02
        frameon=True, fancybox=False, edgecolor="black",
        facecolor="white", framealpha=1.0,
        fontsize=12, title_fontsize=12,  # +2 from 10
        handlelength=3.45, handleheight=1.2, labelspacing=0.7,  # 15% wider: handlelength 3.0→3.45, spacing 0.7→0.8
        columnspacing=1.7,  # Add spacing between columns if needed
    )
    legend.set_zorder(1000)
    
    # Color the legend title red if there are unstable variants
    if legend_title is not None:
        legend.get_title().set_color("red")
        legend.get_title().set_weight("bold")

    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_arch(text: str):
    t = text.lower()
    m1 = re.search(r"lmax(\d+)", t)
    m2 = re.search(r"nlayers(\d+)", t)
    if m1 and m2: return f"lmax{m1.group(1)}-nlayers{m2.group(1)}"
    return "unknown"

def parse_loss_variant(text: str):
    t = text.lower()
    has_ca = bool(re.search(r"config[_\-]?aware|\bca\b", t))
    has_tw = bool(re.search(r"tuned[_\-]?weights|\btw\b", t))
    has_msetw = "msetw" in t or ("mse" in t and has_tw and not has_ca)
    has_mse = "mse" in t
    if has_ca and has_tw: return "CATW"
    if has_ca: return "CA"
    if has_msetw: return "MSETW"
    if has_mse: return "MSE"
    if has_tw and not has_ca: return "MSETW"
    return "unknown"

def extract_if_tar(input_path: Path) -> Path:
    if input_path.is_file() and str(input_path).endswith((".tar.gz",".tgz")):
        out = input_path.with_suffix("").with_suffix("")
        out = Path(str(out) + "_extracted")
        out.mkdir(exist_ok=True, parents=True)
        with tarfile.open(input_path, "r:gz") as tar:
            tar.extractall(path=out)
        # if exactly one dir inside, return it
        entries = [p for p in out.iterdir() if p.is_dir()]
        return entries[0] if len(entries)==1 else out
    return input_path

def load_runs(base: Path) -> pd.DataFrame:
    csvs = sorted(base.glob("**/per_structure.csv"))
    if not csvs:
        raise FileNotFoundError(f"No per_structure.csv under {base}")
    rows = []
    for p in csvs:
        run_dir = p.parent
        run_name = run_dir.name
        hint = "/".join([run_dir.parent.name, run_dir.name])
        df = pd.read_csv(p)
        df["variant"] = run_name
        df["generation"] = parse_generation(hint)
        df["seed"] = parse_seed(hint)
        df["shielding"] = parse_shielding(hint)
        df["arch"] = parse_arch(hint)
        df["loss_variant"] = parse_loss_variant(hint)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    if "section" in out.columns:
        out["section"] = pd.Categorical(out["section"], categories=SECTION_ORDER, ordered=True)
    return out

def summarize_by(df: pd.DataFrame, by_cols):
    g = df.groupby(by_cols, dropna=False)
    out = g.agg(
        n=("f_rmse", "count"),
        e_mean=("e_pa_abs_meV", "mean"),  e_std=("e_pa_abs_meV", "std"),
        f_mean=("f_rmse", "mean"),        f_std=("f_rmse", "std"),
        p_mean=("p_abs_GPa", "mean"),     p_std=("p_abs_GPa", "std"),
        s_mean=("sigma_rmse_GPa", "mean"),s_std=("sigma_rmse_GPa", "std"),
        vm_mean=("von_mises_abs_GPa", "mean"), vm_std=("von_mises_abs_GPa", "std"),
    ).reset_index()
    return out

def config_aware_composite(sum_df: pd.DataFrame, by_cols, metrics=("f_mean","e_mean","s_mean","p_mean"),
                           metric_weights=None):
    if metric_weights is None:
        metric_weights = {"f_mean":0.7, "e_mean":0.3, "s_mean":1.0, "p_mean":1.0}
    id_cols = [c for c in ["variant","generation","seed","loss_variant"] if c in sum_df.columns]
    group_cols = [c for c in by_cols if c not in id_cols]
    rows = []
    for grp_vals, sub in sum_df.groupby(group_cols, dropna=False):
        if "section" in sub.columns:
            if isinstance(grp_vals, tuple):
                sec_val = grp_vals[group_cols.index("section")]
            else:
                sec_val = sub["section"].iloc[0]
            sec_val = str(sec_val)
            mask = SECTION_METRIC_MASK.get(sec_val, {})
        else:
            mask = {}
        zparts = {}
        for m in metrics:
            vals = sub[m].astype(float).values if m in sub.columns else np.full(len(sub), np.nan)
            if not bool(mask.get(m, True)):
                zparts[m] = np.zeros_like(vals, dtype=float)
                continue
            mu = float(np.nanmean(vals))
            sd = float(np.nanstd(vals))
            if not np.isfinite(sd) or sd < 1e-12: sd = 1.0
            zparts[m] = (vals - mu) / sd
        active = [m for m in metrics if bool(mask.get(m, True))]
        denom = sum(float(metric_weights.get(m, 0.0)) for m in active)
        wnorm = {m: (float(metric_weights.get(m,0.0))/denom) if denom>0 else 0.0 for m in metrics}
        combo = np.zeros(len(sub))
        for m in metrics:
            combo += zparts[m] * float(wnorm.get(m,0.0))
        tmp = sub[id_cols].copy().reset_index(drop=True)
        tmp["combo"] = combo
        rows.append(tmp)
    allrows = pd.concat(rows, ignore_index=True)
    grp = allrows.groupby(id_cols, dropna=False)
    scores = grp["combo"].apply(lambda s: float(np.nanmean(s.values))).reset_index(name="score")
    return scores.sort_values("score")

def save_bar_with_err(x_labels, means, errs, ylabel, title, outfile):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(8,4))
    x = np.arange(len(x_labels))
    plt.bar(x, means, yerr=errs, capsize=3)
    plt.xticks(x, x_labels, rotation=0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()

def save_grouped_bar(data_df, x_col, group_col, value_col, x_order=None, group_order=None,
                     ylabel="", title="", outfile="plot.png", rotate_xticks=30):
    import matplotlib.pyplot as plt
    import numpy as np
    d = data_df.copy()
    if x_order is None: x_order = list(d[x_col].dropna().unique())
    if group_order is None: group_order = list(d[group_col].dropna().unique())
    x = np.arange(len(x_order))
    width = 0.8 / max(1, len(group_order))
    offsets = (np.arange(len(group_order)) - (len(group_order)-1)/2.0) * width
    plt.figure(figsize=(12,5))
    for i, grp in enumerate(group_order):
        sub = d[d[group_col] == grp]
        y = []
        for xv in x_order:
            row = sub[sub[x_col] == xv]
            y.append(float(row[value_col].values[0]) if len(row) else np.nan)
        plt.bar(x + offsets[i], y, width=width, label=str(grp))
    plt.xticks(x, [str(v) for v in x_order], rotation=rotate_xticks, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.legend(frameon=False, title=group_col)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def save_grouped_bar_with_overlay(
    data_df: pd.DataFrame,
    x_col: str,
    group_col: str,
    overlay_col: str,
    value_col: str,
    x_order: list | None = None,
    group_order: list | None = None,
    overlay_order: list | None = None,
    ylabel: str = "",
    title: str = "",
    outfile: Path | str = "plot.png",
    rotate_xticks: int = 30,
):
    """Grouped bars with an overlay grouping (e.g., loss_variant as group, hyperparam group as overlay)."""
    import matplotlib.pyplot as plt
    import numpy as np
    d = data_df.copy()
    if x_order is None: x_order = list(d[x_col].dropna().unique())
    if group_order is None: group_order = list(d[group_col].dropna().unique())
    if overlay_order is None: overlay_order = list(d[overlay_col].dropna().unique())
    x = np.arange(len(x_order))
    # width partition: allocate width per overlay, then per group within overlay
    overlays = len(overlay_order)
    groups = len(group_order)
    width_total = 0.85
    width_overlay = width_total / max(1, overlays)
    width_group = width_overlay / max(1, groups)
    base_offsets = (np.arange(overlays) - (overlays - 1)/2.0) * width_overlay
    plt.figure(figsize=(12, 5))
    for j, ov in enumerate(overlay_order):
        for i, grp in enumerate(group_order):
            sub = d[(d[group_col] == grp) & (d[overlay_col] == ov)]
            y = []
            for xv in x_order:
                row = sub[sub[x_col] == xv]
                y.append(float(row[value_col].values[0]) if len(row) else np.nan)
            xpos = x + base_offsets[j] + (i - (groups - 1)/2.0) * width_group
            label = f"{ov}:{grp}"
            plt.bar(xpos, y, width=width_group * 0.95, label=label)
    plt.xticks(x, [str(v) for v in x_order], rotation=rotate_xticks, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.legend(frameon=False, title=f"{overlay_col}:{group_col}")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()

def collect_subset(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    if not names:
        return df
    mask = df["variant"].astype(str).isin(names)
    return df[mask].copy()

def parse_list_map(spec: str) -> dict:
    """
    Parse "10=runA,runB;7=runC" into {10:[runA,runB], 7:[runC]}
    or "MSE=runX;CATW=runY" into {"MSE":[runX], "CATW":[runY]}.
    """
    out = {}
    if not spec:
        return out
    for part in spec.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        vals = [t.strip() for t in v.split(",") if t.strip()]
        # cast generation keys to int when possible
        try:
            k_cast = int(k)
        except Exception:
            k_cast = k
        out[k_cast] = vals
    return out

def main():
    ap = argparse.ArgumentParser(description="Streamlined comparison: generations + loss variants.")
    ap.add_argument("path", nargs="?", default=None, help="Path to runs directory or .tar.gz archive (omit if using --xyz evaluation)")
    ap.add_argument("--outdir", default="streamlined_analysis", help="Where to write outputs")
    ap.add_argument("--gens", default=None, help='Explicit generation groups, e.g. "0=runA,runB;7=runC;10=runD,runE"')
    ap.add_argument("--loss", default=None, help='Explicit loss groups, e.g. "MSE=run1;MSETW=run2;CA=run3;CATW=run4"')
    ap.add_argument("--prefer-gen", type=int, default=10, help="Preferred generation for loss-variant comparison")
    # Optional evaluation mode from XYZ + hardcoded model paths
    ap.add_argument("--xyz", type=str, default=None, help="Path to XYZ or EXTXYZ dataset to evaluate directly (uses hardcoded MODELS_BY_*)")
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu","cuda"], help="Device for model evaluation")
    ap.add_argument("--max-structures", type=int, default=None, help="Optional limit on number of structures from XYZ")
    ap.add_argument("--elasticity_tensors", type=str, default=None, help="Path to JSON file with elastic tensor data for models")
    # Precomputed data support
    ap.add_argument("--precomputed-data", type=str, default=None, help="Path to precomputed results JSON (skips evaluation)")
    ap.add_argument("--save-precomputed", type=str, default=None, help="Save evaluation results to JSON for distribution (after evaluation)")
    args = ap.parse_args()

    # Set defaults for convenience
    if args.precomputed_data is None:
        default_precomputed = Path("precomputed_data.json")
        if default_precomputed.exists():
            args.precomputed_data = str(default_precomputed)
            print(f"[INFO] Found default precomputed data: {default_precomputed}")

    if args.elasticity_tensors is None:
        default_elasticity = Path("allegro_elastic_tensors_summary.json")
        if default_elasticity.exists():
            args.elasticity_tensors = str(default_elasticity)
            print(f"[INFO] Found default elasticity data: {default_elasticity}")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Initialize DataFrames (used by both evaluation and precomputed modes)
    gen_eval_df = pd.DataFrame()
    loss_eval_df = pd.DataFrame()
    loss_groups_eval_df = pd.DataFrame()
    precomputed_mode = False

    # Check for precomputed data first
    if args.precomputed_data is not None:
        print(f"[INFO] Using precomputed data from {args.precomputed_data}")
        gen_eval_df, loss_eval_df, loss_groups_eval_df = load_precomputed_results(Path(args.precomputed_data))
        
        # Add config stress metrics if not already present
        gen_eval_df = add_config_stress_metrics(gen_eval_df) if len(gen_eval_df) else gen_eval_df
        loss_eval_df = add_config_stress_metrics(loss_eval_df) if len(loss_eval_df) else loss_eval_df
        loss_groups_eval_df = add_config_stress_metrics(loss_groups_eval_df) if len(loss_groups_eval_df) else loss_groups_eval_df
        
        # Save as CSV for reference
        if len(gen_eval_df):
            gen_eval_df.to_csv(outdir / "eval_per_structure_by_generation.csv", index=False)
        if len(loss_eval_df):
            loss_eval_df.to_csv(outdir / "eval_per_structure_by_loss.csv", index=False)
        if len(loss_groups_eval_df):
            loss_groups_eval_df.to_csv(outdir / "eval_per_structure_by_loss_groups.csv", index=False)
        
        # Skip to plotting section
        precomputed_mode = True

    # If XYZ is provided, run direct evaluation using hardcoded model paths and write panels
    if args.xyz is not None and not precomputed_mode:
        print(f"[INFO] Loading structures from: {args.xyz}")
        atoms_list = read_structures(Path(args.xyz), max_structures=args.max_structures)
        if len(atoms_list) == 0:
            raise RuntimeError(f"No structures loaded from {args.xyz}")
        truths = [extract_truth(a) for a in atoms_list]

        # Generations panel
        if not MODELS_BY_GENERATION and not MODELS_BY_LOSS:
            print("[WARN] MODELS_BY_GENERATION and MODELS_BY_LOSS are both empty; nothing to evaluate.")
        gen_eval_df = evaluate_by_generation(atoms_list, truths, device=args.device, cache_dir=outdir)
        if len(gen_eval_df):
            print(f"[INFO] Evaluated generations: {sorted(gen_eval_df['generation'].dropna().unique().tolist())}")
            gen_eval_df = add_config_stress_metrics(gen_eval_df)
            gen_eval_df.to_csv(outdir / "eval_per_structure_by_generation.csv", index=False)
            gens_order = sorted([int(g) for g in gen_eval_df["generation"].dropna().unique()])
            save_metric_panel(gen_eval_df, group_col="generation", outfile=outdir / "panel_generations_metrics.png", order=gens_order, title_prefix="Generations")
            # Stress comparison: all-stress vs config-aware by generation
            if len(gens_order):
                save_stress_compare_bars(
                    gen_eval_df,
                    group_col="generation",
                    outfile=outdir / "generations_stress_comparison.png",
                    order=gens_order,
                    title_prefix="Generations",
                )

        # Loss-variants panel
        # Loss variants (single group)
        loss_eval_df = evaluate_by_loss_variant(atoms_list, truths, device=args.device, cache_dir=outdir)
        # Loss variants (multi-group hyperparams)
        loss_groups_eval_df = evaluate_by_loss_groups(atoms_list, truths, device=args.device, cache_dir=outdir)
        if len(loss_eval_df):
            print(f"[INFO] Evaluated loss variants: {sorted(set(loss_eval_df['loss_variant'].astype(str)))}")
            loss_eval_df = add_config_stress_metrics(loss_eval_df)
            loss_eval_df.to_csv(outdir / "eval_per_structure_by_loss.csv", index=False)
            canonical = ["MSE","MSETW","CA","CATW","unknown"]
            lv_order = [v for v in canonical if v in set(loss_eval_df["loss_variant"].astype(str))]
            if len(lv_order):
                save_metric_panel(loss_eval_df, group_col="loss_variant", outfile=outdir / "panel_loss_variants_metrics.png", order=lv_order, title_prefix="Loss variants")
                # Stress comparison by loss variant
                save_stress_compare_bars(
                    loss_eval_df,
                    group_col="loss_variant",
                    outfile=outdir / "loss_variants_stress_comparison.png",
                    order=lv_order,
                    title_prefix="Loss variants",
                )
                # 4-metric horizontal grouped chart by loss variant
                save_four_metric_horizontal_by_loss_variant(
                    loss_eval_df,
                    outfile=outdir / "loss_variants_four_metric_horizontal.png",
                    variants_order=[v for v in ["MSE","MSETW","CA","CATW"] if v in lv_order],
                    elasticity_json_path=Path(args.elasticity_tensors) if args.elasticity_tensors else None,
                )
                # 4-metric vertical grouped chart by loss variant
                save_four_metric_vertical_by_loss_variant(
                    loss_eval_df,
                    outfile=outdir / "loss_variants_four_metric_vertical.png",
                    variants_order=[v for v in ["MSE","MSETW","CA","CATW"] if v in lv_order],
                    elasticity_json_path=Path(args.elasticity_tensors) if args.elasticity_tensors else None,
                )
            # Significance tests overall
            sig_overall = significance_tests_loss(loss_eval_df, outdir=outdir, scope_label="overall")
            if len(sig_overall):
                print("[INFO] Wrote loss_significance.csv (overall)")
        if len(loss_groups_eval_df):
            # Overlay panel by group: grouped bars per loss_group × loss_variant
            loss_groups_eval_df = add_config_stress_metrics(loss_groups_eval_df)
            loss_groups_eval_df.to_csv(outdir / "eval_per_structure_by_loss_groups.csv", index=False)
            print(f"[INFO] Evaluated loss groups: {sorted(set(loss_groups_eval_df['loss_group'].astype(str)))}")
            # Aggregate means by loss_group × loss_variant for E/F/P/S
            agg = loss_groups_eval_df.groupby(["loss_group","loss_variant"], dropna=False).agg(
                e_mean=("e_pa_abs_meV","mean"), f_mean=("f_rmse","mean"), p_mean=("p_abs_GPa","mean"), s_mean=("sigma_rmse_GPa","mean")
            ).reset_index()
            # Plot overlays per metric
            canonical = ["MSE","MSETW","CA","CATW","unknown"]
            lv_order = [v for v in canonical if v in set(agg["loss_variant"].astype(str))]
            group_order = sorted(set(agg["loss_group"].astype(str)))
            for col, ylabel, fname in [
                ("e_mean", "|ΔE| per atom RMSE (meV)", "loss_groups_energy.png"),
                ("f_mean", "Force RMSE (eV/Å)", "loss_groups_forces.png"),
                ("p_mean", "|ΔP| (GPa)", "loss_groups_pressure.png"),
                ("s_mean", "Stress RMSE (GPa)", "loss_groups_stress.png"),
            ]:
                # Use centered grouped bars: x = hyperparameter Group, bars = Variant
                save_grouped_bar(
                    data_df=agg.rename(columns={"loss_variant":"Variant","loss_group":"Group"}),
                    x_col="Group",
                    group_col="Variant",
                    value_col=col,
                    x_order=group_order,
                    group_order=lv_order,
                    ylabel=ylabel,
                    title=f"Loss variants across hyperparameter groups — {ylabel}",
                    outfile=outdir / fname,
                    rotate_xticks=20,
                )
            # Also save stress comparison by hyperparameter group and by loss variant separately
            if len(group_order):
                save_stress_compare_bars(
                    loss_groups_eval_df.rename(columns={"loss_group":"group"}),
                    group_col="group",
                    outfile=outdir / "loss_groups_stress_comparison_by_group.png",
                    order=group_order,
                    title_prefix="Loss groups",
                )
            if len(lv_order):
                save_stress_compare_bars(
                    loss_groups_eval_df,
                    group_col="loss_variant",
                    outfile=outdir / "loss_groups_stress_comparison_by_variant.png",
                    order=lv_order,
                    title_prefix="Loss variants (within groups)",
                )
            # Per-group significance tests
            results = []
            for grp_name, sub in loss_groups_eval_df.groupby("loss_group"):
                res = significance_tests_loss(sub, outdir=outdir, scope_label="group", group_value=str(grp_name))
                if len(res):
                    results.append(res)
            if len(results):
                pd.concat(results, ignore_index=True).to_csv(outdir / "loss_significance_by_group.csv", index=False)
        
        # Save precomputed results if requested
        if args.save_precomputed is not None:
            save_precomputed_results(gen_eval_df, loss_eval_df, loss_groups_eval_df, Path(args.save_precomputed))

        print("Wrote outputs to", outdir.resolve())
        return
    
    # Plotting section for precomputed data
    if precomputed_mode:
        print("[INFO] Generating plots from precomputed data")
        
        # Plot for generations
        if len(gen_eval_df):
            gens_order = sorted([int(g) for g in gen_eval_df["generation"].dropna().unique()])
            save_metric_panel(gen_eval_df, group_col="generation", outfile=outdir / "panel_generations_metrics.png", order=gens_order, title_prefix="Generations")
            if len(gens_order):
                save_stress_compare_bars(
                    gen_eval_df,
                    group_col="generation",
                    outfile=outdir / "generations_stress_comparison.png",
                    order=gens_order,
                    title_prefix="Generations",
                )
        
        # Plot for loss variants
        if len(loss_eval_df):
            canonical = ["MSE","MSETW","CA","CATW","unknown"]
            lv_order = [v for v in canonical if v in set(loss_eval_df["loss_variant"].astype(str))]
            if len(lv_order):
                save_metric_panel(loss_eval_df, group_col="loss_variant", outfile=outdir / "panel_loss_variants_metrics.png", order=lv_order, title_prefix="Loss variants")
                save_stress_compare_bars(
                    loss_eval_df,
                    group_col="loss_variant",
                    outfile=outdir / "loss_variants_stress_comparison.png",
                    order=lv_order,
                    title_prefix="Loss variants",
                )
                save_four_metric_horizontal_by_loss_variant(
                    loss_eval_df,
                    outfile=outdir / "loss_variants_four_metric_horizontal.png",
                    variants_order=[v for v in ["MSE","MSETW","CA","CATW"] if v in lv_order],
                    elasticity_json_path=Path(args.elasticity_tensors) if args.elasticity_tensors else None,
                )
                save_four_metric_vertical_by_loss_variant(
                    loss_eval_df,
                    outfile=outdir / "loss_variants_four_metric_vertical.png",
                    variants_order=[v for v in ["MSE","MSETW","CA","CATW"] if v in lv_order],
                    elasticity_json_path=Path(args.elasticity_tensors) if args.elasticity_tensors else None,
                )
            # Significance tests
            sig_overall = significance_tests_loss(loss_eval_df, outdir=outdir, scope_label="overall")
            if len(sig_overall):
                print("[INFO] Wrote loss_significance.csv (overall)")
        
        # Plot for loss groups
        if len(loss_groups_eval_df):
            print(f"[INFO] Evaluated loss groups: {sorted(set(loss_groups_eval_df['loss_group'].astype(str)))}")
            agg = loss_groups_eval_df.groupby(["loss_group","loss_variant"], dropna=False).agg(
                e_mean=("e_pa_abs_meV","mean"), f_mean=("f_rmse","mean"), p_mean=("p_abs_GPa","mean"), s_mean=("sigma_rmse_GPa","mean")
            ).reset_index()
            canonical = ["MSE","MSETW","CA","CATW","unknown"]
            lv_order = [v for v in canonical if v in set(agg["loss_variant"].astype(str))]
            group_order = sorted(set(agg["loss_group"].astype(str)))
            for col, ylabel, fname in [
                ("e_mean", "|ΔE| per atom RMSE (meV)", "loss_groups_energy.png"),
                ("f_mean", "Force RMSE (eV/Å)", "loss_groups_forces.png"),
                ("p_mean", "|ΔP| (GPa)", "loss_groups_pressure.png"),
                ("s_mean", "Stress RMSE (GPa)", "loss_groups_stress.png"),
            ]:
                save_grouped_bar(
                    data_df=agg.rename(columns={"loss_variant":"Variant","loss_group":"Group"}),
                    x_col="Group",
                    group_col="Variant",
                    value_col=col,
                    x_order=group_order,
                    group_order=lv_order,
                    ylabel=ylabel,
                    title=f"Loss variants across hyperparameter groups — {ylabel}",
                    outfile=outdir / fname,
                    rotate_xticks=20,
                )
            if len(group_order):
                save_stress_compare_bars(
                    loss_groups_eval_df.rename(columns={"loss_group":"group"}),
                    group_col="group",
                    outfile=outdir / "loss_groups_stress_comparison_by_group.png",
                    order=group_order,
                    title_prefix="Loss groups",
                )
            if len(lv_order):
                save_stress_compare_bars(
                    loss_groups_eval_df,
                    group_col="loss_variant",
                    outfile=outdir / "loss_groups_stress_comparison_by_variant.png",
                    order=lv_order,
                    title_prefix="Loss variants (within groups)",
                )
            # Per-group significance tests
            results = []
            for grp_name, sub in loss_groups_eval_df.groupby("loss_group"):
                res = significance_tests_loss(sub, outdir=outdir, scope_label="group", group_value=str(grp_name))
                if len(res):
                    results.append(res)
            if len(results):
                pd.concat(results, ignore_index=True).to_csv(outdir / "loss_significance_by_group.csv", index=False)
        
        print("Wrote outputs to", outdir.resolve())
        return

    if args.path is None:
        raise RuntimeError("Either provide --xyz dataset with evaluation or --precomputed-data for plot reproduction.")

    base = extract_if_tar(Path(args.path))
    all_df = load_runs(base)
    # Ensure derived stress metrics exist for downstream comparisons
    all_df = add_config_stress_metrics(all_df)

    # Restrict to provided lists (if any)
    gen_map = parse_list_map(args.gens)
    loss_map = parse_list_map(args.loss)

    # Generational analysis
    gen_scores_agg = pd.DataFrame()
    sec_gen = pd.DataFrame()
    if gen_map:
        keep = []
        for g, names in gen_map.items():
            keep += names
        sub = all_df[all_df["variant"].isin(keep)].copy()
        # attach explicit generation labels (override inferred)
        def _override_gen(v):
            for g, names in gen_map.items():
                if v in names: return g
            return np.nan
        sub["generation"] = sub["variant"].map(_override_gen)
    else:
        sub = all_df.copy()

    # Summaries and composite per variant
    if len(sub):
        sub_sum = summarize_by(sub, ["variant","generation","section","purity"])
        gen_scores = config_aware_composite(sub_sum, by_cols=["variant","generation","section","purity"])
        # aggregate across variants (seeds) per generation
        gen_scores_agg = gen_scores.groupby("generation").agg(
            score_mean=("score","mean"), score_std=("score","std"), n=("score","count")
        ).reset_index().sort_values("generation")
        gen_scores_agg["score_sem"] = gen_scores_agg["score_std"] / np.sqrt(gen_scores_agg["n"].clip(lower=1))
        sec_metrics = summarize_by(sub, ["variant","generation","section"])
        sec_gen = sec_metrics.groupby(["generation","section"], dropna=False).agg(
            f_mean=("f_mean","mean"), e_mean=("e_mean","mean"),
            p_mean=("p_mean","mean"), s_mean=("s_mean","mean"), n=("n","sum")
        ).reset_index()

    # outdir already created above
    # Plots for generations
    if len(gen_scores_agg):
        save_bar_with_err(
            [str(int(g)) for g in gen_scores_agg["generation"]],
            [float(v) for v in gen_scores_agg["score_mean"]],
            [float(v) if np.isfinite(v) else 0.0 for v in gen_scores_agg["score_sem"]],
            "Composite score (z, lower is better)",
            "Generational progress (config-aware composite)",
            outdir / "generations_composite.png"
        )
    if len(sec_gen):
        sec_gen_sorted = sec_gen.sort_values(["section","generation"])
        x_order = [s for s in SECTION_ORDER if s in set(sec_gen_sorted["section"].astype(str))]
        gens = sorted([int(g) for g in sec_gen_sorted["generation"].dropna().unique()])
        for col, ylabel, fname in [
            ("f_mean", "Force RMSE (eV/Å)", "generations_forces_by_section.png"),
            ("e_mean", "|ΔE| per atom RMSE (meV)", "generations_energy_by_section.png"),
            ("p_mean", "|ΔP| (GPa)", "generations_pressure_by_section.png"),
            ("s_mean", "Stress RMSE (GPa)", "generations_stress_by_section.png"),
        ]:
            save_grouped_bar(
                data_df=sec_gen_sorted.rename(columns={"generation":"Generation","section":"Section"}),
                x_col="Section", group_col="Generation", value_col=col,
                x_order=x_order, group_order=gens,
                ylabel=ylabel, title=f"By section — {ylabel}",
                outfile=outdir / fname, rotate_xticks=20
            )

    # Stress comparison plots from per-structure data directly
    try:
        if "generation" in all_df.columns and all_df["generation"].notna().any():
            gens_order = sorted([int(g) for g in all_df["generation"].dropna().unique()])
            if len(gens_order):
                save_stress_compare_bars(
                    all_df,
                    group_col="generation",
                    outfile=outdir / "generations_stress_comparison_from_runs.png",
                    order=gens_order,
                    title_prefix="Generations (runs)",
                )
    except Exception:
        pass

    # Additional panels from aggregated per-structure data (if available)
    # Panel across generations
    try:
        if "generation" in sub.columns and sub["generation"].notna().any():
            gens_order_panel = sorted([int(g) for g in sub["generation"].dropna().unique()])
            if len(gens_order_panel):
                save_metric_panel(
                    sub,
                    group_col="generation",
                    outfile=outdir / "panel_generations_metrics.png",
                    order=gens_order_panel,
                    title_prefix="Generations",
                )
    except Exception:
        pass

    # Loss-variant analysis at preferred or highest gen
    loss_scores_agg = pd.DataFrame()
    loss_sec = pd.DataFrame()
    tgt = args.prefer_gen
    if gen_map and tgt not in gen_map:
        # fallback to highest present in explicit map
        tgt = max(gen_map.keys())
    elif not gen_map:
        # infer from data if not provided
        gens_present = sorted([g for g in all_df["generation"].dropna().unique()])
        if len(gens_present):
            tgt = args.prefer_gen if args.prefer_gen in gens_present else gens_present[-1]
        else:
            tgt = None

    if tgt is not None:
        loss_df = all_df.copy()
        if gen_map:
            # restrict to specified variants for target gen if provided
            keep_loss = loss_map.get("MSE", []) + loss_map.get("MSETW", []) + loss_map.get("CA", []) + loss_map.get("CATW", [])
            if keep_loss:
                loss_df = loss_df[loss_df["variant"].isin(keep_loss)].copy()
        loss_df = loss_df[(loss_df["generation"].isna()) | (loss_df["generation"]==tgt)].copy()
        loss_sum = summarize_by(loss_df, ["variant","loss_variant","section","purity"])
        loss_scores = config_aware_composite(loss_sum, by_cols=["variant","loss_variant","section","purity"])
        loss_scores_agg = loss_scores.groupby("loss_variant").agg(
            score_mean=("score","mean"), score_std=("score","std"), n=("score","count")
        ).reset_index().sort_values("score_mean")
        loss_scores_agg["score_sem"] = loss_scores_agg["score_std"] / np.sqrt(loss_scores_agg["n"].clip(lower=1))
        # per-section
        sec_metrics_loss = summarize_by(loss_df, ["variant","loss_variant","section"])
        loss_sec = sec_metrics_loss.groupby(["loss_variant","section"], dropna=False).agg(
            f_mean=("f_mean","mean"), e_mean=("e_mean","mean"),
            p_mean=("p_mean","mean"), s_mean=("s_mean","mean"), n=("n","sum")
        ).reset_index()

        if len(loss_scores_agg):
            canonical = ["MSE","MSETW","CA","CATW","unknown"]
            lv_order = [v for v in canonical if v in set(loss_scores_agg["loss_variant"])]
            loss_scores_agg = loss_scores_agg.set_index("loss_variant").loc[lv_order].reset_index()
            save_bar_with_err(
                [str(v) for v in loss_scores_agg["loss_variant"]],
                [float(v) for v in loss_scores_agg["score_mean"]],
                [float(v) if np.isfinite(v) else 0.0 for v in loss_scores_agg["score_sem"]],
                f"Composite score (z, lower is better) — Gen {tgt}",
                f"Loss-function variants @ Gen {tgt}",
                outdir / "loss_variants_composite.png",
            )
        if len(loss_sec):
            loss_sec_sorted = loss_sec.sort_values(["section","loss_variant"])
            x_order = [s for s in SECTION_ORDER if s in set(loss_sec_sorted["section"].astype(str))]
            lv_present = [v for v in ["MSE","MSETW","CA","CATW","unknown"] if v in set(loss_sec_sorted["loss_variant"])]
            for col, ylabel, fname in [
                ("f_mean", "Force RMSE (eV/Å)", "loss_forces_by_section.png"),
                ("e_mean", "|ΔE| per atom RMSE (meV)", "loss_energy_by_section.png"),
                ("p_mean", "|ΔP| (GPa)", "loss_pressure_by_section.png"),
                ("s_mean", "Stress RMSE (GPa)", "loss_stress_by_section.png"),
            ]:
                save_grouped_bar(
                    data_df=loss_sec_sorted.rename(columns={"loss_variant":"Variant","section":"Section"}),
                    x_col="Section", group_col="Variant", value_col=col,
                    x_order=x_order, group_order=lv_present,
                    ylabel=ylabel, title=f"Gen {tgt} — {ylabel} by section",
                    outfile=outdir / fname, rotate_xticks=20
                )

        # Stress comparison by loss variant using per-structure table
        try:
            if "loss_variant" in loss_df.columns and loss_df["loss_variant"].notna().any():
                lv_order = [v for v in ["MSE","MSETW","CA","CATW","unknown"] if v in set(loss_df["loss_variant"]) ]
                if len(lv_order):
                    save_stress_compare_bars(
                        loss_df,
                        group_col="loss_variant",
                        outfile=outdir / "loss_variants_stress_comparison_from_runs.png",
                        order=lv_order,
                        title_prefix=f"Loss variants @ Gen {tgt}",
                    )
                    save_four_metric_horizontal_by_loss_variant(
                        loss_df,
                        outfile=outdir / "loss_variants_four_metric_horizontal_from_runs.png",
                        variants_order=[v for v in ["MSE","MSETW","CA","CATW"] if v in lv_order],
                        elasticity_json_path=Path(args.elasticity_tensors) if args.elasticity_tensors else None,
                    )
                    save_four_metric_vertical_by_loss_variant(
                        loss_df,
                        outfile=outdir / "loss_variants_four_metric_vertical_from_runs.png",
                        variants_order=[v for v in ["MSE","MSETW","CA","CATW"] if v in lv_order],
                        elasticity_json_path=Path(args.elasticity_tensors) if args.elasticity_tensors else None,
                    )
        except Exception:
            pass

        # Panel across loss variants
        try:
            if "loss_variant" in loss_df.columns and loss_df["loss_variant"].notna().any():
                canonical = ["MSE","MSETW","CA","CATW","unknown"]
                lv_order_panel = [v for v in canonical if v in set(loss_df["loss_variant"].astype(str))]
                if len(lv_order_panel):
                    save_metric_panel(
                        loss_df,
                        group_col="loss_variant",
                        outfile=outdir / "panel_loss_variants_metrics.png",
                        order=lv_order_panel,
                        title_prefix=f"Loss variants @ Gen {tgt}",
                    )
        except Exception:
            pass

    # Write CSVs
    all_df.to_csv(Path(outdir) / "all_per_structure_merged.csv", index=False)
    # Save small index of runs seen
    runs_seen = sorted(list(set(all_df["variant"].astype(str).tolist())))
    with open(Path(outdir) / "runs_seen.txt", "w") as fh:
        for r in runs_seen:
            fh.write(r + "\n")
    if len(gen_scores_agg): gen_scores_agg.to_csv(Path(outdir) / "generational_scores_aggregated.csv", index=False)
    if len(loss_scores_agg): loss_scores_agg.to_csv(Path(outdir) / "loss_scores_aggregated.csv", index=False)

    print("Wrote outputs to", Path(outdir).resolve())

if __name__ == "__main__":
    main()
