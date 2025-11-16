"""
Standalone analysis and plotting for per-generation W&B exports.

- Reads a W&B CSV export.
- Extracts generation/seed from the `Name` column (expects patterns like 'genX...seedY').
- Identifies all 'test0*' columns and aggregates per run, then per generation.
- Optionally aggregates RMSE metrics if present: forces, stress, per-atom energy.
- Saves tidy CSV summaries and static PNG figures (no GUI display).
- Optionally evaluates models on test data for parity plots; caches results as JSON.

Basic run:
  python per_generation_analysis.py \
    --csv /path/to/wandb_export_....csv \
    --outdir /path/to/outdir \
    --gen-min 0 --gen-max 10 --seeds 0 1 2

With parity plots (requires NequIP models and test XYZ):
  python per_generation_analysis.py \
    --csv /path/to/wandb_export.csv \
    --outdir /path/to/outdir \
    --parity 0 7 10 \
    --models /path/to/models_root \
    --xyz /path/to/test.xyz

Replotting from cached data (no recomputation):
  python per_generation_analysis.py \
    --outdir ../../data/per_gen_out/ \
    --replot \
    --parity 0 10

  # Or to plot different generations:
  python per_generation_analysis.py \
    --outdir ../../data/per_gen_out/ \
    --replot \
    --parity 0 7 10
"""

import argparse
import logging
import re
import sys
import json
import gzip
from typing import Iterable, Optional, Dict, List

import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from cycler import cycler
import zipfile
import tempfile

# Set custom color cycle to match publication color scheme
matplotlib.rcParams['axes.prop_cycle'] = cycler(color=['#2A33C3', '#A35D00', '#6E8B00'])

# Set Helvetica font family
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# Lazy imports for optional parity evaluation
try:
    from ase.io import read as ase_read
except Exception:
    ase_read = None  # Only needed if --parity is used



def extract_gen_seed(name: str):
    if not isinstance(name, str):
        return (np.nan, np.nan)
    gen_match = re.search(r"gen(\d+)", name, flags=re.IGNORECASE)
    seed_match = None
    for m in re.finditer(r"seed(\d+)", name, flags=re.IGNORECASE):
        seed_match = m
    gen = int(gen_match.group(1)) if gen_match else np.nan
    seed = int(seed_match.group(1)) if seed_match else np.nan
    return (gen, seed)


def find_latest_wandb_csv(default_dir: Path) -> Optional[Path]:
    if not default_dir.exists():
        return None
    candidates = sorted(default_dir.glob("wandb_export_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def flatten_rmse_columns(df_in: pd.DataFrame, group_key: str, present_rmse: dict) -> pd.DataFrame:
    df = df_in.copy()
    # Rename group key if present as a simple column
    if group_key in df.columns:
        df = df.rename(columns={group_key: "generation"})
    # Flatten potential MultiIndex columns
    col_to_key = {v: k for k, v in present_rmse.items()}
    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            # Some pandas versions return ('__gen', '') or similar for the group key
            if len(col) >= 1 and col[0] == group_key:
                new_cols.append("generation")
                continue
            base_col = col[0]
            stat = col[1] if len(col) > 1 else ""
            mapped = col_to_key.get(base_col, base_col)
            new_cols.append(f"{mapped}_{stat}".strip("_"))
        else:
            new_cols.append(col)
    df.columns = new_cols
    # Ensure generation exists (even if empty)
    if "generation" not in df.columns:
        df.insert(0, "generation", pd.Series(dtype=float))
    return df


def make_publication_plot(agg_df: pd.DataFrame, show: bool = False, savepath: Optional[Path] = None):
    required = {"generation", "test0_mean_of_means", "test0_median_of_means"}
    missing = required - set(agg_df.columns)
    if missing or len(agg_df) == 0:
        logging.info("Skipping overall plot; missing columns or no data: %s", sorted(missing))
        return

    matplotlib.use("Agg", force=False)
    x = agg_df["generation"].to_numpy()
    y_mean = agg_df["test0_mean_of_means"].to_numpy()
    y_median = agg_df["test0_median_of_means"].to_numpy()
    if len(x) == 0:
        logging.info("Skipping overall plot; no generations to plot.")
        return

    fig, ax = plt.subplots(figsize=(5.0, 3.5), dpi=300)
    ax.plot(x, y_mean, marker="o", linewidth=1.2, label="Mean of test0* (per-gen)")
    ax.plot(x, y_median, marker="s", linewidth=1.2, label="Median of test0* (per-gen)")
    ax.set_xlabel("Generation (X)", weight='bold')
    ax.set_ylabel("Performance (test0* aggregated)", weight='bold')
    ax.set_title("Active Learning Trend: test0* Mean & Median by Generation")
    ax.set_xticks(x)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    leg = ax.legend(frameon=True)
    leg.get_frame().set_linewidth(0.5)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", pad_inches=0.02)
    if show:
        plt.show()
    else:
        plt.close(fig)


def make_rmse_plot(agg_rmse_df: pd.DataFrame, show: bool = False, savepath: Optional[Path] = None):
    """Plot RMSE with dual y-axes: forces on left, energy+stress on right (log).

    Legend entries:
      - Force RMSE (eV/Å)
      - Energy RMSE (eV/atom)
      - Stress RMSE (eV/Å³)
    """
    if "generation" not in agg_rmse_df.columns or len(agg_rmse_df) == 0:
        logging.info("Skipping RMSE plot; 'generation' missing or no data.")
        return

    matplotlib.use("Agg", force=False)
    x = agg_rmse_df["generation"].to_numpy()
    if len(x) == 0:
        logging.info("Skipping RMSE plot; no generations to plot.")
        return

    # Font sizes
    label_fs = 10
    tick_fs = 9
    legend_fs = 9

    # Publication palette (matches rcParams)
    force_color = "#2A33C3"   # blue
    energy_color = "#A35D00"  # brown/orange
    stress_color = "#6E8B00"  # green

    # Wider figure + reserved right margin for legend
    # To adjust RMSE plot width in the panel: change the first value in figsize (currently 8.5)
    fig, ax_left = plt.subplots(figsize=(9.5, 3.5), dpi=300)
    # Reserve space on the right for the legend
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.95)
    ax_right = ax_left.twinx()
    ax_right.set_yscale("log")

    handles: list = []
    labels: list = []

    def plot_band(
        ax_obj,
        key: str,
        marker: str,
        legend_label: str,
        color: str,
        log_clip: bool,
    ):
        mean_col = f"{key}_mean"
        std_col = f"{key}_std"
        if mean_col not in agg_rmse_df.columns or std_col not in agg_rmse_df.columns:
            return

        y = agg_rmse_df[mean_col].to_numpy()
        ystd = agg_rmse_df[std_col].to_numpy()

        if log_clip:
            positive = y[y > 0]
            if positive.size == 0:
                return
            eps = float(np.nanmin(positive)) * 1e-3
            y_plot = np.clip(y, eps, None)
            y_low = np.clip(y - ystd, eps, None)
            y_high = np.clip(y + ystd, eps * 1.001, None)
        else:
            y_plot = y
            y_low = y - ystd
            y_high = y + ystd

        [line] = ax_obj.plot(x, y_plot, marker=marker, linewidth=1.4, color=color)
        ax_obj.fill_between(x, y_low, y_high, alpha=0.20, color=color, linewidth=0.0)
        handles.append(line)
        labels.append(legend_label)

    # Left axis: forces
    plot_band(
        ax_left,
        key="forces_rmse",
        marker="o",
        legend_label="Force RMSE (eV/Å)",
        color=force_color,
        log_clip=False,
    )

    # Right axis: energy + stress
    plot_band(
        ax_right,
        key="energy_rmse",
        marker="D",
        legend_label="Energy RMSE (eV/atom)",
        color=energy_color,
        log_clip=True,
    )
    plot_band(
        ax_right,
        key="stress_rmse",
        marker="s",
        legend_label="Stress RMSE (eV/Å³)",
        color=stress_color,
        log_clip=True,
    )

    # Axis labels / ticks
    ax_left.set_xlabel("Generation (X)", weight="bold", fontsize=label_fs)
    ax_left.set_ylabel("F RMSE (eV/Å)", weight="bold", fontsize=label_fs)
    ax_right.set_ylabel(
        "(E, $\sigma$) RMSE [log] (eV/atom, eV/Å³)",
        weight="bold",
        fontsize=label_fs,
    )

    ax_left.set_xticks(x)
    ax_left.tick_params(axis="both", labelsize=tick_fs)
    ax_right.tick_params(axis="y", labelsize=tick_fs)

    ax_left.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    # Panel label "(a)"
    ax_left.text(
        0.02,
        0.98,
        "(a)",
        transform=ax_left.transAxes,
        fontsize=label_fs,
        weight="bold",
        color="black",
        va="top",
        ha="left",
    )

    # Legend inside plot area, upper right corner
    if handles:
        leg = ax_left.legend(
            handles,
            labels,
            loc="upper right",
            frameon=True,
            bbox_to_anchor=(0.98, 0.98),
            framealpha=0.95,
            fontsize=legend_fs,
        )
        leg.get_frame().set_linewidth(0.5)

    # Axis arrows: always visible (axes-fraction coordinates)
    arrow_kw = dict(arrowstyle="-|>", lw=2.4)

    # Forces arrow: blue, pointing left toward left y-axis
    ax_left.annotate(
        "",
        xy=(0.04, 0.40),    # head near left axis
        xytext=(0.22, 0.40),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(**arrow_kw, color=force_color),
    )

    # Energy arrow: brown/orange, pointing right toward right y-axis
    ax_right.annotate(
        "",
        xy=(0.96, 0.72),
        xytext=(0.78, 0.72),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(**arrow_kw, color=energy_color),
    )

    # Stress arrow: green, pointing right toward right y-axis
    ax_right.annotate(
        "",
        xy=(0.96, 0.55),
        xytext=(0.78, 0.55),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(**arrow_kw, color=stress_color),
    )

    # Respect reserved right margin
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])

    if savepath is not None:
        # Reduced padding to minimize vertical space in panel
        # CHANGE THIS TO ADJUST VERTICAL SPACE IN MAKE_RMSE_PARITY_PANEL 
        fig.savefig(savepath, bbox_inches="tight", pad_inches=0.01)
    if show:
        plt.show()
    else:
        plt.close(fig)




def _is_packaged_model_dir(path: Path) -> bool:
    """Heuristically determine if a directory is a NequIP packaged model directory."""
    if not path.is_dir():
        return False
    markers = [
        path / "deployed.pth",
        path / "deployed.pth.tar",
        path / "config.json",
        path / "config.yaml",
    ]
    return any(m.exists() for m in markers)


def discover_model_packages_by_generation(models_root: Path, generations: Iterable[int]) -> Dict[int, List[Path]]:
    """Find packaged NequIP models for specified generations.

    Supports two layouts:
      1) Zip packages: files named like genX_seedY.nequip.zip anywhere under models_root
      2) Legacy directories: folders starting with genX containing a packaged model (deployed.pth, config.*)

    Returns a dict: generation -> list of package paths (zip files or package directories), one per seed.
    """
    gens = sorted(set(int(g) for g in generations))
    result: Dict[int, List[Path]] = {g: [] for g in gens}
    if not models_root.exists():
        logging.error("Models directory does not exist: %s", models_root)
        return result

    # First: find zipped packages genX_seedY.nequip.zip
    zip_regex = re.compile(r"gen(\d+)_seed(\d+)\.nequip\.zip$", flags=re.IGNORECASE)
    try:
        for p in models_root.rglob("*.nequip.zip"):
            m = zip_regex.search(p.name)
            if not m:
                continue
            gen = int(m.group(1))
            if gen in result:
                result[gen].append(p)
    except Exception:
        pass

    # Also support legacy packaged directories under genX
    gen_dir_regex = re.compile(r"gen(\d+)", flags=re.IGNORECASE)
    try:
        for entry in models_root.iterdir():
            if not entry.is_dir():
                continue
            m = gen_dir_regex.search(entry.name)
            if not m:
                continue
            gen = int(m.group(1))
            if gen not in result:
                continue
            if _is_packaged_model_dir(entry):
                result[gen].append(entry)
                continue
            try:
                for sub in entry.iterdir():
                    if sub.is_dir() and _is_packaged_model_dir(sub):
                        result[gen].append(sub)
            except Exception:
                pass
    except Exception:
        pass

    # Unique and sort paths per gen
    for gen in result:
        unique_paths = []
        seen = set()
        for p in result[gen]:
            s = str(p.resolve())
            if s not in seen:
                unique_paths.append(p)
                seen.add(s)
        result[gen] = sorted(unique_paths)

    for gen, paths in result.items():
        logging.info("Gen %s: found %d packaged model(s)", gen, len(paths))

    return result


def _get_reference_energy(atoms) -> Optional[float]:
    # Common keys that might store reference energy (eV)
    keys = ["REF_energy", "energy", "E", "total_energy", "ref_energy", "dft_energy"]
    for k in keys:
        val = atoms.info.get(k)
        if val is not None:
            try:
                return float(val)
            except Exception:
                continue
    return None


def _get_reference_forces(atoms) -> Optional[np.ndarray]:
    # Common array names for forces (eV/Ang)
    keys = ["REF_force", "forces", "F", "ref_forces", "dft_forces"]
    for k in keys:
        if k in atoms.arrays:
            arr = atoms.arrays[k]
            try:
                arr = np.asarray(arr, dtype=float)
                if arr.ndim == 2 and arr.shape[1] == 3:
                    return arr
            except Exception:
                continue
    return None


def _get_reference_energy_per_atom(atoms) -> Optional[float]:
    # Prefer explicitly per-atom keys if present
    keys_per_atom = [
        "energy_per_atom", "E_per_atom", "per_atom_energy", "per_atom_E", "E0_per_atom"
    ]
    for k in keys_per_atom:
        val = atoms.info.get(k)
        if val is not None:
            try:
                return float(val)
            except Exception:
                continue
    # Fallback: total energy divided by number of atoms
    e_total = _get_reference_energy(atoms)
    if e_total is None:
        return None
    num_atoms = max(1, len(atoms))
    try:
        return float(e_total) / float(num_atoms)
    except Exception:
        return None


def _to_voigt6(stress: np.ndarray) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(stress, dtype=float)
        if arr.ndim == 1:
            if arr.size == 6:
                return arr.reshape(6)
            if arr.size == 9:
                # Interpret as row-major flattened 3x3
                arr = arr.reshape(3, 3)
        if arr.shape == (3, 3):
            # [xx, yy, zz, yz, xz, xy]
            return np.array([arr[0, 0], arr[1, 1], arr[2, 2], arr[1, 2], arr[0, 2], arr[0, 1]], dtype=float)
    except Exception:
        return None
    return None


def _get_reference_stress(atoms) -> Optional[np.ndarray]:
    # Try info first
    info_keys = ["REF_stress", "stress", "S", "ref_stress", "dft_stress"]
    for k in info_keys:
        val = atoms.info.get(k)
        if val is not None:
            voigt = _to_voigt6(val)
            if voigt is not None:
                return voigt
    # Try arrays (less common for stress)
    array_keys = ["REF_stress", "stress", "S", "ref_stress", "dft_stress"]
    for k in array_keys:
        if k in atoms.arrays:
            voigt = _to_voigt6(atoms.arrays[k])
            if voigt is not None:
                return voigt
    return None


def evaluate_models_on_xyz(
    gen_to_packages: Dict[int, List[Path]], xyz_path: Path, device: str = "cuda"
) -> Dict[str, object]:
    """Evaluate multiple NequIP packaged models on a given XYZ structure file.

    Returns a dict with keys:
      - energy: { 'y_true': np.ndarray, 'y_pred': {gen: np.ndarray} }
      - forces: { 'y_true': np.ndarray, 'y_pred': {gen: np.ndarray} }
    Only includes entries for properties available in the XYZ.
    """
    if ase_read is None:
        raise RuntimeError("ASE is required for parity evaluation; please install ase.")
    try:
        from nequip.ase import NequIPCalculator  # lazy import
    except Exception as e:
        raise RuntimeError(f"NequIP is required for parity evaluation: {e}")

    def _load_calc(path_like: Path):
        # Try loading directly; if it's a zip and fails, unzip to temp and load
        try:
            return NequIPCalculator._from_packaged_model(package_path=str(path_like), device=device)
        except Exception as e1:
            if isinstance(path_like, Path) and path_like.suffix.lower() == ".zip":
                try:
                    with tempfile.TemporaryDirectory() as td:
                        with zipfile.ZipFile(str(path_like), 'r') as zf:
                            zf.extractall(td)
                        return NequIPCalculator._from_packaged_model(package_path=str(td), device=device)
                except Exception as e2:
                    raise e2
            raise e1

    # Load structures; support multi-frame xyz
    atoms_list = None
    try:
        atoms_list = ase_read(str(xyz_path), ":")
    except Exception:
        atoms_list = [ase_read(str(xyz_path))]

    # Collect reference data
    energy_pa_true: List[float] = []
    forces_true_list: List[np.ndarray] = []
    stress_true_list: List[np.ndarray] = []
    valid_energy_idx: List[int] = []
    valid_forces_idx: List[int] = []
    valid_stress_idx: List[int] = []

    for i, atoms in enumerate(atoms_list):
        e_pa = _get_reference_energy_per_atom(atoms)
        if e_pa is not None and np.isfinite(e_pa):
            valid_energy_idx.append(i)
            energy_pa_true.append(float(e_pa))
        f = _get_reference_forces(atoms)
        if f is not None and np.all(np.isfinite(f)):
            valid_forces_idx.append(i)
            forces_true_list.append(f)
        s = _get_reference_stress(atoms)
        if s is not None and np.all(np.isfinite(s)):
            valid_stress_idx.append(i)
            stress_true_list.append(s)

    have_energy = len(energy_pa_true) > 0
    have_forces = len(forces_true_list) > 0
    have_stress = len(stress_true_list) > 0
    if not (have_energy or have_forces or have_stress):
        logging.error("No reference energy/atom, forces, or stress found in %s; cannot make parity plots.", xyz_path)
        return {}

    # Evaluate predictions
    # Accumulate per-frame predictions per generation across seeds, then average
    energy_pa_pred_by_gen: Dict[int, Dict[int, List[float]]] = {g: {} for g in gen_to_packages}
    forces_pred_by_gen: Dict[int, Dict[int, List[np.ndarray]]] = {g: {} for g in gen_to_packages}
    stress_pred_by_gen: Dict[int, Dict[int, List[np.ndarray]]] = {g: {} for g in gen_to_packages}

    for gen, package_paths in gen_to_packages.items():
        if len(package_paths) == 0:
            continue
        logging.info("Evaluating generation %s on %s with %d seed model(s)", gen, xyz_path, len(package_paths))
        for pkg_path in package_paths:
            try:
                calc = _load_calc(pkg_path)
            except Exception as e:
                logging.warning("Failed to load model at %s: %s", pkg_path, e)
                continue

            # For each frame, compute predictions
            if have_energy:
                for idx in valid_energy_idx:
                    atoms = atoms_list[idx].copy()
                    atoms.calc = calc
                    try:
                        e_total = float(atoms.get_potential_energy())
                        num_atoms = max(1, len(atoms))
                        e_pa_pred = e_total / float(num_atoms)
                        energy_pa_pred_by_gen[gen].setdefault(idx, []).append(e_pa_pred)
                    except Exception as e:
                        logging.debug("Energy prediction failed for gen %s model %s: %s", gen, pkg_path, e)
            if have_forces:
                for idx in valid_forces_idx:
                    atoms = atoms_list[idx].copy()
                    atoms.calc = calc
                    try:
                        f_pred = atoms.get_forces()
                        if isinstance(f_pred, np.ndarray) and f_pred.ndim == 2 and f_pred.shape[1] == 3:
                            forces_pred_by_gen[gen].setdefault(idx, []).append(f_pred)
                    except Exception as e:
                        logging.debug("Forces prediction failed for gen %s model %s: %s", gen, pkg_path, e)
            if have_stress:
                for idx in valid_stress_idx:
                    atoms = atoms_list[idx].copy()
                    atoms.calc = calc
                    try:
                        s_pred = atoms.get_stress()
                        s_voigt = _to_voigt6(s_pred)
                        if s_voigt is not None and s_voigt.size == 6:
                            stress_pred_by_gen[gen].setdefault(idx, []).append(s_voigt)
                    except Exception as e:
                        logging.debug("Stress prediction failed for gen %s model %s: %s", gen, pkg_path, e)

    out: Dict[str, object] = {}
    if have_energy:
        # Average across seeds per frame in the order of valid_energy_idx
        y_pred_energy_by_gen_avg: Dict[int, np.ndarray] = {}
        for g, per_frame in energy_pa_pred_by_gen.items():
            if len(per_frame) == 0:
                continue
            vals_in_order: List[float] = []
            for idx in valid_energy_idx:
                preds = per_frame.get(idx, [])
                if len(preds) == 0:
                    continue
                vals_in_order.append(float(np.mean(preds)))
            if len(vals_in_order) > 0:
                y_pred_energy_by_gen_avg[g] = np.asarray(vals_in_order, dtype=float)
        out["energy"] = {
            "y_true": np.asarray(energy_pa_true, dtype=float),
            "y_pred": y_pred_energy_by_gen_avg,
        }
    if have_forces:
        # Average across seeds per frame, then flatten across frames
        f_true_flat = np.concatenate([f.reshape(-1) for f in forces_true_list], axis=0)
        f_pred_by_gen_flat: Dict[int, np.ndarray] = {}
        for g, per_frame in forces_pred_by_gen.items():
            if len(per_frame) == 0:
                continue
            per_frame_avg: List[np.ndarray] = []
            for idx in valid_forces_idx:
                preds_list = per_frame.get(idx, [])
                if len(preds_list) == 0:
                    continue
                try:
                    stacked = np.stack(preds_list, axis=0)  # (n_seeds, N, 3)
                    avg = np.mean(stacked, axis=0)
                    per_frame_avg.append(avg)
                except Exception:
                    continue
            if len(per_frame_avg) > 0:
                f_flat = np.concatenate([f.reshape(-1) for f in per_frame_avg], axis=0)
                f_pred_by_gen_flat[g] = f_flat
        out["forces"] = {
            "y_true": f_true_flat,
            "y_pred": f_pred_by_gen_flat,
        }
    if have_stress:
        s_true_flat = np.concatenate([s.reshape(-1) for s in stress_true_list], axis=0)
        s_pred_by_gen_flat: Dict[int, np.ndarray] = {}
        for g, per_frame in stress_pred_by_gen.items():
            if len(per_frame) == 0:
                continue
            per_frame_avg: List[np.ndarray] = []
            for idx in valid_stress_idx:
                preds_list = per_frame.get(idx, [])
                if len(preds_list) == 0:
                    continue
                try:
                    stacked = np.stack(preds_list, axis=0)  # (n_seeds, 6)
                    avg = np.mean(stacked, axis=0)
                    per_frame_avg.append(avg)
                except Exception:
                    continue
            if len(per_frame_avg) > 0:
                s_flat = np.concatenate([s.reshape(-1) for s in per_frame_avg], axis=0)
                s_pred_by_gen_flat[g] = s_flat
        out["stress"] = {
            "y_true": s_true_flat,
            "y_pred": s_pred_by_gen_flat,
        }
    return out


def make_parity_plot(
    parity_data: Dict[str, object],
    generations: List[int],
    savepath: Optional[Path] = None,
    show: bool = False,
):
    """Create parity plots (energy and/or forces and/or stress) per generation.

    Legends:
      - One legend per subplot.
      - Horizontally centered over each subplot.
      - All legends share a band above the axes, which becomes the
        "between RMSE and parity" region in the final panel.
    """
    if not parity_data:
        logging.info("Skipping parity plot; no data.")
        return

    # Decide which properties we actually have
    props = []
    if "energy" in parity_data and len(parity_data["energy"]["y_pred"]) > 0:
        props.append("energy")
    if "forces" in parity_data and len(parity_data["forces"]["y_pred"]) > 0:
        props.append("forces")
    if "stress" in parity_data and len(parity_data["stress"]["y_pred"]) > 0:
        props.append("stress")
    if len(props) == 0:
        logging.info("Skipping parity plot; no predictions available.")
        return

    matplotlib.use("Agg", force=False)

    ncols = len(props)

    PARITY_SUBPLOT_WIDTH = 3.5

    plot_area_width = PARITY_SUBPLOT_WIDTH * (ncols + (ncols - 1) * 0.50)

    figure_width = plot_area_width / 0.88 
    # Wider figure to avoid squishing; extra vertical space at the top for legends
    fig, axes = plt.subplots(1, ncols, figsize=(figure_width, 5.5), dpi=300)
    if ncols == 1:
        axes = [axes]

    # Leave a band at the top for legends; give more horizontal space between subplots
    fig.subplots_adjust(
        left=0.10, right=0.98,
        bottom=0.14, top=0.78,
        wspace=0.35,
    )

    label_fs = 16
    tick_fs = 12 
    legend_fs = 12

    part_labels = ['b', 'c', 'd']  # (b), (c), (d)
    gens_sorted = sorted(generations)
    n_g = len(gens_sorted)

    def alpha_for_index(i: int, n: int) -> float:
        if n <= 1:
            return 0.9
        return float(0.35 + (0.6 * (i / (n - 1))))

    # Precompute global x positions for legends (figure coordinates)
    for idx, (ax, prop) in enumerate(zip(axes, props)):
        y_true = parity_data[prop]["y_true"]
        y_pred_by_gen: Dict[int, np.ndarray] = parity_data[prop]["y_pred"]
        if len(y_pred_by_gen) == 0:
            continue

        # Determine diagonal limits
        y_min = float(np.nanmin(y_true)) if np.size(y_true) > 0 else 0.0
        y_max = float(np.nanmax(y_true)) if np.size(y_true) > 0 else 1.0

        for i, gen in enumerate(gens_sorted):
            preds = y_pred_by_gen.get(gen)
            if preds is None or np.size(preds) == 0:
                continue

            if preds.shape[0] != y_true.shape[0]:
                reps = int(np.ceil(preds.shape[0] / max(1, y_true.shape[0])))
                x_vals = np.tile(y_true, reps)[: preds.shape[0]]
            else:
                x_vals = y_true

            color = f"C{(i % 3)}"
            # Metrics
            try:
                resid = preds - x_vals
                rmse = float(np.sqrt(np.nanmean(resid ** 2)))
                ss_res = float(np.nansum(resid ** 2))
                ss_tot = float(np.nansum((x_vals - np.nanmean(x_vals)) ** 2)) if len(x_vals) > 1 else np.nan
                r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot and np.isfinite(ss_tot) and ss_tot != 0.0 else np.nan
                metrics_txt = f"RMSE {rmse:.3g}, R² {r2:.3f}" if np.isfinite(r2) else f"RMSE {rmse:.3g}"
            except Exception:
                metrics_txt = None

            label = f"gen{gen}"
            if metrics_txt:
                label = f"{label} — {metrics_txt}"

            ax.scatter(
                x_vals, preds,
                s=20,  # Increased from 10 for better visibility
                alpha=alpha_for_index(i, n_g),
                color=color,
                label=label,
            )

            # Update limits
            y_min = min(y_min, float(np.nanmin(x_vals)), float(np.nanmin(preds)))
            y_max = max(y_max, float(np.nanmax(x_vals)), float(np.nanmax(preds)))

        # Diagonal reference
        pad = 0.02 * (y_max - y_min if y_max > y_min else 1.0)
        lo, hi = y_min - pad, y_max + pad
        ax.plot([lo, hi], [lo, hi], color="#666666", linewidth=1.1, linestyle="--", zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
        ax.tick_params(axis="both", which="both", labelsize=tick_fs)

        # Axis labels per property
        if prop == "energy":
            ax.set_xlabel("E Ref (eV/atom)", fontsize=label_fs, weight='bold')
            ax.set_ylabel("E Pred (eV/atom)", fontsize=label_fs, weight='bold')
        elif prop == "forces":
            ax.set_xlabel("F Ref (eV/Å)", fontsize=label_fs, weight='bold')
            ax.set_ylabel("F Pred (eV/Å)", fontsize=label_fs, weight='bold')
        elif prop == "stress":
            ax.set_xlabel(r"$\boldsymbol{\sigma}$ Ref (eV/Å³)", fontsize=label_fs, weight='bold')
            ax.set_ylabel(r"$\boldsymbol{\sigma}$ Pred (eV/Å³)", fontsize=label_fs, weight='bold')

        # Part label in top-left of each subplot
        part_label = f"({part_labels[idx]})"
        ax.text(
            0.02, 0.98, part_label,
            transform=ax.transAxes,
            fontsize=label_fs,
            weight='bold',
            color='black',
            va='top',
            ha='left',
        )
        

        # Legend: centered horizontally over each subplot,
        # in the band between top of axes and top of figure.
        bbox = ax.get_position()
        subplot_center_x = (bbox.x0 + bbox.x1) / 2.0

        leg = ax.legend(
            frameon=True,
            markerscale=3,
            fontsize=legend_fs,
            loc='center',
            bbox_to_anchor=(subplot_center_x, 0.85),
            bbox_transform=fig.transFigure,
            framealpha=0.95,
        )
        leg.get_frame().set_linewidth(0.5)

    # No tight_layout: we already did subplots_adjust to reserve legend band
    if savepath is not None:
        # Reduced padding to minimize vertical space in panel
        fig.savefig(savepath, bbox_inches="tight", pad_inches=0.01)
    if show:
        plt.show()
    else:
        plt.close(fig)



def make_rmse_parity_panel(rmse_png: Path, parity_png: Path, savepath: Optional[Path] = None):
    """Stack RMSE figure (top) and parity figure (bottom) into a single panel PNG."""
    try:
        img_rmse = mpimg.imread(str(rmse_png))
        img_parity = mpimg.imread(str(parity_png))
    except Exception as e:
        logging.warning("Could not read images for panel: %s", e)
        return

    matplotlib.use("Agg", force=False)

    rmse_hw = float(img_rmse.shape[0]) / float(img_rmse.shape[1])
    parity_hw = float(img_parity.shape[0]) / float(img_parity.shape[1])

    fig_width = 8.5  # match widened plots
    fig_height = fig_width * (rmse_hw + parity_hw)

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[rmse_hw, parity_hw])

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    ax0.imshow(img_rmse, aspect="equal")
    ax0.axis("off")

    ax1.imshow(img_parity, aspect="equal")
    ax1.axis("off")

    # Remove vertical space between panels
    # Use negative hspace to overlap if needed, and tight margins
    fig.subplots_adjust(hspace=0.02, top=0.998, bottom=0.002, left=0.0, right=1.0)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)




def save_parity_data_json(parity_data: Dict[str, object], path: Path):
    """Save parity data to JSON with numpy arrays converted to lists.
    
    Automatically uses gzip compression if path ends with .gz
    """
    json_data = {}
    for prop, data_dict in parity_data.items():
        json_data[prop] = {}
        # Convert y_true numpy array to list
        if "y_true" in data_dict:
            json_data[prop]["y_true"] = data_dict["y_true"].tolist()
        # Convert y_pred dict of numpy arrays to dict of lists
        if "y_pred" in data_dict:
            json_data[prop]["y_pred"] = {
                str(gen): arr.tolist() for gen, arr in data_dict["y_pred"].items()
            }
    
    # Use gzip if path ends with .gz
    if str(path).endswith('.gz'):
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
    else:
        with open(path, 'w') as f:
            json.dump(json_data, f, indent=2)


def load_parity_data_json(path: Path) -> Dict[str, object]:
    """Load parity data from JSON and convert lists back to numpy arrays.
    
    Automatically uses gzip decompression if path ends with .gz
    """
    # Use gzip if path ends with .gz
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            json_data = json.load(f)
    else:
        with open(path, 'r') as f:
            json_data = json.load(f)
    
    parity_data = {}
    for prop, data_dict in json_data.items():
        parity_data[prop] = {}
        # Convert y_true list back to numpy array
        if "y_true" in data_dict:
            parity_data[prop]["y_true"] = np.array(data_dict["y_true"])
        # Convert y_pred dict of lists back to dict of numpy arrays
        if "y_pred" in data_dict:
            parity_data[prop]["y_pred"] = {
                int(gen): np.array(arr) for gen, arr in data_dict["y_pred"].items()
            }
    return parity_data


def aggregate(df: pd.DataFrame, gen_min: int, gen_max: int, seeds: Iterable[int]):
    # Filter by generation/seed
    df = df.copy()
    df[["__gen", "__seed"]] = df["Name"].apply(lambda s: pd.Series(extract_gen_seed(s)))
    df = df[df["__gen"].between(gen_min, gen_max, inclusive="both")]
    df = df[df["__seed"].isin(list(seeds))]

    # Identify test0* columns, coerce numeric
    test_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("test0")]
    for c in test_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Overall aggregation
    if len(test_cols) > 0:
        df["__test0_mean_per_run"] = df[test_cols].mean(axis=1, skipna=True)
        agg_df = (
            df.groupby("__gen")
              .agg(
                  n_runs=("__test0_mean_per_run", "count"),
                  test0_mean_of_means=("__test0_mean_per_run", "mean"),
                  test0_median_of_means=("__test0_mean_per_run", "median"),
              )
              .reset_index()
              .rename(columns={"__gen": "generation"})
              .sort_values("generation")
        )
        agg_df["delta_mean_vs_prev"] = agg_df["test0_mean_of_means"].diff()
        agg_df["delta_median_vs_prev"] = agg_df["test0_median_of_means"].diff()
    else:
        logging.warning("No columns starting with 'test0' were found; overall aggregation will be empty.")
        gens = df["__gen"].dropna().drop_duplicates().sort_values().to_list()
        agg_df = pd.DataFrame({"generation": gens})

    # RMSE aggregation
    rmse_cols_wanted = {
        "forces_rmse": "test0_epoch/forces_rmse",
        "stress_rmse": "test0_epoch/stress_rmse",
        "energy_rmse": "test0_epoch/per_atom_energy_rmse",
    }
    present_rmse = {k: v for k, v in rmse_cols_wanted.items() if v in df.columns}
    if len(present_rmse) == 0:
        gens = df["__gen"].dropna().drop_duplicates().sort_values().to_list()
        agg_rmse_df = pd.DataFrame({"generation": gens})
    else:
        grouped = df.groupby("__gen")[list(present_rmse.values())].agg(["mean", "std"]).reset_index()
        agg_rmse_df = flatten_rmse_columns(grouped, "__gen", present_rmse)
        if "generation" in agg_rmse_df.columns:
            agg_rmse_df = agg_rmse_df.sort_values("generation")

    return agg_df, agg_rmse_df


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Per-generation analysis and plotting for W&B exports.")
    script_dir = Path(__file__).resolve().parent
    base_dir = Path(__file__).resolve().parents[1]
    default_outdir = base_dir / "data" / "per_generation_wandb_data"
    default_csv = find_latest_wandb_csv(default_outdir)

    parser.add_argument("--csv", type=Path, default=default_csv,
                        help="Path to W&B CSV export. Defaults to latest wandb_export_*.csv in data/per_generation_wandb_data.")
    parser.add_argument("--outdir", type=Path, default=default_outdir,
                        help="Directory to write summary CSVs and figures.")
    parser.add_argument("--gen-min", type=int, default=0, help="Minimum generation (inclusive).")
    parser.add_argument("--gen-max", type=int, default=10, help="Maximum generation (inclusive).")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2], help="Seed values to include.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--replot", action="store_true", help="Replot figures from saved summaries/parity cache without recomputing.")
    # Parity / model evaluation options
    parser.add_argument("--parity", type=int, nargs="*", default=None,
                        help="Generations to evaluate for parity plots, e.g. --parity 0 7 10")
    parser.add_argument("--parity-cache", type=Path, default=None, 
                        help="Path to save/load cached parity data as JSON. Defaults to parity_data.json in --outdir.")
    parser.add_argument("--models", type=Path, default=None,
                        help="Root directory containing packaged models in subfolders like genX/seedY/... or genX_...")
    parser.add_argument("--xyz", type=Path, default=None,
                        help="Path to XYZ file with reference energies/forces for parity plots.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for NequIP evaluation (cuda or cpu).")

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if not args.verbose else logging.DEBUG,
                        format="%(levelname)s: %(message)s")

    if args.replot:
        # Replot using existing summaries; skip recomputation
        if args.outdir is None or not args.outdir.exists():
            logging.error("--outdir must exist for --replot to find saved summaries.")
            return 2
        overall_csv = args.outdir / "agg_overall_summary.csv"
        rmse_csv = args.outdir / "agg_rmse_summary.csv"
        if not (overall_csv.exists() and rmse_csv.exists()):
            logging.error("Saved summaries not found in %s; cannot replot.", args.outdir)
            return 2
        logging.info("Replotting from saved summaries: %s, %s", overall_csv, rmse_csv)
        agg_df = pd.read_csv(overall_csv)
        agg_rmse_df = pd.read_csv(rmse_csv)
    else:
        if args.csv is None:
            logging.error("No CSV provided and none found by default. Please pass --csv.")
            return 2
        if not args.csv.exists():
            logging.error("CSV not found: %s", args.csv)
            return 2
        logging.info("Reading CSV: %s", args.csv)
        df = pd.read_csv(args.csv)
        if "Name" not in df.columns:
            logging.error("CSV must include a 'Name' column to extract generation/seed.")
            return 2
        agg_df, agg_rmse_df = aggregate(df, args.gen_min, args.gen_max, args.seeds)

    args.outdir.mkdir(parents=True, exist_ok=True)
    overall_csv = args.outdir / "agg_overall_summary.csv"
    rmse_csv = args.outdir / "agg_rmse_summary.csv"
    overall_png = args.outdir / "overall.png"
    rmse_png = args.outdir / "rmse_by_gen.png"

    if not args.replot:
        agg_df.to_csv(overall_csv, index=False)
        agg_rmse_df.to_csv(rmse_csv, index=False)
        logging.info("Wrote summaries: %s, %s", overall_csv, rmse_csv)

    make_publication_plot(agg_df, show=False, savepath=overall_png)
    make_rmse_plot(agg_rmse_df, show=False, savepath=rmse_png)
    logging.info("Wrote figures: %s, %s", overall_png, rmse_png)

    # Optional parity evaluation and plots
    if args.parity is not None and len(args.parity) > 0:
        gens = [int(g) for g in args.parity]
        
        # Set default parity cache path in script directory
        if args.parity_cache is None:
            args.parity_cache = script_dir / "parity_data.json.gz"
        
        # Try to load from cache first (if replotting or cache exists)
        parity_data = None
        if args.parity_cache.exists():
            logging.info("Loading cached parity data from %s", args.parity_cache)
            try:
                parity_data = load_parity_data_json(args.parity_cache)
                # Check if requested generations exist in cache
                for prop in parity_data.values():
                    cache_gens = set(prop.get("y_pred", {}).keys())
                    missing = set(gens) - cache_gens
                    if missing:
                        logging.warning("Cache is missing generations %s; only has %s", sorted(missing), sorted(cache_gens))
                    break  # Only check first property
            except Exception as e:
                logging.warning("Failed to load parity cache, will recompute: %s", e)
                parity_data = None
        
        # Compute parity data if not loaded from cache
        if parity_data is None:
            if args.models is None:
                logging.error("--models is required when computing parity (no cache loaded)")
                return 2
            if args.xyz is None or not args.xyz.exists():
                logging.error("--xyz must point to a valid file when computing parity (no cache loaded)")
                return 2
            logging.info("Discovering packaged models for generations: %s in %s", gens, args.models)
            gen_to_packages = discover_model_packages_by_generation(args.models, gens)
            try:
                parity_data = evaluate_models_on_xyz(gen_to_packages, args.xyz, device=args.device)
            except Exception as e:
                logging.error("Parity evaluation failed: %s", e)
                return 2
            # Save to cache
            try:
                save_parity_data_json(parity_data, args.parity_cache)
                logging.info("Saved parity cache to %s", args.parity_cache)
            except Exception as e:
                logging.warning("Could not save parity cache: %s", e)

        parity_png = args.outdir / "parity_by_gen.png"
        make_parity_plot(parity_data, generations=gens, savepath=parity_png, show=False)
        logging.info("Wrote parity figure: %s", parity_png)

        panel_png = args.outdir / "rmse_plus_parity_panel.png"
        make_rmse_parity_panel(rmse_png=rmse_png, parity_png=parity_png, savepath=panel_png)
        logging.info("Wrote combined panel figure: %s", panel_png)

    return 0


if __name__ == "__main__":
    sys.exit(main())
