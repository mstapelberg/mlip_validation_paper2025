"""
Identify the highest-error predictions from the parity data cache and
report configuration/composition metadata for the outlier frames.

Parses the test XYZ file directly (no ASE required) to extract per-frame
metadata: config_type, composition, generation, structure_type, n_atoms,
chemical formula, etc.

Usage:
  python outlier_analysis.py \
    --parity-cache parity_data.json.gz \
    --xyz ../../data/fixed_test_global.xyz \
    --gens 0 10 \
    --top-n 5
"""

import argparse
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def parse_xyz_metadata(xyz_path: Path) -> List[dict]:
    """Parse an extended XYZ file and extract per-frame metadata.

    Returns a list of dicts, one per frame, containing:
      - n_atoms, config_type, composition, generation, structure_type,
        source_path, structure_id, formula, species_counts
    """
    frames: List[dict] = []
    with open(xyz_path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                n_atoms = int(line)
            except ValueError:
                continue

            comment = f.readline().strip()
            info = _parse_comment_line(comment)
            info["n_atoms"] = n_atoms

            species_counts: Counter = Counter()
            for _ in range(n_atoms):
                atom_line = f.readline()
                if atom_line:
                    parts = atom_line.split()
                    if parts:
                        species_counts[parts[0]] += 1

            info["species_counts"] = dict(species_counts)
            info["formula"] = _formula_from_counts(species_counts)
            frames.append(info)

    return frames


def _parse_comment_line(comment: str) -> dict:
    """Extract key=value pairs from an extended XYZ comment line."""
    info: dict = {}
    simple_keys = [
        "config_type", "composition", "generation", "structure_type",
        "source_path", "structure_id", "batch_id", "temperature",
        "adversarial_step", "structure_index", "force_norm",
    ]
    for key in simple_keys:
        pattern = rf'{key}=("([^"]*?)"|(\S+))'
        m = re.search(pattern, comment)
        if m:
            info[key] = m.group(2) if m.group(2) is not None else m.group(3)

    ref_e = re.search(r'REF_energy=([-\d.eE+]+)', comment)
    if ref_e:
        try:
            info["REF_energy"] = float(ref_e.group(1))
        except ValueError:
            pass

    return info


def _formula_from_counts(counts: Counter) -> str:
    """Build a chemical formula string sorted alphabetically."""
    parts = []
    for elem in sorted(counts.keys()):
        c = counts[elem]
        parts.append(f"{elem}{c}" if c > 1 else elem)
    return "".join(parts)


def load_parity_cache(path: Path) -> dict:
    """Load parity data JSON (optionally gzipped)."""
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        with open(path, "r") as f:
            raw = json.load(f)

    data = {}
    for prop, d in raw.items():
        data[prop] = {
            "y_true": np.array(d["y_true"]),
            "y_pred": {int(g): np.array(a) for g, a in d["y_pred"].items()},
        }
    return data


def analyse_energy_outliers(
    parity: dict, frames: List[dict], gen: int, top_n: int = 5
) -> List[dict]:
    """Return top_n frames with largest absolute energy-per-atom error."""
    y_true = parity["y_true"]
    y_pred = parity["y_pred"].get(gen)
    if y_pred is None:
        return []

    errors = np.abs(y_pred - y_true)
    idx_sorted = np.argsort(errors)[::-1][:top_n]

    results = []
    for rank, idx in enumerate(idx_sorted, 1):
        meta = frames[idx].copy()
        meta["rank"] = rank
        meta["frame_idx"] = int(idx)
        meta["ref_energy_pa"] = float(y_true[idx])
        meta["pred_energy_pa"] = float(y_pred[idx])
        meta["abs_error"] = float(errors[idx])
        results.append(meta)
    return results


def analyse_force_outliers(
    parity: dict, frames: List[dict], gen: int, top_n: int = 5
) -> List[dict]:
    """Return top_n frames with largest per-frame force RMSE.

    Forces are stored flattened (n_atoms*3 per frame concatenated).
    Reconstruct frame boundaries from atom counts.
    """
    y_true = parity["y_true"]
    y_pred = parity["y_pred"].get(gen)
    if y_pred is None:
        return []

    boundaries = _frame_boundaries(frames, components_per_atom=3)

    per_frame_rmse = np.zeros(len(frames))
    for i, (start, end) in enumerate(boundaries):
        if start >= len(y_true) or end > len(y_true):
            break
        diff = y_pred[start:end] - y_true[start:end]
        per_frame_rmse[i] = float(np.sqrt(np.mean(diff ** 2)))

    idx_sorted = np.argsort(per_frame_rmse)[::-1][:top_n]

    results = []
    for rank, idx in enumerate(idx_sorted, 1):
        meta = frames[idx].copy()
        start, end = boundaries[idx]
        meta["rank"] = rank
        meta["frame_idx"] = int(idx)
        meta["force_rmse"] = float(per_frame_rmse[idx])

        diff = y_pred[start:end] - y_true[start:end]
        max_comp_idx = int(np.argmax(np.abs(diff)))
        meta["max_component_error"] = float(np.abs(diff[max_comp_idx]))
        results.append(meta)
    return results


def analyse_stress_outliers(
    parity: dict, frames: List[dict], gen: int, top_n: int = 5
) -> List[dict]:
    """Return top_n frames with largest per-frame stress RMSE.

    Stresses are stored as 6 Voigt components per frame, flattened.
    """
    y_true = parity["y_true"]
    y_pred = parity["y_pred"].get(gen)
    if y_pred is None:
        return []

    n_frames = len(frames)
    per_frame_rmse = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * 6
        end = start + 6
        if end > len(y_true):
            break
        diff = y_pred[start:end] - y_true[start:end]
        per_frame_rmse[i] = float(np.sqrt(np.mean(diff ** 2)))

    idx_sorted = np.argsort(per_frame_rmse)[::-1][:top_n]

    results = []
    for rank, idx in enumerate(idx_sorted, 1):
        meta = frames[idx].copy()
        start = idx * 6
        meta["rank"] = rank
        meta["frame_idx"] = int(idx)
        meta["stress_rmse"] = float(per_frame_rmse[idx])

        diff = y_pred[start:start + 6] - y_true[start:start + 6]
        meta["max_component_error"] = float(np.max(np.abs(diff)))
        results.append(meta)
    return results


def _frame_boundaries(frames: List[dict], components_per_atom: int = 3):
    """Build (start, end) index pairs for each frame in the flattened array."""
    boundaries = []
    offset = 0
    for f in frames:
        n = f["n_atoms"] * components_per_atom
        boundaries.append((offset, offset + n))
        offset += n
    return boundaries


def _fmt_row(meta: dict, prop: str) -> str:
    """Format one outlier row for pretty-printing."""
    gen_str = meta.get("generation", "?")
    ct = meta.get("config_type", "?")
    st = meta.get("structure_type", "?")
    formula = meta.get("formula", "?")
    n = meta.get("n_atoms", "?")
    sid = meta.get("structure_id", "?")
    fidx = meta["frame_idx"]

    if prop == "energy":
        return (
            f"  #{meta['rank']:>2d}  frame={fidx:<5d}  |err|={meta['abs_error']:.4f} eV/atom  "
            f"ref={meta['ref_energy_pa']:.4f}  pred={meta['pred_energy_pa']:.4f}  "
            f"config_type={ct}  structure_type={st}  gen={gen_str}  "
            f"formula={formula}  n_atoms={n}  structure_id={sid}"
        )
    elif prop == "forces":
        return (
            f"  #{meta['rank']:>2d}  frame={fidx:<5d}  RMSE={meta['force_rmse']:.4f} eV/A  "
            f"max_comp_err={meta['max_component_error']:.4f} eV/A  "
            f"config_type={ct}  structure_type={st}  gen={gen_str}  "
            f"formula={formula}  n_atoms={n}  structure_id={sid}"
        )
    elif prop == "stress":
        return (
            f"  #{meta['rank']:>2d}  frame={fidx:<5d}  RMSE={meta['stress_rmse']:.4f} eV/A^3  "
            f"max_comp_err={meta['max_component_error']:.4f} eV/A^3  "
            f"config_type={ct}  structure_type={st}  gen={gen_str}  "
            f"formula={formula}  n_atoms={n}  structure_id={sid}"
        )
    return ""


def main(argv=None):
    parser = argparse.ArgumentParser(description="Outlier analysis for parity predictions.")
    parser.add_argument("--parity-cache", type=Path,
                        default=Path(__file__).resolve().parent / "parity_data.json.gz",
                        help="Path to gzipped parity JSON cache.")
    parser.add_argument("--xyz", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data" / "fixed_test_global.xyz",
                        help="Path to test XYZ file.")
    parser.add_argument("--gens", type=int, nargs="+", default=[0, 10],
                        help="Generations to analyse.")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of top outliers to report.")
    parser.add_argument("--save-csv", type=Path, default=None,
                        help="Optional: save outlier table as CSV.")
    args = parser.parse_args(argv)

    if not args.parity_cache.exists():
        print(f"ERROR: Parity cache not found: {args.parity_cache}", file=sys.stderr)
        return 1
    if not args.xyz.exists():
        print(f"ERROR: XYZ file not found: {args.xyz}", file=sys.stderr)
        return 1

    print(f"Loading parity cache: {args.parity_cache}")
    parity = load_parity_cache(args.parity_cache)

    print(f"Parsing XYZ metadata: {args.xyz}")
    frames = parse_xyz_metadata(args.xyz)
    print(f"  {len(frames)} frames parsed")

    total_atoms = sum(f["n_atoms"] for f in frames)
    print(f"  Total atoms: {total_atoms}  (force components: {total_atoms * 3})")
    print(f"  Stress components: {len(frames) * 6}")

    csv_rows = []

    for gen in args.gens:
        print(f"\n{'='*80}")
        print(f"  GENERATION {gen} — Top {args.top_n} Outliers")
        print(f"{'='*80}")

        for prop, analyser in [
            ("energy", analyse_energy_outliers),
            ("forces", analyse_force_outliers),
            ("stress", analyse_stress_outliers),
        ]:
            if prop not in parity:
                print(f"\n  [{prop.upper()}] — No data available.")
                continue

            outliers = analyser(parity[prop], frames, gen, args.top_n)
            if not outliers:
                print(f"\n  [{prop.upper()}] — Gen {gen} not in cache.")
                continue

            print(f"\n  [{prop.upper()}]")
            for o in outliers:
                print(_fmt_row(o, prop))
                csv_rows.append({
                    "gen": gen, "property": prop, "rank": o["rank"],
                    "frame_idx": o["frame_idx"],
                    "config_type": o.get("config_type", ""),
                    "structure_type": o.get("structure_type", ""),
                    "generation_data": o.get("generation", ""),
                    "formula": o.get("formula", ""),
                    "n_atoms": o.get("n_atoms", ""),
                    "structure_id": o.get("structure_id", ""),
                    "source_path": o.get("source_path", ""),
                    "error_metric": (
                        o.get("abs_error") or o.get("force_rmse") or o.get("stress_rmse", "")
                    ),
                })

    if args.save_csv and csv_rows:
        import csv as csv_mod
        with open(args.save_csv, "w", newline="") as f:
            w = csv_mod.DictWriter(f, fieldnames=csv_rows[0].keys())
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\nSaved outlier table to {args.save_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
