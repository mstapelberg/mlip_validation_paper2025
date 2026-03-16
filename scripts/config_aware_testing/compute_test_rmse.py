#!/usr/bin/env python3
"""Compute per-config RMSE statistics for gen10 and CATW test sets,
grouped by physical_group and config_type.

Reads:
  - XYZ test file for config_type labels
  - precomputed_data.json for per-structure error metrics

Outputs:
  - test_rmse_by_physical_group.csv
  - test_rmse_by_config_type.csv
  - Markdown tables to stdout
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read as ase_read

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
OUTDIR = SCRIPT_DIR / "results" / "config_analysis"

XYZ_PATH = ROOT / "data/gen_10_data/exploit_q-97.5_rmax5.50_lmax1_layers1_mlp256_seed42_test.xyz"
JSON_PATH = SCRIPT_DIR / "precomputed_data.json"

DEFAULT_EXCLUDE = ["phonon", "phonon_aa"]

PHYSICAL_GROUP_ORDER = [
    "Bulk crystals",
    "Surfaces",
    "Point defects",
    "NEB",
    "Elastic",
    "Liquids",
    "Composition explore",
]


def physical_group(config_type: str) -> str:
    """Classify config_type by physical nature, stripping _aa/_aa-mc suffix."""
    base = re.sub(r"_aa(-mc)?$", "", config_type).strip().lower()

    if base in ("vacancy", "vacancy-alloy", "di-vacancy", "tri-vacancy",
                "sia", "di-sia", "vac"):
        return "Point defects"
    if base.startswith("neb"):
        return "NEB"
    if base in ("liquid", "surf_liquid"):
        return "Liquids"
    if base.startswith("surface_") or base == "gamma_surface":
        return "Surfaces"
    if base in ("elasticity", "elastic"):
        return "Elastic"
    if base.startswith("comp-explore") or base.startswith("comp_explore"):
        return "Composition explore"
    if base.startswith("phonon"):
        return "Phonon"
    if base in ("bcc_distorted", "fcc", "hcp", "dia", "a15", "c15",
                "bcc", "fcc_distorted"):
        return "Bulk crystals"
    return "Other"


def rmse_agg(vals: np.ndarray | pd.Series) -> float:
    v = np.asarray(vals, dtype=float)
    return float(np.sqrt(np.mean(v ** 2)))


def build_config_map(xyz_path: Path) -> pd.DataFrame:
    print(f"Reading {xyz_path.name} ...")
    atoms_list = ase_read(str(xyz_path), index=":", format="extxyz")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    rows = []
    for i, at in enumerate(atoms_list):
        info = getattr(at, "info", {}) or {}
        ct = str(info.get("config_type", "unknown"))
        rows.append({
            "structure_index": i,
            "config_type": ct,
            "physical_group": physical_group(ct),
            "is_adversarial": bool(re.search(r"_aa(-mc)?$", ct)),
        })
    df = pd.DataFrame(rows)
    print(f"  {len(df)} structures, {df['config_type'].nunique()} unique config_types")
    return df


def load_precomputed(json_path: Path) -> dict[str, pd.DataFrame]:
    with open(json_path, "r") as f:
        data = json.load(f)
    frames = {}
    for key in ("gen_eval_df", "loss_eval_df"):
        frames[key] = pd.DataFrame(data[key]) if data.get(key) else pd.DataFrame()
    return frames


def aggregate_group(sub: pd.DataFrame) -> dict:
    """Compute median and RMSE of energy and force errors for a group."""
    e = sub["e_pa_abs_meV"].dropna()
    f = sub["f_rmse"].dropna()
    return {
        "n_configs": len(sub),
        "median_e_meV": float(e.median()) if len(e) > 0 else np.nan,
        "rmse_e_meV": rmse_agg(e) if len(e) > 0 else np.nan,
        "median_f_eVA": float(f.median()) if len(f) > 0 else np.nan,
        "rmse_f_eVA": rmse_agg(f) if len(f) > 0 else np.nan,
    }


def make_tables(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Table 1 (by physical_group, excl adversarial) and Table 2 (by config_type)."""

    # Table 1: by physical_group, excluding adversarial
    base_df = df[~df["is_adversarial"]]
    rows1 = []
    for pg in PHYSICAL_GROUP_ORDER:
        sub = base_df[base_df["physical_group"] == pg]
        if len(sub) == 0:
            continue
        row = {"physical_group": pg, **aggregate_group(sub)}
        rows1.append(row)
    tbl1 = pd.DataFrame(rows1)

    # Table 2: by config_type (all, including adversarial)
    rows2 = []
    for ct in sorted(df["config_type"].unique()):
        sub = df[df["config_type"] == ct]
        if len(sub) == 0:
            continue
        pg = sub["physical_group"].iloc[0]
        is_aa = sub["is_adversarial"].iloc[0]
        row = {"config_type": ct, "physical_group": pg,
               "is_adversarial": is_aa, **aggregate_group(sub)}
        rows2.append(row)
    tbl2 = pd.DataFrame(rows2)

    return tbl1, tbl2


def print_markdown(tbl: pd.DataFrame, title: str) -> None:
    print(f"\n### {title}\n")
    cols = list(tbl.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    print(header)
    print(sep)
    for _, row in tbl.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.2f}" if not np.isnan(v) else "N/A")
            elif isinstance(v, (bool, np.bool_)):
                cells.append(str(v))
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            else:
                cells.append(str(v))
        print("| " + " | ".join(cells) + " |")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    ct_map = build_config_map(XYZ_PATH)

    frames = load_precomputed(JSON_PATH)

    exclude = set(DEFAULT_EXCLUDE)

    analyses = [
        ("gen_eval_df", "generation", 10, "Gen10"),
        ("loss_eval_df", "loss_variant", "CATW", "CATW"),
    ]

    for df_key, model_col, model_val, label in analyses:
        df = frames.get(df_key)
        if df is None or len(df) == 0:
            print(f"\n[SKIP] {df_key} not found or empty")
            continue

        df = df[df[model_col] == model_val].copy()
        if len(df) == 0:
            print(f"\n[SKIP] No records for {model_col}={model_val}")
            continue

        df = df.merge(ct_map, on="structure_index", how="left")
        df = df[~df["config_type"].isin(exclude)]

        print(f"\n{'='*80}")
        print(f"  {label} ({model_col}={model_val}) — {len(df)} structures")
        print(f"{'='*80}")

        tbl1, tbl2 = make_tables(df, label)

        print_markdown(tbl1, f"{label}: By Physical Group (excl. adversarial)")

        if len(tbl1) > 0:
            e_min, e_max = tbl1["median_e_meV"].min(), tbl1["median_e_meV"].max()
            f_min, f_max = tbl1["median_f_eVA"].min(), tbl1["median_f_eVA"].max()
            re_min, re_max = tbl1["rmse_e_meV"].min(), tbl1["rmse_e_meV"].max()
            rf_min, rf_max = tbl1["rmse_f_eVA"].min(), tbl1["rmse_f_eVA"].max()
            n_total = tbl1["n_configs"].sum()
            print(f"\n**Slide summary ({label}):**")
            print(f"  Median energy error: {e_min:.1f} – {e_max:.1f} meV/atom")
            print(f"  Median force RMSE:   {f_min:.2f} – {f_max:.2f} eV/Å")
            print(f"  Energy RMSE:         {re_min:.1f} – {re_max:.1f} meV/atom")
            print(f"  Force RMSE:          {rf_min:.2f} – {rf_max:.2f} eV/Å")
            print(f"  across {n_total:,} test holdout configs")
            print(f"  ({', '.join(tbl1['physical_group'].tolist())})")

        print_markdown(tbl2, f"{label}: By Config Type (all)")

        suffix = label.lower()
        p1 = OUTDIR / f"test_rmse_by_physical_group_{suffix}.csv"
        p2 = OUTDIR / f"test_rmse_by_config_type_{suffix}.csv"
        tbl1.to_csv(p1, index=False)
        tbl2.to_csv(p2, index=False)
        print(f"\nSaved: {p1.name}, {p2.name}")


if __name__ == "__main__":
    main()
