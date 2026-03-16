#!/usr/bin/env python3
"""Analyze prediction errors broken down by raw config_type from the XYZ file.

The precomputed_data.json only stores coarsened 'section' labels (e.g. "NEB",
"Point defects"). This script recovers the fine-grained config_type from the
original XYZ test file, joins it with the error metrics, and produces summary
tables and faceted plots organized by physical group.

Key design choice: the original normalize_config_type() checks for "aa" early,
sending e.g. vacancy_aa to "Adversarial attacks" rather than "Point defects".
Here we classify by *physical nature* first (stripping _aa/_aa-mc suffixes),
then mark the adversarial flag separately, so vacancy_aa stays with vacancies.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from ase.io import read as ase_read

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plotting_utils import set_pub_style, save_fig, TOL_BRIGHT, DOUBLE_COL, SINGLE_COL, fig_size

set_pub_style()

# Physical groups based on the underlying simulation type, NOT the aa suffix.
PHYSICAL_GROUP_ORDER = [
    "Bulk crystals",
    "Surfaces",
    "Point defects",
    "NEB",
    "Elastic",
    "Liquids",
    "Composition explore",
    "Phonon",
]

ERROR_METRICS = ["e_pa_abs_meV", "f_rmse", "p_abs_GPa", "sigma_rmse_GPa"]
METRIC_LABELS = {
    "e_pa_abs_meV": "Energy RMSE (meV/atom)",
    "f_rmse": "Force RMSE (eV/Å)",
    "p_abs_GPa": "Pressure RMSE (GPa)",
    "sigma_rmse_GPa": "Stress RMSE (GPa)",
}


def rmse_agg(vals: pd.Series) -> float:
    """RMSE across structures: sqrt(mean(x^2)).

    Works for both absolute-error columns (e_pa_abs_meV, p_abs_GPa) where
    squaring the absolute value equals squaring the signed error, and for
    per-structure RMSE columns (f_rmse, sigma_rmse_GPa) where this gives
    the correctly pooled RMSE.
    """
    return float(np.sqrt(np.mean(vals.values ** 2)))

DEFAULT_EXCLUDE = ["phonon", "phonon_aa"]


# ── physical grouping ────────────────────────────────────────────────────────

def physical_group(config_type: str) -> str:
    """Classify config_type by its physical nature, ignoring _aa/_aa-mc suffix."""
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
    # Bulk crystal types
    if base in ("bcc_distorted", "fcc", "hcp", "dia", "A15", "C15",
                "a15", "c15", "bcc", "fcc_distorted"):
        return "Bulk crystals"
    return "Other"


# ── data loading ─────────────────────────────────────────────────────────────

def load_precomputed(json_path: Path) -> dict[str, pd.DataFrame]:
    """Load the three DataFrames from precomputed_data.json."""
    with open(json_path, "r") as f:
        data = json.load(f)
    frames = {}
    for key in ("gen_eval_df", "loss_eval_df", "loss_groups_eval_df"):
        frames[key] = pd.DataFrame(data[key]) if data.get(key) else pd.DataFrame()
    return frames


def build_config_type_map(xyz_path: Path) -> pd.DataFrame:
    """Read XYZ and return a DataFrame mapping structure_index -> config_type + metadata."""
    atoms_list = ase_read(str(xyz_path), index=":", format="extxyz")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    rows = []
    for i, at in enumerate(atoms_list):
        info = getattr(at, "info", {}) or {}
        raw_ct = info.get("config_type", None)
        ct = str(raw_ct) if raw_ct is not None else "unknown"
        symbols = sorted(set(at.get_chemical_symbols()))
        rows.append({
            "structure_index": i,
            "config_type": ct,
            "physical_group": physical_group(ct),
            "is_adversarial": bool(re.search(r"_aa(-mc)?$", ct)),
            "composition": "-".join(symbols),
        })
    return pd.DataFrame(rows)


def enrich(df: pd.DataFrame, ct_map: pd.DataFrame) -> pd.DataFrame:
    """Left-join config_type, physical_group, etc. onto an error DataFrame."""
    if df is None or len(df) == 0:
        return df
    return df.merge(ct_map, on="structure_index", how="left")


# ── filtering ────────────────────────────────────────────────────────────────

def apply_exclusions(df: pd.DataFrame, exclude: list[str]) -> pd.DataFrame:
    if not exclude or len(df) == 0:
        return df
    mask = ~df["config_type"].isin(exclude)
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        print(f"  [FILTER] Excluded {n_dropped} rows matching {exclude}")
    return df[mask].copy()


# ── summary statistics ───────────────────────────────────────────────────────

def summarize(df: pd.DataFrame, group_col: str, model_col: str) -> pd.DataFrame:
    """Per-group summary stats for each error metric (long-form)."""
    rows = []
    for (grp, mdl), sub in df.groupby([group_col, model_col], observed=True):
        for m in ERROR_METRICS:
            vals = sub[m].dropna()
            if len(vals) == 0:
                continue
            rows.append({
                group_col: grp,
                model_col: mdl,
                "metric": m,
                "count": len(vals),
                "rmse": rmse_agg(vals),
                "mae": vals.mean(),
                "median": vals.median(),
                "std": vals.std(),
                "max": vals.max(),
            })
    return pd.DataFrame(rows)


def summarize_distribution(df: pd.DataFrame, model_col: str) -> pd.DataFrame:
    """Per-config_type / per-model distribution stats for each error metric.

    Returns a long-form DataFrame with columns:
        physical_group, config_type, <model_col>, metric,
        count, mean, median, q25, q75, p05, p95, max
    """
    rows = []
    for (pg, ct, mdl), sub in df.groupby(
        ["physical_group", "config_type", model_col], observed=True
    ):
        for m in ERROR_METRICS:
            vals = sub[m].dropna()
            if len(vals) == 0:
                continue
            rows.append({
                "physical_group": pg,
                "config_type": ct,
                model_col: mdl,
                "metric": m,
                "count": len(vals),
                "rmse": rmse_agg(vals),
                "mae": vals.mean(),
                "median": vals.median(),
                "q25": vals.quantile(0.25),
                "q75": vals.quantile(0.75),
                "p05": vals.quantile(0.05),
                "p95": vals.quantile(0.95),
                "max": vals.max(),
            })
    return pd.DataFrame(rows)


def print_summary_table(df: pd.DataFrame, model_col: str, label: str) -> None:
    """Print a compact table: physical_group / config_type / metric means across model variants."""
    print(f"\n{'='*90}")
    print(f"  {label}")
    print(f"{'='*90}")
    for metric in ERROR_METRICS:
        sub = df.dropna(subset=[metric])
        if len(sub) == 0:
            continue
        pivot = sub.pivot_table(
            index=["physical_group", "config_type"],
            columns=model_col, values=metric,
            aggfunc=lambda x: rmse_agg(x),
            observed=True,
        )
        pivot["worst"] = pivot.max(axis=1)
        pivot = pivot.sort_values(["physical_group", "worst"], ascending=[True, False])
        pivot = pivot.drop(columns="worst")
        print(f"\n  ── {METRIC_LABELS[metric]} (RMSE across structures) ──")
        with pd.option_context("display.max_rows", 60, "display.float_format",
                               "{:.4f}".format, "display.width", 150):
            print(pivot.to_string())


# ── plotting: faceted by physical group ──────────────────────────────────────

def _get_model_color_map(model_vals):
    return {m: TOL_BRIGHT[i % len(TOL_BRIGHT)] for i, m in enumerate(model_vals)}


def plot_faceted_bars(df: pd.DataFrame, model_col: str, metric: str,
                      outdir: Path, prefix: str) -> None:
    """One figure per metric: subplots faceted by physical_group, bars per config_type.

    Each subplot shows a horizontal grouped-bar chart with one group of bars
    per config_type and one bar per model variant.
    """
    groups_present = [g for g in PHYSICAL_GROUP_ORDER
                      if g in df["physical_group"].unique()]
    if not groups_present:
        return
    model_vals = sorted(df[model_col].dropna().unique(), key=str)
    colors = _get_model_color_map(model_vals)
    n_models = max(len(model_vals), 1)

    group_dfs = {}
    for g in groups_present:
        gsub = df[df["physical_group"] == g].dropna(subset=[metric])
        if len(gsub) == 0:
            continue
        means = gsub.pivot_table(index="config_type", columns=model_col,
                                  values=metric,
                                  aggfunc=lambda x: rmse_agg(x),
                                  observed=True)
        means = means.reindex(columns=model_vals)
        means["_worst"] = means.max(axis=1)
        means = means.sort_values("_worst", ascending=True).drop(columns="_worst")
        group_dfs[g] = means

    if not group_dfs:
        return

    n_panels = len(group_dfs)
    heights = [max(len(v), 1) for v in group_dfs.values()]
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(DOUBLE_COL, sum(h * 0.38 + 0.6 for h in heights) + 1.2),
        gridspec_kw={"height_ratios": [h + 1 for h in heights]},
        squeeze=False,
    )
    axes = axes.ravel()

    bar_height = 0.8 / n_models
    for ax, (grp, means) in zip(axes, group_dfs.items()):
        n_ct = len(means)
        y = np.arange(n_ct)
        for j, m in enumerate(model_vals):
            if m in means.columns:
                vals = means[m].values
                ax.barh(y + j * bar_height - 0.4 + bar_height / 2, vals,
                        bar_height, label=str(m) if ax is axes[0] else None,
                        color=colors[m], alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.set_yticks(y)
        ax.set_yticklabels(means.index, fontsize=8)
        ax.set_title(grp, fontsize=10, fontweight="bold", loc="left")
        ax.set_xlabel(METRIC_LABELS[metric], fontsize=8)
        ax.invert_yaxis()

    axes[0].legend(fontsize=7, title=model_col, title_fontsize=7,
                   bbox_to_anchor=(1.0, 1.0), loc="upper left")
    fig.suptitle(f"{METRIC_LABELS[metric]} by config_type", fontsize=12, y=1.0)
    fig.tight_layout()
    safe = metric.replace("/", "_")
    save_fig(fig, outdir / f"faceted_{prefix}_{safe}")
    plt.close(fig)


def plot_faceted_distribution(df: pd.DataFrame, model_col: str, metric: str,
                              outdir: Path, prefix: str) -> None:
    """Faceted horizontal interval plot showing per-config error distributions.

    For each physical_group panel, each config_type row shows one marker per
    model: a thin whisker spanning p05–p95, a thick segment for IQR (q25–q75),
    and a dot at the median.  Rows are sorted by worst-model median.
    """
    groups_present = [g for g in PHYSICAL_GROUP_ORDER
                      if g in df["physical_group"].unique()]
    if not groups_present:
        return
    model_vals = sorted(df[model_col].dropna().unique(), key=str)
    colors = _get_model_color_map(model_vals)
    n_models = max(len(model_vals), 1)

    dist = summarize_distribution(df, model_col)
    dist = dist[dist["metric"] == metric]
    if len(dist) == 0:
        return

    group_data: dict[str, pd.DataFrame] = {}
    for g in groups_present:
        gsub = dist[dist["physical_group"] == g]
        if len(gsub) == 0:
            continue
        worst_median = gsub.groupby("config_type", observed=True)["median"].max()
        ct_order = worst_median.sort_values(ascending=True).index.tolist()
        gsub = gsub.copy()
        gsub["_ct_rank"] = gsub["config_type"].map(
            {ct: i for i, ct in enumerate(ct_order)})
        group_data[g] = gsub

    if not group_data:
        return

    n_panels = len(group_data)
    heights = [max(gd["config_type"].nunique(), 1) for gd in group_data.values()]
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(DOUBLE_COL, sum(h * 0.4 * n_models + 0.8 for h in heights) + 1.5),
        gridspec_kw={"height_ratios": [h * n_models + 1 for h in heights]},
        squeeze=False,
    )
    axes = axes.ravel()

    model_offset = 0.8 / n_models

    for ax, (grp, gsub) in zip(axes, group_data.items()):
        ct_order = gsub.sort_values("_ct_rank")["config_type"].unique()
        n_ct = len(ct_order)
        ct_to_y = {ct: i for i, ct in enumerate(ct_order)}

        for j, mv in enumerate(model_vals):
            msub = gsub[gsub[model_col] == mv]
            for _, row in msub.iterrows():
                y_base = ct_to_y[row["config_type"]]
                y = y_base + (j - (n_models - 1) / 2) * model_offset
                c = colors[mv]

                ax.plot([row["p05"], row["p95"]], [y, y],
                        color=c, linewidth=0.8, alpha=0.5, solid_capstyle="round")
                ax.plot([row["q25"], row["q75"]], [y, y],
                        color=c, linewidth=3.0, alpha=0.7, solid_capstyle="round")
                ax.plot(row["median"], y, "o", color=c, markersize=4,
                        markeredgecolor="white", markeredgewidth=0.4,
                        label=str(mv) if (ax is axes[0]
                                          and row["config_type"] == ct_order[0])
                        else None)

        ax.set_yticks(range(n_ct))
        ax.set_yticklabels(ct_order, fontsize=8)
        ax.set_title(grp, fontsize=10, fontweight="bold", loc="left")
        ax.set_xlabel(METRIC_LABELS[metric], fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis="x", linewidth=0.3, alpha=0.4)

    axes[0].legend(fontsize=7, title=model_col, title_fontsize=7,
                   bbox_to_anchor=(1.0, 1.0), loc="upper left")
    fig.suptitle(f"{METRIC_LABELS[metric]} distribution by config_type\n"
                 f"(dot = median, thick = IQR, whisker = 5th–95th pctl)",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    safe = metric.replace("/", "_")
    save_fig(fig, outdir / f"faceted_distribution_{prefix}_{safe}")
    plt.close(fig)


def plot_distribution_heatmap(df: pd.DataFrame, model_col: str, metric: str,
                              outdir: Path, prefix: str,
                              stat: str = "median") -> None:
    """Compact heatmap: rows = physical_group / config_type, columns = models.

    Cell values are a robust distribution statistic (default: median).
    One figure per metric.
    """
    model_vals = sorted(df[model_col].dropna().unique(), key=str)
    dist = summarize_distribution(df, model_col)
    dist = dist[dist["metric"] == metric]
    if len(dist) == 0:
        return

    pivot = dist.pivot_table(
        index=["physical_group", "config_type"],
        columns=model_col, values=stat, observed=True,
    )
    pivot = pivot.reindex(columns=model_vals)

    grp_order = {g: i for i, g in enumerate(PHYSICAL_GROUP_ORDER)}
    pivot["_grp_rank"] = pivot.index.get_level_values("physical_group").map(
        lambda g: grp_order.get(g, 99))
    pivot = pivot.sort_values(["_grp_rank"]).drop(columns="_grp_rank")

    labels = [f"{pg} / {ct}" for pg, ct in pivot.index]
    n_rows, n_cols = pivot.shape

    vmax = pivot.values[~np.isnan(pivot.values)].max() if n_rows > 0 else 1.0
    q95 = np.nanquantile(pivot.values, 0.95) if n_rows > 0 else vmax
    clip_val = max(q95, 1e-12)

    normed = pivot.copy()
    for col in model_vals:
        if col in normed.columns:
            normed[col] = normed[col].clip(upper=clip_val) / clip_val

    fig, ax = plt.subplots(
        figsize=(max(SINGLE_COL, n_cols * 1.0 + 2.0),
                 max(4, len(labels) * 0.32 + 1.5))
    )
    im = ax.imshow(normed.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([str(m) for m in model_vals],
                       rotation=30, ha="right", fontsize=8)

    for i in range(normed.shape[0]):
        for j in range(normed.shape[1]):
            raw = pivot.iloc[i, j]
            if not np.isnan(raw):
                txt = f"{raw:.2f}" if raw < 100 else f"{raw:.0f}"
                color = "white" if normed.iloc[i, j] > 0.6 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                        color=color)

    prev_grp = None
    for idx, (pg, _ct) in enumerate(pivot.index):
        if prev_grp is not None and pg != prev_grp:
            ax.axhline(idx - 0.5, color="black", linewidth=0.8)
        prev_grp = pg

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(f"Normalized {stat} (0 = best, 1 = 95th pctl)", fontsize=7)
    ax.set_title(f"{METRIC_LABELS[metric]} — {stat} across models", fontsize=10)
    fig.tight_layout()
    safe = metric.replace("/", "_")
    save_fig(fig, outdir / f"distribution_heatmap_{prefix}_{safe}_{stat}")
    plt.close(fig)


def plot_overview_heatmap(df: pd.DataFrame, model_col: str, model_val,
                          outdir: Path, prefix: str) -> None:
    """Single heatmap for one model variant: rows = config_types grouped by physical_group,
    columns = error metrics (normalized to [0, 1] per metric for colour comparability)."""
    sub = df[df[model_col] == model_val].copy()
    if len(sub) == 0:
        return

    means = sub.groupby(["physical_group", "config_type"], observed=True)[ERROR_METRICS].agg(rmse_agg)
    grp_order = {g: i for i, g in enumerate(PHYSICAL_GROUP_ORDER)}
    means["_grp_rank"] = means.index.get_level_values("physical_group").map(
        lambda g: grp_order.get(g, 99))
    means = means.sort_values(["_grp_rank"]).drop(columns="_grp_rank")

    normed = means.copy()
    for m in ERROR_METRICS:
        col = normed[m]
        vmax = col.quantile(0.95)
        normed[m] = col.clip(upper=vmax) / vmax if vmax > 0 else 0.0

    labels = [f"{pg} / {ct}" for pg, ct in means.index]
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.9,
                                     max(4, len(labels) * 0.32 + 1.5)))
    im = ax.imshow(normed.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xticks(range(len(ERROR_METRICS)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in ERROR_METRICS],
                       rotation=30, ha="right", fontsize=8)

    for i in range(normed.shape[0]):
        for j in range(normed.shape[1]):
            raw = means.iloc[i, j]
            if not np.isnan(raw):
                txt = f"{raw:.2f}" if raw < 100 else f"{raw:.0f}"
                color = "white" if normed.iloc[i, j] > 0.6 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6, color=color)

    # Horizontal lines between physical groups
    prev_grp = None
    for idx, (pg, _ct) in enumerate(means.index):
        if prev_grp is not None and pg != prev_grp:
            ax.axhline(idx - 0.5, color="black", linewidth=0.8)
        prev_grp = pg

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Normalized (0 = best, 1 = 95th pctl)", fontsize=7)
    safe_val = str(model_val).replace(" ", "_")
    ax.set_title(f"Error overview — {model_col} = {model_val}", fontsize=10)
    fig.tight_layout()
    save_fig(fig, outdir / f"overview_heatmap_{prefix}_{safe_val}")
    plt.close(fig)


def plot_adversarial_comparison(df: pd.DataFrame, model_col: str,
                                 outdir: Path, prefix: str) -> None:
    """Compare mean errors: base config_type vs its _aa counterpart, per model variant."""
    model_vals = sorted(df[model_col].dropna().unique(), key=str)
    base_types = sorted(df.loc[~df["is_adversarial"], "config_type"].unique())

    rows = []
    for bt in base_types:
        aa = bt + "_aa"
        if aa not in df["config_type"].values:
            continue
        for mv in model_vals:
            for metric in ERROR_METRICS:
                base_vals = df.loc[(df["config_type"] == bt) & (df[model_col] == mv), metric].dropna()
                aa_vals = df.loc[(df["config_type"] == aa) & (df[model_col] == mv), metric].dropna()
                if len(base_vals) == 0 or len(aa_vals) == 0:
                    continue
                base_rmse = rmse_agg(base_vals)
                aa_rmse = rmse_agg(aa_vals)
                rows.append({
                    "base_type": bt, model_col: mv, "metric": metric,
                    "base_rmse": base_rmse, "aa_rmse": aa_rmse,
                    "ratio": aa_rmse / base_rmse if base_rmse > 0 else np.nan,
                })
    if not rows:
        return
    comp = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 2, figsize=fig_size(DOUBLE_COL * 1.3, 0.9))
    for ax, metric in zip(axes.ravel(), ERROR_METRICS):
        msub = comp[comp["metric"] == metric]
        if len(msub) == 0:
            ax.set_visible(False)
            continue
        pivot = msub.pivot_table(index="base_type", columns=model_col,
                                  values="ratio", observed=True)
        pivot = pivot.reindex(columns=model_vals)
        x = np.arange(len(pivot))
        colors = _get_model_color_map(model_vals)
        w = 0.8 / max(len(model_vals), 1)
        for j, mv in enumerate(model_vals):
            if mv in pivot.columns:
                ax.bar(x + j * w - 0.4 + w / 2, pivot[mv].values, w,
                       color=colors[mv], alpha=0.85,
                       label=str(mv) if metric == ERROR_METRICS[0] else None)
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(f"AA / base ratio")
        ax.set_title(METRIC_LABELS[metric], fontsize=9)
    axes.ravel()[0].legend(fontsize=6, title=model_col, title_fontsize=6)
    fig.suptitle(f"Adversarial degradation ratio ({prefix})", fontsize=11, y=1.01)
    fig.tight_layout()
    save_fig(fig, outdir / f"adversarial_ratio_{prefix}")
    plt.close(fig)


# ── cascade readiness report ─────────────────────────────────────────────────

CASCADE_GROUPS = {
    "POINT DEFECTS (Frenkel pair production & clustering)": [
        "vacancy", "vacancy_aa", "vacancy-alloy", "vacancy-alloy_aa",
        "di-vacancy", "di-vacancy_aa", "tri-vacancy", "tri-vacancy_aa",
        "sia", "di-sia", "vac_aa-mc",
    ],
    "NEB (migration barriers & saddle-point topology)": [
        "neb", "neb_aa", "neb_aa-mc",
    ],
    "ELASTIC (lattice stiffness, channeling, TDE)": [
        "elasticity",
    ],
    "BCC BULK (reference lattice, distorted configs)": [
        "bcc_distorted", "A15", "A15_aa", "C15", "C15_aa",
        "fcc_aa", "hcp", "hcp_aa", "dia", "dia_aa",
    ],
    "SURFACES (near-surface cascades, sputtering)": [
        "surface_100", "surface_100_aa", "surface_110", "surface_110_aa",
        "surface_111", "surface_111_aa", "surface_112", "surface_112_aa",
        "gamma_surface", "gamma_surface_aa",
    ],
    "LIQUIDS (thermal spike phase)": [
        "liquid", "liquid_aa", "surf_liquid", "surf_liquid_aa",
    ],
    "COMPOSITION EXPLORE (off-stoichiometry robustness)": [
        "comp-explore", "comp-explore_aa",
    ],
}


def generate_cascade_report(frames: dict[str, pd.DataFrame],
                            ct_map: pd.DataFrame,
                            exclude: list[str],
                            outpath: Path | None = None) -> str:
    """Generate a compact text report focused on cascade-relevant config_types.

    Designed to be pasted directly into another LLM context window.
    """
    lines: list[str] = []
    w = lines.append
    w("=" * 78)
    w("CASCADE READINESS REPORT — MLIP Error Analysis by Configuration Type")
    w("=" * 78)
    w("")
    w("Units: E = meV/atom, F = eV/Å, P = GPa (pressure), S = GPa (stress)")
    w("Stats: RMSE (median) [max] over N structures in test set")
    w("All values are RMSE across structures in each group.")
    w("")

    analyses = [
        ("gen_eval_df", "generation", "GENERATION MODELS"),
        ("loss_eval_df", "loss_variant", "LOSS VARIANT MODELS"),
    ]

    for df_key, model_col, heading in analyses:
        df = frames.get(df_key)
        if df is None or len(df) == 0:
            continue
        df = enrich(df, ct_map)
        df = apply_exclusions(df, exclude)
        model_vals = sorted(df[model_col].dropna().unique(), key=str)

        w("-" * 78)
        w(f"  {heading}  (variants: {', '.join(str(v) for v in model_vals)})")
        w("-" * 78)

        for section_label, ct_list in CASCADE_GROUPS.items():
            present = [ct for ct in ct_list if ct in df["config_type"].values
                       and ct not in exclude]
            if not present:
                continue
            w(f"\n  {section_label}")

            for ct in present:
                for mv in model_vals:
                    sub = df[(df["config_type"] == ct) & (df[model_col] == mv)]
                    if len(sub) == 0:
                        continue
                    n = len(sub)
                    parts = [f"    {ct:22s} [{model_col}={mv}] (N={n:4d}):"]
                    for m, short in [("e_pa_abs_meV", "E"), ("f_rmse", "F"),
                                     ("p_abs_GPa", "P"), ("sigma_rmse_GPa", "S")]:
                        vals = sub[m].dropna()
                        if len(vals) == 0:
                            parts.append(f"  {short}=N/A")
                        else:
                            r = rmse_agg(vals)
                            parts.append(
                                f"  {short}={r:.2f}({vals.median():.2f})[{vals.max():.2f}]"
                            )
                    w("".join(parts))

        # Adversarial robustness summary
        w(f"\n  ADVERSARIAL ROBUSTNESS (RMSE ratio: aa / base)")
        base_types = sorted(set(
            ct for cts in CASCADE_GROUPS.values() for ct in cts
            if not ct.endswith("_aa") and not ct.endswith("_aa-mc")
            and ct + "_aa" in df["config_type"].values
            and ct not in exclude and ct + "_aa" not in exclude
        ))
        for bt in base_types:
            aa = bt + "_aa"
            for mv in model_vals:
                ratios = []
                for m, short in [("e_pa_abs_meV", "E"), ("f_rmse", "F"),
                                 ("p_abs_GPa", "P"), ("sigma_rmse_GPa", "S")]:
                    bvals = df.loc[(df["config_type"] == bt) & (df[model_col] == mv), m].dropna()
                    avals = df.loc[(df["config_type"] == aa) & (df[model_col] == mv), m].dropna()
                    if len(bvals) > 0 and len(avals) > 0:
                        br, ar = rmse_agg(bvals), rmse_agg(avals)
                        ratios.append(f"{short}={ar/br:.2f}x" if br > 0 else f"{short}=N/A")
                    else:
                        ratios.append(f"{short}=N/A")
                w(f"    {bt:22s} [{model_col}={mv}]:  {'  '.join(ratios)}")
        w("")

    report = "\n".join(lines)
    if outpath:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        outpath.write_text(report)
        print(f"  Saved cascade report: {outpath}")
    return report


# ── main pipeline ────────────────────────────────────────────────────────────

def run_analysis(frames: dict[str, pd.DataFrame], ct_map: pd.DataFrame,
                 outdir: Path, exclude: list[str]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    analyses = [
        ("gen_eval_df", "generation", "gen"),
        ("loss_eval_df", "loss_variant", "loss"),
        ("loss_groups_eval_df", "loss_variant", "loss_groups"),
    ]

    for df_key, model_col, prefix in analyses:
        df = frames.get(df_key)
        if df is None or len(df) == 0:
            print(f"\n[SKIP] {df_key} is empty.")
            continue

        print(f"\n{'#'*80}")
        print(f"# Analyzing: {df_key}  (model_col={model_col})")
        print(f"{'#'*80}")

        df = enrich(df, ct_map)
        df = apply_exclusions(df, exclude)

        ct_covered = sorted(df["config_type"].unique())
        print(f"  config_types in analysis ({len(ct_covered)}): {ct_covered}")

        # ── summary tables ────────────────────────────────────────────────
        print_summary_table(df, model_col, f"{df_key} — all config_types")

        ct_summary = summarize(df, "config_type", model_col)
        pg_summary = summarize(df, "physical_group", model_col)
        ct_summary.to_csv(outdir / f"{prefix}_config_type_summary.csv", index=False)
        pg_summary.to_csv(outdir / f"{prefix}_physical_group_summary.csv", index=False)
        print(f"  Saved: {prefix}_config_type_summary.csv, {prefix}_physical_group_summary.csv")

        # ── faceted bar plots (one per metric) ────────────────────────────
        for metric in ERROR_METRICS:
            plot_faceted_bars(df, model_col, metric, outdir, prefix)

        # ── distribution interval plots (one per metric) ──────────────────
        for metric in ERROR_METRICS:
            plot_faceted_distribution(df, model_col, metric, outdir, prefix)

        # ── distribution heatmaps (one per metric, rmse & median & p95) ────
        for metric in ERROR_METRICS:
            plot_distribution_heatmap(df, model_col, metric, outdir, prefix,
                                     stat="rmse")
            plot_distribution_heatmap(df, model_col, metric, outdir, prefix,
                                     stat="median")
            plot_distribution_heatmap(df, model_col, metric, outdir, prefix,
                                     stat="p95")

        # ── distribution summary CSV ──────────────────────────────────────
        dist_summary = summarize_distribution(df, model_col)
        dist_summary.to_csv(
            outdir / f"{prefix}_config_type_distribution_summary.csv",
            index=False,
        )
        print(f"  Saved: {prefix}_config_type_distribution_summary.csv")

        # ── overview heatmap (one per model variant) ──────────────────────
        for mv in sorted(df[model_col].dropna().unique(), key=str):
            plot_overview_heatmap(df, model_col, mv, outdir, prefix)

        # ── adversarial comparison ────────────────────────────────────────
        plot_adversarial_comparison(df, model_col, outdir, prefix)

        # ── loss_groups: also produce per-group summaries ─────────────────
        if "loss_group" in df.columns:
            for grp_name, grp_df in df.groupby("loss_group", observed=True):
                grp_prefix = f"{prefix}_{grp_name}"
                grp_ct = summarize(grp_df, "config_type", model_col)
                grp_ct.to_csv(outdir / f"{grp_prefix}_config_type_summary.csv", index=False)
                for metric in ERROR_METRICS:
                    plot_faceted_bars(grp_df, model_col, metric, outdir, grp_prefix)


def main():
    ap = argparse.ArgumentParser(
        description="Analyze prediction errors by raw config_type from XYZ.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="By default, phonon/phonon_aa structures are excluded (large force outliers).\n"
               "Use --no-exclude to include everything.",
    )
    ap.add_argument("--json", default="precomputed_data.json",
                    help="Path to precomputed_data.json")
    ap.add_argument("--xyz",
                    default="../../data/gen_10_data/"
                            "exploit_q-97.5_rmax5.50_lmax1_layers1_mlp256_seed42_test.xyz",
                    help="Path to the test XYZ file")
    ap.add_argument("--outdir", default="results/config_analysis",
                    help="Output directory (default: results/config_analysis)")
    ap.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE,
                    help="config_types to exclude (default: phonon phonon_aa)")
    ap.add_argument("--no-exclude", action="store_true",
                    help="Include all config_types (overrides --exclude)")
    ap.add_argument("--cascade-report", action="store_true",
                    help="Generate a compact text report for sharing with LLMs, "
                         "focused on cascade-relevant config_types")
    ap.add_argument("--report-only", action="store_true",
                    help="Only generate the cascade report (skip plots)")
    args = ap.parse_args()

    json_path = Path(args.json)
    xyz_path = Path(args.xyz)
    outdir = Path(args.outdir)
    exclude = [] if args.no_exclude else (args.exclude or [])

    if not json_path.exists():
        print(f"Error: {json_path} not found"); sys.exit(1)
    if not xyz_path.exists():
        print(f"Error: {xyz_path} not found"); sys.exit(1)

    print(f"[INFO] Loading precomputed data from {json_path}")
    frames = load_precomputed(json_path)
    for k, v in frames.items():
        print(f"  {k}: {len(v)} records")

    print(f"[INFO] Reading XYZ file: {xyz_path}")
    ct_map = build_config_type_map(xyz_path)
    n_types = ct_map["config_type"].nunique()
    print(f"  {len(ct_map)} structures, {n_types} unique config_types")
    print(f"  config_types: {sorted(ct_map['config_type'].unique())}")

    # Show physical group mapping
    print(f"\n  Physical group mapping:")
    for pg in PHYSICAL_GROUP_ORDER:
        members = sorted(ct_map.loc[ct_map["physical_group"] == pg, "config_type"].unique())
        if members:
            print(f"    {pg}: {members}")

    if exclude:
        print(f"\n  Excluding: {exclude}")

    if args.cascade_report or args.report_only:
        report = generate_cascade_report(
            frames, ct_map, exclude,
            outpath=outdir / "cascade_readiness_report.txt",
        )
        print(report)

    if not args.report_only:
        run_analysis(frames, ct_map, outdir, exclude)

    print(f"\n[DONE] Results written to {outdir}/")


if __name__ == "__main__":
    main()
