#!/usr/bin/env python3
"""Standalone plotting script for SAW sensitivity analysis paper figures.

Reproduces key figures from pre-computed CSV data only.
No dependency on sawbench — only numpy, pandas, matplotlib, and mpltern.

Usage:
    python plot_paper_figures.py --out_dir figures/
    python plot_paper_figures.py --data_dir data/sawbench-data --out_dir figures/

Data files required (all produced by example_top_n_commands.sh):
    ternary_grid.csv              -> Ternary KS heatmap
    ternary_top_n_cdfs.csv        -> KS bar chart + CDF comparison
    ks_vs_angle.csv               -> Angle sensitivity
    ks_vs_density.csv             -> Density sensitivity
    knob_share_by_scale.csv       -> Sensitivity summary panel (a)
    winner_frequency_by_scale.csv -> Sensitivity summary annotations
    mean_S1_ST_and_gap.csv        -> Sensitivity summary/global sensitivity
    dominance_heatmap.csv         -> Sensitivity summary panels (b, d)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plotting_utils import set_pub_style, save_fig, COLORS, DOUBLE_COL, fig_size

import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
import mpltern  # noqa: F401

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "sawbench-data"

set_pub_style(base_fontsize=10)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

RANK_COLORS = [
    COLORS["blue"],
    COLORS["red"],
    COLORS["green"],
    COLORS["purple"],
    "#332288",   # indigo (Tol Vibrant, colorblind-safe)
]

BASELINE_CFG = {
    'baseline_DFT': {
        'marker': '^', 'color': COLORS["cyan"],
        'label': 'DFT',
        'label_offset': (-0.05, 0.05, 0.00),
        'no_line': True,
    },
    'baseline_PredEXP': {
        'marker': 's', 'color': COLORS["yellow"],
        'label': 'PredEXP',
        'label_offset': (-0.06, -0.02, 0.08),
    },
    'baseline_MLIP': {
        'marker': 'D', 'color': COLORS["grey"],
        'label': 'MLIP',
        'label_offset': (0.03, 0.04, -0.07),
    },
}

METHOD_STYLES = {
    'Top-1':   ('o', COLORS["blue"]),
    'PredEXP': ('s', COLORS["red"]),
    'DFT':     ('^', COLORS["cyan"]),
    'MLIP':    ('D', COLORS["purple"]),
}

SENSITIVITY_COLORS = {
    "G": COLORS["blue"],
    "rho": COLORS["red"],
    "D": COLORS["green"],
    "K": COLORS["purple"],
    "other": COLORS["yellow"],
    "ebsd": COLORS["grey"],
}

DISPLAY = {
    "K": r"$K$", "D": r"$D$", "G": r"$G$", "rho": r"$\rho$",
    "phi1_err_deg": r"$\varphi_1$", "Phi_err_deg": r"$\Phi$",
    "phi2_err_deg": r"$\varphi_2$", "psi_err_deg": r"$\psi$",
}

KS_CMAP = LinearSegmentedColormap.from_list(
    'ks_sequential', ["#0B5E4D", "#1A8C6E", "#5AB89A", "#A8DCC8", "#E8F4F0", "#FFFFFF"], N=256)
KS_NORM = PowerNorm(gamma=0.5, vmin=0, vmax=1)

# Manually-determined barycentric (t, l, r) positions for A_Z contour labels.
# All placed near the bottom edge (low t), spread left-to-right along each
# contour's intercept with the D--G edge.  Slight t-stagger avoids crowding.
AZ_LABEL_POS = {
    0.5: (0.12, 0.58, 0.30),
    0.8: (0.04, 0.47, 0.49),
    1.0: (0.12, 0.32, 0.56),
    1.5: (0.04, 0.20, 0.76),
    2.0: (0.12, 0.06, 0.82),
}


def _save(fig, path, dpi=300):
    out_path = Path(path)
    stem = out_path.with_suffix("") if out_path.suffix else out_path
    save_fig(fig, stem, dpi=dpi)
    plt.close(fig)


def _label_panel(ax, label: str) -> None:
    """Add a bold panel label slightly outside the upper-left corner."""
    ax.text(-0.16, 1.06, label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="left")


def _load_summary_tables(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the CSV tables used by the summary/global sensitivity figures."""
    share = pd.read_csv(os.path.join(data_dir, "knob_share_by_scale.csv"))
    win = pd.read_csv(os.path.join(data_dir, "winner_frequency_by_scale.csv"))
    agg = pd.read_csv(os.path.join(data_dir, "mean_S1_ST_and_gap.csv"))
    dom = pd.read_csv(os.path.join(data_dir, "dominance_heatmap.csv"))
    return share, win, agg, dom


# ---------------------------------------------------------------------------
# Ternary coordinate helper
# ---------------------------------------------------------------------------

def _bary_coords(K, D, G, df):
    """Convert (K, D, G) GPa values to barycentric (t, l, r) via grid medians."""
    base_k, base_d, base_g = df["K_GPa"].median(), df["D_GPa"].median(), df["G_GPa"].median()
    tc = (K / base_k - 0.5) / 1.5 + 1.0 / 3.0
    lc = (D / base_d - 0.5) / 1.5 + 1.0 / 3.0
    rc = (G / base_g - 0.5) / 1.5 + 1.0 / 3.0
    s = tc + lc + rc
    return tc / s, lc / s, rc / s


# ===================================================================
# Ternary heatmap panel drawer
# ===================================================================

# Ternary-coordinate offsets for rank label callouts, chosen to radiate
# outward from the dense cluster so arrows don't cross.
RANK_LABEL_OFFSET = {
    1: (0.14, -0.06, -0.08),
    2: (0.10, -0.12, 0.02),
    3: (0.10, -0.10, 0.00),
    4: (0.06, -0.20, 0.14),
    5: (0.13, -0.10, -0.03),
}


def _draw_ternary(ax, data_dir, rank_colors=None):
    """Draw KS heatmap + A_Z contours + rank/baseline markers on *ax*."""
    df = pd.read_csv(os.path.join(data_dir, "ternary_grid.csv"))
    has_kdg = "K_GPa" in df.columns

    t = df["t"].values
    l_vals = df["l"].values
    r_vals = df["r"].values
    ks = df["KS"].values
    stable = df["born_stable"].values.astype(bool)
    az = df["Zener_A"].values

    mask = stable & np.isfinite(ks)
    ks_plot = np.where(mask, ks, np.nan)

    sc = ax.tripcolor(t, l_vals, r_vals, ks_plot, cmap=KS_CMAP, norm=KS_NORM,
                      shading='gouraud', alpha=0.75, rasterized=True)

    if np.any(~stable):
        ax.tripcolor(t, l_vals, r_vals, (~stable).astype(float),
                     cmap='Greys', vmin=0, vmax=3, alpha=0.10,
                     shading='gouraud', rasterized=True)

    # --- A_Z iso-anisotropy contours ---
    az_plot = np.where(stable & np.isfinite(az), az, np.nan)
    levels = sorted(AZ_LABEL_POS.keys())
    ax.tricontour(t, l_vals, r_vals, az_plot, levels=levels,
                  colors='black', linewidths=1.5, alpha=0.55)

    for level, (lt, ll, lr) in AZ_LABEL_POS.items():
        ax.text(lt, ll, lr, rf'$A_Z = {level:g}$',
                fontsize=11, color='black', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.15', fc='white',
                          ec='none', alpha=0.85),
                zorder=50)

    # --- Rank markers: small dots + arrow callout labels ---
    cdf_path = os.path.join(data_dir, "ternary_top_n_cdfs.csv")
    if not os.path.exists(cdf_path) or not has_kdg:
        return sc

    cdf_df = pd.read_csv(cdf_path)
    ranks = sorted(
        [s for s in cdf_df["source"].unique() if s.startswith("rank_")],
        key=lambda s: int(s.split("_")[1]),
    )
    if rank_colors is None:
        rank_colors = {int(r.split("_")[1]): RANK_COLORS[i % len(RANK_COLORS)]
                       for i, r in enumerate(ranks)}

    for rank_src in reversed(ranks):
        sub = cdf_df[cdf_df["source"] == rank_src].iloc[0]
        rn = int(rank_src.split("_")[1])
        tc, lc, rc = _bary_coords(float(sub["K_GPa"]),
                                   float(sub["D_GPa"]),
                                   float(sub["G_GPa"]), df)
        color = rank_colors.get(rn, 'black')
        z_order = 100 + (len(ranks) - rn)

        ax.scatter(tc, lc, rc, marker='o', s=60, c=color,
                   edgecolors='white', linewidths=0.8, zorder=z_order)

        dt, dl, dr = RANK_LABEL_OFFSET.get(rn, (0.06, -0.06, 0.00))
        lt, ll, lr_ = tc + dt, lc + dl, rc + dr
        ax.plot([tc, lt], [lc, ll], [rc, lr_],
                color=color, linewidth=0.8, alpha=0.6, zorder=z_order - 1)
        ax.text(lt, ll, lr_, str(rn), fontsize=13, fontweight='bold',
                color=color, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.15', fc='white',
                          ec=color, linewidth=0.6, alpha=0.90),
                zorder=z_order + 1)

    # --- Baseline markers ---
    baselines = [s for s in cdf_df["source"].unique() if s.startswith("baseline_")]
    for bl_src in baselines:
        cfg = BASELINE_CFG.get(bl_src)
        if cfg is None:
            continue
        bl_sub = cdf_df[cdf_df["source"] == bl_src].iloc[0]
        tc, lc, rc = _bary_coords(float(bl_sub["K_GPa"]),
                                   float(bl_sub["D_GPa"]),
                                   float(bl_sub["G_GPa"]), df)
        if not (0 <= tc <= 1 and 0 <= lc <= 1 and 0 <= rc <= 1):
            continue
        ax.scatter(tc, lc, rc, marker=cfg['marker'], s=200, c=cfg['color'],
                   edgecolors='white', linewidths=1.5, zorder=99,
                   label=cfg['label'])
        dt, dl, dr = cfg['label_offset']
        lt, ll, lr_ = tc + dt, lc + dl, rc + dr
        if not cfg.get('no_line', False):
            ax.plot([tc, lt], [lc, ll], [rc, lr_],
                    color=cfg['color'], linewidth=0.8, alpha=0.5, zorder=98)
        ax.text(lt, ll, lr_, cfg['label'],
                fontsize=12, fontweight='bold',
                color=cfg['color'], ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.15', fc='white',
                          ec=cfg['color'], linewidth=0.6, alpha=0.90),
                zorder=102)

    ax.legend(loc='upper left', framealpha=0.9, handletextpad=0.3,
              borderpad=0.3, fontsize=11)

    # --- Axis labels & tick marks ---
    if has_kdg:
        ax.set_tlabel("K (GPa)", fontweight='bold', fontsize=14)
        ax.set_llabel("D (GPa)", fontweight='bold', fontsize=14)
        ax.set_rlabel("G (GPa)", fontweight='bold', fontsize=14)
        n_ticks = 5
        bary = np.linspace(0, 1, n_ticks)
        for axis_obj, col in [(ax.taxis, "K_GPa"),
                              (ax.laxis, "D_GPa"),
                              (ax.raxis, "G_GPa")]:
            vals = np.linspace(df[col].min(), df[col].max(), n_ticks)
            axis_obj.set_ticks(bary)
            axis_obj.set_ticklabels([f"{v:.0f}" for v in vals], fontsize=12)

    return sc


# ===================================================================
# Standalone ternary (full-width)
# ===================================================================

def plot_ternary_heatmap(data_dir: str, out_dir: str) -> None:
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.9))
    ax = fig.add_subplot(111, projection='ternary')
    sc = _draw_ternary(ax, data_dir)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.08)
        cbar.set_label("KS metric")
    _save(fig, os.path.join(out_dir, "ternary_ks_heatmap.png"))


# ===================================================================
# KS bar chart comparison (new)
# ===================================================================

def plot_ks_comparison(data_dir: str, out_dir: str) -> None:
    """Horizontal bar chart of KS metric for all methods, sorted best-to-worst."""
    df = pd.read_csv(os.path.join(data_dir, "ternary_top_n_cdfs.csv"))

    rows = []
    for src in df["source"].unique():
        if src == "experimental":
            continue
        sub = df[df["source"] == src].iloc[0]
        ks_val = float(sub["KS"])
        if src.startswith("rank_"):
            rn = int(src.split("_")[1])
            color = RANK_COLORS[rn - 1] if rn <= len(RANK_COLORS) else RANK_COLORS[-1]
            rows.append((f"Rank {rn}", ks_val, color))
        elif src in BASELINE_CFG:
            cfg = BASELINE_CFG[src]
            rows.append((cfg["label"], ks_val, cfg["color"]))

    rows.sort(key=lambda x: x[1])
    labels = [r[0] for r in rows]
    ks_vals = [r[1] for r in rows]
    colors = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=fig_size(DOUBLE_COL, 0.45))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, ks_vals, color=colors, edgecolor='white',
                   linewidth=0.8, height=0.7)

    for i, (bar, ks) in enumerate(zip(bars, ks_vals)):
        ax.text(ks + 0.015, i, f"{ks:.3f}", va='center', fontsize=9,
                fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("KS Metric (lower = better)", fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8)
    plt.tight_layout()

    _save(fig, os.path.join(out_dir, "ks_comparison.png"))


# ===================================================================
# Combined ternary + KS bar chart (main publication figure)
# ===================================================================

def plot_ternary_and_ks(data_dir: str, out_dir: str) -> None:
    """Side-by-side (a) ternary KS heatmap + (b) KS bar chart."""
    cdf_df = pd.read_csv(os.path.join(data_dir, "ternary_top_n_cdfs.csv"))
    ranks = sorted(
        [s for s in cdf_df["source"].unique() if s.startswith("rank_")],
        key=lambda s: int(s.split("_")[1]),
    )
    rank_colors = {int(r.split("_")[1]): RANK_COLORS[i % len(RANK_COLORS)]
                   for i, r in enumerate(ranks)}

    fig = plt.figure(figsize=(DOUBLE_COL * 2, DOUBLE_COL * 0.95))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.0, 1], wspace=0.25)

    # --- (a) Ternary heatmap, colorbar on the left ---
    ax_tern = fig.add_subplot(gs[0], projection='ternary')
    sc = _draw_ternary(ax_tern, data_dir, rank_colors)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax_tern, location='left',
                            shrink=0.6, pad=0.10)
        cbar.set_label("KS metric", fontweight='bold', fontsize=13)
        cbar.ax.tick_params(labelsize=12)
    ax_tern.text(-0.15, 1.05, "(a)", transform=ax_tern.transAxes,
                 fontsize=14, fontweight='bold', va='top')

    # --- (b) KS bar chart ---
    ax_bar = fig.add_subplot(gs[1])
    rows = []
    for src in cdf_df["source"].unique():
        if src == "experimental":
            continue
        sub = cdf_df[cdf_df["source"] == src].iloc[0]
        ks_val = float(sub["KS"])
        if src.startswith("rank_"):
            rn = int(src.split("_")[1])
            color = rank_colors.get(rn, RANK_COLORS[-1])
            rows.append((f"Rank {rn}", ks_val, color))
        elif src in BASELINE_CFG:
            cfg = BASELINE_CFG[src]
            rows.append((cfg["label"], ks_val, cfg["color"]))
    rows.sort(key=lambda x: x[1])
    labels = [r[0] for r in rows]
    ks_vals = [r[1] for r in rows]
    colors = [r[2] for r in rows]

    y_pos = np.arange(len(labels))
    bars = ax_bar.barh(y_pos, ks_vals, color=colors, edgecolor='white',
                       linewidth=0.8, height=0.7)
    for i, (bar, ks) in enumerate(zip(bars, ks_vals)):
        ax_bar.text(ks + 0.015, i, f"{ks:.3f}", va='center', fontsize=11,
                    fontweight='bold')
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, fontweight='bold', fontsize=12)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 1.18)
    ax_bar.set_xlabel("KS Metric (lower = better)", fontweight='bold', fontsize=13)
    ax_bar.tick_params(axis='x', labelsize=12)
    ax_bar.axvline(x=0, color='black', linewidth=0.8)
    ax_bar.text(-0.18, 1.05, "(b)", transform=ax_bar.transAxes,
                fontsize=14, fontweight='bold', va='top')

    _save(fig, os.path.join(out_dir, "ternary_and_ks.png"))


# ===================================================================
# Combined ternary + CDF (kept as option, not default)
# ===================================================================

def plot_combined_ternary_cdfs(data_dir: str, out_dir: str) -> None:
    cdf_path = os.path.join(data_dir, "ternary_top_n_cdfs.csv")
    cdf_df = pd.read_csv(cdf_path)
    ranks = sorted(
        [s for s in cdf_df["source"].unique() if s.startswith("rank_")],
        key=lambda s: int(s.split("_")[1]),
    )
    rank_colors = {int(r.split("_")[1]): RANK_COLORS[i % len(RANK_COLORS)]
                   for i, r in enumerate(ranks)}

    fig = plt.figure(figsize=(DOUBLE_COL * 2, DOUBLE_COL * 0.85))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.15, 1], wspace=0.32)

    ax_tern = fig.add_subplot(gs[0], projection='ternary')
    sc = _draw_ternary(ax_tern, data_dir, rank_colors)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax_tern, shrink=0.6, pad=0.08)
        cbar.set_label("KS metric")
    ax_tern.text(-0.05, 1.05, "(a)", transform=ax_tern.transAxes,
                 fontsize=12, fontweight='bold', va='top')

    ax_cdf = fig.add_subplot(gs[1])
    _draw_cdfs(ax_cdf, data_dir, rank_colors)
    ax_cdf.text(-0.12, 1.05, "(b)", transform=ax_cdf.transAxes,
                fontsize=12, fontweight='bold', va='top')

    _save(fig, os.path.join(out_dir, "ternary_and_cdfs.png"))


# ===================================================================
# Standalone CDFs (kept as option, not default)
# ===================================================================

def _draw_cdfs(ax, data_dir, rank_colors=None):
    """Draw experimental + baseline + top-N CDFs on *ax*."""
    df = pd.read_csv(os.path.join(data_dir, "ternary_top_n_cdfs.csv"))

    exp = df[df["source"] == "experimental"]["frequency_mhz"].dropna().sort_values().values
    exp_cdf = np.arange(1, len(exp) + 1) / len(exp)
    ax.plot(exp, exp_cdf, 'k-', linewidth=2.5, label='Experimental', zorder=100)

    for src in sorted(df["source"].unique()):
        if not src.startswith("baseline_"):
            continue
        cfg = BASELINE_CFG.get(src)
        if cfg is None:
            continue
        sub = df[df["source"] == src]
        freqs = sub["frequency_mhz"].dropna().sort_values().values
        if len(freqs) == 0:
            continue
        cdf_vals = np.arange(1, len(freqs) + 1) / len(freqs)
        ks_val = sub["KS"].iloc[0]
        ax.plot(freqs, cdf_vals, color=cfg['color'], linestyle=':',
                linewidth=2.0, alpha=0.9,
                label=f"{cfg['label']} (KS={ks_val:.3f})", zorder=90)

    ranks = sorted(
        [s for s in df["source"].unique() if s.startswith("rank_")],
        key=lambda s: int(s.split("_")[1]),
    )
    if rank_colors is None:
        rank_colors = {int(r.split("_")[1]): RANK_COLORS[i % len(RANK_COLORS)]
                       for i, r in enumerate(ranks)}

    for rank_src in ranks:
        sub = df[df["source"] == rank_src]
        freqs = sub["frequency_mhz"].dropna().sort_values().values
        if len(freqs) == 0:
            continue
        cdf_vals = np.arange(1, len(freqs) + 1) / len(freqs)
        rn = int(rank_src.split("_")[1])
        ks_val = sub["KS"].iloc[0]
        color = rank_colors.get(rn, 'black')
        ax.plot(freqs, cdf_vals, color=color, linewidth=1.8, alpha=0.9,
                label=f"Rank {rn} (KS={ks_val:.3f})")

    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Cumulative Probability')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_ylim([0, 1.02])
    ax.set_xlim([190, 400])
    ax.legend(loc='upper left', framealpha=0.9, handlelength=1.5)


def plot_top_n_cdfs(data_dir: str, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    _draw_cdfs(ax, data_dir)
    _save(fig, os.path.join(out_dir, "top_n_cdfs.png"))


# ===================================================================
# Angle sensitivity
# ===================================================================

def plot_angle_sensitivity(data_dir: str, out_dir: str) -> None:
    df = pd.read_csv(os.path.join(data_dir, "ks_vs_angle.csv"))
    fig, ax = plt.subplots(figsize=fig_size(DOUBLE_COL, 0.55))

    for method in df["method"].unique():
        sub = df[df["method"] == method].sort_values("angle_deg")
        marker, color = METHOD_STYLES.get(method, ('o', 'black'))
        angles = sub["angle_deg"].values
        ks = sub["KS"].values

        ax.plot(angles, ks, marker=marker, color=color,
                linewidth=1.5, markersize=6, label=method, alpha=0.9,
                markeredgewidth=1.0, markeredgecolor='white', zorder=2)

        min_idx = np.argmin(ks)
        y_off = 25 if ks[min_idx] < 0.2 else -25
        ax.annotate(f'{angles[min_idx]:.0f}°\n{ks[min_idx]:.4f}',
                    xy=(angles[min_idx], ks[min_idx]),
                    xytext=(0, y_off), textcoords='offset points',
                    fontsize=7, ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen',
                              edgecolor='darkgreen', linewidth=1.5, alpha=0.95),
                    zorder=1000,
                    arrowprops=dict(arrowstyle='->', color='darkgreen',
                                   lw=1.0, alpha=0.7))

    best_ks = df["KS"].min()
    ax.axhline(y=best_ks, color='green', linestyle='--', linewidth=1.0,
               alpha=0.3, zorder=1)
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.set_xlabel(r'In-Plane Angle $\psi$ (degrees)', fontweight='bold')
    ax.set_ylabel('KS Metric (lower = better)', fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8, zorder=0)
    ax.legend(loc='best')
    plt.tight_layout()

    _save(fig, os.path.join(out_dir, "angle_sensitivity.png"))


# ===================================================================
# Density sensitivity
# ===================================================================

def plot_density_sensitivity(data_dir: str, out_dir: str) -> None:
    df = pd.read_csv(os.path.join(data_dir, "ks_vs_density.csv"))
    fig, ax = plt.subplots(figsize=fig_size(DOUBLE_COL, 0.55))

    for method in df["method"].unique():
        sub = df[df["method"] == method].sort_values("density_pct")
        marker, color = METHOD_STYLES.get(method, ('o', 'black'))
        pcts = sub["density_pct"].values
        ks = sub["KS"].values

        ax.plot(pcts, ks, marker=marker, color=color,
                linewidth=1.5, markersize=6, label=method, alpha=0.9,
                markeredgewidth=1.0, markeredgecolor='white', zorder=2)

        min_idx = np.argmin(ks)
        pct_label = f"{pcts[min_idx]:+.1f}%" if abs(pcts[min_idx]) > 0.01 else "0%"
        y_off = 25 if ks[min_idx] < 0.2 else -25
        ax.annotate(f'{pct_label}\n{ks[min_idx]:.4f}',
                    xy=(pcts[min_idx], ks[min_idx]),
                    xytext=(0, y_off), textcoords='offset points',
                    fontsize=7, ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen',
                              edgecolor='darkgreen', linewidth=1.5, alpha=0.95),
                    zorder=1000,
                    arrowprops=dict(arrowstyle='->', color='darkgreen',
                                   lw=1.0, alpha=0.7))

    best_ks = df["KS"].min()
    ax.axhline(y=best_ks, color='green', linestyle='--', linewidth=1.0,
               alpha=0.3, zorder=1)
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.set_xlabel('Density Variation (%)', fontweight='bold')
    ax.set_ylabel('KS Metric (lower = better)', fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8, zorder=0)
    ax.legend(loc='best')
    plt.tight_layout()

    _save(fig, os.path.join(out_dir, "density_sensitivity.png"))


# ===================================================================
# Sensitivity summary
# ===================================================================

def plot_sensitivity_summary(data_dir: str, out_dir: str,
                             interaction_threshold: float = 0.05) -> None:
    share, win, agg, dom = _load_summary_tables(data_dir)

    scales = sorted(share["scale"].unique())
    scale_labels = [f"±{s}°" for s in scales]
    x = np.arange(len(scales))

    fig = plt.figure(figsize=fig_size(DOUBLE_COL * 1.35, 0.88))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.30)

    ax_a = fig.add_subplot(gs[0, 0])
    share_by_scale = share.set_index("scale").loc[scales]
    minor_groups = ["rho", "D", "other"]
    width = 0.22
    offsets = np.linspace(-width, width, len(minor_groups))
    for offset, gname in zip(offsets, minor_groups):
        vals = 100.0 * share_by_scale[gname].values
        ax_a.bar(x + offset, vals, width=width,
                 color=SENSITIVITY_COLORS[gname], edgecolor="white",
                 linewidth=0.8, label=DISPLAY.get(gname, gname))
    g_vals = 100.0 * share_by_scale["G"].values
    for xpos, g_val in zip(x, g_vals):
        ax_a.text(xpos, 1.30, f"$G$ = {g_val:.1f}%", ha="center", va="top",
                  fontsize=9, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                            edgecolor="0.85", alpha=0.95))
    ax_a.set_ylim(0, 1.4)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(scale_labels)
    ax_a.set_ylabel("Non-$G$ share of total $S_T$ (%)")
    ax_a.set_title("Minor contributors after dominant $G$")
    ax_a.grid(True, axis="y", alpha=0.2, linestyle="--", linewidth=0.8)
    ax_a.legend(title="Group", loc="lower right", fontsize=8)
    _label_panel(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1])
    dom_by_scale = [
        100.0 * dom.loc[dom["scale_deg"] == s, "G_share"].values for s in scales
    ]
    box = ax_b.boxplot(dom_by_scale, positions=x, widths=0.55, patch_artist=True,
                       showfliers=False,
                       medianprops=dict(color=SENSITIVITY_COLORS["G"], linewidth=1.5),
                       whiskerprops=dict(color=SENSITIVITY_COLORS["G"], linewidth=1.0),
                       capprops=dict(color=SENSITIVITY_COLORS["G"], linewidth=1.0))
    for patch in box["boxes"]:
        patch.set_facecolor(SENSITIVITY_COLORS["G"])
        patch.set_alpha(0.22)
        patch.set_edgecolor(SENSITIVITY_COLORS["G"])
        patch.set_linewidth(1.0)
    for xpos, values in zip(x, dom_by_scale):
        jitter = np.linspace(-0.10, 0.10, len(values))
        ax_b.scatter(np.full(len(values), xpos) + jitter, values,
                     s=16, color=SENSITIVITY_COLORS["G"], alpha=0.45,
                     edgecolor="white", linewidth=0.3, zorder=3)
    winner_note = []
    for s in scales:
        sub = win[win["scale"] == s].sort_values("pct", ascending=False)
        if not sub.empty:
            winner_note.append(
                f"±{s:g}°: {DISPLAY.get(sub.iloc[0]['param'], sub.iloc[0]['param'])} is #1 in {sub.iloc[0]['pct']:.0f}%"
            )
    ax_b.text(0.03, 0.05, "\n".join(winner_note), transform=ax_b.transAxes,
              fontsize=8, va="bottom", ha="left",
              bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                        edgecolor="0.85", alpha=0.95))
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(scale_labels)
    ax_b.set_ylim(94, 100.2)
    ax_b.set_ylabel("$G$ share across orientations (%)")
    ax_b.set_title("Orientation-level dominance remains high")
    ax_b.grid(True, axis="y", alpha=0.2, linestyle="--", linewidth=0.8)
    _label_panel(ax_b, "b")

    ax_c = fig.add_subplot(gs[1, 0])
    agg_by_param = agg.set_index("param")
    ebsd_params = ["phi1_err_deg", "Phi_err_deg", "phi2_err_deg", "psi_err_deg"]
    summary_rows = [
        ("G", float(agg_by_param.loc["G", "S1"]), float(agg_by_param.loc["G", "ST"])),
        ("rho", float(agg_by_param.loc["rho", "S1"]), float(agg_by_param.loc["rho", "ST"])),
        ("D", float(agg_by_param.loc["D", "S1"]), float(agg_by_param.loc["D", "ST"])),
        ("K", float(agg_by_param.loc["K", "S1"]), float(agg_by_param.loc["K", "ST"])),
        ("EBSD", float(agg_by_param.loc[ebsd_params, "S1"].sum()),
         float(agg_by_param.loc[ebsd_params, "ST"].sum())),
    ]
    labels = [r"$G$", r"$\rho$", r"$D$", r"$K$", "EBSD"]
    y = np.arange(len(summary_rows))
    st_vals = np.array([row[2] for row in summary_rows])
    s1_vals = np.array([row[1] for row in summary_rows])
    st_colors = [
        SENSITIVITY_COLORS["G"],
        SENSITIVITY_COLORS["rho"],
        SENSITIVITY_COLORS["D"],
        SENSITIVITY_COLORS["K"],
        SENSITIVITY_COLORS["ebsd"],
    ]
    s1_colors = [matplotlib.colors.to_rgba(c, alpha=0.55) for c in st_colors]
    ax_c.barh(y + 0.18, st_vals, height=0.34,
              color=st_colors, edgecolor="white", linewidth=0.8,
              label="$S_T$ (mean)")
    ax_c.barh(y - 0.18, s1_vals, height=0.34,
              color=s1_colors, edgecolor="white", linewidth=0.8,
              label="$S_1$ (mean)")
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(labels)
    ax_c.invert_yaxis()
    ax_c.set_xlim(0, 1.05)
    ax_c.set_xlabel("Sensitivity index (averaged)")
    ax_c.set_title("Elastic parameters vs combined EBSD terms")
    ax_c.grid(True, axis="x", alpha=0.2, linestyle="--", linewidth=0.8)
    ax_c.legend(loc="lower left", fontsize=9)
    ax_c.text(0.98, 0.06, f"Interaction gap for $G$: {agg_by_param.loc['G', 'gap']:.3f}",
              transform=ax_c.transAxes, ha="right", va="bottom", fontsize=8,
              bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                        edgecolor="0.85", alpha=0.95))
    _label_panel(ax_c, "c")

    ax_d = fig.add_subplot(gs[1, 1])
    dom_wide = dom.pivot(index="orientation_idx", columns="scale_deg", values="G_share")
    weakest = dom_wide.mean(axis=1).sort_values().head(5).index.tolist()
    weak_colors = [COLORS["purple"], COLORS["cyan"], COLORS["green"], COLORS["red"], COLORS["blue"]]
    for ori, color in zip(weakest, weak_colors):
        vals = 100.0 * dom_wide.loc[ori, scales].values
        ax_d.plot(x, vals, marker="o", color=color, linewidth=1.6,
                  markersize=5, label=f"ori {ori}")
    min_curve = 100.0 * dom_wide[scales].min(axis=0).values
    ax_d.plot(x, min_curve, color="0.25", linewidth=1.2, linestyle="--",
              label="minimum")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(scale_labels)
    ax_d.set_xlabel("EBSD error scale")
    ax_d.set_ylabel("$G$ share of total $S_T$ (%)")
    ax_d.set_title("Weakest orientations still remain $G$-dominated")
    ax_d.set_ylim(94, 100)
    ax_d.grid(True, axis="y", alpha=0.2, linestyle="--", linewidth=0.8)
    ax_d.legend(loc="upper right", fontsize=8, ncol=2)
    _label_panel(ax_d, "d")

    #fig.suptitle("Sensitivity summary: stiffness anisotropy is overwhelmingly shear-modulus driven",
    #             y=0.98, fontsize=13)
    fig.subplots_adjust(top=0.90)

    _save(fig, os.path.join(out_dir, "sensitivity_summary.png"))


def plot_global_sensitivity(data_dir: str, out_dir: str) -> None:
    """Single-panel global sensitivity view focused on elastic parameters."""
    _, _, agg, _ = _load_summary_tables(data_dir)
    params = ["G", "rho", "D", "K", "psi_err_deg"]
    elastic = agg.set_index("param").loc[params].sort_values("ST", ascending=False)
    y = np.arange(len(params))
    values = elastic["ST"].values
    ref = float(elastic.loc["G", "ST"])
    baseline = max(values.min() * 0.6, 1e-4)
    bold_display = {
        "G": r"$\mathbf{G}$",
        "rho": r"$\boldsymbol{\rho}$",
        "D": r"$\mathbf{D}$",
        "K": r"$\mathbf{K}$",
        "psi_err_deg": r"$\boldsymbol{\psi}$",
    }
    tick_labels = [bold_display[p] for p in elastic.index]

    fig, ax = plt.subplots(figsize=fig_size(DOUBLE_COL, 0.62))
    for ypos, param, value in zip(y, params, values):
        color = SENSITIVITY_COLORS.get(param, COLORS["grey"])
        ax.hlines(ypos, baseline, value, color=color, linewidth=4.0, alpha=0.95)
        ax.scatter(value, ypos, s=90, color=color, edgecolor="white",
                   linewidth=0.9, zorder=3)
        if param != "G":
            label = f"{ref / value:.0f}x smaller than $G$"
            x_text = value * 1.12
            ax.text(x_text, ypos, label, va="center", ha="left", fontsize=10)

    ax.set_xscale("log")
    ax.set_xlim(baseline, ref * 2.2)
    ax.set_yticks(y)
    ax.set_yticklabels(tick_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean total-order sensitivity $S_T$ (log scale)", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.2, linestyle="--", linewidth=0.8)
    ax.text(0.97, 0.05, f"$G$ accounts for {100.0 * ref:.1f}% of mean total sensitivity.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="0.85", alpha=0.95))
    plt.tight_layout()

    _save(fig, os.path.join(out_dir, "global_sensitivity.png"))


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce SAW sensitivity paper figures from CSV data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help=f"Directory containing CSV data (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--out_dir", type=str, default="figures",
                        help="Directory to write output PNGs (default: figures)")
    parser.add_argument("--interaction_threshold", type=float, default=0.05,
                        help="Threshold for interaction gap star in summary fig")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dispatch = {
        "ternary_ks": ("ternary_top_n_cdfs.csv", plot_ternary_and_ks),
        "ternary":    ("ternary_grid.csv",        plot_ternary_heatmap),
        "ks_compare": ("ternary_top_n_cdfs.csv",  plot_ks_comparison),
        "combined":   ("ternary_top_n_cdfs.csv",  plot_combined_ternary_cdfs),
        "cdfs":       ("ternary_top_n_cdfs.csv",  plot_top_n_cdfs),
        "angle":      ("ks_vs_angle.csv",         plot_angle_sensitivity),
        "density":    ("ks_vs_density.csv",        plot_density_sensitivity),
        "summary":    ("knob_share_by_scale.csv",
                       lambda d, o: plot_sensitivity_summary(
                           d, o, args.interaction_threshold)),
        "global_sensitivity": ("mean_S1_ST_and_gap.csv", plot_global_sensitivity),
    }

    for fig_name, (required_file, plot_fn) in dispatch.items():
        data_path = os.path.join(args.data_dir, required_file)
        if not os.path.exists(data_path):
            print(f"[skip] {fig_name}: missing {data_path}")
            continue
        print(f"[plot] {fig_name}...")
        plot_fn(args.data_dir, args.out_dir)

    print(f"\nDone. Figures saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
