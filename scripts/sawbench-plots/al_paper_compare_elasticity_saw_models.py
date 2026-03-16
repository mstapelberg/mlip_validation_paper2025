#!/usr/bin/env python3
"""Compare elastic constants and SAW frequency distributions across NequIP models.

Reads pre-computed SAW/elasticity data from a JSON summary file and produces
three publication-quality figures:
  1. Horizontal bar chart of cubic elastic constants (C11, C12, C44)
  2. Overlaid frequency histograms
  3. Overlaid cumulative distribution functions (CDFs)

Usage:
    python al_paper_compare_elasticity_saw_models.py
    python al_paper_compare_elasticity_saw_models.py --out_dir figures/
    python al_paper_compare_elasticity_saw_models.py --figures elastic cdf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plotting_utils import (
    set_pub_style, save_fig, COLORS, TOL_BRIGHT, DOUBLE_COL, fig_size,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "sawbench-data"
DEFAULT_JSON = "saw_cdf_nequip_summary.json"

set_pub_style(base_fontsize=10)

MODEL_NAMES = {
    'mse':     'mse_lmax2_nlayers2_mlp512_zbl_epoch107.nequip.zip',
    'msetw':   'msetw_lmax2_nlayers2_mlp512_nlh.nequip.zip',
    'ca':      'ca_lmax2_nlayers2_mlp512_nlh_epoch169.nequip.zip',
    'catw':    'catw_lmax2_nlayers2_mlp512_nlh_epoch128.nequip.zip',
    'DFT':     'DFT',
    'PredEXP': 'PredEXP',
    'EXP':     'EXP',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, path, dpi=300):
    out_path = Path(path)
    stem = out_path.with_suffix("") if out_path.suffix else out_path
    save_fig(fig, stem, dpi=dpi)
    plt.close(fig)


def _load_models(data_dir: str) -> dict:
    """Load JSON and return the subset of models defined in MODEL_NAMES."""
    json_path = os.path.join(data_dir, DEFAULT_JSON)
    with open(json_path) as f:
        data = json.load(f)

    models_data = {}
    for key, model_name in MODEL_NAMES.items():
        if model_name in data['model_results']:
            models_data[key] = data['model_results'][model_name]
        elif model_name in data['reference_data']:
            models_data[key] = data['reference_data'][model_name]
        else:
            print(f"  [warn] Model {model_name} not found in JSON")
    return models_data


def extract_cubic_constants(model_data):
    """Return dict with keys C11, C12, C44 if available, else None."""
    if 'cubic_constants_gpa' in model_data:
        cc = model_data['cubic_constants_gpa']
        if all(k in cc and cc[k] is not None for k in ('C11', 'C12', 'C44')):
            return {'C11': cc['C11'], 'C12': cc['C12'], 'C44': cc['C44']}
    if 'elasticity' in model_data:
        el = model_data['elasticity']
        c11, c12, c44 = el.get('C11'), el.get('C12'), el.get('C44')
        if c11 is not None and c12 is not None and c44 is not None:
            return {'C11': c11, 'C12': c12, 'C44': c44}
    return None


def is_mechanically_stable(constants):
    """Check cubic stability: C44>0, C11-C12>0, C11+2*C12>0."""
    if constants is None:
        return False
    c11 = constants.get('C11')
    c12 = constants.get('C12')
    c44 = constants.get('C44')
    if c11 is None or c12 is None or c44 is None:
        return False
    return (c44 > 0) and ((c11 - c12) > 0) and ((c11 + 2 * c12) > 0)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_elastic_constants(models_data, out_dir: str) -> None:
    """Horizontal bar chart of C11, C12, C44 for each model."""
    c11_values, c12_values, c44_values, model_labels = [], [], [], []

    for model_name, model_data in models_data.items():
        constants = extract_cubic_constants(model_data)
        if constants:
            c11_values.append(constants['C11'])
            c12_values.append(constants['C12'])
            c44_values.append(constants['C44'])
            model_labels.append(model_name.upper())

    fig, ax = plt.subplots(figsize=fig_size(DOUBLE_COL, 0.55))

    y_pos = np.arange(len(model_labels))
    bar_height = 0.25

    ax.barh(y_pos - bar_height, c11_values, bar_height,
            label='C11', color=COLORS["blue"])
    ax.barh(y_pos, c12_values, bar_height,
            label='C12', color=COLORS["purple"])
    ax.barh(y_pos + bar_height, c44_values, bar_height,
            label='C44', color=COLORS["green"])

    for i, (c11, c12, c44) in enumerate(zip(c11_values, c12_values, c44_values)):
        ax.text(c11 + 1, y_pos[i] - bar_height, f'{c11:.1f}', va='center')
        ax.text(c12 + 1, y_pos[i], f'{c12:.1f}', va='center')
        ax.text(c44 + 1, y_pos[i] + bar_height, f'{c44:.1f}', va='center')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_labels, fontweight='bold')
    ax.set_xlabel('Elastic Constants (GPa)', fontweight='bold')
    ax.legend()
    plt.tight_layout()

    _save(fig, os.path.join(out_dir, "elastic_constants_comparison.png"))


def plot_histograms(models_data, out_dir: str) -> None:
    """Overlaid frequency histograms for all mechanically-stable models."""
    fig, ax = plt.subplots(figsize=fig_size(DOUBLE_COL, 0.62))

    exp_mean = exp_std = None
    if 'EXP' in models_data and 'frequencies_mhz' in models_data['EXP']:
        exp_freqs = np.array(models_data['EXP']['frequencies_mhz'])
        exp_mean, exp_std = np.mean(exp_freqs), np.std(exp_freqs)
        ax.hist(exp_freqs, bins=50, alpha=0.2,
                color='black', edgecolor='black', linewidth=0.5,
                label='EXP')

    plotted = 0
    for model_name, model_data in models_data.items():
        if model_name == 'EXP' or 'frequencies_mhz' not in model_data:
            continue
        if not is_mechanically_stable(extract_cubic_constants(model_data)):
            continue

        freqs = np.array(model_data['frequencies_mhz'])
        color = TOL_BRIGHT[plotted % len(TOL_BRIGHT)]

        ax.hist(freqs, bins=50, alpha=0.7,
                color=color, edgecolor='none', linewidth=0,
                label=model_name.upper())

        mean_freq, std_freq = np.mean(freqs), np.std(freqs)
        ax.text(0.02, 0.98 - plotted * 0.08,
                f'{model_name.upper()}: \u03bc={mean_freq:.1f}\u00b1{std_freq:.1f}',
                transform=ax.transAxes, ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='white',
                          alpha=0.8, edgecolor=color))
        plotted += 1

    if exp_mean is not None:
        num_stable = sum(
            1 for m, d in models_data.items()
            if m != 'EXP' and 'frequencies_mhz' in d
            and is_mechanically_stable(extract_cubic_constants(d))
        )
        ax.text(0.02, 0.98 - num_stable * 0.08 - 0.08,
                f'EXP: \u03bc={exp_mean:.1f}\u00b1{exp_std:.1f}',
                transform=ax.transAxes, ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='white',
                          alpha=0.8, edgecolor='black'))

    ax.set_xlabel('Frequency (MHz)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()

    _save(fig, os.path.join(out_dir, "frequency_histograms.png"))


def plot_cdfs(models_data, out_dir: str) -> None:
    """Overlaid CDFs for all mechanically-stable models with KS stats vs EXP."""
    fig, ax = plt.subplots(figsize=fig_size(DOUBLE_COL, 0.62))

    exp_freqs = None
    if 'EXP' in models_data and 'frequencies_mhz' in models_data['EXP']:
        exp_freqs = np.array(models_data['EXP']['frequencies_mhz'])
        exp_sorted = np.sort(exp_freqs)
        y_exp = np.arange(1, len(exp_freqs) + 1) / len(exp_freqs)
        ax.plot(exp_sorted, y_exp, color='black', linewidth=2.5, zorder=10)

    # Collect stable models, sorted by mean frequency for consistent ordering
    stable_models = []
    for name, md in models_data.items():
        if name == 'EXP' or 'frequencies_mhz' not in md:
            continue
        if not is_mechanically_stable(extract_cubic_constants(md)):
            continue
        freqs = np.array(md['frequencies_mhz'])
        stable_models.append((name, freqs))
    stable_models.sort(key=lambda x: np.mean(x[1]))

    # Per-model label positions along the CDF curve (hand-tuned to spread
    # labels vertically: leftmost curves labelled low, rightmost high).
    cdf_label_frac = [0.25, 0.35, 0.45, 0.55, 0.65]
    # EXP label placed at CDF 0.90 (upper region, well right of DFT)
    exp_label_frac = 0.90

    curves = {}  # name -> (freqs_sorted, y, color) for KS table colouring
    for i, (name, freqs) in enumerate(stable_models):
        freqs_sorted = np.sort(freqs)
        y = np.arange(1, len(freqs) + 1) / len(freqs)
        color = TOL_BRIGHT[i % len(TOL_BRIGHT)]
        curves[name] = (freqs_sorted, y, color)

        ax.plot(freqs_sorted, y, color=color, linewidth=2)

        frac = cdf_label_frac[i % len(cdf_label_frac)]
        idx = int(len(freqs_sorted) * frac)
        ax.text(freqs_sorted[idx], y[idx], name.upper(),
                fontsize=9, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=color, linewidth=0.6, alpha=0.85),
                zorder=20)

    # EXP direct label
    if exp_freqs is not None:
        idx = int(len(exp_sorted) * exp_label_frac)
        ax.text(exp_sorted[idx], y_exp[idx], 'EXP',
                fontsize=9, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='black', linewidth=0.6, alpha=0.85),
                zorder=20)

    ax.set_xlabel('Frequency (MHz)', fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontweight='bold')
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.2, linestyle='--')

    # KS stats table (bottom-right), sorted best (lowest KS) first
    if exp_freqs is not None:
        ks_rows = []
        for name, freqs in stable_models:
            ks_stat = ks_2samp(freqs, exp_freqs).statistic
            color = curves[name][2]
            ks_rows.append((name.upper(), ks_stat, color))
        ks_rows.sort(key=lambda r: r[1])

        ks_lines = ["KS vs EXP:"]
        for label, ks_val, _ in ks_rows:
            ks_lines.append(f"  {label}: {ks_val:.3f}")

        ax.text(0.98, 0.05, "\n".join(ks_lines), transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                fontsize=9, family='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          edgecolor='0.7', alpha=0.9))

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "frequency_cdfs.png"))


def plot_hist_cdf_combined(models_data, out_dir: str) -> None:
    """Two-row figure: (a) histogram on top, (b) CDF on bottom, shared x-axis."""
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=fig_size(DOUBLE_COL, 1.1))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.08)
    ax_hist = fig.add_subplot(gs[0])
    ax_cdf = fig.add_subplot(gs[1], sharex=ax_hist)

    # --- Collect stable models sorted by mean frequency ---
    exp_freqs = None
    if 'EXP' in models_data and 'frequencies_mhz' in models_data['EXP']:
        exp_freqs = np.array(models_data['EXP']['frequencies_mhz'])

    stable_models = []
    for name, md in models_data.items():
        if name == 'EXP' or 'frequencies_mhz' not in md:
            continue
        if not is_mechanically_stable(extract_cubic_constants(md)):
            continue
        freqs = np.array(md['frequencies_mhz'])
        stable_models.append((name, freqs))
    stable_models.sort(key=lambda x: np.mean(x[1]))

    # --- (a) Histogram ---
    if exp_freqs is not None:
        ax_hist.hist(exp_freqs, bins=50, alpha=0.2,
                     color='black', edgecolor='black', linewidth=0.5,
                     label='EXP')

    plotted = 0
    for name, freqs in stable_models:
        color = TOL_BRIGHT[plotted % len(TOL_BRIGHT)]
        ax_hist.hist(freqs, bins=50, alpha=0.7,
                     color=color, edgecolor='none', linewidth=0,
                     label=name.upper())

        mean_freq, std_freq = np.mean(freqs), np.std(freqs)
        ax_hist.text(0.02, 0.98 - plotted * 0.10,
                     f'{name.upper()}: \u03bc={mean_freq:.1f}\u00b1{std_freq:.1f}',
                     transform=ax_hist.transAxes, ha='left', va='top',
                     fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='white',
                               alpha=0.8, edgecolor=color))
        plotted += 1

    if exp_freqs is not None:
        exp_mean, exp_std = np.mean(exp_freqs), np.std(exp_freqs)
        ax_hist.text(0.02, 0.98 - plotted * 0.10,
                     f'EXP: \u03bc={exp_mean:.1f}\u00b1{exp_std:.1f}',
                     transform=ax_hist.transAxes, ha='left', va='top',
                     fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='white',
                               alpha=0.8, edgecolor='black'))

    ax_hist.set_ylabel('Count', fontweight='bold')
    ax_hist.legend(loc='lower right', fontsize=8)
    plt.setp(ax_hist.get_xticklabels(), visible=False)
    ax_hist.text(-0.10, 1.04, "(a)", transform=ax_hist.transAxes,
                 fontsize=12, fontweight='bold', va='bottom', ha='left')

    # --- (b) CDF ---
    if exp_freqs is not None:
        exp_sorted = np.sort(exp_freqs)
        y_exp = np.arange(1, len(exp_freqs) + 1) / len(exp_freqs)
        ax_cdf.plot(exp_sorted, y_exp, color='black', linewidth=2.5, zorder=10)

    cdf_label_frac = [0.25, 0.35, 0.45, 0.55, 0.65]
    curves = {}
    for i, (name, freqs) in enumerate(stable_models):
        freqs_sorted = np.sort(freqs)
        y = np.arange(1, len(freqs) + 1) / len(freqs)
        color = TOL_BRIGHT[i % len(TOL_BRIGHT)]
        curves[name] = (freqs_sorted, y, color)
        ax_cdf.plot(freqs_sorted, y, color=color, linewidth=2)

        frac = cdf_label_frac[i % len(cdf_label_frac)]
        idx = int(len(freqs_sorted) * frac)
        ax_cdf.text(freqs_sorted[idx], y[idx], name.upper(),
                    fontsize=9, fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=color, linewidth=0.6, alpha=0.85),
                    zorder=20)

    if exp_freqs is not None:
        idx = int(len(exp_sorted) * 0.90)
        ax_cdf.text(exp_sorted[idx], y_exp[idx], 'EXP',
                    fontsize=9, fontweight='bold', color='black',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='black', linewidth=0.6, alpha=0.85),
                    zorder=20)

    ax_cdf.set_xlabel('Frequency (MHz)', fontweight='bold')
    ax_cdf.set_ylabel('Cumulative Probability', fontweight='bold')
    ax_cdf.set_ylim([0, 1.02])
    ax_cdf.grid(True, alpha=0.2, linestyle='--')
    ax_cdf.text(-0.10, 1.04, "(b)", transform=ax_cdf.transAxes,
                fontsize=12, fontweight='bold', va='bottom', ha='left')

    if exp_freqs is not None:
        ks_rows = []
        for name, freqs in stable_models:
            ks_stat = ks_2samp(freqs, exp_freqs).statistic
            color = curves[name][2]
            ks_rows.append((name.upper(), ks_stat, color))
        ks_rows.sort(key=lambda r: r[1])
        ks_lines = ["KS vs EXP:"]
        for label, ks_val, _ in ks_rows:
            ks_lines.append(f"  {label}: {ks_val:.3f}")
        ax_cdf.text(0.98, 0.05, "\n".join(ks_lines), transform=ax_cdf.transAxes,
                    verticalalignment='bottom', horizontalalignment='right',
                    fontsize=9, family='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                              edgecolor='0.7', alpha=0.9))

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "frequency_hist_cdf_combined.png"))


def plot_cdf_defense(models_data, out_dir: str) -> None:
    """Presentation-style CDF: only PredEXP vs EXP with KS shading."""
    set_pub_style(base_fontsize=14)
    fig, ax = plt.subplots(figsize=fig_size(DOUBLE_COL, 0.62))

    exp_freqs = np.array(models_data['EXP']['frequencies_mhz'])
    pred_freqs = np.array(models_data['PredEXP']['frequencies_mhz'])

    exp_sorted = np.sort(exp_freqs)
    pred_sorted = np.sort(pred_freqs)
    y_exp = np.arange(1, len(exp_sorted) + 1) / len(exp_sorted)
    y_pred = np.arange(1, len(pred_sorted) + 1) / len(pred_sorted)

    all_freqs = np.sort(np.unique(np.concatenate([exp_sorted, pred_sorted])))
    exp_interp = np.interp(all_freqs, exp_sorted, y_exp, left=0, right=1)
    pred_interp = np.interp(all_freqs, pred_sorted, y_pred, left=0, right=1)

    ax.fill_between(all_freqs, exp_interp, pred_interp,
                    color=COLORS["blue"], alpha=0.15)

    ax.plot(exp_sorted, y_exp, color='black', linewidth=3, zorder=10)
    ax.plot(pred_sorted, y_pred, color=COLORS["blue"], linewidth=3, zorder=10)

    ks_stat = ks_2samp(pred_freqs, exp_freqs).statistic

    # Direct labels on curves
    exp_idx = int(len(exp_sorted) * 0.80)
    ax.annotate('Experimental', xy=(exp_sorted[exp_idx], y_exp[exp_idx]),
                xytext=(18, -8), textcoords='offset points',
                fontsize=13, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.4', fc='white',
                          ec='black', lw=1.2, alpha=0.9),
                zorder=30)

    pred_idx = int(len(pred_sorted) * 0.55)
    ax.annotate('PredEXP', xy=(pred_sorted[pred_idx], y_pred[pred_idx]),
                xytext=(-20, 14), textcoords='offset points',
                fontsize=13, fontweight='bold', color=COLORS["blue"],
                bbox=dict(boxstyle='round,pad=0.4', fc='white',
                          ec=COLORS["blue"], lw=1.2, alpha=0.9),
                zorder=30)

    ax.text(0.97, 0.06, f'KS = {ks_stat:.3f}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', fc='white',
                      ec=COLORS["blue"], lw=1.5, alpha=0.95))

    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.15, linestyle='--')
    plt.tight_layout()

    _save(fig, os.path.join(out_dir, "cdf_defense_predexp_vs_exp.png"))
    set_pub_style(base_fontsize=10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare elastic constants and SAW frequencies across NequIP models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help=f"Directory containing JSON data (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--out_dir", type=str, default="figures",
                        help="Directory to write output figures (default: figures)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    models_data = _load_models(args.data_dir)
    print(f"Loaded {len(models_data)} models: {list(models_data.keys())}")

    dispatch = {
        "elastic":      ("Elastic constants bar chart",     plot_elastic_constants),
        "histogram":    ("Frequency histograms",            plot_histograms),
        "cdf":          ("Frequency CDFs",                  plot_cdfs),
        "hist_cdf":     ("Combined histogram + CDF",        plot_hist_cdf_combined),
        "cdf_defense":  ("CDF defense (PredEXP vs EXP)",   plot_cdf_defense),
    }

    for fig_name, (desc, plot_fn) in dispatch.items():
        print(f"[plot] {desc}...")
        plot_fn(models_data, args.out_dir)

    print(f"\nDone. Figures saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
