import math
import re
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Shared publication style
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plotting_utils import set_pub_style, save_fig, TOL_BRIGHT, DOUBLE_COL, fig_size

set_pub_style()

# Semantic color mapping for this script
COLORS = {
    'primary':   TOL_BRIGHT[0],  # blue
    'secondary': TOL_BRIGHT[1],  # red/coral
    'tertiary':  TOL_BRIGHT[2],  # green
    'highlight': TOL_BRIGHT[5],  # purple
    'accent':    TOL_BRIGHT[3],  # yellow
}

# ---------- CLI ARGUMENTS ----------
parser = argparse.ArgumentParser(description='HPO Analysis: Compare error vs throughput')
parser.add_argument('--selected-model', type=int, default=None, metavar='N',
                    help='Highlight a specific model (e.g., --selected-model 53 for hpo_053)')
parser.add_argument('--throughput-requirement', type=float, default=5.0, metavar='T',
                    help='Minimum throughput requirement in timesteps/s (default: 5.0)')
args = parser.parse_args()

# ---------- CONFIG ----------
CSV_PATH = "/home/myless/Packages/mlip_validation_paper2025/data/wandb_data/hpo_training_results.csv"
INFERENCE_DATA_DIR = "/home/myless/Packages/mlip_validation_paper2025/data/wandb_data/hpo_inference_test_analysis"

# Primary error metric to optimize (lower is better)
ERROR_COL = "test0_epoch/forces_rmse"  # Primary: forces RMSE
# Alternative metric for comparison
ALTERNATIVE_ERROR_COL = "test0_epoch/weighted_sum"  # Composite metric
# Secondary metrics for multi-objective analysis (optional)
SECONDARY_METRICS = [
    "test0_epoch/per_atom_energy_rmse",
    "test0_epoch/stress_rmse",
    "test0_epoch/weighted_sum"
]

# Throughput configuration
# Choose which supercell size to use for throughput comparison
TARGET_SUPERCELL = "16x16x16"  # Options: "3x3x3", "4x4x4", "5x5x5", "6x6x6", "7x7x7", "8x8x8", "16x16x16"
THROUGHPUT_METRIC = "timesteps_per_s_log"  # or "katom_steps_per_s_log"

# Hyperparameters that define a unique configuration
CONFIG_COLS = [
    "model.r_max",
    "model.l_max", 
    "model.num_layers",
    "model.num_tensor_features",
    "model.num_scalar_features",
    "model.allegro_mlp_hidden_layers_width"
]

# Throughput requirement (from CLI argument)
REQUIRED_T = args.throughput_requirement
SELECTED_MODEL = args.selected_model  # User-selected model to highlight
ALPHA = 0.05  # for 95% CI

# Output directory
OUTPUT_DIR = Path("/home/myless/Packages/mlip_validation_paper2025/results/hpo_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- HELPER FUNCTIONS ----------
def extract_hpo_number(name_string):
    """Extract hpo_XXX from the Name column."""
    match = re.search(r'hpo_(\d+)', name_string)
    if match:
        return int(match.group(1))
    return None

def load_benchmark_data(inference_dir, target_supercell):
    """Load all benchmark JSON files and extract throughput for target supercell."""
    benchmark_data = []
    
    # Find all benchmark result directories
    pattern = str(Path(inference_dir) / "benchmark_results_hpo_*_compiled.nequip")
    dirs = glob.glob(pattern)
    
    for dir_path in dirs:
        # Extract hpo number from directory name
        match = re.search(r'hpo_(\d+)', dir_path)
        if not match:
            continue
        hpo_num = int(match.group(1))
        
        # Load the benchmark summary JSON
        json_path = Path(dir_path) / "benchmark_summary_all_supercells.json"
        if not json_path.exists():
            print(f"Warning: No benchmark file found for hpo_{hpo_num:03d}")
            continue
            
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Find the target supercell
        for entry in data:
            if entry['supercell'] == target_supercell:
                benchmark_data.append({
                    'hpo_num': hpo_num,
                    'supercell': entry['supercell'],
                    'num_atoms': entry['num_atoms'],
                    'timesteps_per_s': entry['timesteps_per_s_log'],
                    'katom_steps_per_s': entry['katom_steps_per_s_log'],
                    'peak_gpu_memory_mib': entry['peak_gpu_memory_mib'],
                    'wall_time_s': entry['wall_time_s']
                })
                break
    
    return pd.DataFrame(benchmark_data)

def agg_ci(series, alpha=ALPHA):
    """Compute mean, std, n, and CI halfwidth for a series."""
    n = series.count()
    mean = series.mean()
    std = series.std(ddof=1)
    if n <= 1 or pd.isna(std):
        halfwidth = np.nan
    else:
        # Normal approx; for small n you may use t-crit
        z = 1.96 if abs(alpha - 0.05) < 1e-9 else 1.96
        halfwidth = z * std / math.sqrt(n)
    return pd.Series({"mean": mean, "std": std, "n": n, "ci_halfwidth": halfwidth})

def pareto_nondominated(df2d, err_col="err_u95", thr_col="thr_l95"):
    """Find Pareto non-dominated points (minimize error, maximize throughput)."""
    if len(df2d) == 0:
        return df2d.copy()
    
    A = df2d[[err_col, thr_col]].values
    n = A.shape[0]
    is_dom = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dom[i]:
            continue
        e1, t1 = A[i]
        # dominated if any j has e2 <= e1 and t2 >= t1 with at least one strict
        mask = (A[:,0] <= e1) & (A[:,1] >= t1) & ((A[:,0] < e1) | (A[:,1] > t1))
        mask[i] = False
        if mask.any():
            is_dom[i] = True
    return df2d.loc[~is_dom].copy()

def apply_publication_legend(ax):
    """Make legends more readable (bigger text and less cramped handles/markers)."""
    leg = ax.legend(
        loc="best",
        fontsize=11,
        framealpha=0.9,
        scatterpoints=1,
        markerscale=1.4,
        handlelength=2.4,
        handletextpad=0.8,
        labelspacing=0.7,
        borderpad=0.7,
        borderaxespad=0.6,
    )
    # Special-case: shrink the "Selected" legend marker only (it is intentionally large on-plot)
    try:
        labels = [t.get_text() for t in leg.get_texts()]
        handles = getattr(leg, "legend_handles", None)
        if handles is None:
            handles = leg.legendHandles  # older Matplotlib
        for h, lab in zip(handles, labels):
            if "Selected (" in lab or lab.startswith("Selected"):
                # Scatter -> PathCollection; Line -> Line2D
                if hasattr(h, "set_sizes"):
                    h.set_sizes([70.0])  # points^2; tuned for readability
                if hasattr(h, "set_markersize"):
                    h.set_markersize(6.0)
    except Exception:
        pass
    return leg


# ---------- LOAD TRAINING DATA ----------
print("Loading training data...")
df = pd.read_csv(CSV_PATH)

# Extract hpo number from Name column
df['hpo_num'] = df['Name'].apply(extract_hpo_number)
df = df.dropna(subset=['hpo_num'])
df['hpo_num'] = df['hpo_num'].astype(int)

# Basic cleaning
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[ERROR_COL])

# Verify all config columns exist
present_config_cols = [c for c in CONFIG_COLS if c in df.columns]
if not present_config_cols:
    raise ValueError("No hyperparameter columns from CONFIG_COLS were found in the CSV.")

print(f"Loaded {len(df)} training runs across {df['hpo_num'].nunique()} unique HPO configs")

# ---------- GROUP AND AGGREGATE REPLICATES ----------
print("Aggregating cross-validation folds and seeds...")
group_cols = present_config_cols + ['hpo_num']

# Aggregate across replicates (folds and seeds)
# Build list of metrics to aggregate
metrics_to_agg = [ERROR_COL]
for metric in SECONDARY_METRICS:
    if metric in df.columns and metric not in metrics_to_agg:
        metrics_to_agg.append(metric)

# Build aggregation dict - aggregate each statistic separately
agg_dict = {}
for metric in metrics_to_agg:
    agg_dict[metric] = ['mean', 'std', 'count']

agg = df.groupby(group_cols).agg(agg_dict)

# Flatten multiindex columns
agg.columns = ['_'.join([c for c in col if c]) for col in agg.columns.values]
agg = agg.reset_index()

# Manually compute CI halfwidth for each metric
for metric in metrics_to_agg:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    n_col = f"{metric}_count"
    ci_col = f"{metric}_ci_halfwidth"
    
    n = agg[n_col]
    std = agg[std_col]
    
    # Normal approx CI halfwidth
    z = 1.96  # for 95% CI
    halfwidth = np.where(
        (n > 1) & (~pd.isna(std)),
        z * std / np.sqrt(n),
        np.nan
    )
    agg[ci_col] = halfwidth

# Create robust estimates: worst-case within 95% CI
agg["err_mean"] = agg[f"{ERROR_COL}_mean"]
agg["err_u95"] = agg[f"{ERROR_COL}_mean"] + agg[f"{ERROR_COL}_ci_halfwidth"]

# Also create for alternative metric
if ALTERNATIVE_ERROR_COL and f"{ALTERNATIVE_ERROR_COL}_mean" in agg.columns:
    agg["err_alt_mean"] = agg[f"{ALTERNATIVE_ERROR_COL}_mean"]
    agg["err_alt_u95"] = agg[f"{ALTERNATIVE_ERROR_COL}_mean"] + agg[f"{ALTERNATIVE_ERROR_COL}_ci_halfwidth"]

print(f"Aggregated to {len(agg)} unique configurations")

# ---------- LOAD INFERENCE/THROUGHPUT DATA ----------
print(f"\nLoading benchmark data for supercell {TARGET_SUPERCELL}...")
benchmark_df = load_benchmark_data(INFERENCE_DATA_DIR, TARGET_SUPERCELL)

if len(benchmark_df) == 0:
    raise ValueError(f"No benchmark data found for supercell {TARGET_SUPERCELL}")

print(f"Loaded benchmark data for {len(benchmark_df)} configurations")

# ---------- MERGE TRAINING AND BENCHMARK DATA ----------
print("\nMerging training and benchmark data...")
merged = agg.merge(benchmark_df, on='hpo_num', how='inner')

if len(merged) == 0:
    raise ValueError("No matching data between training CSV and benchmark JSONs!")

print(f"Successfully merged {len(merged)} configurations with both training and benchmark data")

# Add throughput columns based on selected metric
merged["throughput"] = merged[THROUGHPUT_METRIC.replace('_log', '')]
merged["thr_mean"] = merged["throughput"]
merged["thr_l95"] = merged["throughput"]  # No CI for throughput (single measurement per config)

# ---------- FEASIBILITY FILTER ----------
if REQUIRED_T is not None:
    feasible = merged[merged["thr_l95"] >= REQUIRED_T].copy()
    infeasible = merged[merged["thr_l95"] < REQUIRED_T].copy()
    
    print(f"\n" + "="*70)
    print(f"FEASIBILITY FILTER (Throughput >= {REQUIRED_T} timesteps/s)")
    print("="*70)
    print(f"Feasible configs: {len(feasible)} / {len(merged)} ({100*len(feasible)/len(merged):.1f}%)")
    print(f"Infeasible configs: {len(infeasible)} / {len(merged)} ({100*len(infeasible)/len(merged):.1f}%)")
    
    if len(infeasible) > 0:
        print(f"\nConfigurations that FAILED throughput requirement:")
        print("-" * 70)
        infeasible_sorted = infeasible.sort_values("thr_l95", ascending=False)
        for idx, row in infeasible_sorted.iterrows():
            print(f"  hpo_{int(row['hpo_num']):03d}: {row['thr_l95']:.2f} timesteps/s "
                  f"(Forces RMSE: {row['err_u95']:.6f})")
        print("-" * 70)
else:
    feasible = merged.copy()
    print(f"\nNo throughput requirement set. All {len(merged)} configs considered feasible.")

# ---------- PARETO FRONT ANALYSIS ----------
print("\nComputing Pareto fronts...")
front_all = pareto_nondominated(merged, "err_u95", "thr_l95")
front_feasible = pareto_nondominated(feasible, "err_u95", "thr_l95")

print(f"Pareto front (all): {len(front_all)} configurations")
print(f"Pareto front (feasible): {len(front_feasible)} configurations")

# Best feasible (smallest pessimistic error)
best_row = None
if len(feasible):
    best_idx = feasible["err_u95"].idxmin()
    best_row = feasible.loc[best_idx]
    print(f"\nBest feasible config (min forces_rmse): hpo_{best_row['hpo_num']:03d}")
    print(f"  Forces RMSE (95% upper): {best_row['err_u95']:.6f}")
    print(f"  Throughput: {best_row['thr_l95']:.2f} {THROUGHPUT_METRIC}")

# Also analyze alternative metric if available
best_row_alt = None
front_all_alt = None
front_feasible_alt = None
if 'err_alt_u95' in merged.columns:
    front_all_alt = pareto_nondominated(merged, "err_alt_u95", "thr_l95")
    front_feasible_alt = pareto_nondominated(feasible, "err_alt_u95", "thr_l95")
    
    print(f"\nAlternative metric ({ALTERNATIVE_ERROR_COL}):")
    print(f"Pareto front (all): {len(front_all_alt)} configurations")
    print(f"Pareto front (feasible): {len(front_feasible_alt)} configurations")
    
    if len(feasible):
        best_idx_alt = feasible["err_alt_u95"].idxmin()
        best_row_alt = feasible.loc[best_idx_alt]
        print(f"\nBest feasible config (min weighted_sum): hpo_{best_row_alt['hpo_num']:03d}")
        print(f"  Weighted sum (95% upper): {best_row_alt['err_alt_u95']:.6f}")
        print(f"  Throughput: {best_row_alt['thr_l95']:.2f} {THROUGHPUT_METRIC}")

# ---------- SELECTED MODEL ANALYSIS ----------
selected_row = None
if SELECTED_MODEL is not None:
    selected_match = merged[merged['hpo_num'] == SELECTED_MODEL]
    if len(selected_match) == 0:
        print(f"\nWARNING: Selected model hpo_{SELECTED_MODEL:03d} not found in dataset!")
    else:
        selected_row = selected_match.iloc[0]
        is_feasible = selected_row['thr_l95'] >= REQUIRED_T if REQUIRED_T is not None else True
        
        print(f"\n" + "="*70)
        print(f"SELECTED MODEL: hpo_{SELECTED_MODEL:03d}")
        print("="*70)
        print(f"  Forces RMSE (95% upper): {selected_row['err_u95']:.6f}")
        if 'err_alt_u95' in selected_row:
            print(f"  Weighted Sum (95% upper): {selected_row['err_alt_u95']:.6f}")
        print(f"  Throughput: {selected_row['thr_l95']:.2f} {THROUGHPUT_METRIC}")
        print(f"  GPU Memory: {selected_row['peak_gpu_memory_mib']:.0f} MiB")
        print(f"  Feasible: {'✓ Yes' if is_feasible else '✗ No (below threshold)'}")
        
        # Rank among all configs
        error_rank = (merged['err_u95'] < selected_row['err_u95']).sum() + 1
        throughput_rank = (merged['thr_l95'] > selected_row['thr_l95']).sum() + 1
        print(f"  Error Rank: {error_rank} / {len(merged)} (lower is better)")
        print(f"  Throughput Rank: {throughput_rank} / {len(merged)} (lower is faster)")
        
        # Compare to best
        if best_row is not None:
            error_diff = selected_row['err_u95'] - best_row['err_u95']
            throughput_diff = selected_row['thr_l95'] - best_row['thr_l95']
            print(f"\n  vs Best Config (hpo_{best_row['hpo_num']:03d}):")
            print(f"    Error difference: {error_diff:+.6f} ({100*error_diff/best_row['err_u95']:+.2f}%)")
            print(f"    Throughput difference: {throughput_diff:+.2f} timesteps/s ({100*throughput_diff/best_row['thr_l95']:+.2f}%)")
        
        print(f"  Hyperparameters:")
        for col in present_config_cols:
            print(f"    {col}: {selected_row[col]}")
        print("="*70)

# ---------- PLOT ----------
print("\nGenerating plots...")

# Create comparison plot if we have alternative metric, otherwise single plot
if 'err_alt_u95' in merged.columns:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax1, ax2 = axes
else:
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax2 = None

# PLOT 1: Forces RMSE
ax = ax1
ax.scatter(merged["thr_l95"], merged["err_u95"], 
           s=80, alpha=0.4, label="All configs", color=COLORS['primary'], edgecolor='none')

if len(front_all):
    sorted_front = front_all.sort_values("thr_l95")
    ax.plot(sorted_front["thr_l95"], sorted_front["err_u95"], 
            'o-', linewidth=2.5, markersize=6, label="Pareto front (all)", 
            color=COLORS['tertiary'], alpha=0.9)

if REQUIRED_T is not None:
    ax.axvline(REQUIRED_T, linestyle="--", linewidth=2, color=COLORS['secondary'],
               label=f"Min throughput = {REQUIRED_T}", alpha=0.8)

if len(front_feasible) and REQUIRED_T is not None:
    sorted_feas = front_feasible.sort_values("thr_l95")
    ax.plot(sorted_feas["thr_l95"], sorted_feas["err_u95"],
            'D-', linewidth=3, markersize=8, label="Pareto front (feasible)", 
            color=COLORS['accent'], markeredgecolor='black', markeredgewidth=1.2)

if best_row is not None:
    x, y = best_row["thr_l95"], best_row["err_u95"]
    ax.scatter([x], [y], s=350, marker="*", color=COLORS['secondary'], 
               edgecolor='black', linewidth=2.5, zorder=10,
               label=f"Best (hpo_{best_row['hpo_num']:03d})")
    
    config_str = f"L={int(best_row['model.num_layers'])}, "
    config_str += f"l_max={int(best_row['model.l_max'])}, "
    config_str += f"r_max={best_row['model.r_max']:.1f}"
    ax.annotate(config_str, (x, y), 
                xytext=(10, 10), textcoords="offset points",
                fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['secondary'], 
                                      alpha=0.8, edgecolor='black'),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5))

# Highlight selected model
if selected_row is not None:
    x, y = selected_row["thr_l95"], selected_row["err_u95"]
    ax.scatter([x], [y], s=450, marker="D", color=COLORS['highlight'], 
               edgecolor='black', linewidth=2.5, zorder=11,
               label=f"Selected (hpo_{SELECTED_MODEL:03d})")
    
    # Compact annotation near the point (like "Best" label)
    config_str = f"L={int(selected_row['model.num_layers'])}, "
    config_str += f"l_max={int(selected_row['model.l_max'])}, "
    config_str += f"r_max={selected_row['model.r_max']:.1f}"
    ax.annotate(config_str, (x, y), 
                xytext=(80, -30), textcoords="offset points",
                fontsize=9, ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['highlight'], 
                          alpha=0.7, edgecolor='black'),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', lw=1.5))

ax.set_xlabel(f"Throughput (timesteps/s) @ {TARGET_SUPERCELL}", fontsize=12)
ax.set_ylabel(f"Forces RMSE (95% upper CI)", fontsize=12)
ax.set_title(f"Forces RMSE vs Throughput", fontsize=13, fontweight='bold')
apply_publication_legend(ax)
ax.grid(True, alpha=0.3, linestyle='--')

# PLOT 2: Weighted Sum (if available)
if ax2 is not None and 'err_alt_u95' in merged.columns:
    ax2.scatter(merged["thr_l95"], merged["err_alt_u95"], 
               s=80, alpha=0.4, label="All configs", color=COLORS['primary'], edgecolor='none')
    
    if len(front_all_alt):
        sorted_front_alt = front_all_alt.sort_values("thr_l95")
        ax2.plot(sorted_front_alt["thr_l95"], sorted_front_alt["err_alt_u95"], 
                'o-', linewidth=2.5, markersize=6, label="Pareto front (all)", 
                color=COLORS['tertiary'], alpha=0.9)
    
    if REQUIRED_T is not None:
        ax2.axvline(REQUIRED_T, linestyle="--", linewidth=2, color=COLORS['secondary'],
                   label=f"Min throughput = {REQUIRED_T}", alpha=0.8)
    
    if len(front_feasible_alt) and REQUIRED_T is not None:
        sorted_feas_alt = front_feasible_alt.sort_values("thr_l95")
        ax2.plot(sorted_feas_alt["thr_l95"], sorted_feas_alt["err_alt_u95"],
                'D-', linewidth=3, markersize=8, label="Pareto front (feasible)", 
                color=COLORS['accent'], markeredgecolor='black', markeredgewidth=1.2)
    
    if best_row_alt is not None:
        x, y = best_row_alt["thr_l95"], best_row_alt["err_alt_u95"]
        ax2.scatter([x], [y], s=350, marker="*", color=COLORS['secondary'], 
                   edgecolor='black', linewidth=2.5, zorder=10,
                   label=f"Best (hpo_{best_row_alt['hpo_num']:03d})")
        
        config_str = f"L={int(best_row_alt['model.num_layers'])}, "
        config_str += f"l_max={int(best_row_alt['model.l_max'])}, "
        config_str += f"r_max={best_row_alt['model.r_max']:.1f}"
        ax2.annotate(config_str, (x, y), 
                    xytext=(10, 10), textcoords="offset points",
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['secondary'], 
                                         alpha=0.8, edgecolor='black'),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5))
    
    # Highlight selected model on weighted sum plot
    if selected_row is not None:
        x, y = selected_row["thr_l95"], selected_row["err_alt_u95"]
        ax2.scatter([x], [y], s=450, marker="D", color=COLORS['highlight'], 
                   edgecolor='black', linewidth=2.5, zorder=11,
                   label=f"Selected (hpo_{SELECTED_MODEL:03d})")
        
        # Compact annotation near the point (like "Best" label)
        config_str = f"L={int(selected_row['model.num_layers'])}, "
        config_str += f"l_max={int(selected_row['model.l_max'])}, "
        config_str += f"r_max={selected_row['model.r_max']:.1f}"
        ax2.annotate(config_str, (x, y), 
                    xytext=(80, -30), textcoords="offset points",
                    fontsize=9, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['highlight'], 
                              alpha=0.7, edgecolor='black'),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', lw=1.5))
    
    ax2.set_xlabel(f"Throughput (timesteps/s) @ {TARGET_SUPERCELL}", fontsize=12)
    ax2.set_ylabel(f"Weighted Sum (95% upper CI)", fontsize=12)
    ax2.set_title(f"Weighted Sum vs Throughput", fontsize=13, fontweight='bold')
    apply_publication_legend(ax2)
    ax2.grid(True, alpha=0.3, linestyle='--')

title = f"HPO Analysis: Error vs Throughput @ {TARGET_SUPERCELL} ({benchmark_df['num_atoms'].iloc[0]:,} atoms)"
if SELECTED_MODEL is not None:
    title += f" | Selected: hpo_{SELECTED_MODEL:03d}"
# plt.suptitle(title, fontsize=15, fontweight='bold', y=0.995)

plt.tight_layout()

save_fig(plt.gcf(), OUTPUT_DIR / "hpo_pareto_analysis")
plt.show()

# ---------- EXPORT RESULTS ----------
print("\nExporting results...")

# Full merged data
merged_output = merged.copy()
merged_output = merged_output.sort_values(['err_u95', 'thr_l95'], ascending=[True, False])
merged_path = OUTPUT_DIR / "hpo_merged_results.csv"
merged_output.to_csv(merged_path, index=False)
print(f"Full results: {merged_path}")

# Feasible Pareto front
if len(front_feasible):
    front_output = front_feasible.sort_values(['err_u95', 'thr_l95'], ascending=[True, False])
    front_path = OUTPUT_DIR / "hpo_feasible_pareto.csv"
    front_output.to_csv(front_path, index=False)
    print(f"Feasible Pareto front: {front_path}")

# Best choice
if best_row is not None:
    best_path = OUTPUT_DIR / "hpo_best_choice.csv"
    pd.DataFrame([best_row]).to_csv(best_path, index=False)
    print(f"Best feasible choice: {best_path}")

# Summary statistics
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total unique configs: {len(merged)}")
print(f"Supercell size: {TARGET_SUPERCELL} ({benchmark_df['num_atoms'].iloc[0]} atoms)")
print(f"Throughput metric: {THROUGHPUT_METRIC}")
if REQUIRED_T is not None:
    print(f"Throughput requirement: {REQUIRED_T}")
    print(f"Feasible configs: {len(feasible)} ({100*len(feasible)/len(merged):.1f}%)")
print(f"Pareto optimal configs: {len(front_feasible)}")

if best_row is not None:
    print(f"\nBEST CONFIG (Forces RMSE): hpo_{best_row['hpo_num']:03d}")
    print(f"  Forces RMSE (95% upper): {best_row['err_u95']:.6f}")
    if 'err_alt_u95' in best_row:
        print(f"  Weighted Sum (95% upper): {best_row['err_alt_u95']:.6f}")
    print(f"  Throughput: {best_row['thr_l95']:.2f} {THROUGHPUT_METRIC}")
    print(f"  GPU Memory: {best_row['peak_gpu_memory_mib']:.0f} MiB")
    print(f"  Hyperparameters:")
    for col in present_config_cols:
        print(f"    {col}: {best_row[col]}")

if best_row_alt is not None and best_row_alt['hpo_num'] != (best_row['hpo_num'] if best_row is not None else -1):
    print(f"\nBEST CONFIG (Weighted Sum): hpo_{best_row_alt['hpo_num']:03d}")
    print(f"  Weighted Sum (95% upper): {best_row_alt['err_alt_u95']:.6f}")
    print(f"  Forces RMSE (95% upper): {best_row_alt['err_u95']:.6f}")
    print(f"  Throughput: {best_row_alt['thr_l95']:.2f} {THROUGHPUT_METRIC}")
    print(f"  GPU Memory: {best_row_alt['peak_gpu_memory_mib']:.0f} MiB")
    print(f"  Hyperparameters:")
    for col in present_config_cols:
        print(f"    {col}: {best_row_alt[col]}")

print("="*60)

# ---------- SCALING ANALYSIS (OPTIONAL) ----------
print("\n" + "="*60)
print("SCALING ANALYSIS")
print("="*60)
print(f"WARNING: Benchmarks were run up to {benchmark_df['num_atoms'].max()} atoms")
print(f"         Target deployment is 250,000 atoms (~{250000/benchmark_df['num_atoms'].max():.0f}× larger)")
print("\nRecommendations:")
print("1. Benchmark the top configs at larger supercell sizes (e.g., 32x32x32)")
print("2. Perform scaling analysis to extrapolate performance")
print("3. Consider memory constraints for 250k atom systems")
print("="*60)

