"""
Throughput scaling analysis across different supercell sizes.
Helps extrapolate performance to larger systems (e.g., 250k atoms).
"""

import re
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

# Shared publication style
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plotting_utils import set_pub_style, save_fig, DOUBLE_COL, fig_size

set_pub_style()

# ---------- CONFIG ----------
CSV_PATH = "/home/myless/Packages/mlip_validation_paper2025/data/wandb_data/hpo_training_results.csv"
INFERENCE_DATA_DIR = "/home/myless/Packages/mlip_validation_paper2025/data/wandb_data/hpo_inference_test_analysis"

# Target atom count for extrapolation
TARGET_ATOMS = 250000

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

def load_all_benchmark_data(inference_dir):
    """Load all benchmark data for all supercell sizes."""
    all_data = []
    
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
            continue
            
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Add all supercell results
        for entry in data:
            all_data.append({
                'hpo_num': hpo_num,
                'supercell': entry['supercell'],
                'num_atoms': entry['num_atoms'],
                'timesteps_per_s': entry['timesteps_per_s_log'],
                'katom_steps_per_s': entry['katom_steps_per_s_log'],
                'peak_gpu_memory_mib': entry['peak_gpu_memory_mib'],
                'wall_time_s': entry['wall_time_s']
            })
    
    return pd.DataFrame(all_data)

def power_law(x, a, b):
    """Power law: y = a * x^b"""
    return a * np.power(x, b)

def inverse_linear(x, a, b):
    """Inverse linear: y = a / (x + b)"""
    return a / (x + b)

def fit_scaling(df_config, metric='timesteps_per_s'):
    """Fit scaling law for a single configuration."""
    x = df_config['num_atoms'].values
    y = df_config[metric].values
    
    if len(x) < 3:
        return None, None, None
    
    try:
        # Try power law fit
        popt_power, _ = curve_fit(power_law, x, y, p0=[1000, -0.5], maxfev=5000)
        y_pred_power = power_law(x, *popt_power)
        r2_power = 1 - np.sum((y - y_pred_power)**2) / np.sum((y - y.mean())**2)
        
        return 'power', popt_power, r2_power
    except:
        return None, None, None

def extrapolate_throughput(df_config, target_atoms, metric='timesteps_per_s'):
    """Extrapolate throughput to target atom count."""
    fit_type, params, r2 = fit_scaling(df_config, metric)
    
    if fit_type is None:
        return None, None, None
    
    if fit_type == 'power':
        prediction = power_law(target_atoms, *params)
        return prediction, fit_type, r2
    
    return None, None, None

# ---------- LOAD DATA ----------
print("Loading benchmark data for all supercell sizes...")
benchmark_df = load_all_benchmark_data(INFERENCE_DATA_DIR)

if len(benchmark_df) == 0:
    raise ValueError("No benchmark data found!")

print(f"Loaded {len(benchmark_df)} benchmark measurements")
print(f"Unique configs: {benchmark_df['hpo_num'].nunique()}")
print(f"Supercell sizes: {sorted(benchmark_df['supercell'].unique())}")
print(f"Atom counts: {sorted(benchmark_df['num_atoms'].unique())}")

# ---------- LOAD TRAINING DATA (for error metrics) ----------
print("\nLoading training data...")
train_df = pd.read_csv(CSV_PATH)
train_df['hpo_num'] = train_df['Name'].apply(extract_hpo_number)
train_df = train_df.dropna(subset=['hpo_num'])
train_df['hpo_num'] = train_df['hpo_num'].astype(int)

# Aggregate to get mean error per config
error_agg = train_df.groupby('hpo_num').agg({
    'test0_epoch/forces_rmse': 'mean',
    'test0_epoch/per_atom_energy_rmse': 'mean',
    'test0_epoch/stress_rmse': 'mean'
}).reset_index()

# ---------- SCALING ANALYSIS ----------
print("\nPerforming scaling analysis...")

scaling_results = []
for hpo_num in benchmark_df['hpo_num'].unique():
    df_config = benchmark_df[benchmark_df['hpo_num'] == hpo_num].sort_values('num_atoms')
    
    # Fit scaling for timesteps/s
    pred_ts, fit_type, r2 = extrapolate_throughput(df_config, TARGET_ATOMS, 'timesteps_per_s')
    
    # Fit scaling for katom_steps/s
    pred_katom, _, _ = extrapolate_throughput(df_config, TARGET_ATOMS, 'katom_steps_per_s')
    
    if pred_ts is not None:
        scaling_results.append({
            'hpo_num': hpo_num,
            'fit_type': fit_type,
            'r2_score': r2,
            'extrapolated_timesteps_per_s': pred_ts,
            'extrapolated_katom_steps_per_s': pred_katom,
            'max_tested_atoms': df_config['num_atoms'].max(),
            'num_datapoints': len(df_config)
        })

scaling_df = pd.DataFrame(scaling_results)

# Merge with error metrics
scaling_df = scaling_df.merge(error_agg, on='hpo_num', how='left')

print(f"Successfully fit scaling laws for {len(scaling_df)} configurations")
print(f"Mean R² score: {scaling_df['r2_score'].mean():.3f}")

# ---------- VISUALIZATION ----------
print("\nGenerating visualizations...")

# 1. Scaling curves for top 5 configs (by forces RMSE)
top_n = 5
top_configs = scaling_df.nsmallest(top_n, 'test0_epoch/forces_rmse')

fig, axes = plt.subplots(1, 2, figsize=fig_size(DOUBLE_COL, 0.45))

# Timesteps/s scaling
ax = axes[0]
colors = plt.cm.viridis(np.linspace(0, 1, top_n))

for i, (_, row) in enumerate(top_configs.iterrows()):
    hpo_num = int(row['hpo_num'])
    df_config = benchmark_df[benchmark_df['hpo_num'] == hpo_num].sort_values('num_atoms')
    
    # Plot measured data
    ax.loglog(df_config['num_atoms'], df_config['timesteps_per_s'], 
              'o', markersize=8, color=colors[i], 
              label=f"hpo_{hpo_num:03d} (err={row['test0_epoch/forces_rmse']:.4f})")
    
    # Plot fitted curve
    x_fit = np.logspace(np.log10(df_config['num_atoms'].min()), 
                        np.log10(TARGET_ATOMS), 100)
    fit_type, params, _ = fit_scaling(df_config, 'timesteps_per_s')
    if fit_type == 'power':
        y_fit = power_law(x_fit, *params)
        ax.loglog(x_fit, y_fit, '--', color=colors[i], alpha=0.6)

# Mark target atom count
ax.axvline(TARGET_ATOMS, color='red', linestyle=':', linewidth=2, 
           label=f'Target: {TARGET_ATOMS:,} atoms')
ax.set_xlabel('Number of Atoms', fontsize=12)
ax.set_ylabel('Timesteps/s', fontsize=12)
ax.set_title('Throughput Scaling: Top Configs by Force Error', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3, which='both')

# Katom_steps/s scaling  
ax = axes[1]
for i, (_, row) in enumerate(top_configs.iterrows()):
    hpo_num = int(row['hpo_num'])
    df_config = benchmark_df[benchmark_df['hpo_num'] == hpo_num].sort_values('num_atoms')
    
    ax.loglog(df_config['num_atoms'], df_config['katom_steps_per_s'],
              'o', markersize=8, color=colors[i],
              label=f"hpo_{hpo_num:03d}")
    
    x_fit = np.logspace(np.log10(df_config['num_atoms'].min()),
                        np.log10(TARGET_ATOMS), 100)
    fit_type, params, _ = fit_scaling(df_config, 'katom_steps_per_s')
    if fit_type == 'power':
        y_fit = power_law(x_fit, *params)
        ax.loglog(x_fit, y_fit, '--', color=colors[i], alpha=0.6)

ax.axvline(TARGET_ATOMS, color='red', linestyle=':', linewidth=2,
           label=f'Target: {TARGET_ATOMS:,} atoms')
ax.set_xlabel('Number of Atoms', fontsize=12)
ax.set_ylabel('kAtom-steps/s', fontsize=12)
ax.set_title('Computational Efficiency Scaling', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
save_fig(plt.gcf(), OUTPUT_DIR / "throughput_scaling_analysis")
plt.show()

# 2. Pareto front with extrapolated throughput
fig, ax = plt.subplots(figsize=(10, 7))

# Plot all configs
ax.scatter(scaling_df['extrapolated_timesteps_per_s'], 
           scaling_df['test0_epoch/forces_rmse'],
           s=100, alpha=0.6, c=scaling_df['r2_score'], 
           cmap='RdYlGn', vmin=0.9, vmax=1.0,
           edgecolor='black', linewidth=0.5)

cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label('R² Score (fit quality)', fontsize=11)

# Highlight top configs
for _, row in top_configs.iterrows():
    hpo_num = int(row['hpo_num'])
    ax.scatter(row['extrapolated_timesteps_per_s'], 
               row['test0_epoch/forces_rmse'],
               s=300, marker='*', color='gold', 
               edgecolor='black', linewidth=2, zorder=10)
    ax.annotate(f"hpo_{hpo_num:03d}", 
                (row['extrapolated_timesteps_per_s'], row['test0_epoch/forces_rmse']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel(f'Extrapolated Throughput at {TARGET_ATOMS:,} atoms (timesteps/s)', fontsize=12)
ax.set_ylabel('Forces RMSE (test)', fontsize=12)
ax.set_title(f'Error vs Extrapolated Throughput\n(Extrapolation to {TARGET_ATOMS:,} atoms)', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()

save_fig(plt.gcf(), OUTPUT_DIR / "pareto_extrapolated_throughput")
plt.show()

# ---------- EXPORT RESULTS ----------
print("\nExporting scaling results...")
scaling_output = scaling_df.sort_values('test0_epoch/forces_rmse')
scaling_path = OUTPUT_DIR / "scaling_extrapolation_results.csv"
scaling_output.to_csv(scaling_path, index=False)
print(f"Scaling results: {scaling_path}")

# ---------- SUMMARY ----------
print("\n" + "="*70)
print("SCALING ANALYSIS SUMMARY")
print("="*70)
print(f"Target system size: {TARGET_ATOMS:,} atoms")
print(f"Largest tested size: {benchmark_df['num_atoms'].max():,} atoms")
print(f"Extrapolation factor: {TARGET_ATOMS/benchmark_df['num_atoms'].max():.1f}×")
print(f"\nConfigurations analyzed: {len(scaling_df)}")
print(f"Mean R² score: {scaling_df['r2_score'].mean():.4f}")
print(f"Min R² score: {scaling_df['r2_score'].min():.4f}")

print("\nTOP 5 CONFIGURATIONS (by forces RMSE):")
print("-" * 70)
for i, (_, row) in enumerate(top_configs.iterrows(), 1):
    print(f"{i}. hpo_{int(row['hpo_num']):03d}")
    print(f"   Forces RMSE: {row['test0_epoch/forces_rmse']:.6f}")
    print(f"   Extrapolated throughput: {row['extrapolated_timesteps_per_s']:.2f} timesteps/s")
    print(f"   Scaling fit R²: {row['r2_score']:.4f}")
    print()

print("="*70)
print("\nWARNING: Extrapolation beyond tested regime carries uncertainty!")
print("Recommendations:")
print("1. Validate predictions with actual benchmarks at larger sizes")
print("2. Check if R² scores are high (> 0.95) for reliable extrapolation")
print("3. Consider memory constraints not captured in throughput scaling")
print("="*70)

