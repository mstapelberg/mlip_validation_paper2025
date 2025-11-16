#!/usr/bin/env python
"""
Plot Ensemble Comparison with Error Bars
=========================================

This script takes results from compare_loss_functions.py (where you ran all
18 models = 6 loss types × 3 seeds) and creates grouped plots with error bars.

Usage:
------
    # Step 1: Run all models
    python compare_loss_functions.py --models \
      "MSE_seed1:..." "MSE_seed2:..." "MSE_seed3:..." \
      "TailHuber_seed1:..." "TailHuber_seed2:..." "TailHuber_seed3:..." \
      --output results/all_18_models
    
    # Step 2: Group by loss type and plot with error bars
    python plot_ensemble_comparison.py \
      --detailed_results results/all_18_models/detailed_results.csv \
      --output results/ensemble_comparison
      
Model Naming Convention:
------------------------
Model names should follow: LOSSTYPE_seedN or LOSSTYPE_runN
Examples:
  - MSE_seed1, MSE_seed2, MSE_seed3
  - TailHuber_run1, TailHuber_run2, TailHuber_run3
  
The script extracts the loss type by removing "_seed*" or "_run*" suffix.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re


def extract_loss_type(model_name: str) -> str:
    """
    Extract loss type from model name.
    
    Examples:
        MSE_seed1 -> MSE
        TailHuber_run2 -> TailHuber
        RMCE_v1 -> RMCE
    """
    # Remove common suffixes: _seed*, _run*, _v*, _model*
    base = re.sub(r'_(seed|run|v|model)\d+$', '', model_name, flags=re.IGNORECASE)
    return base


def group_by_loss_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group results by loss type and compute mean ± std across ensemble.
    
    Returns:
        DataFrame with columns:
        - loss_type
        - defect_mae_mean, defect_mae_std
        - bulk_mae_mean, bulk_mae_std
        - ratio_mean, ratio_std
        - n_models (number of ensemble members)
    """
    # Extract loss type for each model
    df['loss_type'] = df['model'].apply(extract_loss_type)
    
    # Group by loss type and compute statistics
    grouped = df.groupby('loss_type').agg({
        'defect_mae': ['mean', 'std', 'count'],
        'bulk_mae': ['mean', 'std'],
        'all_mae': ['mean', 'std'],
        'defect_rmse': ['mean', 'std'],
        'bulk_rmse': ['mean', 'std'],
        'ratio': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    grouped = grouped.rename(columns={'loss_type': 'loss_type'})
    
    return grouped


def plot_ensemble_comparison(grouped_df: pd.DataFrame, output_dir: Path):
    """
    Create comparison plots with error bars showing ensemble statistics.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    loss_types = grouped_df['loss_type'].values
    n_types = len(loss_types)
    x = np.arange(n_types)
    width = 0.35
    
    # Define colors for each loss type
    color_map = {
        'MSE': '#2A33C3',        # Blue
        'MAE': '#A35D00',         # Orange/Brown
        'RMSE': '#A35D00',        # Orange/Brown
        'RMCE': '#6E8B00',        # Green
        'RMQE': '#E2E8F0',        # Slate
        'TailHuber': '#0B7285',   # Teal/Cyan
        'StratifiedHuber': '#8F2D56',  # Magenta/Pink
        'StratHuber': '#8F2D56',  # Magenta/Pink
    }
    colors = [color_map.get(lt, 'gray') for lt in loss_types]
    
    # 1. Defect vs Bulk MAE
    ax = axes[0]
    
    defect_means = grouped_df['defect_mae_mean'].values
    defect_stds = grouped_df['defect_mae_std'].values
    bulk_means = grouped_df['bulk_mae_mean'].values
    bulk_stds = grouped_df['bulk_mae_std'].values
    
    bars1 = ax.bar(x - width/2, defect_means, width, 
                   yerr=defect_stds, capsize=5,
                   label='Near Defect', color='cyan', alpha=0.58, 
                   edgecolor='k', linewidth=1.5)
    bars2 = ax.bar(x + width/2, bulk_means, width,
                   yerr=bulk_stds, capsize=5,
                   label='Bulk', color='gray', alpha=0.58,
                   edgecolor='k', linewidth=1.5)
    
    ax.set_ylabel('Mean Absolute Error (eV/Å)', fontsize=13, fontweight='bold')
    ax.set_title('Defect vs Bulk Error\n(Mean ± Std across ensemble)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(loss_types, rotation=15, ha='right', fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Defect/Bulk Ratio
    ax = axes[1]
    
    ratio_means = grouped_df['ratio_mean'].values
    ratio_stds = grouped_df['ratio_std'].values
    
    bars = ax.bar(loss_types, ratio_means, yerr=ratio_stds, capsize=5,
                 color=colors, alpha=0.58, edgecolor='k', linewidth=1.5)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, 
              label='Equal error', alpha=0.7)
    
    ax.set_ylabel('Defect/Bulk Error Ratio', fontsize=13, fontweight='bold')
    ax.set_title('Error Localization\n(Lower = Better defect learning)', 
                fontsize=14, fontweight='bold')
    ax.set_xticklabels(loss_types, rotation=15, ha='right', fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels with ± std
    for i, (bar, mean, std) in enumerate(zip(bars, ratio_means, ratio_stds)):
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.2f}±{std:.2f}', ha='center', va='bottom',
                   fontsize=9, fontweight='bold')
    
    # 3. Overall MAE (across all atoms)
    ax = axes[2]
    
    all_mae_means = grouped_df['all_mae_mean'].values
    all_mae_stds = grouped_df['all_mae_std'].values
    
    bars = ax.bar(loss_types, all_mae_means, yerr=all_mae_stds, capsize=5,
                 color=colors, alpha=0.58, edgecolor='k', linewidth=1.5)
    
    ax.set_ylabel('Mean Absolute Error (eV/Å)', fontsize=13, fontweight='bold')
    ax.set_title('Overall Force Error\n(All atoms)', 
                fontsize=14, fontweight='bold')
    ax.set_xticklabels(loss_types, rotation=15, ha='right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ensemble_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved ensemble comparison: {output_dir / 'ensemble_comparison.png'}")
    plt.close()


def plot_variance_comparison(grouped_df: pd.DataFrame, output_dir: Path):
    """
    Plot coefficient of variation (CV) to show which loss functions are stable.
    CV = std / mean (lower = more stable across seeds)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    loss_types = grouped_df['loss_type'].values
    
    # Compute coefficient of variation for defect error
    cv_defect = (grouped_df['defect_mae_std'] / grouped_df['defect_mae_mean']) * 100
    cv_bulk = (grouped_df['bulk_mae_std'] / grouped_df['bulk_mae_mean']) * 100
    
    x = np.arange(len(loss_types))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cv_defect, width, label='Defect Error CV', 
                   color='cyan', alpha=0.58, edgecolor='k')
    bars2 = ax.bar(x + width/2, cv_bulk, width, label='Bulk Error CV',
                   color='gray', alpha=0.58, edgecolor='k')
    
    ax.set_ylabel('Coefficient of Variation (%)', fontsize=13, fontweight='bold')
    ax.set_title('Training Reproducibility\n(Lower = More consistent across seeds)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(loss_types, rotation=15, ha='right', fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reproducibility_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved reproducibility comparison: {output_dir / 'reproducibility_comparison.png'}")
    plt.close()


def create_ensemble_summary_table(grouped_df: pd.DataFrame, output_dir: Path):
    """
    Create a clean summary table for the paper.
    """
    # Select key columns
    summary = grouped_df[[
        'loss_type',
        'defect_mae_mean', 'defect_mae_std',
        'bulk_mae_mean', 'bulk_mae_std',
        'ratio_mean', 'ratio_std',
        'defect_mae_count'  # number of ensemble members
    ]].copy()
    
    summary = summary.rename(columns={'defect_mae_count': 'n_models'})
    summary = summary.round(4)
    
    # Sort by ratio_mean descending (highest first = most sensitive to defects)
    # Winner should have: highest defect MAE (most sensitive), lowest bulk MAE, highest ratio
    summary = summary.sort_values('ratio_mean', ascending=False)
    
    # Save
    summary.to_csv(output_dir / 'ensemble_summary.csv', index=False)
    print(f"✓ Saved ensemble summary: {output_dir / 'ensemble_summary.csv'}")
    
    # Print to console
    print("\n" + "="*80)
    print("ENSEMBLE SUMMARY (mean ± std across seeds)")
    print("Sorted by defect sensitivity (ratio_mean, highest first)")
    print("="*80)
    print(summary.to_string(index=False))
    print("\n")
    
    # Highlight winner (highest ratio = most sensitive to defects)
    best_loss = summary.iloc[0]['loss_type']
    best_ratio = summary.iloc[0]['ratio_mean']
    best_ratio_std = summary.iloc[0]['ratio_std']
    best_defect = summary.iloc[0]['defect_mae_mean']
    best_defect_std = summary.iloc[0]['defect_mae_std']
    best_bulk = summary.iloc[0]['bulk_mae_mean']
    best_bulk_std = summary.iloc[0]['bulk_mae_std']
    
    print("="*80)
    print(f"WINNER (Most Defect-Sensitive): {best_loss}")
    print(f"  Defect/Bulk Ratio: {best_ratio:.4f} ± {best_ratio_std:.4f}")
    print(f"  Defect MAE: {best_defect:.4f} ± {best_defect_std:.4f} eV/Å")
    print(f"  Bulk MAE: {best_bulk:.4f} ± {best_bulk_std:.4f} eV/Å")
    print("="*80)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Create ensemble comparison plots with error bars',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--detailed_results', type=str, required=True,
                       help='Path to detailed_results.csv from compare_loss_functions.py')
    
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for ensemble plots')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("ENSEMBLE COMPARISON WITH ERROR BARS")
    print("="*80)
    
    # Load results
    print(f"\nLoading: {args.detailed_results}")
    df = pd.read_csv(args.detailed_results)
    print(f"  Total rows: {len(df)}")
    print(f"  Models: {df['model'].nunique()}")
    print(f"  Unique model names: {list(df['model'].unique())}")
    
    # Group by loss type
    print("\nGrouping models by loss type...")
    grouped = group_by_loss_type(df)
    print(f"  Loss types found: {list(grouped['loss_type'].values)}")
    print(f"  Models per type: {grouped['defect_mae_count'].values}")
    
    # Create plots
    print("\nCreating ensemble comparison plots...")
    plot_ensemble_comparison(grouped, output_dir)
    plot_variance_comparison(grouped, output_dir)
    create_ensemble_summary_table(grouped, output_dir)
    
    print("\n" + "="*80)
    print("ENSEMBLE ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nKey files:")
    print("  - ensemble_comparison.png         (defect vs bulk with error bars)")
    print("  - reproducibility_comparison.png  (coefficient of variation)")
    print("  - ensemble_summary.csv            (statistics table)")


if __name__ == '__main__':
    main()

