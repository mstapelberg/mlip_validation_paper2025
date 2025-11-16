#!/usr/bin/env python
"""
3-Region Error Analysis: Core / Shell / Bulk
=============================================

Physics-motivated splitting based on Allegro's 5 Å radial cutoff:
1. Core: 8 nearest neighbors to vacancy (directly affected)
2. Shell: Within 5 Å cutoff but not nearest neighbors (indirect perturbation)
3. Bulk: Beyond 5 Å (should be unaffected due to locality)

This respects the strict locality of the Allegro potential.

Usage:
------
    python analyze_3region_errors.py \
      --data_path ../../data/good_atoms_objects_fixed.xyz \
      --models "MSE_model0:path/to/mse.pth" "TailHuber_model0:path/to/tail.pth" \
      --output_dir ../../results/3region_analysis \
      --cutoff 5.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict, List
import sys

from loss_testing import (
    load_by_config,
    locate_vacancy,
    vacancy_center_mic,
    get_reference_forces,
    ASECalculatorPredictor
)
from compare_loss_functions import load_model


def compute_3region_errors(atoms, predict_forces, cutoff: float = 5.0) -> Dict:
    """
    Compute force errors split into 3 physics-motivated regions.
    
    Args:
        cutoff: Radial cutoff of the potential (5.0 Å for Allegro)
    
    Returns dict with:
        - core_errors: 8 nearest neighbors to vacancy
        - shell_errors: within cutoff but not nearest neighbors  
        - bulk_errors: beyond cutoff (should be unaffected)
    """
    # Get forces
    F_true = get_reference_forces(atoms)
    F_pred = predict_forces(atoms)
    errors = np.linalg.norm(F_pred - F_true, axis=1)
    
    # Auto-detect defect (vacancy or interstitial)
    from detect_defects import detect_defect_auto
    from compare_loss_functions import _first_shell_cutoff
    rc = _first_shell_cutoff(atoms)
    center, top_idx, defect_type = detect_defect_auto(atoms, rc=rc)
    center, pos_unwrapped = vacancy_center_mic(atoms, top_idx)
    dists = np.linalg.norm(pos_unwrapped - center, axis=1)
    
    # Find 8 nearest neighbors (core)
    nearest_8_idx = np.argsort(dists)[:8]
    core_mask = np.zeros(len(atoms), dtype=bool)
    core_mask[nearest_8_idx] = True
    
    # Shell: within cutoff but not in core
    shell_mask = (dists <= cutoff) & (~core_mask)
    
    # Bulk: beyond cutoff (strictly local model cannot "see" vacancy)
    bulk_mask = dists > cutoff
    
    return {
        'all_errors': errors,
        'core_errors': errors[core_mask],
        'shell_errors': errors[shell_mask],
        'bulk_errors': errors[bulk_mask],
        'core_mask': core_mask,
        'shell_mask': shell_mask,
        'bulk_mask': bulk_mask,
        'dists': dists,
        'n_core': core_mask.sum(),
        'n_shell': shell_mask.sum(),
        'n_bulk': bulk_mask.sum(),
        'n_total': len(atoms)
    }


def analyze_model_3region(frames: List, predict_forces, model_name: str, 
                          cutoff: float = 5.0) -> pd.DataFrame:
    """Analyze model with 3-region split."""
    results = []
    
    print(f"\nAnalyzing {model_name} (3-region) on {len(frames)} structures...")
    
    for i, atoms in enumerate(frames):
        print(f"  [{i+1}/{len(frames)}] {atoms.get_chemical_formula()}", end=' ')
        
        try:
            result = compute_3region_errors(atoms, predict_forces, cutoff=cutoff)
            
            row = {
                'model': model_name,
                'structure_idx': i,
                'formula': atoms.get_chemical_formula(),
                'n_atoms': len(atoms),
                'n_core': result['n_core'],
                'n_shell': result['n_shell'],
                'n_bulk': result['n_bulk'],
                
                # MAE for each region
                'core_mae': result['core_errors'].mean() if len(result['core_errors']) > 0 else np.nan,
                'shell_mae': result['shell_errors'].mean() if len(result['shell_errors']) > 0 else np.nan,
                'bulk_mae': result['bulk_errors'].mean() if len(result['bulk_errors']) > 0 else np.nan,
                'all_mae': result['all_errors'].mean(),
                
                # Ratios
                'core_shell_ratio': (result['core_errors'].mean() / result['shell_errors'].mean()) 
                                   if len(result['shell_errors']) > 0 and result['shell_errors'].mean() > 0 else np.nan,
                'core_bulk_ratio': (result['core_errors'].mean() / result['bulk_errors'].mean())
                                  if len(result['bulk_errors']) > 0 and result['bulk_errors'].mean() > 0 else np.nan,
            }
            
            results.append(row)
            print("✓")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    return pd.DataFrame(results)


def plot_3region_comparison(summary_df: pd.DataFrame, output_dir: Path):
    """Create 3-region comparison plot."""
    
    models = summary_df['model'].unique()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. MAE by region
    ax = axes[0]
    x = np.arange(len(models))
    width = 0.25
    
    core_mae = summary_df.groupby('model')['core_mae'].mean()
    shell_mae = summary_df.groupby('model')['shell_mae'].mean()
    bulk_mae = summary_df.groupby('model')['bulk_mae'].mean()
    
    ax.bar(x - width, core_mae, width, label='Core (8 NN)', 
           color='red', alpha=0.8, edgecolor='k')
    ax.bar(x, shell_mae, width, label='Shell (< 5 Å)', 
           color='orange', alpha=0.8, edgecolor='k')
    ax.bar(x + width, bulk_mae, width, label='Bulk (> 5 Å)', 
           color='gray', alpha=0.8, edgecolor='k')
    
    ax.set_ylabel('Mean Absolute Error (eV/Å)', fontsize=13, fontweight='bold')
    ax.set_title('3-Region Force Error Analysis\n(Physics-motivated split)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=11)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Core/Bulk ratio (should be >1 if model captures defect effect)
    ax = axes[1]
    
    ratio = summary_df.groupby('model')['core_bulk_ratio'].mean()
    
    bars = ax.bar(models, ratio, color='purple', alpha=0.8, edgecolor='k', linewidth=1.5)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, 
              label='Equal error', alpha=0.7)
    
    ax.set_ylabel('Core/Bulk Error Ratio', fontsize=13, fontweight='bold')
    ax.set_title('Defect Sensitivity\n(Higher = model captures defect physics)', 
                fontsize=14, fontweight='bold')
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}×', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '3region_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved 3-region comparison: {output_dir / '3region_comparison.png'}")
    plt.close()


def create_3region_summary(summary_df: pd.DataFrame, output_dir: Path):
    """Create summary statistics table."""
    
    summary = summary_df.groupby('model').agg({
        'core_mae': ['mean', 'std'],
        'shell_mae': ['mean', 'std'],
        'bulk_mae': ['mean', 'std'],
        'core_bulk_ratio': ['mean', 'std']
    }).round(4)
    
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    summary.to_csv(output_dir / '3region_summary.csv', index=False)
    print(f"✓ Saved 3-region summary: {output_dir / '3region_summary.csv'}")
    
    print("\n" + "="*80)
    print("3-REGION ERROR ANALYSIS")
    print("="*80)
    print(summary.to_string(index=False))
    print("\n")
    
    # Highlight key findings
    print("="*80)
    print("KEY FINDINGS:")
    print("="*80)
    
    best_core = summary.loc[summary['core_mae_mean'].idxmin(), 'model']
    best_core_val = summary['core_mae_mean'].min()
    print(f"✓ Best core error: {best_core} ({best_core_val:.4f} eV/Å)")
    
    best_bulk = summary.loc[summary['bulk_mae_mean'].idxmin(), 'model']
    best_bulk_val = summary['bulk_mae_mean'].min()
    print(f"✓ Best bulk error: {best_bulk} ({best_bulk_val:.4f} eV/Å)")
    
    highest_ratio = summary.loc[summary['core_bulk_ratio_mean'].idxmax(), 'model']
    highest_ratio_val = summary['core_bulk_ratio_mean'].max()
    print(f"✓ Highest core/bulk ratio: {highest_ratio} ({highest_ratio_val:.2f}×)")
    print("  → This model best captures that defects are harder to predict")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='3-region error analysis (core/shell/bulk)',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to vacancy structures dataset')
    
    parser.add_argument('--models', nargs='+', type=str, required=True,
                       help='Models as "NAME:PATH" pairs')
    
    parser.add_argument('--cutoff', type=float, default=5.0,
                       help='Radial cutoff of potential (default: 5.0 Å)')
    
    parser.add_argument('--n_structures', type=int, default=None,
                       help='Number of structures to analyze')
    
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for inference (cpu/cuda)')
    
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Parse models
    models_dict = {}
    for model_spec in args.models:
        try:
            name, path = model_spec.split(':', 1)
            models_dict[name] = path
        except ValueError:
            print(f"Error: Invalid model specification '{model_spec}'")
            sys.exit(1)
    
    print("="*80)
    print("3-REGION ERROR ANALYSIS")
    print("="*80)
    print(f"\nPhysics-motivated split:")
    print(f"  Core:  8 nearest neighbors (direct vacancy interaction)")
    print(f"  Shell: Within {args.cutoff} Å (indirect perturbation)")
    print(f"  Bulk:  Beyond {args.cutoff} Å (should be unaffected by locality)")
    print()
    
    # Load data
    print("Loading data...")
    if 'fixed_test_global.xyz' in args.data_path:
        print("Using filtered loading (vac*, neb*, sia* from gen <= 7)")
        from load_filtered_data import load_filtered_structures
        frames = load_filtered_structures(
            args.data_path,
            config_patterns=['vac*', 'neb*', 'sia*'],
            max_generation=7,
            max_structures=args.n_structures
        )
    else:
        from ase.io import read
        frames = read(args.data_path, ':')
        print(f"Loaded {len(frames)} structures")
        if args.n_structures:
            frames = frames[:args.n_structures]
            print(f"Using first {len(frames)} structures")
    
    # Analyze each model
    all_results = []
    
    for model_name, model_path in models_dict.items():
        print("\n" + "="*80)
        print(f"ANALYZING: {model_name}")
        print("="*80)
        
        try:
            predictor = load_model(model_path, device=args.device)
            df = analyze_model_3region(frames, predictor, model_name, cutoff=args.cutoff)
            all_results.append(df)
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_results:
        print("\n✗ No models were successfully analyzed!")
        sys.exit(1)
    
    # Combine results
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(output_dir / '3region_detailed_results.csv', index=False)
    print(f"\n✓ Saved detailed results: {output_dir / '3region_detailed_results.csv'}")
    
    # Create visualizations
    plot_3region_comparison(combined_df, output_dir)
    create_3region_summary(combined_df, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()

