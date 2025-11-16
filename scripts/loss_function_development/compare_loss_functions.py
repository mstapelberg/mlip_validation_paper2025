#!/usr/bin/env python
"""
Compare Loss Functions: Defect vs Bulk Error Analysis
======================================================

This script tests the hypothesis that different force loss functions (MSE, RMSE, 
RMCE, RMQE, TailHuber, StratifiedHuber) produce models with different accuracy 
on defects vs bulk regions.

Hypothesis:
-----------
TailHuber loss produces models that:
1. Have lower force errors near defects (where forces are high)
2. Maintain bulk accuracy (don't sacrifice equilibrium regions)
3. Train stably (unlike RMCE/RMQE which may diverge)

Test:
-----
For each model trained with a different loss function:
1. Compute per-atom force errors on vacancy structures
2. Split atoms: "near defect" (<1.2×rc) vs "bulk" (>3×rc)
3. Compare mean errors in each region

Usage:
------
    python compare_loss_functions.py \
      --data_path ../../data/good_atoms_objects.xyz \
      --models "MSE:path/to/mse.pth" "TailHuber:path/to/tail.pth" \
      --output_dir ../../results/loss_comparison \
      --n_structures 20
      
    # Or use a config file (recommended):
    python compare_loss_functions.py --config models_config.yaml
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict, List, Optional
import sys

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback: create a dummy tqdm that just returns the iterable
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Import utilities from loss_testing.py
from loss_testing import (
    load_by_config,
    vacancy_center_mic,
    get_reference_forces,
    ASECalculatorPredictor
)
from cache_predictions import (
    get_cache_dir,
    save_model_predictions,
    load_model_predictions,
    compute_predictions_for_frame,
    compute_errors_from_cache
)


# =============================================================================
# Model Loading
# =============================================================================

def load_model(model_path: str, device: str = 'cuda'):
    """Load a NequIP/Allegro model."""
    import torch
    
    # Print CUDA diagnostics
    if device == 'cuda':
        if torch.cuda.is_available():
            print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  PyTorch version: {torch.__version__}")
        else:
            print("  ⚠ CUDA requested but not available, falling back to CPU")
            device = 'cpu'
    
    try:
        from nequip.ase import NequIPCalculator
        
        # Check NequIP version
        try:
            import nequip
            nequip_version = nequip.__version__
            print(f"  NequIP version: {nequip_version}")
            # Parse version string (handle formats like "0.15.0" or "0.15")
            version_parts = nequip_version.split('.')
            major = int(version_parts[0])
            minor = int(version_parts[1]) if len(version_parts) > 1 else 0
            use_new_api = (major > 0) or (major == 0 and minor >= 15)
        except (ImportError, AttributeError, ValueError):
            # If version check fails, try new API first
            print("  ⚠ Could not determine NequIP version, trying new API first")
            use_new_api = True
        
        print(f"  Loading model from {model_path}")
        
        # Determine loading method based on file extension
        if model_path.endswith('.pt2') or model_path.endswith('.nequip.pt2'):
            print(f"    → Compiled model (.pt2)")
            calc = NequIPCalculator.from_compiled_model(
                compile_path=model_path,
                device=device
            )
        else:
            # Set global options if available
            try:
                from nequip.utils._global_options import _set_global_options
                _set_global_options(dict(allow_tf32=False))
            except ImportError:
                pass
            
            if use_new_api:
                # NequIP >= 0.15: use _from_saved_model for both packaged and deployed models
                if model_path.endswith('.nequip.zip') or model_path.endswith('.zip'):
                    print(f"    → Packaged model (.nequip.zip) [using _from_saved_model]")
                else:
                    print(f"    → Deployed/saved model (.pth or other) [using _from_saved_model]")
                
                calc = NequIPCalculator._from_saved_model(
                    model_path=model_path,
                    device=device
                )
            else:
                # NequIP < 0.15: use separate methods for packaged vs deployed
                if model_path.endswith('.nequip.zip') or model_path.endswith('.zip'):
                    print(f"    → Packaged model (.nequip.zip) [trying _from_packaged_model]")
                    # Try both _from_packaged_model and from_packaged_model (developers changed it)
                    try:
                        calc = NequIPCalculator._from_packaged_model(
                            package_path=model_path,
                            device=device
                        )
                    except AttributeError:
                        print(f"    → Trying from_packaged_model (public method)")
                        calc = NequIPCalculator.from_packaged_model(
                            package_path=model_path,
                            device=device
                        )
                else:
                    print(f"    → Deployed model (.pth) [using from_deployed_model]")
                    calc = NequIPCalculator.from_deployed_model(
                        model_path=model_path,
                        device=device
                    )
        
        return ASECalculatorPredictor(calc)
    
    except RuntimeError as e:
        error_msg = str(e)
        # Check for CUDA compute capability errors
        if 'no kernel image is available' in error_msg or 'CUDA' in error_msg:
            if device == 'cuda':
                print(f"    ✗ CUDA error: {error_msg}")
                print(f"    → Falling back to CPU...")
                return load_model(model_path, device='cpu')
            else:
                print(f"    ✗ Error loading model on CPU: {e}")
                raise
        else:
            raise
    except Exception as e:
        print(f"    ✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise


# =============================================================================
# Error Analysis
# =============================================================================

def compute_defect_bulk_errors(atoms, predict_forces, 
                               near_mult: float = 1.2,
                               far_mult: float = 3.0) -> Dict:
    """
    Compute force errors split by proximity to defect.
    
    Auto-detects defect type (vacancy or interstitial) from config_type.
    
    Returns dict with:
    - defect_errors: errors for atoms near defect (< near_mult × rc)
    - bulk_errors: errors for atoms far from defect (> far_mult × rc)
    - all_errors: all per-atom errors
    - defect_mask: boolean mask for defect atoms
    - bulk_mask: boolean mask for bulk atoms
    - defect_type: 'vacancy' or 'interstitial'
    """
    # Get reference and predicted forces
    F_true = get_reference_forces(atoms)
    F_pred = predict_forces(atoms)
    
    # Compute per-atom force error magnitude
    errors = np.linalg.norm(F_pred - F_true, axis=1)
    
    # Auto-detect defect type and locate it
    from detect_defects import detect_defect_auto
    rc = _first_shell_cutoff(atoms)
    center, top_idx, defect_type = detect_defect_auto(atoms, rc=rc)
    
    # Get unwrapped positions and distances (vacancy_center_mic works for both vacancy and interstitial)
    center, pos_unwrapped = vacancy_center_mic(atoms, top_idx)
    dists = np.linalg.norm(pos_unwrapped - center, axis=1)
    
    # Create masks
    defect_mask = dists < (near_mult * rc)
    bulk_mask = dists > (far_mult * rc)
    
    return {
        'all_errors': errors,
        'defect_errors': errors[defect_mask],
        'bulk_errors': errors[bulk_mask],
        'defect_mask': defect_mask,
        'bulk_mask': bulk_mask,
        'dists': dists,
        'rc': rc,
        'n_defect': defect_mask.sum(),
        'n_bulk': bulk_mask.sum(),
        'n_total': len(atoms),
        'defect_type': defect_type,
        'config_type': atoms.info.get('config_type', 'unknown')
    }


def _first_shell_cutoff(atoms, use_simple=True):
    """Estimate first-shell cutoff (simplified from loss_testing.py)."""
    D = atoms.get_all_distances(mic=True)
    iu = np.triu_indices(len(atoms), 1)
    d = D[iu]
    d = d[d > 1e-8]
    
    if use_simple or len(atoms) < 200:
        rc = float(np.percentile(d, 5) * 1.20)
        rc = np.clip(rc, 2.5, 3.5)
        return rc
    
    # Fallback
    rc = float(np.percentile(d, 7.5) * 1.15)
    rc = np.clip(rc, 2.5, 3.5)
    return rc


def analyze_model_on_dataset(frames: List, predict_forces, 
                             model_name: str,
                             cache_dir: Optional[Path] = None,
                             environment: str = 'standard',
                             use_cache: bool = True,
                             cached_predictions: Optional[List] = None) -> pd.DataFrame:
    """
    Analyze a model on multiple structures, using cache if available.
    
    Args:
        frames: List of ASE Atoms objects
        predict_forces: Predictor object (can be None if using cache)
        model_name: Name identifier for the model
        cache_dir: Directory for cache files
        environment: 'standard' or 'custom'
        use_cache: Whether to use cache
        cached_predictions: Pre-loaded cache (optional, will load if None)
    
    Returns DataFrame with one row per structure containing:
    - structure_idx
    - formula
    - defect_mae, bulk_mae, all_mae
    - defect_rmse, bulk_rmse, all_rmse
    - ratio (defect_mae / bulk_mae)
    """
    results = []
    
    # Try to load from cache if not provided
    if cached_predictions is None and use_cache and cache_dir is not None:
        cached_predictions = load_model_predictions(cache_dir, model_name, environment)
        if cached_predictions is not None:
            print(f"\n  ✓ Loaded {len(cached_predictions)} cached predictions for {model_name}")
            # Verify cache matches frame count
            if len(cached_predictions) != len(frames):
                print(f"  ⚠ Cache size mismatch ({len(cached_predictions)} vs {len(frames)}), re-running inference")
                cached_predictions = None
    
    # If no cache, compute predictions (requires predictor)
    if cached_predictions is None:
        if predict_forces is None:
            raise ValueError(f"No cache found for {model_name} and no predictor provided")
        
        print(f"\nAnalyzing {model_name} on {len(frames)} structures...")
        cached_predictions = []
        
        # Use tqdm for progress bar if available
        frame_iter = tqdm(enumerate(frames), total=len(frames), 
                         desc=f"  {model_name}", unit="struct", 
                         disable=not TQDM_AVAILABLE)
        
        for i, atoms in frame_iter:
            if TQDM_AVAILABLE:
                frame_iter.set_postfix(formula=atoms.get_chemical_formula())
            else:
                print(f"  [{i+1}/{len(frames)}] {atoms.get_chemical_formula()}", end=' ', flush=True)
            
            try:
                pred_data = compute_predictions_for_frame(atoms, predict_forces)
                pred_data['structure_idx'] = i
                cached_predictions.append(pred_data)
                if not TQDM_AVAILABLE:
                    print("✓", flush=True)
            except Exception as e:
                if not TQDM_AVAILABLE:
                    print(f"✗ Error: {e}", flush=True)
                else:
                    frame_iter.write(f"  ✗ Error on structure {i+1}: {e}")
                continue
        
        # Save to cache
        if use_cache and cache_dir is not None:
            save_model_predictions(cache_dir, model_name, cached_predictions, environment)
    
    # Now compute error metrics from cached predictions
    print(f"\nComputing error metrics from {'cache' if cached_predictions else 'predictions'}...")
    
    # Use tqdm for progress bar
    metrics_iter = tqdm(enumerate(frames), total=len(frames),
                       desc=f"  Computing metrics", unit="struct",
                       disable=not TQDM_AVAILABLE or len(frames) == 0)
    
    for i, atoms in metrics_iter:
        if i >= len(cached_predictions):
            continue
        
        try:
            # Compute defect/bulk split (fast, no model inference needed)
            from detect_defects import detect_defect_auto
            rc = _first_shell_cutoff(atoms)
            center, top_idx, defect_type = detect_defect_auto(atoms, rc=rc)
            center, pos_unwrapped = vacancy_center_mic(atoms, top_idx)
            dists = np.linalg.norm(pos_unwrapped - center, axis=1)
            
            # Create masks
            defect_mask = dists < (1.2 * rc)
            bulk_mask = dists > (3.0 * rc)
            
            # Get errors from cache
            result = compute_errors_from_cache(
                cached_predictions, i, defect_mask, bulk_mask
            )
            result['defect_type'] = defect_type
            
            # Compute statistics
            row = {
                'model': model_name,
                'structure_idx': i,
                'formula': cached_predictions[i]['formula'],
                'config_type': result['config_type'],
                'defect_type': result['defect_type'],
                'n_atoms': len(atoms),
                'n_defect': result['n_defect'],
                'n_bulk': result['n_bulk'],
                
                # MAE
                'defect_mae': result['defect_errors'].mean() if len(result['defect_errors']) > 0 else np.nan,
                'bulk_mae': result['bulk_errors'].mean() if len(result['bulk_errors']) > 0 else np.nan,
                'all_mae': result['all_errors'].mean(),
                
                # RMSE
                'defect_rmse': np.sqrt((result['defect_errors']**2).mean()) if len(result['defect_errors']) > 0 else np.nan,
                'bulk_rmse': np.sqrt((result['bulk_errors']**2).mean()) if len(result['bulk_errors']) > 0 else np.nan,
                'all_rmse': np.sqrt((result['all_errors']**2).mean()),
            }
            
            # Ratio
            if row['bulk_mae'] > 0:
                row['ratio'] = row['defect_mae'] / row['bulk_mae']
            else:
                row['ratio'] = np.nan
            
            results.append(row)
            
        except Exception as e:
            if TQDM_AVAILABLE:
                metrics_iter.write(f"  ✗ Error computing metrics for structure {i}: {e}")
            else:
                print(f"  ✗ Error computing metrics for structure {i}: {e}", flush=True)
            continue
    
    return pd.DataFrame(results)


# =============================================================================
# Visualization
# =============================================================================

def plot_error_distributions(results_dict: Dict[str, List], 
                             output_dir: Path,
                             max_error: float = 0.5):
    """
    Plot error distributions grouped by loss type (averaged over ensemble seeds).
    Shows histogram of errors with defect atoms in cyan, bulk in gray.
    """
    # Group models by loss type
    import re
    
    loss_type_results = {}
    for model_name, struct_results in results_dict.items():
        # Extract loss type (remove _model0, _seed1, etc.)
        loss_type = re.sub(r'_(seed|run|v|model)\d+$', '', model_name, flags=re.IGNORECASE)
        
        if loss_type not in loss_type_results:
            loss_type_results[loss_type] = []
        loss_type_results[loss_type].extend(struct_results)
    
    n_types = len(loss_type_results)
    fig, axes = plt.subplots(2, (n_types + 1) // 2, figsize=(5 * ((n_types + 1) // 2), 8))
    axes = axes.flatten() if n_types > 1 else [axes]
    
    for idx, (loss_type, struct_results) in enumerate(loss_type_results.items()):
        ax = axes[idx]
        
        # Collect all errors across structures
        all_defect_errors = []
        all_bulk_errors = []
        
        for result in struct_results:
            all_defect_errors.extend(result['defect_errors'])
            all_bulk_errors.extend(result['bulk_errors'])
        
        all_defect_errors = np.array(all_defect_errors)
        all_bulk_errors = np.array(all_bulk_errors)
        
        # Plot histograms
        bins = np.linspace(0, max_error, 50)
        
        ax.hist(all_bulk_errors, bins=bins, alpha=0.6, color='gray', 
               label=f'Bulk (n={len(all_bulk_errors)})', density=True)
        ax.hist(all_defect_errors, bins=bins, alpha=0.7, color='cyan', 
               label=f'Defect (n={len(all_defect_errors)})', density=True)
        
        # Mark means
        if len(all_defect_errors) > 0:
            defect_mean = all_defect_errors.mean()
            ax.axvline(defect_mean, color='cyan', linestyle='--', linewidth=2,
                      label=f'Defect μ={defect_mean:.3f}')
        
        if len(all_bulk_errors) > 0:
            bulk_mean = all_bulk_errors.mean()
            ax.axvline(bulk_mean, color='gray', linestyle='--', linewidth=2,
                      label=f'Bulk μ={bulk_mean:.3f}')
        
        ax.set_xlabel('Force Error (eV/Å)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{loss_type} (n={len(struct_results)} seeds)', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, max_error])
    
    # Hide extra subplots
    for idx in range(n_types, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distributions.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved error distributions: {output_dir / 'error_distributions.png'}")
    plt.close()


def plot_defect_vs_bulk_comparison(summary_df: pd.DataFrame, 
                                   output_dir: Path):
    """
    Bar chart comparing defect vs bulk errors across models.
    """
    models = summary_df['model'].unique()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    # MAE comparison
    ax = axes[0]
    defect_mae = summary_df.groupby('model')['defect_mae'].mean()
    bulk_mae = summary_df.groupby('model')['bulk_mae'].mean()
    
    bars1 = ax.bar(x - width/2, defect_mae, width, label='Near Defect', 
                   color='cyan', alpha=0.8, edgecolor='k')
    bars2 = ax.bar(x + width/2, bulk_mae, width, label='Bulk', 
                   color='gray', alpha=0.8, edgecolor='k')
    
    ax.set_ylabel('Mean Absolute Error (eV/Å)', fontsize=12, fontweight='bold')
    ax.set_title('Defect vs Bulk Force Error (MAE)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Ratio comparison
    ax = axes[1]
    ratio = summary_df.groupby('model')['ratio'].mean()
    
    bars = ax.bar(models, ratio, color='orange', alpha=0.8, edgecolor='k', linewidth=1.5)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='Equal error')
    ax.set_ylabel('Defect/Bulk Error Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Error Localization to Defects', fontsize=13, fontweight='bold')
    ax.set_xticklabels(models, rotation=15, ha='right')
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
    plt.savefig(output_dir / 'defect_vs_bulk_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison: {output_dir / 'defect_vs_bulk_comparison.png'}")
    plt.close()


def create_summary_table(summary_df: pd.DataFrame, output_dir: Path):
    """
    Create summary statistics table.
    """
    # Group by model
    summary = summary_df.groupby('model').agg({
        'defect_mae': ['mean', 'std'],
        'bulk_mae': ['mean', 'std'],
        'all_mae': ['mean', 'std'],
        'ratio': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    # Save to CSV
    summary.to_csv(output_dir / 'summary_statistics.csv', index=False)
    print(f"✓ Saved summary: {output_dir / 'summary_statistics.csv'}")
    
    # Print to console
    print("\n" + "="*80)
    print("SUMMARY: Defect vs Bulk Force Errors")
    print("="*80)
    print(summary.to_string(index=False))
    print()
    
    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    # Immediate output to show script started
    print("Starting compare_loss_functions.py...", flush=True)
    sys.stdout.flush()
    
    parser = argparse.ArgumentParser(
        description='Compare loss functions: defect vs bulk error analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--data_path', type=str, 
                       default='../../data/good_atoms_objects.xyz',
                       help='Path to vacancy structures dataset')
    
    parser.add_argument('--models', nargs='+', type=str,
                       help='Models as "NAME:PATH" pairs, e.g., "MSE:path/to/mse.pth"')
    
    parser.add_argument('--config_type', type=str, default='vacancy-alloy',
                       help='Config type to analyze (default: vacancy-alloy)')
    
    parser.add_argument('--n_structures', type=int, default=None,
                       help='Number of structures to analyze (default: all)')
    
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for inference (cpu/cuda, default: cpu)')
    
    parser.add_argument('--output_dir', type=str, 
                       default='../../results/loss_comparison',
                       help='Output directory')
    
    parser.add_argument('--near_mult', type=float, default=1.2,
                       help='Multiplier for defect region (default: 1.2)')
    
    parser.add_argument('--far_mult', type=float, default=3.0,
                       help='Multiplier for bulk region (default: 3.0)')
    
    parser.add_argument('--use_cache', action='store_true', default=True,
                       help='Use prediction cache if available (default: True)')
    
    parser.add_argument('--no_cache', dest='use_cache', action='store_false',
                       help='Disable cache (force re-run inference)')
    
    parser.add_argument('--environment', type=str, default='standard',
                       choices=['standard', 'custom'],
                       help='Environment type (standard or custom for RMCE/RMQE)')
    
    args = parser.parse_args()
    
    # Immediate output after parsing args
    print("Arguments parsed successfully", flush=True)
    sys.stdout.flush()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup cache
    cache_dir = get_cache_dir(output_dir) if args.use_cache else None
    
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Cache directory: {cache_dir}", flush=True)
    sys.stdout.flush()
    
    # Parse models
    if not args.models:
        print("Error: Must provide --models argument")
        print('Example: --models "MSE:path/to/mse.pth" "TailHuber:path/to/tail.pth"')
        sys.exit(1)
    
    models_dict = {}
    for model_spec in args.models:
        try:
            name, path = model_spec.split(':', 1)
            models_dict[name] = path
        except ValueError:
            print(f"Error: Invalid model specification '{model_spec}'")
            print("Format should be 'NAME:PATH'")
            sys.exit(1)
    
    print("="*80)
    print("LOSS FUNCTION COMPARISON: Defect vs Bulk Error Analysis")
    print("="*80)
    if TQDM_AVAILABLE:
        print("(Progress bars enabled)")
    else:
        print("(Install 'tqdm' for progress bars: pip install tqdm)")
    print(f"\nModels to compare:")
    for name, path in models_dict.items():
        print(f"  - {name}: {path}")
    sys.stdout.flush()
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    print(f"Data path: {args.data_path}")
    sys.stdout.flush()  # Ensure output is visible
    
    # Use filtered loading for fixed_test_global.xyz
    if 'fixed_test_global.xyz' in args.data_path:
        print("Using filtered loading (vac*, neb*, sia* from gen <= 7)")
        sys.stdout.flush()
        from load_filtered_data import load_filtered_structures
        frames = load_filtered_structures(
            args.data_path,
            config_patterns=['vac*', 'neb*', 'sia*'],
            max_generation=7,
            max_structures=args.n_structures
        )
        print(f"✓ Loaded {len(frames)} structures")
    elif 'good_atoms_objects' in args.data_path:
        print("Loading all structures from good_atoms_objects...")
        sys.stdout.flush()
        from ase.io import read
        frames = read(args.data_path, ':')
        print(f"✓ Loaded {len(frames)} structures")
        if args.n_structures:
            frames = frames[:args.n_structures]
            print(f"Using first {len(frames)} structures")
    else:
        # Try with config_type filter
        print(f"Config type: {args.config_type}")
        sys.stdout.flush()
        try:
            frames = load_by_config(args.data_path, config=args.config_type)
            print(f"✓ Loaded {len(frames)} structures with config_type='{args.config_type}'")
        except Exception:
            from ase.io import read
            frames = read(args.data_path, ':')
            print(f"✓ Loaded {len(frames)} structures (no config_type filter)")
        
        if args.n_structures:
            frames = frames[:args.n_structures]
            print(f"Using first {len(frames)} structures")
    
    sys.stdout.flush()
    
    # Analyze each model
    all_results = []
    results_by_model = {}
    
    # Progress bar for models
    models_iter = tqdm(models_dict.items(), total=len(models_dict),
                       desc="Processing models", unit="model",
                       disable=not TQDM_AVAILABLE)
    
    for model_name, model_path in models_iter:
        if TQDM_AVAILABLE:
            models_iter.set_description(f"Processing {model_name}")
        else:
            print("\n" + "="*80)
            print(f"ANALYZING: {model_name}")
            print("="*80)
        
        try:
            # Check cache BEFORE loading model
            cached_predictions = None
            need_inference = True
            
            if args.use_cache and cache_dir is not None:
                cached_predictions = load_model_predictions(cache_dir, model_name, args.environment)
                if cached_predictions is not None:
                    if len(cached_predictions) == len(frames):
                        print(f"\n  ✓ Found valid cache for {model_name} ({len(cached_predictions)} predictions)")
                        print(f"  → Skipping model loading and inference")
                        need_inference = False
                    else:
                        print(f"\n  ⚠ Cache size mismatch ({len(cached_predictions)} vs {len(frames)})")
                        print(f"  → Will re-run inference")
                        cached_predictions = None
            
            # Only load model if we need to run inference
            predictor = None
            if need_inference:
                print(f"\n  Loading model: {model_path}")
                predictor = load_model(model_path, device=args.device)
            
            # Get per-structure results (with caching)
            # Pass cached_predictions if we already loaded it to avoid reloading
            df = analyze_model_on_dataset(
                frames, predictor, model_name,
                cache_dir=cache_dir,
                environment=args.environment,
                use_cache=args.use_cache,
                cached_predictions=cached_predictions if not need_inference else None
            )
            all_results.append(df)
            
            # Store raw results for distribution plots
            # Use cached_predictions we already loaded above, or reload if needed
            struct_results = []
            # Reuse cached_predictions if we already loaded it, otherwise load now
            if cached_predictions is None and args.use_cache and cache_dir is not None:
                cached_predictions = load_model_predictions(cache_dir, model_name, args.environment)
            
            # Progress bar for computing distribution results
            dist_iter = tqdm(enumerate(frames), total=len(frames),
                            desc=f"  Computing distributions", unit="struct",
                            disable=not TQDM_AVAILABLE or len(frames) == 0)
            
            for i, atoms in dist_iter:
                try:
                    if cached_predictions and i < len(cached_predictions):
                        # Use cached predictions
                        from detect_defects import detect_defect_auto
                        rc = _first_shell_cutoff(atoms)
                        center, top_idx, defect_type = detect_defect_auto(atoms, rc=rc)
                        center, pos_unwrapped = vacancy_center_mic(atoms, top_idx)
                        dists = np.linalg.norm(pos_unwrapped - center, axis=1)
                        defect_mask = dists < (args.near_mult * rc)
                        bulk_mask = dists > (args.far_mult * rc)
                        
                        result = compute_errors_from_cache(
                            cached_predictions, i, defect_mask, bulk_mask
                        )
                        result['defect_type'] = defect_type
                        result['rc'] = rc
                        result['dists'] = dists
                    else:
                        # Compute on-the-fly
                        result = compute_defect_bulk_errors(
                            atoms, predictor,
                            near_mult=args.near_mult,
                            far_mult=args.far_mult
                        )
                    struct_results.append(result)
                except Exception:
                    continue
            
            results_by_model[model_name] = struct_results
            
        except Exception as e:
            print(f"✗ Failed to analyze {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_results:
        print("\n✗ No models were successfully analyzed!")
        sys.exit(1)
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(output_dir / 'detailed_results.csv', index=False)
    print(f"\n✓ Saved detailed results: {output_dir / 'detailed_results.csv'}")
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    plot_error_distributions(results_by_model, output_dir)
    plot_defect_vs_bulk_comparison(combined_df, output_dir)
    create_summary_table(combined_df, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nKey files:")
    print(f"  - error_distributions.png     (per-model histograms)")
    print(f"  - defect_vs_bulk_comparison.png  (bar charts)")
    print(f"  - summary_statistics.csv      (mean ± std)")
    print(f"  - detailed_results.csv        (per-structure data)")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

