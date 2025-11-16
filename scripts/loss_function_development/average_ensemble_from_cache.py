#!/usr/bin/env python
"""
Average Ensemble Predictions from Cache
========================================

Load cached predictions for multiple model seeds and average them to create
ensemble predictions. This allows re-analyzing with different metrics without
re-running inference.

Usage:
------
    python average_ensemble_from_cache.py \
      --cache_dir results/loss_comparison/cache \
      --loss_types MSE TailHuber RMSE \
      --output_dir results/ensemble_averaged
"""

import json
import numpy as np
from pathlib import Path
import argparse
import re
from typing import Dict, List, Optional


def extract_loss_type(model_name: str) -> str:
    """Extract loss type from model name (remove _seed*, _model*, etc.)."""
    base = re.sub(r'_(seed|run|v|model)\d+$', '', model_name, flags=re.IGNORECASE)
    return base


def group_models_by_loss_type(cached_models: List[str]) -> Dict[str, List[str]]:
    """Group cached model names by loss type."""
    grouped = {}
    for model_name in cached_models:
        loss_type = extract_loss_type(model_name)
        if loss_type not in grouped:
            grouped[loss_type] = []
        grouped[loss_type].append(model_name)
    return grouped


def load_cached_predictions(cache_dir: Path, model_name: str, environment: str = 'standard') -> Optional[List[Dict]]:
    """Load cached predictions for a model."""
    from cache_predictions import load_model_predictions
    return load_model_predictions(cache_dir, model_name, environment)


def average_ensemble_predictions(predictions_list: List[List[Dict]]) -> List[Dict]:
    """
    Average predictions across multiple model seeds.
    
    Args:
        predictions_list: List of prediction lists (one per seed)
    
    Returns:
        Averaged predictions (one per frame)
    """
    n_seeds = len(predictions_list)
    n_frames = len(predictions_list[0])
    
    averaged = []
    
    for frame_idx in range(n_frames):
        # Collect predictions for this frame across all seeds
        frame_preds = [preds[frame_idx] for preds in predictions_list]
        
        # Average energy
        energies = [p['prediction']['energy'] for p in frame_preds]
        avg_energy = np.mean(energies)
        
        # Average forces (N, 3)
        forces_list = [np.array(p['prediction']['forces']) for p in frame_preds]
        avg_forces = np.mean(forces_list, axis=0)
        
        # Average stress if available
        avg_stress = None
        stresses = [p['prediction']['stress'] for p in frame_preds if p['prediction']['stress'] is not None]
        if stresses:
            stresses_array = [np.array(s) for s in stresses]
            avg_stress = np.mean(stresses_array, axis=0).tolist()
        
        # Use metadata from first seed (should be same across seeds)
        averaged_frame = {
            'structure_idx': frame_preds[0]['structure_idx'],
            'formula': frame_preds[0]['formula'],
            'config_type': frame_preds[0]['config_type'],
            'generation': frame_preds[0]['generation'],
            'n_atoms': frame_preds[0]['n_atoms'],
            'reference': frame_preds[0]['reference'],  # Same across seeds
            'prediction': {
                'energy': float(avg_energy),
                'forces': avg_forces.tolist(),
                'stress': avg_stress
            },
            'n_seeds': n_seeds
        }
        
        averaged.append(averaged_frame)
    
    return averaged


def save_averaged_predictions(output_dir: Path, loss_type: str, averaged_predictions: List[Dict]):
    """Save averaged ensemble predictions."""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_file = output_dir / f"{loss_type}_ensemble.json"
    
    with open(output_file, 'w') as f:
        json.dump(averaged_predictions, f, indent=2)
    
    print(f"  ✓ Saved averaged predictions: {output_file} ({len(averaged_predictions)} frames)")


def main():
    parser = argparse.ArgumentParser(
        description='Average ensemble predictions from cache',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--cache_dir', type=str, required=True,
                       help='Cache directory containing model predictions')
    
    parser.add_argument('--loss_types', nargs='+', type=str,
                       help='Loss types to average (e.g., MSE TailHuber). If not provided, averages all found.')
    
    parser.add_argument('--environment', type=str, default='standard',
                       choices=['standard', 'custom'],
                       help='Environment type (standard or custom)')
    
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for averaged predictions')
    
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        print(f"Error: Cache directory does not exist: {cache_dir}")
        return
    
    output_dir = Path(args.output_dir)
    
    # Get all cached models
    from cache_predictions import get_cached_model_names
    cached_models = get_cached_model_names(cache_dir, args.environment)
    
    if not cached_models:
        print(f"No cached models found in {cache_dir} for environment '{args.environment}'")
        return
    
    print(f"Found {len(cached_models)} cached models")
    
    # Group by loss type
    grouped = group_models_by_loss_type(cached_models)
    
    print(f"\nGrouped into {len(grouped)} loss types:")
    for loss_type, models in grouped.items():
        print(f"  {loss_type}: {len(models)} seeds ({', '.join(models)})")
    
    # Determine which loss types to process
    loss_types_to_process = args.loss_types if args.loss_types else list(grouped.keys())
    
    print(f"\nAveraging predictions for: {loss_types_to_process}")
    
    for loss_type in loss_types_to_process:
        if loss_type not in grouped:
            print(f"  ⚠ Loss type '{loss_type}' not found in cache, skipping")
            continue
        
        model_names = grouped[loss_type]
        print(f"\n  Processing {loss_type} ({len(model_names)} seeds)...")
        
        # Load predictions for all seeds
        predictions_list = []
        for model_name in model_names:
            preds = load_cached_predictions(cache_dir, model_name, args.environment)
            if preds is None:
                print(f"    ⚠ Failed to load {model_name}, skipping")
                continue
            predictions_list.append(preds)
        
        if not predictions_list:
            print(f"    ✗ No valid predictions found for {loss_type}")
            continue
        
        # Verify all have same number of frames
        n_frames = len(predictions_list[0])
        if not all(len(p) == n_frames for p in predictions_list):
            print(f"    ⚠ Frame count mismatch, using minimum")
            n_frames = min(len(p) for p in predictions_list)
            predictions_list = [p[:n_frames] for p in predictions_list]
        
        # Average predictions
        averaged = average_ensemble_predictions(predictions_list)
        
        # Save averaged predictions
        save_averaged_predictions(output_dir, loss_type, averaged)
    
    print(f"\n{'='*80}")
    print("ENSEMBLE AVERAGING COMPLETE!")
    print('='*80)
    print(f"\nAveraged predictions saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()

