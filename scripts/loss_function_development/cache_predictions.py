#!/usr/bin/env python
"""
Prediction Cache System
=======================

Cache model predictions (energy, forces, stress) to avoid re-running expensive inference
when tweaking analysis/plotting code.

Cache Structure:
----------------
cache/
  MSE_model0.json      # Individual model cache files
  MSE_model1.json
  RMCE_model0.json
  TailHuber_model0.json
  ...

Each JSON file contains a list of predictions:
  [
    {
      "formula": "Al64",
      "config_type": "vacancy-alloy",
      "generation": 7,
      "n_atoms": 64,
      "info": {...},  # Full atoms.info dict (using monty for serialization)
      "reference": {
        "energy": 1234.56,
        "forces": [[...], [...]],  # (N, 3)
        "stress": [[...], [...]]   # (3, 3) or (6,)
      },
      "prediction": {
        "energy": 1234.50,
        "forces": [[...], [...]],
        "stress": [[...], [...]]
      }
    },
    ...
  ]

This structure reduces file size and makes it easier to manage individual model caches.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import os

try:
    from monty.json import MontyEncoder, MontyDecoder
    MONTY_AVAILABLE = True
except ImportError:
    MONTY_AVAILABLE = False
    print("Warning: monty not available, falling back to basic JSON serialization")


def get_cache_dir(output_dir: Path) -> Path:
    """Get cache directory path."""
    cache_dir = output_dir / 'cache'
    cache_dir.mkdir(exist_ok=True, parents=True)
    return cache_dir


def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize model name for use as filename.
    
    Args:
        model_name: Model identifier (e.g., "MSE_model0", "RMCE_model1")
    
    Returns:
        Sanitized filename-safe string
    """
    # Replace any potentially problematic characters
    sanitized = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    return sanitized


def get_cache_file(cache_dir: Path, model_name: str, environment: str = 'standard') -> Path:
    """
    Get cache file path for a specific model.
    
    Args:
        cache_dir: Cache directory
        model_name: Model identifier (e.g., "MSE_model0")
        environment: 'standard' or 'custom' (for backward compatibility, not used in new format)
    
    Returns:
        Path to individual model cache file
    """
    sanitized_name = sanitize_model_name(model_name)
    return cache_dir / f"{sanitized_name}.json"


def save_model_predictions(cache_dir: Path, model_name: str, predictions: List[Dict],
                          environment: str = 'standard'):
    """
    Save predictions for a model to an individual cache file.
    
    Args:
        cache_dir: Cache directory
        model_name: Model identifier (e.g., "MSE_model0")
        predictions: List of prediction dicts (one per frame)
        environment: 'standard' or 'custom' (for backward compatibility, not used in new format)
    """
    cache_file = get_cache_file(cache_dir, model_name, environment)
    
    # Save using monty if available
    if MONTY_AVAILABLE:
        from monty.json import jsanitize
        sanitized = jsanitize(predictions)
        with open(cache_file, 'w') as f:
            import json
            json.dump(sanitized, f, indent=2, cls=MontyEncoder)
    else:
        # Fallback: manual conversion
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            return obj
        
        sanitized = [convert_to_json_serializable(p) for p in predictions]
        import json
        with open(cache_file, 'w') as f:
            json.dump(sanitized, f, indent=2)
    
    file_size_mb = cache_file.stat().st_size / (1024 * 1024)
    print(f"  ✓ Cached {len(predictions)} predictions for {model_name} in {cache_file.name} ({file_size_mb:.2f} MB)")


def load_model_predictions(cache_dir: Path, model_name: str,
                          environment: str = 'standard') -> Optional[List[Dict]]:
    """
    Load cached predictions for a model from individual cache file.
    
    Returns:
        List of prediction dicts or None if not found
    """
    cache_file = get_cache_file(cache_dir, model_name, environment)
    if not cache_file.exists():
        # Try loading from old combined format for backward compatibility
        return _load_from_old_format(cache_dir, model_name, environment)
    
    import json
    
    if MONTY_AVAILABLE:
        with open(cache_file, 'r') as f:
            predictions = json.load(f, cls=MontyDecoder)
    else:
        with open(cache_file, 'r') as f:
            predictions = json.load(f)
        
        # Convert lists back to numpy arrays
        def convert_from_json(obj):
            if isinstance(obj, dict):
                return {k: convert_from_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                arr = np.array(obj)
                if arr.ndim > 1:
                    return arr
                return arr
            return obj
        
        predictions = [convert_from_json(p) for p in predictions]
    
    return predictions


def _load_from_old_format(cache_dir: Path, model_name: str, environment: str) -> Optional[List[Dict]]:
    """
    Load from old combined cache format for backward compatibility.
    
    Returns:
        List of prediction dicts or None if not found
    """
    # Check for old format files
    if environment == 'custom':
        old_cache_file = cache_dir / 'custom_models.json'
    else:
        old_cache_file = cache_dir / 'standard_models.json'
    
    if not old_cache_file.exists():
        return None
    
    import json
    
    if MONTY_AVAILABLE:
        with open(old_cache_file, 'r') as f:
            all_predictions = json.load(f, cls=MontyDecoder)
    else:
        with open(old_cache_file, 'r') as f:
            all_predictions = json.load(f)
        
        # Convert lists back to numpy arrays
        def convert_from_json(obj):
            if isinstance(obj, dict):
                return {k: convert_from_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                arr = np.array(obj)
                if arr.ndim > 1:
                    return arr
                return arr
            return obj
        
        all_predictions = {k: [convert_from_json(p) for p in v] 
                          for k, v in all_predictions.items()}
    
    return all_predictions.get(model_name)


def load_all_predictions(cache_dir: Path, environment: str = 'standard') -> Dict[str, List[Dict]]:
    """
    Load all cached predictions for an environment.
    Scans for individual model cache files and also checks old format for backward compatibility.
    
    Returns:
        Dict mapping model_name -> list of predictions
    """
    all_predictions = {}
    
    # Load from new individual files format
    if cache_dir.exists():
        for cache_file in cache_dir.glob('*.json'):
            # Skip old format files
            if cache_file.name in ['standard_models.json', 'custom_models.json']:
                continue
            
            # Extract model name from filename (remove .json extension)
            model_name = cache_file.stem
            predictions = load_model_predictions(cache_dir, model_name, environment)
            if predictions is not None:
                all_predictions[model_name] = predictions
    
    # Also check old format for backward compatibility
    if environment == 'custom':
        old_cache_file = cache_dir / 'custom_models.json'
    else:
        old_cache_file = cache_dir / 'standard_models.json'
    
    if old_cache_file.exists():
        import json
        if MONTY_AVAILABLE:
            with open(old_cache_file, 'r') as f:
                old_data = json.load(f, cls=MontyDecoder)
        else:
            with open(old_cache_file, 'r') as f:
                old_data = json.load(f)
            
            # Convert lists back to numpy arrays
            def convert_from_json(obj):
                if isinstance(obj, dict):
                    return {k: convert_from_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    arr = np.array(obj)
                    if arr.ndim > 1:
                        return arr
                    return arr
                return obj
            
            old_data = {k: [convert_from_json(p) for p in v] for k, v in old_data.items()}
            
            # Merge old data (don't overwrite if new format exists)
            for model_name, predictions in old_data.items():
                if model_name not in all_predictions:
                    all_predictions[model_name] = predictions
    
    return all_predictions


def get_cached_model_names(cache_dir: Path, environment: str = 'standard') -> List[str]:
    """Get list of model names that have cached predictions."""
    all_predictions = load_all_predictions(cache_dir, environment)
    return sorted(all_predictions.keys())


def compute_predictions_for_frame(atoms, predictor) -> Dict[str, Any]:
    """
    Compute predictions for a single frame.
    
    Returns dict with:
        - structure_idx: index in dataset
        - formula: chemical formula
        - config_type: config type from atoms.info
        - generation: generation from atoms.info
        - n_atoms: number of atoms
        - info: full atoms.info dict (using monty for serialization)
        - reference: dict with energy, forces, stress
        - prediction: dict with energy, forces, stress
    """
    from loss_testing import get_reference_forces
    
    # Get reference values
    F_ref = get_reference_forces(atoms)
    
    # Try to get reference energy
    E_ref = None
    for key in ['REF_energy', 'energy', 'ref_energy', 'dft_energy', 'E']:
        if key in atoms.info:
            E_ref = atoms.info[key]
            break
    
    # Try to get reference stress
    stress_ref = None
    for key in ['REF_stress', 'stress', 'ref_stress', 'dft_stress']:
        if key in atoms.arrays or key in atoms.info:
            stress_data = atoms.arrays.get(key) or atoms.info.get(key)
            if stress_data is not None:
                stress_ref = np.array(stress_data)
                break
    
    # Get predictions
    atoms_copy = atoms.copy()
    atoms_copy.calc = predictor.calculator
    E_pred = atoms_copy.get_potential_energy()
    F_pred = atoms_copy.get_forces()
    
    # Try to get stress prediction
    stress_pred = None
    try:
        stress_pred = atoms_copy.get_stress()
    except:
        pass
    
    # Store full info dict (monty will handle numpy arrays, etc.)
    info_dict = dict(atoms.info)
    
    return {
        'formula': atoms.get_chemical_formula(),
        'config_type': atoms.info.get('config_type', 'unknown'),
        'generation': atoms.info.get('generation', None),
        'n_atoms': len(atoms),
        'info': info_dict,  # Full info dict
        'reference': {
            'energy': float(E_ref) if E_ref is not None else None,
            'forces': F_ref,
            'stress': stress_ref if stress_ref is not None else None
        },
        'prediction': {
            'energy': float(E_pred),
            'forces': F_pred,
            'stress': stress_pred if stress_pred is not None else None
        }
    }


def compute_errors_from_cache(cached_predictions: List[Dict], 
                              structure_idx: int,
                              defect_mask: np.ndarray,
                              bulk_mask: np.ndarray) -> Dict:
    """
    Compute error metrics from cached predictions.
    
    Args:
        cached_predictions: List of cached prediction dicts
        structure_idx: Index of structure to analyze
        defect_mask: Boolean mask for defect atoms
        bulk_mask: Boolean mask for bulk atoms
    
    Returns:
        Dict with error metrics
    """
    pred_data = cached_predictions[structure_idx]
    
    F_ref = np.array(pred_data['reference']['forces'])
    F_pred = np.array(pred_data['prediction']['forces'])
    
    # Ensure arrays are 2D (N, 3)
    if F_ref.ndim == 1:
        F_ref = F_ref.reshape(-1, 3)
    if F_pred.ndim == 1:
        F_pred = F_pred.reshape(-1, 3)
    
    # Compute per-atom force errors
    errors = np.linalg.norm(F_pred - F_ref, axis=1)
    
    defect_errors = errors[defect_mask] if defect_mask.any() else np.array([])
    bulk_errors = errors[bulk_mask] if bulk_mask.any() else np.array([])
    
    return {
        'all_errors': errors,
        'defect_errors': defect_errors,
        'bulk_errors': bulk_errors,
        'defect_mask': defect_mask,
        'bulk_mask': bulk_mask,
        'config_type': pred_data['config_type'],
        'n_defect': defect_mask.sum(),
        'n_bulk': bulk_mask.sum(),
        'n_total': len(errors)
    }


if __name__ == '__main__':
    # Test the cache system
    import argparse
    
    parser = argparse.ArgumentParser(description='Test cache system')
    parser.add_argument('--cache_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--environment', type=str, default='standard',
                       choices=['standard', 'custom'])
    
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    cached = load_model_predictions(cache_dir, args.model_name, args.environment)
    
    if cached:
        print(f"Found {len(cached)} cached predictions for {args.model_name}")
        print(f"First frame: {cached[0]['formula']}")
        cache_file = get_cache_file(cache_dir, args.model_name, args.environment)
        if cache_file.exists():
            file_size_mb = cache_file.stat().st_size / (1024 * 1024)
            print(f"Cache file size: {file_size_mb:.2f} MB")
    else:
        print(f"No cache found for {args.model_name}")
