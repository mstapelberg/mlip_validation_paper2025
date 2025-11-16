#!/usr/bin/env python
"""
Load and Filter Test Data
==========================

Load structures from fixed_test_global.xyz with filtering by:
- Config type (vac*, neb*, sia*)
- Generation (<= 7)
"""

import numpy as np
from ase.io import read
from typing import List
import re


def load_filtered_structures(data_path: str, 
                             config_patterns: List[str] = None,
                             max_generation: int = 7,
                             max_structures: int = None) -> List:
    """
    Load structures with filtering.
    
    Args:
        data_path: Path to .xyz file
        config_patterns: List of glob patterns (e.g., ['vac*', 'neb*', 'sia*'])
                        If None, loads all
        max_generation: Maximum generation to include (default: 7)
        max_structures: Maximum number of structures to return
    
    Returns:
        List of ASE Atoms objects
    """
    print(f"Loading structures from: {data_path}")
    all_frames = read(data_path, ':')
    print(f"  Total structures in file: {len(all_frames)}")
    
    filtered = []
    
    for atoms in all_frames:
        # Filter by generation
        gen = atoms.info.get('generation', None)
        if gen is not None and gen > max_generation:
            continue
        
        # Filter by config_type
        if config_patterns is not None:
            config_type = atoms.info.get('config_type', '')
            matched = False
            for pattern in config_patterns:
                # Simple glob matching
                pattern_re = pattern.replace('*', '.*')
                if re.match(pattern_re, config_type):
                    matched = True
                    break
            
            if not matched:
                continue
        
        filtered.append(atoms)
    
    print(f"  After filtering:")
    print(f"    Generation <= {max_generation}: {len(filtered)} structures")
    
    if config_patterns:
        print(f"    Config patterns: {config_patterns}")
        # Show breakdown by config_type
        from collections import Counter
        config_counts = Counter([f.info.get('config_type', 'unknown') for f in filtered])
        print(f"    Config type breakdown:")
        for config, count in sorted(config_counts.items()):
            print(f"      {config:20s}: {count:4d}")
    
    if max_structures and len(filtered) > max_structures:
        print(f"  Limiting to first {max_structures} structures")
        filtered = filtered[:max_structures]
    
    return filtered


def get_defect_types_summary(frames: List) -> dict:
    """Get summary of defect types in dataset."""
    from collections import Counter
    
    config_counts = Counter([f.info.get('config_type', 'unknown') for f in frames])
    gen_counts = Counter([f.info.get('generation', 'unknown') for f in frames])
    
    return {
        'config_types': dict(config_counts),
        'generations': dict(gen_counts),
        'total': len(frames)
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test data loading and filtering')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--config_patterns', nargs='+', default=['vac*', 'neb*', 'sia*'])
    parser.add_argument('--max_generation', type=int, default=7)
    
    args = parser.parse_args()
    
    frames = load_filtered_structures(
        args.data_path,
        config_patterns=args.config_patterns,
        max_generation=args.max_generation
    )
    
    print(f"\n{'='*80}")
    print("FILTERED DATASET SUMMARY")
    print('='*80)
    summary = get_defect_types_summary(frames)
    print(f"Total: {summary['total']}")
    print(f"\nConfig types: {summary['config_types']}")
    print(f"\nGenerations: {summary['generations']}")

