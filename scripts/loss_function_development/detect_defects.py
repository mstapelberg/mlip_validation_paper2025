#!/usr/bin/env python
"""
Unified Defect Detection
=========================

Detect vacancies and interstitials in structures.

- Vacancies: Under-coordinated atoms
- Interstitials (SIA): Over-coordinated atoms
"""

import numpy as np
from ase import Atoms
from typing import Tuple

# Import from loss_testing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from loss_testing import _first_shell_cutoff, _neighbor_stats, vacancy_center_mic, locate_vacancy


def detect_vacancy(atoms: Atoms, rc: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect vacancy in structure.
    
    Returns:
        center: vacancy center position (PBC-aware)
        top_idx: indices of atoms near vacancy
        score: per-atom vacancy scores
    """
    # Use the existing locate_vacancy from loss_testing.py
    center, top_idx, score = locate_vacancy(atoms, rc=rc)
    return center, top_idx, score


def detect_interstitial(atoms: Atoms, rc: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect interstitial (SIA) in structure.
    
    Looks for over-coordinated atoms (opposite of vacancy).
    
    Returns:
        center: interstitial center position
        top_idx: indices of atoms near interstitial
        score: per-atom interstitial scores
    """
    if rc is None:
        rc = _first_shell_cutoff(atoms)
    
    z, dbar, z_mode = _neighbor_stats(atoms, rc)
    
    # Over-coordination score (opposite of vacancy)
    excess = np.maximum(0, z - z_mode).astype(float)
    
    # Atoms with high coordination are candidates
    score = excess.astype(float)
    
    k = max(6, int(0.05 * len(atoms)))
    top_idx = np.argsort(score)[-k:]
    
    # Get PBC-aware center
    center, pos_unwrapped = vacancy_center_mic(atoms, top_idx)
    
    return center, top_idx, score


def detect_defect_auto(atoms: Atoms, config_type: str = None, rc: float = None) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Automatically detect defect type based on config_type.
    
    Args:
        atoms: Structure
        config_type: Config type string (if None, uses atoms.info['config_type'])
        rc: First shell cutoff
    
    Returns:
        center: defect center position
        top_idx: indices of atoms near defect
        defect_type: 'vacancy' or 'interstitial'
    """
    if config_type is None:
        config_type = atoms.info.get('config_type', '')
    
    # Determine defect type from config
    if 'sia' in config_type.lower():
        center, top_idx, _ = detect_interstitial(atoms, rc=rc)
        return center, top_idx, 'interstitial'
    else:
        # Default: vacancy (includes vac*, neb*, and others)
        center, top_idx, _ = detect_vacancy(atoms, rc=rc)
        return center, top_idx, 'vacancy'


if __name__ == '__main__':
    # Test the detection
    from ase.io import read
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python detect_defects.py <data.xyz> <structure_index>")
        sys.exit(1)
    
    frames = read(sys.argv[1], ':')
    idx = int(sys.argv[2])
    
    atoms = frames[idx]
    config_type = atoms.info.get('config_type', 'unknown')
    
    print(f"Structure {idx}: {atoms.get_chemical_formula()}")
    print(f"Config type: {config_type}")
    
    center, top_idx, defect_type = detect_defect_auto(atoms)
    
    print(f"\nDetected: {defect_type}")
    print(f"Center: {center}")
    print(f"Affected atoms: {len(top_idx)}")

