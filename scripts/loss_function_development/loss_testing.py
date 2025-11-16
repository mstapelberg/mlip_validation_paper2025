import numpy as np
from typing import Callable, Dict, List, Any, Optional, Tuple
from ase.io import read
from ase.neighborlist import NeighborList
import torch

# --------------------------
# Data loading and vacancy localization
# --------------------------

def load_by_config(path, config='vacancy-alloy'):
    frames = read(path, ':')  # read all frames
    return [a for a in frames if a.info.get('config_type') == config]

def _first_shell_cutoff(atoms, bins=200, use_simple=True):
    """Find first shell cutoff.
    
    For small/defective structures, histogram-based detection often fails because
    most distances are at "medium" range (half box size) rather than first neighbors.
    
    Args:
        use_simple: If True, use simple percentile-based approach (robust for small systems)
    """
    D = atoms.get_all_distances(mic=True)
    iu = np.triu_indices(len(atoms), 1)
    d = D[iu]
    d = d[d > 1e-8]
    
    if use_simple or len(atoms) < 200:
        # Simple approach: first shell is around 5th-10th percentile of distances
        # For vacancy structures, this captures the nearest-neighbor peak
        rc = float(np.percentile(d, 5) * 1.20)  # 5th percentile + 20% buffer
        # Clamp to reasonable range for metals (2.5-3.5 Å typically)
        rc = np.clip(rc, 2.5, 3.5)
        return rc
    
    # Full histogram approach (for large structures)
    hist, edges = np.histogram(d, bins=bins)
    k_peak = hist.argmax()
    
    # Find minimum after first peak
    search_window = min(bins//10, 20)
    k_min = k_peak + 1 + np.argmin(hist[k_peak+1 : k_peak + search_window])
    rmin = 0.5*(edges[k_min] + edges[k_min+1])
    rc = float(1.05*rmin)
    
    # Sanity check
    if rc > 4.0 or rc < 2.0:
        # Fall back to simple method
        rc = float(np.percentile(d, 7.5) * 1.15)
        rc = np.clip(rc, 2.5, 3.5)
    
    return rc

def _neighbor_stats(atoms, rc):
    n = len(atoms)
    radii = np.full(n, rc/2.0)
    nl = NeighborList(radii, self_interaction=False, bothways=True, skin=0.0)
    nl.update(atoms)
    z = np.zeros(n, dtype=int)
    dbar = np.zeros(n)
    for i in range(n):
        idxs, _ = nl.get_neighbors(i)
        z[i] = len(idxs)
        if z[i] > 0:
            d = atoms.get_distances(i, idxs, mic=True)
            dbar[i] = d.mean()
    vals, counts = np.unique(z, return_counts=True)
    z_mode = int(vals[counts.argmax()])
    return z, dbar, z_mode

def locate_vacancy(atoms, rc=None, k_frac=0.05, w_dist=0.5):
    """Heuristic: under-coordination + enlarged NN spacing."""
    if rc is None:
        rc = _first_shell_cutoff(atoms)
    z, dbar, z_mode = _neighbor_stats(atoms, rc)
    deficit = np.maximum(0, z_mode - z).astype(float)
    dbar_z = (dbar - np.median(dbar)) / (np.std(dbar) + 1e-12)
    score = deficit + w_dist*dbar_z
    k = max(6, int(k_frac*len(atoms)))
    top_idx = np.argsort(score)[-k:]
    # Naive center (may be off near PBC):
    center = atoms.positions[top_idx].mean(axis=0)
    return center, top_idx, score  # vacancy center (approx), shell indices, per-atom scores

# --- PBC-safe helpers ---

def unwrap_positions(atoms, anchor: int) -> np.ndarray:
    """Unwrap positions around an anchor atom using minimum-image vectors."""
    vecs = atoms.get_distances(anchor, np.arange(len(atoms)), mic=True, vector=True)
    return atoms.positions[anchor] + vecs  # (N,3)

def vacancy_center_mic(atoms, top_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a PBC-consistent vacancy center from top_idx cluster."""
    pos_unwrapped = unwrap_positions(atoms, anchor=int(top_idx[0]))
    center_unwrapped = pos_unwrapped[top_idx].mean(axis=0)
    return center_unwrapped, pos_unwrapped

def label_vacancy_neighbors(dists_to_center: np.ndarray, rc: float, shell_mult: float = 1.2) -> np.ndarray:
    """Boolean labels: atoms within shell_mult*rc of the vacancy center."""
    return dists_to_center <= (shell_mult * rc)

# --------------------------
# Reference/prediction utilities
# --------------------------

def get_reference_forces(atoms, keys=('REF_force', 'forces', 'ref_forces', 'dft_forces', 'F')):
    """Pull ground-truth forces from common array keys."""
    for k in keys:
        if k in atoms.arrays:
            F = np.asarray(atoms.arrays[k])
            if F.ndim == 2 and F.shape[1] == 3:
                return F
    raise KeyError(
        f"No reference forces found in atoms.arrays under {keys}. "
        "Store DFT forces there, or pass a calculator to atoms and call atoms.get_forces()."
    )

class ASECalculatorPredictor:
    """Wrap an ASE calculator as a callable that returns predicted forces."""
    def __init__(self, calculator):
        self.calculator = calculator
    def __call__(self, atoms) -> np.ndarray:
        a = atoms.copy()
        a.calc = self.calculator
        return a.get_forces()

# --------------------------
# Per-atom error fields (your implementation, lightly wrapped)
# --------------------------

def per_atom_error_fields(F_pred, F_true, quantile=0.9, delta='auto', min_tail=8):
    as_torch = isinstance(F_pred, torch.Tensor)
    if not as_torch:
        Fp, Ft = np.asarray(F_pred), np.asarray(F_true)
        err = Fp - Ft
        r = np.linalg.norm(err, axis=-1)
        s_mae, s_mse, s_rmse, s_l3, s_l4 = r, r**2, r, r**3, r**4
        if r.size == 0:
            return dict(fields=dict(MAE=r, MSE=r, RMSE=r, L3=r, L4=r, Huber=r, TailHuber=r),
                        global_metrics={}, tail_mask=np.zeros_like(r, dtype=bool))
        q = np.quantile(r.astype(np.float32), quantile)
        k = max(min_tail, int(np.ceil((1-quantile)*r.size)))
        topk_thresh = np.partition(r, -k)[-k] if k>0 else q
        mask = r >= min(q, topk_thresh)
        med = np.median(r[mask]) if mask.any() else np.median(r)
        d = float(np.clip(1.5*med, 0.1, 5.0)) if delta=='auto' else float(delta)
        # Pure Huber (all atoms)
        s_huber_all = d**2*(np.sqrt(1+(r/d)**2)-1.0)
        # TailHuber (only tail atoms)
        s_tail = np.where(mask, s_huber_all, 0.0)
        globals_ = dict(MAE=s_mae.mean(), MSE=s_mse.mean(),
                        RMSE=np.sqrt(s_mse.mean()),
                        L3=(s_l3.mean())**(1/3), L4=(s_l4.mean())**(1/4),
                        Huber=s_huber_all.mean(),
                        TailHuber=s_tail.mean())
        return dict(fields=dict(MAE=s_mae, MSE=s_mse, RMSE=s_rmse, L3=s_l3, L4=s_l4, Huber=s_huber_all, TailHuber=s_tail),
                    global_metrics=globals_, tail_mask=mask)
    else:
        err = F_pred - F_true
        r = torch.linalg.norm(err, dim=-1)         # (N,)
        s_mae, s_mse, s_rmse, s_l3, s_l4 = r, r**2, r, r**3, r**4
        if r.numel() == 0:
            empty = dict(fields=dict(MAE=r, MSE=r, RMSE=r, L3=r, L4=r, Huber=r, TailHuber=r),
                         global_metrics={}, tail_mask=torch.zeros_like(r, dtype=torch.bool))
            return empty
        q = torch.quantile(r.float(), quantile).to(r.device)
        k = max(min_tail, int((1-quantile)*r.numel()))
        if k>0:
            topk_val, _ = torch.topk(r, k)
            thr = min(q, topk_val.min())
        else:
            thr = q
        mask = r >= thr
        if delta == 'auto':
            med = torch.median(r[mask]) if mask.any() else torch.median(r)
            d = torch.clamp(1.5*med, 0.1, 5.0)
        else:
            d = torch.tensor(float(delta), device=r.device)
        # Pure Huber (all atoms)
        s_huber_all = d**2*(torch.sqrt(1+(r/d)**2)-1.0)
        # TailHuber (only tail atoms)
        s_tail = torch.where(mask, s_huber_all, torch.zeros_like(r))
        globals_ = dict(MAE=s_mae.mean(), MSE=s_mse.mean(),
                        RMSE=torch.sqrt(s_mse.mean()),
                        L3=(s_l3.mean())**(1/3), L4=(s_l4.mean())**(1/4),
                        Huber=s_huber_all.mean(),
                        TailHuber=s_tail.mean())
        return dict(fields=dict(MAE=s_mae, MSE=s_mse, RMSE=s_rmse, L3=s_l3, L4=s_l4, Huber=s_huber_all, TailHuber=s_tail),
                    global_metrics=globals_, tail_mask=mask)

# --------------------------
# Sensitivity metrics
# --------------------------

def precision_recall_ap(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute precision, recall, and Average Precision (AP) without sklearn."""
    y = y_true.astype(np.int64)
    order = np.argsort(scores)[::-1]
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    denom = tp + fp + 1e-12
    precision = tp / denom
    P = y.sum()
    if P == 0:
        return precision, np.zeros_like(precision), np.nan
    recall = tp / P
    # AP = sum over recall increments of precision at that threshold (rectangle rule)
    dR = recall - np.r_[0.0, recall[:-1]]
    ap = np.sum(precision * dR)
    return precision, recall, float(ap)

def lift_at_frac(y_true: np.ndarray, scores: np.ndarray, frac: float = 0.10) -> float:
    k = max(1, int(np.ceil(frac * len(scores))))
    idx = np.argsort(scores)[::-1][:k]
    hits = y_true[idx].sum()
    expected = y_true.mean() * k + 1e-12
    return float(hits / expected)

def radial_contrast(scores: np.ndarray, dists: np.ndarray, rc: float,
                    near_mult: float = 1.2, far_mult: float = 3.0,
                    exclude_zeros: bool = False) -> float:
    """Compute radial contrast (near/far ratio).
    
    Args:
        exclude_zeros: If True, ignore zero-valued scores (useful for TailHuber)
    """
    near = dists <= near_mult * rc
    far = dists >= far_mult * rc
    
    if exclude_zeros:
        # Only consider non-zero scores (for tail-based metrics)
        scores_near = scores[near]
        scores_far = scores[far]
        scores_near = scores_near[scores_near > 1e-10]
        scores_far = scores_far[scores_far > 1e-10]
        
        if len(scores_near) < 3 or len(scores_far) < 3:
            return np.nan
        
        return float(scores_near.mean() / (scores_far.mean() + 1e-12))
    
    if not near.any() or not far.any():
        return np.nan
    
    mean_far = scores[far].mean()
    if mean_far < 1e-9:  # Avoid division by near-zero
        return np.nan
    
    return float(scores[near].mean() / mean_far)

# --------------------------
# Per-structure analysis
# --------------------------

def analyze_structure(atoms,
                      predict_forces: Callable[[Any], np.ndarray],
                      quantile: float = 0.75,
                      delta: float = 1.0,
                      min_tail: int = 8,
                      shell_mult: float = 1.2,
                      bulk_mult: float = 3.0,
                      verbose: bool = False) -> Dict[str, Any]:
    """Compute per-atom fields and sensitivity metrics for one structure."""
    rc = _first_shell_cutoff(atoms)
    center0, top_idx, _ = locate_vacancy(atoms, rc=rc)
    center, pos_unwrapped = vacancy_center_mic(atoms, top_idx)
    dists = np.linalg.norm(pos_unwrapped - center, axis=1)
    y_pos = label_vacancy_neighbors(dists, rc, shell_mult=shell_mult).astype(np.int32)

    if verbose:
        print(f"  DEBUG: rc = {rc:.3f} Å")
        print(f"  DEBUG: Found {y_pos.sum()} vacancy neighbors out of {len(atoms)} atoms")
        print(f"  DEBUG: Distance range: {dists.min():.3f} - {dists.max():.3f} Å")
        print(f"  DEBUG: Vacancy shell cutoff: {shell_mult * rc:.3f} Å")
        print(f"  DEBUG: Bulk region starts at: {bulk_mult * rc:.3f} Å")
        print(f"  DEBUG: Atoms in bulk region: {(dists > bulk_mult * rc).sum()}")

    F_true = get_reference_forces(atoms)
    F_pred = predict_forces(atoms)
    
    if verbose:
        errors = np.linalg.norm(F_pred - F_true, axis=1)
        print(f"  DEBUG: Force error range: {errors.min():.4f} - {errors.max():.4f} eV/Å")
        print(f"  DEBUG: Force error (mean): {errors.mean():.4f} eV/Å")
        print(f"  DEBUG: Force error near vacancy: {errors[y_pos.astype(bool)].mean():.4f} eV/Å")
        if (dists > bulk_mult * rc).sum() > 0:
            print(f"  DEBUG: Force error in bulk: {errors[dists > bulk_mult * rc].mean():.4f} eV/Å")

    res = per_atom_error_fields(F_pred, F_true, quantile=quantile, delta=delta, min_tail=min_tail)
    fields = res['fields']  # dict of arrays: MAE, MSE, L3, L4, TailHuber

    metrics = {}
    for name, s in fields.items():
        s_np = s.detach().cpu().numpy() if isinstance(s, torch.Tensor) else s
        _, _, ap = precision_recall_ap(y_pos, s_np)
        lift10 = lift_at_frac(y_pos, s_np, frac=0.10)
        # For TailHuber, exclude zeros when computing contrast
        exclude_zeros = (name == 'TailHuber')
        contrast = radial_contrast(s_np, dists, rc, near_mult=shell_mult, far_mult=bulk_mult,
                                   exclude_zeros=exclude_zeros)
        metrics[name] = dict(AP=ap, Lift10=lift10, RadialContrast=contrast)

    return dict(
        rc=rc,
        center=center,
        dists=dists,
        labels=y_pos,
        global_metrics=res['global_metrics'],
        per_field_metrics=metrics,
        tail_mask=res['tail_mask']
    )

# --------------------------
# Dataset analysis and summary
# --------------------------

def analyze_dataset(frames: List[Any],
                    predict_forces: Callable[[Any], np.ndarray],
                    quantile: float = 0.75,
                    delta: float = 1.0,
                    min_tail: int = 8,
                    shell_mult: float = 1.2,
                    bulk_mult: float = 3.0) -> List[Dict[str, Any]]:
    results = []
    for atoms in frames:
        results.append(
            analyze_structure(atoms, predict_forces,
                              quantile=quantile, delta=delta, min_tail=min_tail,
                              shell_mult=shell_mult, bulk_mult=bulk_mult)
        )
    return results

def summarize_results(per_structure: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Mean metrics across structures for each field."""
    fields = ['MAE', 'MSE', 'RMSE', 'L3', 'L4', 'Huber', 'TailHuber']
    out: Dict[str, Dict[str, float]] = {}
    for f in fields:
        ap, lift10, contrast = [], [], []
        for r in per_structure:
            m = r['per_field_metrics'].get(f, {})
            if np.isfinite(m.get('AP', np.nan)):
                ap.append(m['AP'])
            if np.isfinite(m.get('Lift10', np.nan)):
                lift10.append(m['Lift10'])
            if np.isfinite(m.get('RadialContrast', np.nan)):
                contrast.append(m['RadialContrast'])
        out[f] = dict(
            AP=np.nanmean(ap) if ap else np.nan,
            Lift10=np.nanmean(lift10) if lift10 else np.nan,
            RadialContrast=np.nanmean(contrast) if contrast else np.nan
        )
    return out

# --------------------------
# Example usage (fill in your path and predictor)
# --------------------------

if __name__ == "__main__":
    # 1) Load frames
    # frames = load_by_config("your_dataset.extxyz", config="vacancy-alloy")

    # 2) Define a predictor that returns predicted forces for an Atoms object.
    #    Option A: any ASE calculator
    # from your_model_pkg import build_calculator
    # calc = build_calculator("path/to/checkpoint")
    # predict = ASECalculatorPredictor(calc)

    #    Option B: your own callable
    # def predict(atoms): 
    #     return your_model_forces(atoms)  # (N,3) numpy array

    # 3) Run analysis (set your TailHuber params)
    # results = analyze_dataset(frames, predict, quantile=0.75, delta=1.0, min_tail=8)

    # 4) Summarize
    # summary = summarize_results(results)
    # print(summary)
    pass
