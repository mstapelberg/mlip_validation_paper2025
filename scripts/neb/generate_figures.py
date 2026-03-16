#!/usr/bin/env python3
"""
Map + NEB (Gen0 vs Gen10) figure — parity spans both columns; smart manifold zoom.

Left: composition map (PCA on ILR/CLR/raw) with novelty numbers and feasible polytope overlay.
Right: stacked NEB trajectories per SIMPLE composition with VASP (dashed), MLIP Gen0, MLIP Gen10.
Bottom (optional): parity panel spanning BOTH columns.

Usage:
  python generate_figures.py \
      --summary summary_barriers.csv \
      --transform ilr \
      --include_parity \
      --zoom manifold --zoom_buffer 5 \
      --save map_plus_neb.png
"""

import ast, csv, argparse, sys, numpy as np, matplotlib.pyplot as plt
import os, json
from pathlib import Path
from matplotlib import gridspec
from matplotlib.patches import Rectangle, ConnectionPatch, Polygon
from typing import Dict, List, Tuple
from sklearn.decomposition import PCA

# Shared publication style
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plotting_utils import set_pub_style, save_fig, TOL_BRIGHT, COLORS as _COLORS, DOUBLE_COL, fig_size

set_pub_style(base_fontsize=10)

# Try to import adjustText for smart label positioning
try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False

# ---- novelty analyzer (your API). Falls back to local 'composition.py' if needed. ----
from forge.analysis.composition import CompositionAnalyzer, analyze_composition_distribution

# ---------------------- 5-element utilities ----------------------
TARGET_ELEMENTS = ['V','Cr','Ti','W','Zr']
PSEUDOCOUNT = 1e-9

# ---------------------- Color scheme (from shared palette) ----------------------
COLOR_BLUE = _COLORS["blue"]
COLOR_ORANGE = _COLORS["red"]      # coral/red replaces old brown-orange
COLOR_RED = _COLORS["purple"]      # purple replaces old magenta
COLOR_GREEN = _COLORS["green"]

def parse_formula_to_counts(formula: str) -> Dict[str, int]:
    from collections import Counter
    c, i = Counter(), 0
    while i < len(formula):
        if formula[i].isupper():
            if i+1 < len(formula) and formula[i+1].islower():
                el, i = formula[i:i+2], i+2
            else:
                el, i = formula[i], i+1
            j = i
            while j < len(formula) and formula[j].isdigit(): j += 1
            n = int(formula[i:j]) if j > i else 1
            c[el] += n; i = j
        else:
            i += 1
    return c

def fractions_on_target_elements(formula: str, els=TARGET_ELEMENTS, eps=PSEUDOCOUNT) -> np.ndarray:
    cnt = parse_formula_to_counts(formula)
    v = np.array([cnt.get(e, 0.0) for e in els], float) + eps
    return v / v.sum()

def build_fraction_matrix(names: List[str], els=TARGET_ELEMENTS) -> np.ndarray:
    return np.vstack([fractions_on_target_elements(n, els) for n in names])

# --------------------------- CLR and ILR ---------------------------
def clr_transform(X: np.ndarray, eps=PSEUDOCOUNT) -> np.ndarray:
    Xp = X + eps
    Xp = Xp / Xp.sum(axis=1, keepdims=True)
    L = np.log(Xp)
    return L - L.mean(axis=1, keepdims=True)

def ilr_transform_pivot(X: np.ndarray, eps=PSEUDOCOUNT) -> np.ndarray:
    """
    Pivot ILR (orthonormal): ilr_k = sqrt(k/(k+1))*(mean(log x_1..x_k) - log x_{k+1}), k=1..D-1
    """
    Xp = X + eps
    Xp = Xp / Xp.sum(axis=1, keepdims=True)
    L = np.log(Xp)
    n, D = L.shape
    Z = np.zeros((n, D-1))
    for k in range(1, D):
        a = np.sqrt(k/(k+1))
        Z[:, k-1] = a * (L[:, :k].mean(axis=1) - L[:, k])
    return Z

def get_preproc(transform: str):
    t = transform.lower()
    if t == 'ilr': return ilr_transform_pivot, 'ILR→PCA'
    if t == 'clr': return clr_transform, 'CLR→PCA'
    if t == 'raw': return (lambda X: X), 'PCA (raw fractions)'
    raise ValueError("transform must be one of: ilr, clr, raw")

# ------------------------- Dataset background -------------------------
def get_dataset_compositions(max_generation: int = 10) -> Dict[str, int]:
    """Return {formula: count} for generation <= max_generation (if DB available)."""
    try:
        from forge.core.database import DatabaseManager
    except Exception:
        print("Warning: forge DB not available; continuing without dataset background.")
        return {}
    db = DatabaseManager()
    ids = db.find_structures_by_metadata(metadata_filters={'generation': max_generation}, operator='<=')
    atoms = db.get_structures_batch(ids)
    out = {}
    for _, a in atoms.items():
        f = a.get_chemical_formula()
        out[f] = out.get(f, 0) + 1
    print(f"Dataset compositions (≤ gen {max_generation}): {len(out)} unique")
    return out

# --------------------------- Novelty ranking ---------------------------
def dicts_from_formula_list(formulas: List[str]) -> List[Dict[str, float]]:
    return [dict(zip(TARGET_ELEMENTS, fractions_on_target_elements(s))) for s in formulas]

def novelty_rank(existing_comp: Dict[str, int], simple_formulas: List[str],
                 dim_method: str = 'PCA', n_neighbors: int = 10) -> List[str]:
    analyzer = CompositionAnalyzer(n_components=2, dim_method=dim_method, random_state=42)
    existing = dicts_from_formula_list(list(existing_comp.keys()))
    new = dicts_from_formula_list(simple_formulas)
    res = analyze_composition_distribution(
        analyzer,
        existing_compositions=existing,
        new_compositions=new,
        n_clusters=5, n_neighbors=n_neighbors, top_n=len(new),
        weights=(0.4, 0.3, 0.3)
    )
    return [simple_formulas[tc['index']] for tc in sorted(res['top_compositions'], key=lambda x: x['rank'])]

# --------------------------- Load summary_barriers ---------------------------
def load_summary(path: str):
    """
    Returns:
      keys: list of 'Cr..._a_to_b'
      neb:  dict[key] = {'vasp':[7], 'mlip_gen0':[7], 'mlip_gen10':[7],
                         'bar_vasp':float, 'bar_gen0':float, 'bar_gen10':float}
    """
    per_comp = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            key = r['composition']
            lab = r['label'].strip().lower()
            bar_vasp = float(r['barrier_vasp'])
            bar_mlip = float(r['barrier_mlip'])
            neb_vasp = list(ast.literal_eval(r['neb_vasp']))
            neb_mlip = list(ast.literal_eval(r['neb_mlip']))
            if key not in per_comp:
                per_comp[key] = {'vasp': neb_vasp, 'bar_vasp': bar_vasp}
            if len(neb_vasp) > len(per_comp[key]['vasp']):
                per_comp[key]['vasp'] = neb_vasp
            if lab == 'gen0':
                per_comp[key]['mlip_gen0'] = neb_mlip; per_comp[key]['bar_gen0'] = bar_mlip
            elif lab in ('gen10','gen_10','gen-10'):
                per_comp[key]['mlip_gen10'] = neb_mlip; per_comp[key]['bar_gen10'] = bar_mlip
    for k, v in per_comp.items():
        assert 'mlip_gen0' in v and 'mlip_gen10' in v, f"Missing gen0/gen10 for {k}"
    return sorted(per_comp.keys()), per_comp

def load_summary_json(path: str):
    """
    Accepts a minimal JSON produced by analyse_neb_results or converted from CSV.
    Expected shapes:
      {"records": [
          {"composition": str,
           "vasp": [..],
           "mlip_gen0": [..], "mlip_gen10": [..],
           "bar_vasp": float, "bar_gen0": float, "bar_gen10": float}
      ]}
    or list of such records directly.
    Returns (keys, per_comp) like load_summary.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    records = data.get('records', data if isinstance(data, list) else [])
    per_comp = {}
    for r in records:
        key = r['composition']
        per_comp[key] = {
            'vasp': list(r.get('vasp', [])),
            'bar_vasp': float(r.get('bar_vasp', float('nan')))
        }
        if 'mlip_gen0' in r:
            per_comp[key]['mlip_gen0'] = list(r['mlip_gen0'])
            if 'bar_gen0' in r:
                per_comp[key]['bar_gen0'] = float(r['bar_gen0'])
        if 'mlip_gen10' in r:
            per_comp[key]['mlip_gen10'] = list(r['mlip_gen10'])
            if 'bar_gen10' in r:
                per_comp[key]['bar_gen10'] = float(r['bar_gen10'])
    return sorted(per_comp.keys()), per_comp

def load_summary_any(path: str):
    p = (path or '').lower()
    if p.endswith('.json'):
        return load_summary_json(path)
    return load_summary(path)

def resolve_default_summary():
    """Look for a default summary next to this script, preferring minimal JSON."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        'summary_barriers.min.json',
        'summary_barriers.json',
        'summary_barriers.csv',
    ]
    for name in candidates:
        p = os.path.join(here, name)
        if os.path.isfile(p):
            return p
    # Fallback to current working directory CSV
    if os.path.isfile('summary_barriers.csv'):
        return 'summary_barriers.csv'
    # Return first candidate path (likely missing) to trigger a clear error upstream
    return os.path.join(here, 'summary_barriers.min.json')

# --------------------------- Feasible polytope overlay ---------------------------
def sample_feasible_region(n: int = 6000, seed: int = 0):
    """V≥0.80, Zr≤0.05, (Cr+Ti+W+Zr)≤0.20, sum=1 → (n,5) fractions."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        T = rng.uniform(0.0, 0.20)                  # total non-V
        zr = rng.uniform(0.0, min(0.05, T))
        rest = T - zr
        cr, ti, w = (rng.dirichlet(np.ones(3)) * rest) if rest > 1e-12 else (0.0, 0.0, 0.0)
        v = 1.0 - T
        if v >= 0.80:
            pts.append([v, cr, ti, w, zr])
    return np.asarray(pts)

def overlay_manifold(ax, pca: PCA, preproc_fn, facecolor=COLOR_GREEN, alpha=0.07):
    """Project feasible-region samples with same transform and draw outline. Returns Z (N,2)."""
    S = sample_feasible_region(6000)
    Z = pca.transform(preproc_fn(S))
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(Z)
        verts = hull.vertices
        # Draw outline only (no fill) - use Polygon with edgecolor only
        poly_verts = Z[verts]
        poly = Polygon(poly_verts, closed=True, facecolor='none', edgecolor=facecolor, 
                      linewidth=2.0, zorder=1)
        ax.add_patch(poly)
    except Exception:
        ax.scatter(Z[:,0], Z[:,1], s=5, c=facecolor, alpha=alpha, zorder=1)
    return Z

# --------------------------- Embedding + zoom helpers ---------------------------
def embed_pca_2d(formulas_all: List[str], transform: str = 'ilr', random_state: int = 42):
    preproc_fn, tname = get_preproc(transform)
    X = build_fraction_matrix(formulas_all)
    X_in = preproc_fn(X)
    pca = PCA(n_components=2, random_state=random_state)
    Z = pca.fit_transform(X_in)
    coords = {f: z for f, z in zip(formulas_all, Z)}
    var = float(pca.explained_variance_ratio_.sum())
    return coords, var, pca, preproc_fn, tname

def manifold_zoom_limits(Z_poly: np.ndarray, simple_xy: np.ndarray,
                         buffer_left_bottom: float = 5.0,
                         pad_right: float = 1.0,
                         pad_top: float = 1.0):
    """
    Frame the feasible polytope and SIMPLE points with a fixed buffer on
    the LEFT and BOTTOM (default 5 units), and small pads on the TOP/RIGHT.
    Ensures far-right SIMPLE outliers remain visible.
    """
    zx, zy = Z_poly[:,0], Z_poly[:,1]
    sx, sy = simple_xy[:,0], simple_xy[:,1]
    x_min = min(zx.min(), sx.min()) - buffer_left_bottom
    y_min = min(zy.min(), sy.min()) - buffer_left_bottom
    x_max = max(zx.max(), sx.max()) + pad_right
    y_max = max(zy.max(), sy.max()) + pad_top
    return (x_min, x_max), (y_min, y_max)

def smart_zoom_limits(simple_xy: np.ndarray,
                      Z_poly: np.ndarray = None,
                      pad_frac: float = 0.35,
                      min_span: float = 3.0,
                      clamp_to_poly: bool = False):
    """
    Fit a square-ish window tightly around SIMPLE points with fractional padding.
    Optionally clamp to feasible polytope bounds to avoid empty regions.
    """
    x, y = simple_xy[:,0], simple_xy[:,1]
    cx, cy = x.mean(), y.mean()
    span_x = x.max() - x.min()
    span_y = y.max() - y.min()
    span = max(span_x, span_y, min_span)
    pad = pad_frac * span
    half = 0.5 * span
    xl, xh = cx - half - pad, cx + half + pad
    yl, yh = cy - half - pad, cy + half + pad

    if clamp_to_poly and Z_poly is not None and len(Z_poly) > 0:
        px_min, px_max = float(np.min(Z_poly[:,0])), float(np.max(Z_poly[:,0]))
        py_min, py_max = float(np.min(Z_poly[:,1])), float(np.max(Z_poly[:,1]))
        xl = max(xl, px_min); xh = min(xh, px_max)
        yl = max(yl, py_min); yh = min(yh, py_max)

    return (xl, xh), (yl, yh)

# --------------------------- Inset zoom ---------------------------
def add_inset_zoom(ax, ordered_keys: List[str], coords: Dict[str, Tuple[float, float]], 
                   indices_to_zoom: List[int], inset_bounds: Tuple[float, float, float, float],
                   zoom_padding: float = 0.5, dataset_formulas: List[str] = None):
    """
    Add an inset axes showing a zoomed view of specific numbered points.
    
    Args:
        ax: main axes to add inset to
        ordered_keys: list of composition keys in display order
        coords: {formula: (x, y)} mapping
        indices_to_zoom: list of 1-based indices to zoom on (e.g., [1, 2, 4, 5, 6])
        inset_bounds: (x0, y0, width, height) in axes fraction coordinates
        zoom_padding: padding around zoomed points in data coordinates
        dataset_formulas: optional list of dataset formulas to show in background
    """
    # Get coordinates of points to zoom
    zoom_xy = []
    for idx in indices_to_zoom:
        if 1 <= idx <= len(ordered_keys):
            k = ordered_keys[idx - 1]
            base = k.split('_')[0]
            if base in coords:
                zoom_xy.append(coords[base])
    
    if not zoom_xy:
        return None
    
    zoom_xy = np.array(zoom_xy)
    x_min, y_min = zoom_xy.min(axis=0)
    x_max, y_max = zoom_xy.max(axis=0)
    
    # Add padding (ensure minimum span)
    if x_max - x_min < 0.5:
        x_center = (x_min + x_max) / 2
        x_min, x_max = x_center - 0.25, x_center + 0.25
    if y_max - y_min < 0.5:
        y_center = (y_min + y_max) / 2
        y_min, y_max = y_center - 0.25, y_center + 0.25
    
    x_min -= zoom_padding
    x_max += zoom_padding
    y_min -= zoom_padding
    y_max += zoom_padding
    
    # Create inset axes
    axins = ax.inset_axes(inset_bounds)
    
    # Add dataset background points within the zoomed region
    if dataset_formulas:
        for f in dataset_formulas:
            if f in coords:
                x, y = coords[f]
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    axins.scatter([x], [y], c='#CFCFCF', s=30, alpha=0.65, edgecolors='none', zorder=2)
    
    # Show the zoomed SIMPLE points with their labels
    for idx in indices_to_zoom:
        if 1 <= idx <= len(ordered_keys):
            k = ordered_keys[idx - 1]
            base = k.split('_')[0]
            if base in coords:
                x, y = coords[base]
                axins.scatter([x], [y], c=COLOR_RED, s=64, alpha=0.92, edgecolors='black', linewidths=0.6, zorder=3)
                axins.text(x, y, str(idx), ha='center', va='center', fontsize=12.2,
                          bbox=dict(boxstyle='circle,pad=0.22', fc='white', ec=COLOR_RED, lw=1.0, alpha=0.95),
                          color='black', zorder=4)
    
    axins.set_xlim(x_min, x_max)
    axins.set_ylim(y_min, y_max)
    axins.grid(alpha=0.25, linestyle=':', zorder=2)
    axins.set_aspect('equal', adjustable='box')
    # Remove axis labels from inset
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    # Draw rectangle on main plot showing zoomed region (lower zorder so axes appear on top)
    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                     linewidth=1.5, edgecolor=COLOR_BLUE, facecolor='none', linestyle='--', zorder=2)
    ax.add_patch(rect)
    
    # Add connection lines from rectangle to inset (lower zorder)
    # Connect bottom-left of rect to bottom-left of inset
    con1 = ConnectionPatch(xyA=(x_min, y_min), xyB=(0, 0), 
                          coordsA='data', coordsB='axes fraction',
                          axesA=ax, axesB=axins, 
                          color=COLOR_BLUE, linewidth=1, linestyle='--', alpha=0.6, zorder=2)
    ax.add_artist(con1)
    
    # Connect bottom-right of rect to bottom-right of inset
    con2 = ConnectionPatch(xyA=(x_max, y_min), xyB=(1, 0),
                          coordsA='data', coordsB='axes fraction',
                          axesA=ax, axesB=axins,
                          color=COLOR_BLUE, linewidth=1, linestyle='--', alpha=0.6, zorder=2)
    ax.add_artist(con2)
    
    return axins

# --------------------------- Pretty label ---------------------------
def pretty_comp_label(key: str) -> str:
    if "_to_" in key and key.rsplit('_',2)[-2] == 'to':
        base = key.rsplit('_',2)[0]
        a = key.split('_')[-3]; b = key.split('_')[-1]
        return f"{base}: {a} → {b}"
    return key.replace('_', ': ')

_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

def format_formula_subscript(key: str) -> str:
    """Convert 'Cr2Ti2V120W2Zr2_98_to_118' → 'V₁₂₀Cr₂Ti₂W₂Zr₂' (V first, unicode subscripts)."""
    base = key.split('_')[0]  # strip vacancy indices
    counts = parse_formula_to_counts(base)
    # V first, then alphabetical
    elements = sorted(counts.keys(), key=lambda e: (e != 'V', e))
    parts = []
    for el in elements:
        n = counts[el]
        parts.append(el if n == 1 else el + str(n).translate(_SUB))
    return ''.join(parts)

# --------------------------- Parity panel ---------------------------
def parity_panel(ax, vasp: List[float], gen0: List[float], gen10: List[float], labels: List[int]):
    v = np.array(vasp); g0 = np.array(gen0); g10 = np.array(gen10)
    lo = float(min(v.min(), g0.min(), g10.min())); hi = float(max(v.max(), g0.max(), g10.max()))
    ax.plot([lo, hi], [lo, hi], ls='--', c='k', lw=1, zorder=1)
    
    # Calculate metrics
    rmse0 = float(np.sqrt(np.mean((g0 - v)**2)))
    rmse10 = float(np.sqrt(np.mean((g10 - v)**2)))
    # R² = 1 - (SS_res / SS_tot)
    ss_tot0 = np.sum((v - v.mean())**2)
    ss_res0 = np.sum((v - g0)**2)
    r2_0 = float(1 - (ss_res0 / ss_tot0)) if ss_tot0 > 0 else 0.0
    
    ss_tot10 = np.sum((v - v.mean())**2)
    ss_res10 = np.sum((v - g10)**2)
    r2_10 = float(1 - (ss_res10 / ss_tot10)) if ss_tot10 > 0 else 0.0
    
    # draw markers with larger size
    ax.scatter(v, g0, marker='o', s=180, c=COLOR_ORANGE, edgecolor='black', linewidth=0.8, 
               label=f'MLIP Gen0 (RMSE={rmse0:.2f}, R²={r2_0:.3f})', zorder=3, alpha=0.85)
    ax.scatter(v, g10, marker='^', s=200, c=COLOR_BLUE, edgecolor='black', linewidth=0.8, 
               label=f'MLIP Gen10 (RMSE={rmse10:.2f}, R²={r2_10:.3f})', zorder=3, alpha=0.85)
    
    # Create text annotations with vertical arrows
    for x, y, n in zip(v, g0, labels):
        # For marker #4, flip positions (Gen10 is below Gen0)
        if n == 4:
            label_y = y + 1.00  # Position ABOVE for #4
        else:
            label_y = y - 1.00  # Position below for others
        ax.annotate(str(n), xytext=(x, label_y), xy=(x, y),  # xytext=label position, xy=arrow target
                   ha='center', va='center', fontsize=13, fontweight='bold',
                   bbox=dict(boxstyle='circle,pad=0.22', fc='white', ec='black', lw=0.6, alpha=0.95),
                   color='black', zorder=4,
                   arrowprops=dict(arrowstyle='->', color=COLOR_ORANGE, lw=1.5, alpha=0.9))
    
    for x, y, n in zip(v, g10, labels):
        # For marker #4, flip positions (Gen10 is below Gen0)
        if n == 4:
            label_y = y - 1.00  # Position BELOW for #4
        else:
            label_y = y + 1.00  # Position above for others
        ax.annotate(str(n), xytext=(x, label_y), xy=(x, y),  # xytext=label position, xy=arrow target
                   ha='center', va='center', fontsize=13, fontweight='bold',
                   bbox=dict(boxstyle='circle,pad=0.22', fc='white', ec='black', lw=0.6, alpha=0.95),
                   color='black', zorder=4,
                   arrowprops=dict(arrowstyle='->', color=COLOR_BLUE, lw=1.5, alpha=0.9))
    
    ax.set_xlabel('VASP barrier (eV)', fontweight='bold')
    ax.set_ylabel('MLIP barrier (eV)', fontweight='bold')
    ax.tick_params()
    
    # Set fixed y-axis limits for better spacing
    ax.set_ylim(-1.0, 6.0)
    ax.set_xlim(lo - 0.3, hi + 0.3)
    
    ax.legend(frameon=False, loc='upper left')


def error_comparison_panel(ax, vasp_bars, gen0_bars, gen10_bars, labels,
                           comp_labels=None):
    """Horizontal grouped bar chart of |barrier error| per composition."""
    v = np.array(vasp_bars)
    g0 = np.array(gen0_bars)
    g10 = np.array(gen10_bars)
    err0 = np.abs(g0 - v)
    err10 = np.abs(g10 - v)

    rmse0 = float(np.sqrt(np.mean((g0 - v) ** 2)))
    rmse10 = float(np.sqrt(np.mean((g10 - v) ** 2)))

    h = 0.40  # bar thickness — don't change
    group_spacing = 1.05  # distance between group centres (>2*h to leave gap between groups)
    y = np.arange(len(labels)) * group_spacing
    offset = h / 2  # no gap within a pair — bars touch

    ax.barh(y + offset, err0, h, color=COLOR_ORANGE, edgecolor='black',
            linewidth=0.5, label=f'Gen 0  (RMSE = {rmse0:.2f} eV)', zorder=3)
    ax.barh(y - offset, err10, h, color=COLOR_BLUE, edgecolor='black',
            linewidth=0.5, label=f'Gen 10 (RMSE = {rmse10:.2f} eV)', zorder=3)

    # value labels to the right of each bar
    for yi, e0, e10 in zip(y, err0, err10):
        ax.text(e0 + 0.01, yi + offset, f'{e0:.2f}', ha='left', va='center',
                fontsize=6.5, fontweight='bold')
        ax.text(e10 + 0.01, yi - offset, f'{e10:.2f}', ha='left', va='center',
                fontsize=6.5, fontweight='bold')

    tick_labels = [f'#{n}' for n in labels]
    ax.set_yticks(y)
    ax.set_yticklabels(tick_labels, fontsize=8, fontweight='bold')
    ax.set_xlabel('|Barrier error| (eV)', fontweight='bold', fontsize=9)
    ax.invert_yaxis()  # #1 at top
    # Extend y-limit to create blank space below #6 for the legend
    ax.set_ylim(y[-1] + 1.8, y[0] - 0.8)
    ax.grid(True, axis='x', linestyle=':', alpha=0.5, zorder=0)
    ax.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=7,
              loc='lower center')
    # pad right for labels
    xhi = max(err0.max(), err10.max())
    ax.set_xlim(0, xhi * 1.3)


# --------------------------- Main figure ---------------------------
def plot_map_and_neb(summary_csv: str,
                     transform: str = 'ilr',
                     include_parity: bool = False,
                     novelty_dim_method: str = 'PCA',
                     save_path: str = 'map_plus_neb.png',
                     max_generation_bg: int = 10,
                     # layout
                     row_height: float = 1.75,
                     right_col_ratio: float = 0.85,
                     parity_height: float = 4.5,
                     height_scale: float = 1.2,
                     map_scale: float = 1.15,
                     # zoom
                     zoom: str = 'manifold',   # 'smart' | 'manifold' | 'tight' | 'dataset' | 'none'
                     zoom_buffer: float = 5.0,
                     smart_pad_frac: float = 0.35,
                     smart_min_span: float = 3.0,
                     smart_clamp_to_poly: bool = False,
                     overlay_polytope: bool = True):

    # 1) Load
    simple_keys, neb = load_summary_any(summary_csv)
    base_formulas = [k.split('_')[0] for k in simple_keys]
    dataset = get_dataset_compositions(max_generation=max_generation_bg)

    # 2) Novelty order
    try:
        order = novelty_rank(dataset, base_formulas, dim_method=novelty_dim_method, n_neighbors=10)
    except Exception as e:
        print(f"Novelty ranking failed ({e}); using CSV order.")
        order = base_formulas
    key_by_base = {k.split('_')[0]: k for k in simple_keys}
    ordered_keys = [key_by_base[b] for b in order]

    # 3) Embedding
    all_for_map = list(dataset.keys()) + base_formulas
    coords, var, pca, preproc_fn, tname = embed_pca_2d(all_for_map, transform=transform)

    # 4) Grid — new layout: top row = [map | error], bottom = 2×3 NEB grid
    n = len(ordered_keys)
    from matplotlib.ticker import FormatStrFormatter

    top_h = 4.0                              # inches for map + error row
    neb_rows = int(np.ceil(n / 3))           # 2 rows for 6 compositions
    neb_row_h = 2.2                          # height per NEB row
    neb_block_h = neb_rows * neb_row_h
    total_height_in = top_h + neb_block_h
    fig_width_in = DOUBLE_COL                # 7.0 inches

    fig = plt.figure(figsize=(fig_width_in, total_height_in))

    outer_gs = gridspec.GridSpec(
        nrows=2, ncols=1,
        height_ratios=[top_h, neb_block_h],
        hspace=0.38, figure=fig,
    )

    # --- Top row: map (left) + error comparison (right) ---
    top_gs = gridspec.GridSpecFromSubplotSpec(
        nrows=1, ncols=2, subplot_spec=outer_gs[0],
        width_ratios=[1.0, 1.0], wspace=0.35,
    )

    # --- Bottom: 2×3 NEB grid ---
    neb_gs = gridspec.GridSpecFromSubplotSpec(
        nrows=neb_rows, ncols=3, subplot_spec=outer_gs[1],
        wspace=0.35, hspace=0.45,
    )

    # ============================================================
    # MAP PANEL (top-left)
    # ============================================================
    axL = fig.add_subplot(top_gs[0, 0])

    Z_poly = None
    if overlay_polytope:
        Z_poly = overlay_manifold(axL, pca, preproc_fn, facecolor=COLOR_GREEN, alpha=0.10)
        if len(Z_poly) > 0:
            cx = Z_poly[:, 0].mean()
            cy_top = Z_poly[:, 1].max()
            axL.text(cx, cy_top + 0.5, 'Feasible region', ha='center', va='bottom',
                    color=COLOR_GREEN, fontweight='bold', alpha=0.9, zorder=10, fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))
    else:
        if zoom == 'manifold':
            Z_poly = pca.transform(preproc_fn(sample_feasible_region(4000)))

    if dataset:
        ds = [f for f in dataset.keys() if f in coords]
        axL.scatter([coords[f][0] for f in ds], [coords[f][1] for f in ds],
                    c='#CFCFCF', s=30, alpha=0.75, edgecolors='none', zorder=2)

    # Fan-out offsets (in points) so clustered labels don't overlap
    _label_offsets = [
        (0, 22),     # #1: up
        (22, 12),    # #2: up-right
        (18, -5),    # #3: right (isolated point, minimal offset)
        (-22, 12),   # #4: up-left
        (-22, -12),  # #5: down-left
        (0, -22),    # #6: down
    ]
    simple_xy = []
    for idx, k in enumerate(ordered_keys, start=1):
        base = k.split('_')[0]; x, y = coords[base]
        simple_xy.append([x, y])
        axL.scatter([x], [y], c=COLOR_RED, s=50, alpha=0.92,
                    edgecolors='black', linewidths=0.6, zorder=3)
        off = _label_offsets[idx - 1] if idx <= len(_label_offsets) else (0, 18)
        axL.annotate(
            str(idx), xy=(x, y), xytext=off,
            textcoords='offset points', ha='center', va='center',
            fontsize=9, fontweight='bold', color='black', zorder=5,
            bbox=dict(boxstyle='circle,pad=0.25', fc='white', ec=COLOR_RED,
                      lw=1.2, alpha=0.95),
            arrowprops=dict(arrowstyle='-', color=COLOR_RED, lw=0.8,
                            shrinkA=0, shrinkB=5),
        )
    simple_xy = np.asarray(simple_xy)

    axL.set_xlabel('PC 1', fontweight='bold', fontsize=9)
    axL.set_ylabel('PC 2', fontweight='bold', fontsize=9)
    axL.tick_params(labelsize=8)
    axL.grid(alpha=0.25, linestyle=':', zorder=1)
    axL.set_aspect('equal', adjustable='datalim')
    axL.set_title('(a)', fontweight='bold', fontsize=10, loc='left')

    # Auto-scale to show all points + feasible region
    if Z_poly is not None and len(simple_xy):
        all_pts = np.vstack([simple_xy, Z_poly])
    elif len(simple_xy):
        all_pts = simple_xy
    else:
        all_pts = None
    if all_pts is not None:
        xpad = (all_pts[:, 0].max() - all_pts[:, 0].min()) * 0.15 + 1.5
        ypad = (all_pts[:, 1].max() - all_pts[:, 1].min()) * 0.15 + 1.5
        axL.set_xlim(all_pts[:, 0].min() - xpad, all_pts[:, 0].max() + xpad)
        axL.set_ylim(all_pts[:, 1].min() - ypad, all_pts[:, 1].max() + ypad)

    # ============================================================
    # NEB PANELS (bottom 2×3 grid)
    # ============================================================
    vasp_bars, gen0_bars, gen10_bars, labels_num, comp_labels = [], [], [], [], []
    first_neb_ax = None
    for r, comp_key in enumerate(ordered_keys):
        row_idx, col_idx = divmod(r, 3)
        ax = fig.add_subplot(neb_gs[row_idx, col_idx])
        if r == 0:
            first_neb_ax = ax
        vasp = neb[comp_key]['vasp']
        g0 = neb[comp_key]['mlip_gen0']
        g10 = neb[comp_key]['mlip_gen10']
        ximg = np.arange(len(vasp))
        ax.plot(ximg, vasp, '--', c='k', lw=1.4, label='VASP')
        ax.plot(ximg, g0, '-', c=COLOR_ORANGE, lw=1.5, label='MLIP Gen 0')
        ax.plot(ximg, g10, '-', c=COLOR_BLUE, lw=1.5, label='MLIP Gen 10')
        formula = format_formula_subscript(comp_key)
        ax.set_title(f"#{r+1}  {formula}", loc='left', fontweight='bold',
                     fontsize=8, pad=3)
        ax.set_ylabel('ΔE (eV)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # Only show x-label on bottom row
        if row_idx == neb_rows - 1:
            ax.set_xlabel('Image', fontsize=8)
        else:
            ax.tick_params(labelbottom=False)
        vasp_bars.append(neb[comp_key]['bar_vasp'])
        gen0_bars.append(neb[comp_key]['bar_gen0'])
        gen10_bars.append(neb[comp_key]['bar_gen10'])
        labels_num.append(r + 1)
        comp_labels.append(formula)

    # Shared NEB legend — horizontal row above the 2×3 grid, with (c) label
    if first_neb_ax is not None:
        handles, leg_labels = first_neb_ax.get_legend_handles_labels()
        fig.legend(handles, leg_labels, loc='lower center',
                   bbox_to_anchor=(0.55, 0.465), ncol=3,
                   frameon=False, fontsize=8)
        fig.text(0.06, 0.475, '(c)', fontweight='bold', fontsize=10,
                 va='center', ha='left')

    # ============================================================
    # ERROR COMPARISON PANEL (top-right)
    # ============================================================
    if include_parity:
        axE = fig.add_subplot(top_gs[0, 1])
        axE.set_title('(b)', fontweight='bold', fontsize=10, loc='left')
        error_comparison_panel(axE, vasp_bars, gen0_bars, gen10_bars,
                               labels_num, comp_labels=None)

    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig

# --------------------------- Individual panels ---------------------------
def plot_map_only(summary_csv: str,
                  transform: str = 'ilr',
                  novelty_dim_method: str = 'PCA',
                  save_path: str = 'map_only.png',
                  max_generation_bg: int = 10,
                  row_height: float = 1.75,
                  zoom: str = 'smart',
                  zoom_buffer: float = 5.0,
                  smart_pad_frac: float = 0.35,
                  smart_min_span: float = 3.0,
                  smart_clamp_to_poly: bool = False,
                  overlay_polytope: bool = True):
    keys, neb = load_summary_any(summary_csv)
    base_formulas = [k.split('_')[0] for k in keys]
    dataset = get_dataset_compositions(max_generation=max_generation_bg)
    try:
        order = novelty_rank(dataset, base_formulas, dim_method=novelty_dim_method, n_neighbors=10)
    except Exception as e:
        print(f"Novelty ranking failed ({e}); using CSV order.")
        order = base_formulas
    key_by_base = {k.split('_')[0]: k for k in keys}
    ordered_keys = [key_by_base[b] for b in order]

    all_for_map = list(dataset.keys()) + base_formulas
    coords, var, pca, preproc_fn, tname = embed_pca_2d(all_for_map, transform=transform)

    n = len(ordered_keys)
    fig = plt.figure(figsize=fig_size(DOUBLE_COL, 1.0))
    axL = fig.add_subplot(111)

    Z_poly = None
    if overlay_polytope:
        Z_poly = overlay_manifold(axL, pca, preproc_fn, facecolor=COLOR_GREEN, alpha=0.07)
        if len(Z_poly) > 0:
            cx = Z_poly[:, 0].mean()
            cy_top = Z_poly[:, 1].max()
            axL.text(cx, cy_top + 0.5, 'Feasible region', ha='center', va='bottom',
                    color=COLOR_GREEN, fontweight='bold', alpha=0.8, zorder=2, fontsize=9)
    else:
        if zoom == 'manifold':
            Z_poly = pca.transform(preproc_fn(sample_feasible_region(4000)))

    if dataset:
        ds = [f for f in dataset.keys() if f in coords]
        axL.scatter([coords[f][0] for f in ds], [coords[f][1] for f in ds],
                    c='#CFCFCF', s=48, alpha=0.65, edgecolors='none', label='Dataset (≤ Gen 10)', zorder=2)

    simple_xy = []
    for idx, k in enumerate(ordered_keys, start=1):
        base = k.split('_')[0]; x, y = coords[base]
        simple_xy.append([x, y])
        axL.scatter([x], [y], c=COLOR_RED, s=80, alpha=0.92,
                    edgecolors='black', linewidths=0.6, zorder=3)
        axL.text(x, y, str(idx), ha='center', va='center', fontsize=9,
                 bbox=dict(boxstyle='circle,pad=0.25', fc='white', ec=COLOR_RED,
                           lw=1.2, alpha=0.95),
                 color='black', fontweight='bold', zorder=4)
    simple_xy = np.asarray(simple_xy)

    axL.set_xlabel('PC 1', fontweight='bold', fontsize=9)
    axL.set_ylabel('PC 2', fontweight='bold', fontsize=9)
    axL.tick_params(labelsize=8)
    axL.grid(alpha=0.25, linestyle=':', zorder=1)
    axL.legend(frameon=False, loc='upper center', ncol=2, fontsize=7)
    axL.set_aspect('equal', adjustable='box')

    # Auto-scale to show all points + feasible region
    if Z_poly is not None and len(simple_xy):
        all_pts = np.vstack([simple_xy, Z_poly])
    elif len(simple_xy):
        all_pts = simple_xy
    else:
        all_pts = None
    if all_pts is not None:
        xpad = (all_pts[:, 0].max() - all_pts[:, 0].min()) * 0.12 + 1.0
        ypad = (all_pts[:, 1].max() - all_pts[:, 1].min()) * 0.12 + 1.0
        axL.set_xlim(all_pts[:, 0].min() - xpad, all_pts[:, 0].max() + xpad)
        axL.set_ylim(all_pts[:, 1].min() - ypad, all_pts[:, 1].max() + ypad)

    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig

def plot_neb_only(summary_csv: str,
                  save_path: str = 'neb_only.png',
                  novelty_dim_method: str = 'PCA',
                  row_height: float = 1.75,
                  right_col_ratio: float = 0.85,
                  neb_legend_outside: bool = True,
                  neb_legend_pad: float = 1.0,
                  max_generation_bg: int = 10):
    from matplotlib.ticker import FormatStrFormatter
    keys, neb = load_summary_any(summary_csv)
    base_formulas = [k.split('_')[0] for k in keys]
    dataset = get_dataset_compositions(max_generation=max_generation_bg)
    try:
        order = novelty_rank(dataset, base_formulas, dim_method=novelty_dim_method, n_neighbors=10)
    except Exception:
        order = base_formulas
    key_by_base = {k.split('_')[0]: k for k in keys}
    ordered_keys = [key_by_base[b] for b in order]

    n = len(ordered_keys)
    left_width_in = n * row_height
    width_in = right_col_ratio * left_width_in + (neb_legend_pad if neb_legend_outside else 0.0)
    height_in = n * row_height
    fig, axes = plt.subplots(n, 1, figsize=(width_in, height_in), sharex=False)
    if n == 1:
        axes = [axes]
    for r, (ax, comp_key) in enumerate(zip(axes, ordered_keys)):
        vasp = neb[comp_key]['vasp']
        g0 = neb[comp_key]['mlip_gen0']
        g10 = neb[comp_key]['mlip_gen10']
        ximg = np.arange(len(vasp))
        ax.plot(ximg, vasp, '--', c='k', lw=1.4, label='VASP')
        ax.plot(ximg, g0, '-', c=COLOR_ORANGE, lw=1.5, label='MLIP Gen 0')
        ax.plot(ximg, g10, '-', c=COLOR_BLUE, lw=1.5, label='MLIP Gen 10')
        formula = format_formula_subscript(comp_key)
        ax.set_title(f"#{r+1}  {formula}", loc='left', fontweight='bold',
                     fontsize=8, pad=2)
        ax.set_ylabel('ΔE (eV)', fontweight='bold', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if r == n - 1:
            ax.set_xlabel('Image', fontweight='bold', fontsize=8)
        else:
            ax.tick_params(labelbottom=False)
        if r == 0:
            if neb_legend_outside:
                ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
                          borderaxespad=0.0, frameon=False, fontsize=7)
            else:
                ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.08),
                          ncol=3, frameon=False, fontsize=7)

    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig

def plot_parity_only(summary_csv: str,
                     save_path: str = 'parity_only.png',
                     novelty_dim_method: str = 'PCA',
                     row_height: float = 1.75,
                     right_col_ratio: float = 0.85,
                     parity_only_height: float = 4.5,
                     max_generation_bg: int = 10):
    keys, neb = load_summary_any(summary_csv)
    base_formulas = [k.split('_')[0] for k in keys]
    dataset = get_dataset_compositions(max_generation=max_generation_bg)
    try:
        order = novelty_rank(dataset, base_formulas, dim_method=novelty_dim_method, n_neighbors=10)
    except Exception:
        order = base_formulas
    key_by_base = {k.split('_')[0]: k for k in keys}
    ordered_keys = [key_by_base[b] for b in order]

    n = len(ordered_keys)
    left_width_in = n * row_height
    width_in = (1.0 + right_col_ratio) * left_width_in
    height_in = parity_only_height
    fig = plt.figure(figsize=(width_in, height_in))
    ax = fig.add_subplot(111)

    vasp_bars, gen0_bars, gen10_bars, labels_num = [], [], [], []
    for r, comp_key in enumerate(ordered_keys):
        vasp_bars.append(neb[comp_key]['bar_vasp'])
        gen0_bars.append(neb[comp_key]['bar_gen0'])
        gen10_bars.append(neb[comp_key]['bar_gen10'])
        labels_num.append(r+1)
    parity_panel(ax, vasp_bars, gen0_bars, gen10_bars, labels_num)

    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig

# -------------------------------- CLI --------------------------------
def main():
    p = argparse.ArgumentParser(description="Map + NEB trajectories (Gen0 vs Gen10), manifold zoom, parity spans both columns")
    p.add_argument("--summary", type=str, required=False, default=None, help="Path to summary (.json or .csv). Defaults to JSON if present.")
    p.add_argument("--transform", type=str, default="ilr", choices=["ilr","clr","raw"], help="Transform for the map")
    p.add_argument("--include_parity", action="store_true", help="Add parity row spanning both columns (only for panel=combined)")
    p.add_argument("--novelty_dim_method", type=str, default="PCA", help="Dimensionality reduction used by novelty analyzer")
    p.add_argument("--save", type=str, default="map_plus_neb.png", help="Output figure path")
    p.add_argument("--max_generation_bg", type=int, default=10, help="Background dataset generation upper bound (<=)")
    p.add_argument("--panel", type=str, default="combined", choices=["combined","map","neb","parity"], help="Which panel to render")
    # layout/zoom knobs
    p.add_argument("--row_height", type=float, default=1.75, help="Height per NEB row (inches)")
    p.add_argument("--right_col_ratio", type=float, default=0.85, help="Right/left column width ratio")
    p.add_argument("--parity_height", type=float, default=4.5, help="Parity row height (inches) for combined panel")
    p.add_argument("--height_scale", type=float, default=1.2, help="Overall height scaling factor for combined panel")
    p.add_argument("--map_scale", type=float, default=1.15, help="Scaling factor for composition map size")
    p.add_argument("--neb_legend_inside", action="store_true", help="Place NEB legend inside axis for panel=neb")
    p.add_argument("--neb_legend_pad", type=float, default=1.0, help="Right padding when legend is outside (panel=neb)")
    p.add_argument("--zoom", type=str, default="smart", choices=["smart","manifold","tight","dataset","none"], help="Zoom mode for map panels")
    p.add_argument("--smart_pad_frac", type=float, default=0.35, help="Padding fraction for smart zoom around SIMPLE points")
    p.add_argument("--smart_min_span", type=float, default=3.0, help="Minimum span for smart zoom window")
    p.add_argument("--smart_clamp_to_poly", action="store_true", help="Clamp smart zoom window to feasible polytope bounds")
    p.add_argument("--zoom_buffer", type=float, default=5.0, help="Buffer (units) on LEFT and BOTTOM when zoom='manifold'")
    p.add_argument("--no_overlay_polytope", dest="overlay_polytope", action="store_false", help="Disable polytope overlay on map panels")
    args = p.parse_args()

    summary_path = args.summary or resolve_default_summary()

    if args.panel == 'combined':
        plot_map_and_neb(summary_csv=summary_path,
                         transform=args.transform,
                         include_parity=args.include_parity,
                         novelty_dim_method=args.novelty_dim_method,
                         save_path=args.save,
                         max_generation_bg=args.max_generation_bg,
                         row_height=args.row_height,
                         right_col_ratio=args.right_col_ratio,
                         parity_height=args.parity_height,
                         height_scale=args.height_scale,
                         map_scale=args.map_scale,
                         zoom=args.zoom,
                         zoom_buffer=args.zoom_buffer,
                         overlay_polytope=args.overlay_polytope,
                         smart_pad_frac=args.smart_pad_frac,
                         smart_min_span=args.smart_min_span,
                         smart_clamp_to_poly=args.smart_clamp_to_poly)
    elif args.panel == 'map':
        plot_map_only(summary_csv=summary_path,
                      transform=args.transform,
                      novelty_dim_method=args.novelty_dim_method,
                      save_path=args.save,
                      max_generation_bg=args.max_generation_bg,
                      row_height=args.row_height,
                      zoom=args.zoom,
                      zoom_buffer=args.zoom_buffer,
                      overlay_polytope=args.overlay_polytope,
                      smart_pad_frac=args.smart_pad_frac,
                      smart_min_span=args.smart_min_span,
                      smart_clamp_to_poly=args.smart_clamp_to_poly)
    elif args.panel == 'neb':
        plot_neb_only(summary_csv=summary_path,
                      save_path=args.save,
                      novelty_dim_method=args.novelty_dim_method,
                      row_height=args.row_height,
                      right_col_ratio=args.right_col_ratio,
                      neb_legend_outside=not args.neb_legend_inside,
                      neb_legend_pad=args.neb_legend_pad,
                      max_generation_bg=args.max_generation_bg)
    elif args.panel == 'parity':
        plot_parity_only(summary_csv=summary_path,
                         save_path=args.save,
                         novelty_dim_method=args.novelty_dim_method,
                         row_height=args.row_height,
                         right_col_ratio=args.right_col_ratio,
                         parity_only_height=max(4.5, args.parity_height),
                         max_generation_bg=args.max_generation_bg)

if __name__ == "__main__":
    main()

