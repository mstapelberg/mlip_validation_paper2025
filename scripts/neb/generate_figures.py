#!/usr/bin/env python3
"""
Map + NEB (Gen0 vs Gen10) figure — parity spans both columns; smart manifold zoom.

Left: composition map (PCA on ILR/CLR/raw) with novelty numbers and feasible polytope overlay.
Right: stacked NEB trajectories per SIMPLE composition with VASP (dashed), MLIP Gen0, MLIP Gen10.
Bottom (optional): parity panel spanning BOTH columns.

Usage:
  python map_plus_neb_gen_compare.py \
      --summary summary_barriers.csv \
      --transform ilr \
      --include_parity \
      --zoom manifold --zoom_buffer 5 \
      --save map_plus_neb.png
"""

import ast, csv, argparse, numpy as np, matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle, ConnectionPatch
from typing import Dict, List, Tuple
from sklearn.decomposition import PCA

# Set global font and font sizes - use Helvetica and increase all by 2 points
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 12  # Default is usually 10
plt.rcParams['axes.titlesize'] = 14  # Default is usually 12
plt.rcParams['axes.labelsize'] = 14  # Default is usually 12
plt.rcParams['xtick.labelsize'] = 12  # Default is usually 10
plt.rcParams['ytick.labelsize'] = 12  # Default is usually 10
plt.rcParams['legend.fontsize'] = 14  # Increased from 12 to 14 (2 points bigger)

# Try to import adjustText for smart label positioning
try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False

# ---- novelty analyzer (your API). Falls back to local 'composition.py' if needed. ----
try:
    from forge.analysis.composition import CompositionAnalyzer, analyze_composition_distribution
except Exception:
    from composition import CompositionAnalyzer, analyze_composition_distribution  # :contentReference[oaicite:0]{index=0}

# ---------------------- 5-element utilities ----------------------
TARGET_ELEMENTS = ['V','Cr','Ti','W','Zr']
PSEUDOCOUNT = 1e-9

# ---------------------- Color scheme ----------------------
COLOR_BLUE = '#2A33C3'
COLOR_ORANGE = '#A35D00'
COLOR_RED = '#8F2D56'
COLOR_GREEN = '#6E8B00'

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
    """Project feasible-region samples with same transform and shade convex hull. Returns Z (N,2)."""
    S = sample_feasible_region(6000)
    Z = pca.transform(preproc_fn(S))
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(Z)
        verts = hull.vertices
        ax.fill(Z[verts,0], Z[verts,1], alpha=alpha, facecolor=facecolor, edgecolor='none', zorder=1)
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
                    axins.scatter([x], [y], c='#CFCFCF', s=18, alpha=0.65, edgecolors='none', zorder=2)
    
    # Show the zoomed SIMPLE points with their labels
    for idx in indices_to_zoom:
        if 1 <= idx <= len(ordered_keys):
            k = ordered_keys[idx - 1]
            base = k.split('_')[0]
            if base in coords:
                x, y = coords[base]
                axins.scatter([x], [y], c=COLOR_RED, s=62, alpha=0.92, edgecolors='black', linewidths=0.6, zorder=3)
                axins.text(x, y, str(idx), ha='center', va='center', fontsize=12.2,
                          bbox=dict(boxstyle='circle,pad=0.22', fc='white', ec=COLOR_RED, lw=1.0, alpha=0.95),
                          color='black', zorder=4)
    
    axins.set_xlim(x_min, x_max)
    axins.set_ylim(y_min, y_max)
    axins.grid(alpha=0.25, linestyle=':', zorder=2)
    axins.set_aspect('equal', adjustable='box')
    
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
    
    ax.set_xlabel('VASP barrier (eV)', fontsize=16, fontweight='bold')
    ax.set_ylabel('MLIP barrier (eV)', fontsize=16, fontweight='bold')
    ax.tick_params(labelsize=14)
    
    # Set fixed y-axis limits for better spacing
    ax.set_ylim(-1.0, 6.0)
    ax.set_xlim(lo - 0.3, hi + 0.3)
    
    ax.legend(frameon=False, loc='upper left', fontsize=14)

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
    simple_keys, neb = load_summary(summary_csv)
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

    # 4) Grid (parity spans BOTH columns at FULL width)
    n = len(ordered_keys)
    left_width_in = n * row_height * map_scale  # Scale map size
    fig_width_in = left_width_in * (1.0 + right_col_ratio)
    total_height_in = (n * row_height + (parity_height if include_parity else 0.0)) * height_scale
    fig = plt.figure(figsize=(fig_width_in, total_height_in))
    
    if include_parity:
        # Use nested GridSpec: outer has 2 rows (map+NEBs, then parity)
        outer_gs = gridspec.GridSpec(nrows=2, ncols=1, 
                                      height_ratios=[n * row_height, parity_height],
                                      hspace=0.15)
        # Inner grid for map + NEBs
        gs = gridspec.GridSpecFromSubplotSpec(nrows=n, ncols=2,
                                               subplot_spec=outer_gs[0],
                                               width_ratios=[1.0, right_col_ratio],
                                               height_ratios=[1.0]*n,
                                               wspace=0.28, hspace=0.75)
        # Parity will use outer_gs[1] to span full width
    else:
        # No parity: use simple GridSpec
        gs = gridspec.GridSpec(
            nrows=n, ncols=2,
            width_ratios=[1.0, right_col_ratio],
            height_ratios=[1.0]*n, wspace=0.28, hspace=0.75
        )

    # ----- Left: composition map (span NEB rows) -----
    axL = plt.subplot(gs[:, 0])

    # Polytope overlay; keep projected points for zoom control
    Z_poly = None
    if overlay_polytope:
        Z_poly = overlay_manifold(axL, pca, preproc_fn, facecolor=COLOR_GREEN, alpha=0.10)
        # Add text label for feasible region above the top center
        if len(Z_poly) > 0:
            cx = Z_poly[:, 0].mean()
            cy_top = Z_poly[:, 1].max()
            axL.text(cx, cy_top + 0.8, 'Feasible region', ha='center', va='bottom',
                    fontsize=18, color=COLOR_GREEN, fontweight='bold', alpha=0.8, zorder=2)
    else:
        # still compute Z_poly if zoom='manifold'
        if zoom == 'manifold':
            Z_poly = pca.transform(preproc_fn(sample_feasible_region(4000)))

    # dataset background
    if dataset:
        ds = [f for f in dataset.keys() if f in coords]
        axL.scatter([coords[f][0] for f in ds], [coords[f][1] for f in ds],
                    c='#CFCFCF', s=18, alpha=0.75, edgecolors='none', label='Dataset (≤ Gen 10)', zorder=2)

    # SIMPLE points + numbers
    simple_xy = []
    for idx, k in enumerate(ordered_keys, start=1):
        base = k.split('_')[0]; x, y = coords[base]
        simple_xy.append([x,y])
        axL.scatter([x],[y], c=COLOR_RED, s=62, alpha=0.92, edgecolors='black', linewidths=0.6, zorder=3)
        axL.text(x, y, str(idx), ha='center', va='center', fontsize=12.2,
                 bbox=dict(boxstyle='circle,pad=0.22', fc='white', ec=COLOR_RED, lw=1.0, alpha=0.95),
                 color='black', zorder=4)
    simple_xy = np.asarray(simple_xy)

    axL.set_title(f'Composition map ({tname}, var≈{100*var:.1f}%)', fontsize=14, fontweight='bold')
    axL.set_xlabel('PC 1', fontsize=14, fontweight='bold')
    axL.set_ylabel('PC 2', fontsize=14, fontweight='bold')
    axL.tick_params(labelsize=12)
    axL.grid(alpha=0.25, linestyle=':', zorder=1)
    axL.legend(frameon=False, loc='upper center', ncol=2, fontsize=12)
    axL.set_aspect('equal', adjustable='box')

    # ---- Zoom control ----
    if zoom == 'smart' and len(simple_xy):
        (xl, xh), (yl, yh) = smart_zoom_limits(simple_xy, Z_poly=Z_poly,
                                               pad_frac=smart_pad_frac,
                                               min_span=smart_min_span,
                                               clamp_to_poly=smart_clamp_to_poly)
        axL.set_xlim(xl, xh); axL.set_ylim(yl, yh)
    elif zoom == 'manifold' and Z_poly is not None and len(simple_xy):
        (xl, xh), (yl, yh) = manifold_zoom_limits(
            Z_poly, simple_xy, buffer_left_bottom=zoom_buffer, pad_right=1.0, pad_top=1.0
        )
        axL.set_xlim(xl, xh); axL.set_ylim(yl, yh)
    elif zoom == 'tight' and len(simple_xy):
        # center on SIMPLE with modest padding
        x, y = simple_xy[:,0], simple_xy[:,1]
        cx, cy = x.mean(), y.mean()
        span = max(x.max()-x.min(), y.max()-y.min(), 3.0)
        pad = 0.45*span
        axL.set_xlim(cx-span/2-pad, cx+span/2+pad)
        axL.set_ylim(cy-span/2-pad, cy+span/2+pad)
    elif zoom == 'dataset' and dataset:
        XY = np.array([coords[f] for f in dataset.keys() if f in coords])
        xl, xh = np.percentile(XY[:,0], [2.0, 98.0]); yl, yh = np.percentile(XY[:,1], [2.0, 98.0])
        axL.set_xlim(xl-0.1*(xh-xl), xh+0.1*(xh-xl))
        axL.set_ylim(yl-0.1*(yh-yl), yh+0.1*(yh-yl))
    # 'none' → leave autoscale

    # ----- Add inset zoom for clustered points -----
    # Inset in bottom-right: 50% of width/height
    dataset_list = [f for f in dataset.keys() if f in coords] if dataset else []
    add_inset_zoom(axL, ordered_keys, coords, 
                   indices_to_zoom=[1, 2, 4, 5, 6],
                   inset_bounds=(0.48, 0.02, 0.50, 0.50),
                   zoom_padding=0.5,
                   dataset_formulas=dataset_list)

    # ----- Right: stacked NEB trajectories -----
    from matplotlib.ticker import FormatStrFormatter
    vasp_bars, gen0_bars, gen10_bars, labels_num = [], [], [], []
    for r, comp_key in enumerate(ordered_keys):
        ax = plt.subplot(gs[r, 1])
        vasp = neb[comp_key]['vasp']; g0 = neb[comp_key]['mlip_gen0']; g10 = neb[comp_key]['mlip_gen10']
        x = np.arange(len(vasp))
        ax.plot(x, vasp, '--', c='k', lw=1.4, label='VASP')
        ax.plot(x, g0, '-', c=COLOR_ORANGE, lw=1.5, label='MLIP Gen0')
        ax.plot(x, g10, '-', c=COLOR_BLUE, lw=1.5, label='MLIP Gen10')
        ax.set_title(f"#{r+1}  {pretty_comp_label(comp_key)}", loc='left', fontsize=13, fontweight='bold')
        ax.set_ylabel('ΔE (eV)', fontsize=14, fontweight='bold')
        ax.tick_params(labelsize=12)
        # Format y-axis to 2 decimal places
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if r == n-1: 
            ax.set_xlabel('Image', fontsize=14, fontweight='bold')
        else: 
            ax.tick_params(labelbottom=False)
        if r == 0:
            # Place legend horizontally above the first plot with more clearance
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.18), ncol=3, 
                     frameon=False, fontsize=13)
        vasp_bars.append(neb[comp_key]['bar_vasp'])
        gen0_bars.append(neb[comp_key]['bar_gen0'])
        gen10_bars.append(neb[comp_key]['bar_gen10'])
        labels_num.append(r+1)

    # ----- Parity: now SPANS FULL FIGURE WIDTH -----
    if include_parity:
        axP = plt.subplot(outer_gs[1])  # uses full-width outer grid
        parity_panel(axP, vasp_bars, gen0_bars, gen10_bars, labels_num)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
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
    keys, neb = load_summary(summary_csv)
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
    side_in = n * row_height
    fig = plt.figure(figsize=(side_in, side_in))
    axL = fig.add_subplot(111)

    Z_poly = None
    if overlay_polytope:
        Z_poly = overlay_manifold(axL, pca, preproc_fn, facecolor=COLOR_GREEN, alpha=0.07)
        # Add text label for feasible region above the top center
        if len(Z_poly) > 0:
            cx = Z_poly[:, 0].mean()
            cy_top = Z_poly[:, 1].max()
            axL.text(cx, cy_top + 0.8, 'Feasible region', ha='center', va='bottom',
                    fontsize=13, color=COLOR_GREEN, fontweight='bold', alpha=0.8, zorder=2)
    else:
        if zoom == 'manifold':
            Z_poly = pca.transform(preproc_fn(sample_feasible_region(4000)))

    if dataset:
        ds = [f for f in dataset.keys() if f in coords]
        axL.scatter([coords[f][0] for f in ds], [coords[f][1] for f in ds],
                    c='#CFCFCF', s=18, alpha=0.65, edgecolors='none', label='Dataset (≤ Gen 10)', zorder=2)

    simple_xy = []
    for idx, k in enumerate(ordered_keys, start=1):
        base = k.split('_')[0]; x, y = coords[base]
        simple_xy.append([x,y])
        axL.scatter([x],[y], c=COLOR_RED, s=62, alpha=0.92, edgecolors='black', linewidths=0.6, zorder=3)
        axL.text(x, y, str(idx), ha='center', va='center', fontsize=12.2,
                 bbox=dict(boxstyle='circle,pad=0.22', fc='white', ec=COLOR_RED, lw=1.0, alpha=0.95),
                 color='black', zorder=4)
    simple_xy = np.asarray(simple_xy)

    axL.set_title(f'Composition map ({tname}, var≈{100*var:.1f}%)', fontsize=14, fontweight='bold')
    axL.set_xlabel('PC 1', fontsize=14, fontweight='bold')
    axL.set_ylabel('PC 2', fontsize=14, fontweight='bold')
    axL.tick_params(labelsize=12)
    axL.grid(alpha=0.25, linestyle=':', zorder=1)
    axL.legend(frameon=False, loc='upper center', ncol=2, fontsize=12)
    axL.set_aspect('equal', adjustable='box')

    if zoom == 'smart' and len(simple_xy):
        (xl, xh), (yl, yh) = smart_zoom_limits(simple_xy, Z_poly=Z_poly,
                                               pad_frac=smart_pad_frac,
                                               min_span=smart_min_span,
                                               clamp_to_poly=smart_clamp_to_poly)
        axL.set_xlim(xl, xh); axL.set_ylim(yl, yh)
    elif zoom == 'manifold' and Z_poly is not None and len(simple_xy):
        (xl, xh), (yl, yh) = manifold_zoom_limits(
            Z_poly, simple_xy, buffer_left_bottom=zoom_buffer, pad_right=1.0, pad_top=1.0
        )
        axL.set_xlim(xl, xh); axL.set_ylim(yl, yh)
    elif zoom == 'tight' and len(simple_xy):
        x, y = simple_xy[:,0], simple_xy[:,1]
        cx, cy = x.mean(), y.mean()
        span = max(x.max()-x.min(), y.max()-y.min(), 3.0)
        pad = 0.45*span
        axL.set_xlim(cx-span/2-pad, cx+span/2+pad)
        axL.set_ylim(cy-span/2-pad, cy+span/2+pad)
    elif zoom == 'dataset' and dataset:
        XY = np.array([coords[f] for f in dataset.keys() if f in coords])
        xl, xh = np.percentile(XY[:,0], [2.0, 98.0]); yl, yh = np.percentile(XY[:,1], [2.0, 98.0])
        axL.set_xlim(xl-0.1*(xh-xl), xh+0.1*(xh-xl))
        axL.set_ylim(yl-0.1*(yh-yl), yh+0.1*(yh-yl))

    # ----- Add inset zoom for clustered points -----
    dataset_list = [f for f in dataset.keys() if f in coords] if dataset else []
    add_inset_zoom(axL, ordered_keys, coords, 
                   indices_to_zoom=[1, 2, 4, 5, 6],
                   inset_bounds=(0.48, 0.02, 0.50, 0.50),
                   zoom_padding=0.5,
                   dataset_formulas=dataset_list)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
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
    keys, neb = load_summary(summary_csv)
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
        vasp = neb[comp_key]['vasp']; g0 = neb[comp_key]['mlip_gen0']; g10 = neb[comp_key]['mlip_gen10']
        x = np.arange(len(vasp))
        ax.plot(x, vasp, '--', c='k', lw=1.4, label='VASP')
        ax.plot(x, g0, '-', c=COLOR_ORANGE, lw=1.5, label='MLIP Gen0')
        ax.plot(x, g10, '-', c=COLOR_BLUE, lw=1.5, label='MLIP Gen10')
        ax.set_title(f"#{r+1}  {pretty_comp_label(comp_key)}", loc='left', fontsize=13, fontweight='bold')
        ax.set_ylabel('ΔE (eV)', fontsize=14, fontweight='bold')
        ax.tick_params(labelsize=12)
        # Format y-axis to 2 decimal places
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if r == n-1:
            ax.set_xlabel('Image', fontsize=14, fontweight='bold')
        else:
            ax.tick_params(labelbottom=False)
        if r == 0:
            if neb_legend_outside:
                ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False, fontsize=13)
            else:
                # Horizontal legend above with more clearance
                ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False, fontsize=13)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    return fig

def plot_parity_only(summary_csv: str,
                     save_path: str = 'parity_only.png',
                     novelty_dim_method: str = 'PCA',
                     row_height: float = 1.75,
                     right_col_ratio: float = 0.85,
                     parity_only_height: float = 4.5,
                     max_generation_bg: int = 10):
    keys, neb = load_summary(summary_csv)
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
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    return fig

# -------------------------------- CLI --------------------------------
def main():
    p = argparse.ArgumentParser(description="Map + NEB trajectories (Gen0 vs Gen10), manifold zoom, parity spans both columns")
    p.add_argument("--summary", type=str, required=True, help="Path to updated summary_barriers.csv")
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

    if args.panel == 'combined':
        plot_map_and_neb(summary_csv=args.summary,
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
        plot_map_only(summary_csv=args.summary,
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
        plot_neb_only(summary_csv=args.summary,
                      save_path=args.save,
                      novelty_dim_method=args.novelty_dim_method,
                      row_height=args.row_height,
                      right_col_ratio=args.right_col_ratio,
                      neb_legend_outside=not args.neb_legend_inside,
                      neb_legend_pad=args.neb_legend_pad,
                      max_generation_bg=args.max_generation_bg)
    elif args.panel == 'parity':
        plot_parity_only(summary_csv=args.summary,
                         save_path=args.save,
                         novelty_dim_method=args.novelty_dim_method,
                         row_height=args.row_height,
                         right_col_ratio=args.right_col_ratio,
                         parity_only_height=max(4.5, args.parity_height),
                         max_generation_bg=args.max_generation_bg)

if __name__ == "__main__":
    main()

