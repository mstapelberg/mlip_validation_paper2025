#!/usr/bin/env python3
"""
Analyze distributions of forces, per-atom energy, and stress components
from xyz files in data/gen_10_data.

Creates publication-quality plots showing distributions with key statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from ase.io import read

# Publication quality settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['figure.dpi'] = 300

# Color scheme
COLORS = ['#2A33C3', '#A35D00', '#0B7285', '#8F2D56', '#6E8B00']

# Unit conversion
GPA_PER_EVA3 = 160.21766208

def eva3_to_gpa(x):
    return x * GPA_PER_EVA3

def mat33(x):
    """Convert various formats to (N,3,3) stress tensor."""
    x = np.asarray(x)
    if x.ndim == 2 and x.shape == (3, 3):
        return x[None, ...]
    if x.ndim == 3 and x.shape[-2:] == (3, 3):
        return x
    if x.ndim == 1:
        if x.shape[0] == 9:
            return x.reshape(1, 3, 3)
        elif x.shape[0] == 6:
            # Voigt notation
            xx, yy, zz, yz, xz, xy = x
            S = np.array([[xx, xy, xz],
                          [xy, yy, yz],
                          [xz, yz, zz]])
            return S[None, ...]
    raise ValueError(f"Cannot interpret stress shape: {x.shape}")

def decompose_stress(S_GPa):
    """
    Decompose stress tensor into hydrostatic and deviatoric components.
    
    Args:
        S_GPa: Stress tensor in GPa, shape (N, 3, 3) or (3, 3)
    
    Returns:
        p: Hydrostatic pressure (scalar per structure), shape (N,)
        dev: Deviatoric stress tensor, shape (N, 3, 3)
    """
    S_GPa = mat33(S_GPa)
    tr = S_GPa[..., 0, 0] + S_GPa[..., 1, 1] + S_GPa[..., 2, 2]
    p = -tr / 3.0  # Pressure (positive in compression)
    identity_matrix = np.eye(3)[None, ...]
    dev = S_GPa + p[..., None, None] * identity_matrix
    return p, dev

def read_xyz_files(data_dir):
    """Read all xyz files and extract forces, energies, and stress."""
    data_dir = Path(data_dir)
    xyz_files = sorted(data_dir.glob("*.xyz"))
    
    if not xyz_files:
        raise ValueError(f"No .xyz files found in {data_dir}")
    
    print(f"Found {len(xyz_files)} xyz files")
    
    all_force_magnitudes = []
    all_energies_per_atom = []
    all_stress_hydrostatic = []
    all_stress_deviatoric = []
    
    for xyz_file in xyz_files:
        print(f"Reading {xyz_file.name}...")
        try:
            frames = read(str(xyz_file), index=":")
            print(f"  Found {len(frames)} structures")
            
            for atoms in frames:
                n_atoms = len(atoms)
                
                # Forces - compute magnitude per atom
                F = atoms.arrays.get("REF_force", atoms.arrays.get("force", None))
                if F is not None:
                    F = np.asarray(F, dtype=float)
                    # Compute L2 norm (magnitude) for each atom
                    force_mags = np.linalg.norm(F, axis=1)
                    all_force_magnitudes.extend(force_mags)
                
                # Per-atom energy
                E = atoms.info.get("REF_energy", atoms.info.get("total_energy", None))
                if E is None:
                    E = atoms.info.get("free_energy", None)
                if E is not None:
                    all_energies_per_atom.append(float(E) / n_atoms)
                
                # Stress
                S_raw = atoms.info.get("REF_stress", atoms.info.get("stress", None))
                if S_raw is not None:
                    # Handle string format (space-separated values)
                    if isinstance(S_raw, str):
                        S_vals = [float(x) for x in S_raw.split()]
                        S = np.array(S_vals)
                    else:
                        S = np.asarray(S_raw, dtype=float)
                    
                    # Convert to 3x3 tensor
                    S_3x3 = mat33(S)[0]  # Get first (and only) structure
                    
                    # Detect units: if magnitudes are large (>100), likely GPa, else eV/Å³
                    m = np.nanmedian(np.abs(S_3x3))
                    if m > 100.0:
                        S_GPa = S_3x3
                    else:
                        S_GPa = eva3_to_gpa(S_3x3)
                    
                    # Decompose
                    p, dev = decompose_stress(S_GPa)
                    all_stress_hydrostatic.append(float(p))
                    
                    # Use Frobenius norm of deviatoric tensor
                    dev_frob = np.sqrt((dev[0] * dev[0]).sum())
                    all_stress_deviatoric.append(float(dev_frob))
        
        except Exception as e:
            print(f"  Error reading {xyz_file.name}: {e}")
            continue
    
    return {
        'force_magnitudes': np.array(all_force_magnitudes),
        'energies_per_atom': np.array(all_energies_per_atom),
        'stress_hydrostatic': np.array(all_stress_hydrostatic),
        'stress_deviatoric': np.array(all_stress_deviatoric)
    }

def calculate_statistics(data):
    """Calculate q10, q50 (median), mean, q75, q90, and kurtosis."""
    if len(data) == 0:
        return None
    
    # Calculate kurtosis (Fisher's definition, excess kurtosis)
    # kurtosis = E[(X - μ)^4] / σ^4 - 3
    # where 3 is subtracted to make normal distribution have kurtosis = 0
    mean = np.mean(data)
    std = np.std(data)
    if std > 0:
        kurtosis = np.mean(((data - mean) / std) ** 4) - 3.0
    else:
        kurtosis = np.nan
    
    stats = {
        'q10': np.percentile(data, 10),
        'q50': np.percentile(data, 50),  # median
        'mean': np.mean(data),
        'q75': np.percentile(data, 75),
        'q90': np.percentile(data, 90),
        'kurtosis': kurtosis,
        'count': len(data)
    }
    return stats

def create_adaptive_bins(data, n_bins=80, center_focus=True, zero_focus=True):
    """
    Create adaptive bins that are denser near zero for data with outliers.
    
    Uses quantile-based binning: more bins in the IQR, fewer in the tails.
    If zero_focus=True, creates even denser bins symmetrically around zero.
    
    Args:
        data: Array of data values
        n_bins: Target number of bins
        center_focus: If True, use more bins in central region
        zero_focus: If True, create extra-dense bins symmetrically around zero
    
    Returns:
        Array of bin edges
    """
    if center_focus:
        if zero_focus:
            # Focus bins symmetrically around zero
            abs_data = np.abs(data)
            q50_abs = np.percentile(abs_data, 50)  # Median absolute value
            q75_abs = np.percentile(abs_data, 75)
            q90_abs = np.percentile(abs_data, 90)
            
            # Define regions: very near zero, near zero, moderate, far
            # Use percentiles of absolute values to define symmetric regions
            if q50_abs > 0:
                # Very dense near zero: ±q50_abs gets 50% of bins
                # Moderate: q50_abs to q75_abs gets 25% of bins
                # Far: q75_abs to q90_abs gets 15% of bins
                # Very far: beyond q90_abs gets 10% of bins
                n_near_zero = int(n_bins * 0.5)
                n_moderate = int(n_bins * 0.25)
                n_far = int(n_bins * 0.15)
                n_very_far = n_bins - n_near_zero - n_moderate - n_far
                
                bins_list = []
                
                # Negative side (symmetric)
                # Very far negative
                if data.min() < -q90_abs:
                    bins_list.append(np.linspace(data.min(), -q90_abs, n_very_far + 1)[:-1])
                
                # Far negative
                if q90_abs > q75_abs:
                    bins_list.append(np.linspace(-q90_abs, -q75_abs, n_far + 1)[:-1])
                
                # Moderate negative
                if q75_abs > q50_abs:
                    bins_list.append(np.linspace(-q75_abs, -q50_abs, n_moderate + 1)[:-1])
                
                # Near zero (negative side) - densest
                bins_list.append(np.linspace(-q50_abs, 0, n_near_zero // 2 + 1)[:-1])
                
                # Near zero (positive side) - densest
                bins_list.append(np.linspace(0, q50_abs, n_near_zero // 2 + 1))
                
                # Moderate positive
                if q75_abs > q50_abs:
                    bins_list.append(np.linspace(q50_abs, q75_abs, n_moderate + 1)[1:])
                
                # Far positive
                if q90_abs > q75_abs:
                    bins_list.append(np.linspace(q75_abs, q90_abs, n_far + 1)[1:])
                
                # Very far positive
                if data.max() > q90_abs:
                    bins_list.append(np.linspace(q90_abs, data.max(), n_very_far + 1)[1:])
                
                bins = np.concatenate(bins_list)
                bins = np.unique(np.sort(bins))
                
                # Ensure min, max, and zero are included
                bins = np.unique(np.concatenate([[data.min()], bins, [0.0], [data.max()]]))
                bins = np.sort(bins)
                
                return bins
            else:
                # Fallback if all values are zero
                return np.linspace(data.min(), data.max(), n_bins + 1)
        else:
            # Original IQR-based approach
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            
            # Determine range to focus on (IQR + some padding)
            if iqr > 0:
                # Focus on IQR ± 1.5*IQR (typical outlier definition)
                focus_range = 1.5 * iqr
                center_min = max(q1 - focus_range, data.min())
                center_max = min(q3 + focus_range, data.max())
            else:
                # If IQR is zero, focus on median ± small range
                median = np.median(data)
                center_min = max(median - 0.1, data.min())
                center_max = min(median + 0.1, data.max())
            
            # Allocate bins: 70% for center, 15% each for tails
            n_center = int(n_bins * 0.7)
            n_lower = int(n_bins * 0.15)
            n_upper = int(n_bins * 0.15)
            
            # Create bins
            bins_list = []
            
            # Lower tail
            if center_min > data.min():
                bins_list.append(np.linspace(data.min(), center_min, n_lower + 1)[:-1])
            
            # Center region (most bins)
            bins_list.append(np.linspace(center_min, center_max, n_center + 1))
            
            # Upper tail
            if center_max < data.max():
                bins_list.append(np.linspace(center_max, data.max(), n_upper + 1)[1:])
            
            # Combine and ensure sorted
            bins = np.concatenate(bins_list)
            bins = np.unique(np.sort(bins))
            
            # Ensure min and max are included
            bins[0] = data.min()
            bins[-1] = data.max()
            
            return bins
    else:
        # Standard linear bins
        return np.linspace(data.min(), data.max(), n_bins + 1)

def plot_distribution(ax, data, label, color, xlabel, unit="", use_log=False, use_symlog=False, 
                     adaptive_bins=False, focus_range=None, use_asinh=False):
    """Plot distribution with statistics.
    
    Args:
        focus_range: If provided, limit x-axis to this range (tuple of min, max).
                    Useful for data with outliers where you want to focus on central region.
        use_asinh: If True, use asinh (inverse hyperbolic sine) transform for axis.
                   Great for data centered around zero with outliers - compresses outliers
                   while preserving detail near zero. Works with negative values.
    """
    if len(data) == 0:
        ax.text(0.5, 0.5, f'No data for {label}', 
                transform=ax.transAxes, ha='center', va='center')
        return None
    
    # Filter out zeros and negatives for log scale
    if use_log:
        data_plot = data[data > 0]
        if len(data_plot) == 0:
            ax.text(0.5, 0.5, f'No positive data for {label}', 
                    transform=ax.transAxes, ha='center', va='center')
            return None
    else:
        data_plot = data
    
    stats = calculate_statistics(data)  # Use full data for stats
    
    # Histogram binning
    n_bins = min(100, max(30, len(data_plot) // 50))
    
    if use_asinh:
        # Use asinh transform: compresses outliers while preserving detail near zero
        # Fixed scale parameter: p0 = 5.0 GPa
        # This means [-5, 5] GPa is nearly linear, larger values get compressed
        p0 = 5.0
        data_transformed = np.arcsinh(data_plot / p0)
        
        # Plot histogram in transformed space
        counts, bins_edges_transformed, patches = ax.hist(data_transformed, bins=n_bins, 
                                                         alpha=0.6, color=color, 
                                                         edgecolor='black', linewidth=0.5, density=True)
        
        # Choose a small, readable tick set in original space
        tick_vals = np.array([-50, -20, -10, -5, 0, 5, 10, 20, 50])
        tick_vals = tick_vals[(tick_vals >= data_plot.min()) & (tick_vals <= data_plot.max())]
        
        # Transform ticks to transformed space
        tick_positions = np.arcsinh(tick_vals / p0)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{v:.0f}" for v in tick_vals])
        
        ax.set_xlim(data_transformed.min(), data_transformed.max())
        
    elif use_log:
        # Log scale: use log-spaced bins
        log_min = np.log10(data_plot.min())
        log_max = np.log10(data_plot.max())
        bins = np.logspace(log_min, log_max, n_bins)
        ax.set_xscale('log')
        counts, bins_edges, patches = ax.hist(data_plot, bins=bins, alpha=0.6, color=color, 
                                             edgecolor='black', linewidth=0.5, density=True)
    elif use_symlog:
        if adaptive_bins:
            # Use adaptive bins for better visualization near zero
            bins = create_adaptive_bins(data_plot, n_bins=n_bins, center_focus=True, zero_focus=True)
            # Still use symlog scale for axis, but with larger linthresh to reduce tick density
            # Use a larger linthresh to reduce number of ticks near zero
            linthresh = max(0.1, np.percentile(np.abs(data_plot), 30))
            ax.set_xscale('symlog', linthresh=linthresh, linscale=2.0)
            
            # Reduce tick density by limiting number of major ticks
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        else:
            # Symmetric log scale for data that can be negative
            linthresh = max(0.1, np.percentile(np.abs(data_plot), 10))
            ax.set_xscale('symlog', linthresh=linthresh, linscale=1.0)
            # Reduce tick density by limiting number of major ticks
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
            bins = n_bins
        counts, bins_edges, patches = ax.hist(data_plot, bins=bins, alpha=0.6, color=color, 
                                             edgecolor='black', linewidth=0.5, density=True)
    else:
        if adaptive_bins:
            bins = create_adaptive_bins(data_plot, n_bins=n_bins, center_focus=True, zero_focus=True)
        else:
            bins = n_bins
        counts, bins_edges, patches = ax.hist(data_plot, bins=bins, alpha=0.6, color=color, 
                                             edgecolor='black', linewidth=0.5, density=True)
    
    # Set x-axis limits if focus_range is provided (only for non-asinh plots)
    if focus_range is not None and not use_asinh:
        ax.set_xlim(focus_range)
        # Add text indicating data is zoomed
        n_outliers = np.sum((data_plot < focus_range[0]) | (data_plot > focus_range[1]))
        if n_outliers > 0:
            ax.text(0.98, 0.98, f'{n_outliers} outliers\noutside range', 
                   transform=ax.transAxes, ha='right', va='top',
                   fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add vertical lines for statistics (only if they're in the plotted range)
    if use_asinh:
        # Transform statistics to asinh space (use fixed scale p0 = 5.0)
        p0 = 5.0
        ax.axvline(np.arcsinh(stats['q10'] / p0), color='red', linestyle='--', linewidth=1.5, 
                  label=f"q10 = {stats['q10']:.4f} {unit}")
        ax.axvline(np.arcsinh(stats['q50'] / p0), color='blue', linestyle='--', linewidth=1.5, 
                  label=f"q50 (median) = {stats['q50']:.4f} {unit}")
        ax.axvline(np.arcsinh(stats['mean'] / p0), color='green', linestyle='--', linewidth=1.5, 
                  label=f"mean = {stats['mean']:.4f} {unit}")
        ax.axvline(np.arcsinh(stats['q75'] / p0), color='orange', linestyle='--', linewidth=1.5, 
                  label=f"q75 = {stats['q75']:.4f} {unit}")
        ax.axvline(np.arcsinh(stats['q90'] / p0), color='purple', linestyle='--', linewidth=1.5, 
                  label=f"q90 = {stats['q90']:.4f} {unit}")
    elif use_log:
        # For log scale, only show if positive
        if stats['q10'] > 0:
            ax.axvline(stats['q10'], color='red', linestyle='--', linewidth=1.5, 
                      label=f"q10 = {stats['q10']:.4f} {unit}")
        if stats['q50'] > 0:
            ax.axvline(stats['q50'], color='blue', linestyle='--', linewidth=1.5, 
                      label=f"q50 (median) = {stats['q50']:.4f} {unit}")
        if stats['mean'] > 0:
            ax.axvline(stats['mean'], color='green', linestyle='--', linewidth=1.5, 
                      label=f"mean = {stats['mean']:.4f} {unit}")
        if stats['q75'] > 0:
            ax.axvline(stats['q75'], color='orange', linestyle='--', linewidth=1.5, 
                      label=f"q75 = {stats['q75']:.4f} {unit}")
        if stats['q90'] > 0:
            ax.axvline(stats['q90'], color='purple', linestyle='--', linewidth=1.5, 
                      label=f"q90 = {stats['q90']:.4f} {unit}")
    else:
        # Regular scale
        ax.axvline(stats['q10'], color='red', linestyle='--', linewidth=1.5, 
                  label=f"q10 = {stats['q10']:.4f} {unit}")
        ax.axvline(stats['q50'], color='blue', linestyle='--', linewidth=1.5, 
                  label=f"q50 (median) = {stats['q50']:.4f} {unit}")
        ax.axvline(stats['mean'], color='green', linestyle='--', linewidth=1.5, 
                  label=f"mean = {stats['mean']:.4f} {unit}")
        ax.axvline(stats['q75'], color='orange', linestyle='--', linewidth=1.5, 
                  label=f"q75 = {stats['q75']:.4f} {unit}")
        ax.axvline(stats['q90'], color='purple', linestyle='--', linewidth=1.5, 
                  label=f"q90 = {stats['q90']:.4f} {unit}")
    
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=11)
    ax.set_ylabel('Probability Density', fontweight='bold', fontsize=11)
    #ax.set_title(label, fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-axis labels if needed to prevent overlap
    if use_symlog or (focus_range is not None) or use_asinh:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    return stats

def main():
    # Paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent.parent / "data" / "gen_10_data"
    output_dir = script_dir.parent.parent / "results" / "loss_function_development" / "misc_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading data from: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Read data
    data = read_xyz_files(data_dir)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for key, values in data.items():
        if len(values) > 0:
            stats = calculate_statistics(values)
            print(f"\n{key.upper().replace('_', ' ')}:")
            print(f"  Count:    {stats['count']}")
            print(f"  q10:      {stats['q10']:.6f}")
            print(f"  q50:      {stats['q50']:.6f} (median)")
            print(f"  mean:     {stats['mean']:.6f}")
            print(f"  q75:      {stats['q75']:.6f}")
            print(f"  q90:      {stats['q90']:.6f}")
            print(f"  kurtosis: {stats['kurtosis']:.6f}")
    
    # Create plots
    fig = plt.figure(figsize=(16, 12))
    # Subplot spacing parameters:
    #   hspace: vertical spacing between subplots (as fraction of average subplot height)
    #           Smaller values = less vertical spacing (default ~0.2-0.4)
    #   wspace: horizontal spacing between subplots (as fraction of average subplot width)
    #           Smaller values = less horizontal spacing (default ~0.2-0.4)
    # Tune these values to adjust spacing:
    gs = fig.add_gridspec(2, 2, hspace=0.15, wspace=0.15)
    
    # Force magnitudes (log scale)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_distribution(ax1, data['force_magnitudes'], 
                     'Force Magnitude Distribution', 
                     COLORS[0], 
                     'Force Magnitude (eV/Å)', 
                     unit="eV/Å",
                     use_log=True)
    
    # Per-atom energy
    ax2 = fig.add_subplot(gs[0, 1])
    plot_distribution(ax2, data['energies_per_atom'], 
                     'Per-Atom Energy Distribution', 
                     COLORS[1], 
                     'Energy per Atom (eV/atom)', 
                     unit="eV/atom")
    
    # Hydrostatic stress (use asinh transform to handle data centered around zero with outliers)
    ax3 = fig.add_subplot(gs[1, 0])
    hydrostatic_data = data['stress_hydrostatic']
    
    plot_distribution(ax3, hydrostatic_data, 
                     'Hydrostatic Pressure Distribution', 
                     COLORS[2], 
                     'Hydrostatic Pressure (GPa)', 
                     unit="GPa",
                     use_asinh=True)
    
    # Deviatoric stress (log scale)
    ax4 = fig.add_subplot(gs[1, 1])
    plot_distribution(ax4, data['stress_deviatoric'], 
                     'Deviatoric Stress (Frobenius Norm) Distribution', 
                     COLORS[3], 
                     'Deviatoric Stress Frobenius Norm (GPa)', 
                     unit="GPa",
                     use_log=True)
    
    #plt.suptitle('Distribution Analysis of Forces, Energies, and Stress Components', fontsize=14, fontweight='bold', y=0.995)
    
    output_file = output_dir / "force_energy_stress_distributions.png"
    # pad_inches: padding around the entire figure when saving
    #             Smaller values = tighter margins (default 0.1-0.2)
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.05, dpi=300)
    print(f"\nPlot saved to: {output_file}")
    plt.close()

if __name__ == "__main__":
    main()

