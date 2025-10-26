# NEB Analysis and Plotting

This directory contains scripts for analyzing Nudged Elastic Band (NEB) calculations comparing VASP, MLIP Gen0, and MLIP Gen10 results.

## Quick Start (No MLIP Models Required)

To generate publication-quality figures without running any MLIP calculations:

```bash
# Generate the combined map + NEB + parity figure
python generate_figures.py --panel combined --include_parity --save map_plus_neb.png

# Generate NEB profiles only
python generate_figures.py --panel neb --save neb_only.png

# Generate parity plot only
python generate_figures.py --panel parity --save parity_only.png
```

The script automatically uses `summary_barriers.min.json` which contains all the necessary data for plotting.

## Files

- **`generate_figures.py`** - Main plotting script that creates composition maps, NEB profiles, and parity plots
- **`summary_barriers.min.json`** - Minimal dataset (≈3KB) with barrier energies and NEB profiles for all 6 compositions
- **`analyze_neb_results.py`** - Processes raw VASP/MLIP data (requires full dataset, not needed for plotting)
- **`calc_barriers.py`** - Runs MLIP calculations (requires models, not needed for plotting)

## Plotting Options

### Panels
- `--panel combined` - Map + NEB profiles (default)
- `--panel map` - Composition map only
- `--panel neb` - NEB trajectories only
- `--panel parity` - Parity plot only

### Map Transformations
- `--transform ilr` - Isometric log-ratio (default, recommended)
- `--transform clr` - Centered log-ratio
- `--transform raw` - Raw fractions

### Zoom Options
- `--zoom smart` - Tight zoom around data points (default)
- `--zoom manifold` - Include feasible polytope region
- `--zoom tight` - Very tight zoom
- `--zoom dataset` - Zoom to full dataset extent
- `--zoom none` - No zoom limits

### Other Options
- `--include_parity` - Add parity panel (for combined panel only)
- `--no_overlay_polytope` - Disable feasible region overlay on map
- `--max_generation_bg 10` - Background dataset generation cutoff

## Example Commands

```bash
# Full combined figure with parity
python generate_figures.py --panel combined --include_parity --save full_figure.png

# Map with manifold zoom showing feasible region
python generate_figures.py --panel map --zoom manifold --save map_manifold.png

# NEB trajectories with custom dimensions
python generate_figures.py --panel neb --row_height 2.0 --save neb_tall.png
```

## Data Structure

The `summary_barriers.min.json` file contains:
- 6 compositions (Cr5Ti5V115_17_to_23, Cr6Ti11V102W6Zr3_60_to_54, etc.)
- VASP reference barriers and NEB energy profiles
- MLIP Gen0 and Gen10 barriers and NEB profiles
- All data needed to reproduce the figures

## Requirements

- Python 3.7+
- matplotlib
- numpy
- pandas
- scikit-learn
- scipy (optional, for convex hull calculations)
- forge analysis library (optional, for dataset background)

## For Developers

To regenerate `summary_barriers.min.json` from raw data:

```bash
# Requires access to data/simple_data and simple_results_gen_*/ directories
python analyze_neb_results.py
```

This will create both `summary_barriers.csv` and `summary_barriers.min.json`.

