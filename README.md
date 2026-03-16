# MLIP Validation Paper 2025

Repository for machine learning interatomic potential validation and analysis.

## Quick Start -- Reproducing Figures

Every plotting script shares a single style module (`scripts/plotting_utils.py`)
that sets colours, fonts, and figure sizes. No extra install is needed beyond the
conda environment below.

### 1. Create the environment

```bash
# from the repo root
./setup_forge_environment.sh
conda activate forge_allegro_paper_env
```

This installs Python 3.11, matplotlib, pandas, scikit-learn, seaborn, and all
other dependencies. See `FORGE_ENVIRONMENT_SETUP.md` if you hit issues.

### 2. Run any script

Each script is self-contained. `cd` into its directory and run it:

```bash
cd scripts/config_aware_testing
python streamlined_compare.py --precomputed-data precomputed_data.json

cd scripts/convergence_study
python convergence_analysis.py

cd scripts/neb
python generate_figures.py --include_parity

cd scripts/per_gen_analysis
python per_generation_analysis.py --replot --outdir ../../data/per_gen_out --parity 0 10

cd scripts/hyper-parameter-analysis
python main_analysis_adapted.py
```

Every figure is saved as **both** a 300 DPI PNG and a vector PDF in the same
output directory.

### 3. Colour palette

All plots use the **Tol Bright** palette (Paul Tol), which is colourblind-safe
for deuteranopia and protanopia. The seven colours are:

| Index | Hex       | Name    |
|-------|-----------|---------|
| 0     | `#4477AA` | blue    |
| 1     | `#EE6677` | red     |
| 2     | `#228833` | green   |
| 3     | `#CCBB44` | yellow  |
| 4     | `#66CCEE` | cyan    |
| 5     | `#AA3377` | purple  |
| 6     | `#BBBBBB` | grey    |

If you need to use the palette in your own code:

```python
from plotting_utils import TOL_BRIGHT, set_pub_style
set_pub_style()  # call once at the top of your script
```

## Environment Setup (detailed)

This repository uses two conda environments:

### `forge_allegro_paper_env` (repository-wide)

General-purpose environment for running FORGE workflows and all analysis/plotting
scripts.

```bash
./setup_forge_environment.sh
conda activate forge_allegro_paper_env
```

Includes: Python 3.11, PyTorch 2.7.1, FORGE, NequIP, Allegro, WandB,
matplotlib, seaborn, scikit-learn, and more.

See `FORGE_ENVIRONMENT_SETUP.md` for detailed instructions.

### `custom_allegro_env` (loss function development only)

Required only for scripts in `scripts/loss_function_development/` that use
custom loss functions (RMCE, RMQE).

```bash
cd scripts/loss_function_development
./setup_custom_allegro_environment.sh
conda activate custom_allegro_env
```

Includes: PyTorch 2.7.1, custom NequIP fork (`feature/L3_and_L4_loss`),
Allegro v0.6.0, WandB 0.19.0.

## Repository Structure

```
scripts/
    plotting_utils.py               <-- shared colours, fonts, save_fig()
    config_aware_testing/            <-- config-aware loss comparison
    convergence_study/               <-- DFT convergence (ENCUT, k-spacing)
    neb/                             <-- NEB barrier analysis + composition maps
    per_gen_analysis/                <-- per-generation active learning curves
    hyper-parameter-analysis/        <-- HPO Pareto analysis
    loss_function_development/       <-- loss function comparison & 3-region analysis
data/                                <-- datasets, model files, precomputed results
results/                             <-- analysis outputs (CSV, PNG, PDF)
```

