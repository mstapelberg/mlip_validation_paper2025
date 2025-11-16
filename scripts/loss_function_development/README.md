# Loss Function Comparison for Defect Prediction

**Comprehensive evaluation of force loss functions for MLIP models on defect structures.**

This repository contains a complete analysis pipeline comparing 6 loss functions (MSE, RMSE, RMCE, RMQE, TailHuber, StratHuber) across 18 trained models (3 seeds each) to determine which produces better models for defect prediction in materials.

---

## Quick Start

```bash
cd /path/to/mlip_validation_paper2025/scripts/loss_function_development

# Run complete analysis (auto-switches environments, downloads training curves)
./run_full_analysis.sh
> Enter choice: 6
> Enter wandb project: mnm-shortlab-mit/MLIP2025-Loss-Testing

# Takes ~30-60 minutes, produces all figures and statistics
```

**That's it!** Option 6 runs everything automatically (including training stability analysis).

---

## Overview

### Research Question
Which force loss function produces MLIP models with the best accuracy on defect structures while maintaining bulk performance?

### Methodology
1. **Model Evaluation:** Test 18 trained models (6 loss types × 3 seeds) on vacancy/interstitial structures
2. **Error Analysis:** Compute force errors split by defect proximity (defect vs bulk, or core/shell/bulk)
3. **Ensemble Statistics:** Aggregate across seeds to compute mean ± std for each loss type
4. **Training Analysis:** Examine convergence and stability from wandb training curves
5. **Statistical Tests:** Perform t-tests to assess significance of differences

### Loss Functions Tested
- **MSE** (Mean Squared Error) - Standard baseline
- **RMSE** (Root Mean Squared Error) - Standard baseline
- **TailHuber** - Huber loss focusing on high-force tail (hypothesized to help with defects)
- **StratHuber** - Stratified Huber loss
- **RMCE** (Root Mean Cubic Error) - Higher-order loss (requires custom NequIP fork)
- **RMQE** (Root Mean Quartic Error) - Higher-order loss (requires custom NequIP fork)

---

## Pipeline Workflow

The `run_full_analysis.sh` script orchestrates the complete analysis:

```
0. Download training curves (wandb) → training_history.csv
1. Test 12 standard models           → defect/bulk errors (MSE, RMSE, TailHuber, StratHuber)
2. Test 6 custom models              → defect/bulk errors (RMCE, RMQE)
3. Combine + ensemble plots          → mean ± std with error bars across seeds
4. 3-region analysis                 → core/shell/bulk split (5 Å cutoff, physics-motivated)
5. Deep analysis                     → training curves + statistical tests + publication figures
```

**Key Features:**
- Automatic conda environment switching (`forge_allegro_paper_env` ↔ `custom_allegro_env`)
- Prediction caching to avoid redundant inference
- Physics-aware 3-region analysis respecting Allegro's 5 Å cutoff
- Publication-quality figures with statistical significance tests

---

## Codebase Structure

### Main Orchestration
- **`run_full_analysis.sh`** - Master script with 7 workflow options (0-6)
  - Handles environment switching, data flow, and error checking
  - Option 6 runs complete pipeline automatically

### Core Analysis Scripts
- **`compare_loss_functions.py`** - Evaluates models on defect structures
  - Computes per-atom force errors, splits into defect/bulk regions
  - Outputs: `detailed_results.csv` with per-structure metrics
  - Supports prediction caching for faster iteration

- **`analyze_3region_errors.py`** - Physics-motivated 3-region analysis
  - Core: 8 nearest neighbors (direct defect interaction)
  - Shell: Within 5 Å cutoff (indirect perturbation)
  - Bulk: Beyond 5 Å (should be unaffected due to locality)
  - Outputs: `3region_summary.csv`, `3region_comparison.png`

- **`plot_ensemble_comparison.py`** - Ensemble statistics visualization
  - Groups models by loss type (extracts from names like `MSE_seed1`)
  - Computes mean ± std across seeds
  - Creates bar charts with error bars

- **`plot_combined_analysis.py`** - Publication-quality combined analysis
  - Integrates training curves, test errors, distributions, statistical tests
  - Creates multi-panel publication figures
  - Outputs: `publication_figure.png`, `statistical_tests.csv`

### Data & Utilities
- **`get_wandb_training_curves.py`** - Downloads training history from wandb
  - Outputs: `training_history.csv` (used for training stability analysis)

- **`combine_results.py`** - Merges results from standard and custom model runs
  - Needed because models run in different conda environments

- **`load_filtered_data.py`** - Filters test dataset by config type and generation
  - Loads structures matching patterns (`vac*`, `neb*`, `sia*`) with gen ≤ 7

- **`loss_testing.py`** - Core utilities library
  - Vacancy detection, force/energy extraction, model predictor wrappers
  - Shared by multiple analysis scripts

- **`detect_defects.py`** - Defect detection utilities (vacancies, interstitials)
- **`cache_predictions.py`** - Prediction caching system (optional, speeds up iteration)
- **`average_ensemble_from_cache.py`** - Ensemble averaging from cached predictions

### Configuration
- **`models_config_standard.txt`** - Paths to standard models (MSE, RMSE, TailHuber, StratHuber)
- **`models_config_custom.txt`** - Paths to custom models (RMCE, RMQE)
- **`environment_custom_allegro.yml`** - Conda environment spec for custom NequIP fork
- **`setup_custom_allegro_environment.sh`** - Setup script for custom environment

### Documentation
- **`README.md`** - This file (quick start and overview)
- **`GUIDE.md`** - Complete reference with detailed explanations

---

## Results Location

After running the pipeline, results are saved to:

```
../../results/loss_function_development/
├── loss_comparison_ensemble/
│   ├── ensemble_summary.csv          ← Main results table (mean ± std per loss type)
│   ├── ensemble_comparison.png       ← Bar charts with error bars
│   └── reproducibility_comparison.png
├── 3region_analysis/
│   ├── 3region_summary.csv           ← Core/Shell/Bulk error breakdown
│   ├── 3region_comparison.png        ← Visualization
│   └── 3region_detailed_results.csv
└── combined_analysis/
    ├── publication_figure.png        ← Multi-panel publication figure
    ├── statistical_tests.csv         ← p-values for all comparisons
    ├── error_distributions.png       ← Mean, max, 95th percentile errors
    └── training_analysis.png         ← Training convergence curves (if available)
```

---

## Key Metrics

1. **Defect vs Bulk Error:** Mean absolute error (MAE) in eV/Å for atoms near defects vs far from defects
2. **3-Region Errors:** Core (8 NN), Shell (<5 Å), Bulk (>5 Å) - respects Allegro's locality
3. **Ensemble Statistics:** Mean ± std across 3 seeds (assesses reproducibility)
4. **Training Stability:** Convergence curves, final loss values, variance across seeds
5. **Statistical Significance:** t-tests and p-values comparing loss functions

---

## Dependencies & Environment Setup

### Standard Models (MSE, RMSE, TailHuber, StratHuber)
- Environment: `forge_allegro_paper_env` (standard NequIP/Allegro)
- Models: Standard Allegro models trained with different loss functions

### Custom Models (RMCE, RMQE)
- Environment: `custom_allegro_env` (custom NequIP fork with L3/L4 loss support)
- Setup: Run `./setup_custom_allegro_environment.sh` to create environment
- Models: Require custom NequIP fork from `mstapelberg/nequip` (feature/L3_and_L4_loss branch)

---

## Troubleshooting

**Environment names don't match?**  
Edit `run_full_analysis.sh` lines 24-25

**Want to use CPU instead of GPU?**  
Edit `run_full_analysis.sh` line 10: `DEVICE="cpu"`

**Custom environment not set up?**  
Run `./setup_custom_allegro_environment.sh [CUDA_VERSION] [PYTORCH_VERSION]`

**More help?**  
See `GUIDE.md` for detailed documentation

---

## Current Findings

Based on initial results:
- **MSE, RMSE, TailHuber, StratHuber:** Similar performance (~0.16-0.17 eV/Å defect MAE)
- **RMCE, RMQE:** Failed catastrophically (0.32-0.64 eV/Å, 2-4× worse, training instability)

**Interpretation:** Standard loss functions (MSE, RMSE) and Huber variants perform equivalently. Higher-order losses (RMCE, RMQE) are unstable and produce poor models.

**Next Steps:** Deep statistical analysis to identify subtle differences between MSE/RMSE/TailHuber/StratHuber and assess significance.
