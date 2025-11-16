# Complete Guide

## Overview

You have 18 models (6 loss types × 3 seeds) trained on the same data.  
**Goal:** Determine which loss function produces better models for defect prediction.

---

## The 7-Option Pipeline

Run `./run_full_analysis.sh` and select:

| Option | What It Does | Environment | Time |
|--------|-------------|-------------|------|
| 0 | **Get training curves from wandb** | any | ~2-5 min |
| 1 | Standard models (MSE, RMSE, TailHuber, StratHuber) | forge_allegro_paper_env | ~15 min |
| 2 | Custom models (RMCE, RMQE) | custom_allegro_env | ~8 min |
| 3 | Combine results + ensemble plots | forge_allegro_paper_env | ~2 min |
| 4 | 3-region analysis (core/shell/bulk) | forge_allegro_paper_env | ~20 min |
| 5 | Deep statistics + publication figure | forge_allegro_paper_env | ~2 min |
| 6 | **Run all automatically** (includes option 0-5) | both | ~50 min |

**Recommended:** Option 6 for first complete run.

**Note:** Option 0 downloads training curves (convergence, stability) from wandb. Required for option 5 to include training analysis. Option 6 runs this automatically.

---

## What You Get

### After Option 3 (Basic Analysis)
**Location:** `../../results/loss_function_development/loss_comparison_ensemble/`

**Key file:** `ensemble_summary.csv`
```csv
loss_type    defect_mae_mean  defect_mae_std  ratio_mean
MSE          0.164           0.076           1.10
TailHuber    0.172           0.070           1.16
RMCE         0.320           0.121           1.35  ← Failed
RMQE         0.641           0.169           1.31  ← Failed
```

**Plots:**
- `ensemble_comparison.png` - Defect vs bulk with error bars
- `reproducibility_comparison.png` - Variance across seeds

### After Option 4 (3-Region Analysis)
**Location:** `../../results/loss_function_development/3region_analysis/`

**Key file:** `3region_summary.csv`  
Shows errors split by:
- **Core:** 8 nearest neighbors (direct interaction)
- **Shell:** Within 5 Å (indirect perturbation)  
- **Bulk:** Beyond 5 Å (should be ~zero due to locality!)

**Plot:** `3region_comparison.png`

### After Option 5 (Deep Analysis)
**Location:** `../../results/loss_function_development/combined_analysis/`

**Key files:**
- `statistical_tests.csv` - p-values for all comparisons
- `publication_figure.png` - Comprehensive multi-panel figure
- `training_analysis.png` - Shows RMCE/RMQE instability
- `error_distributions.png` - Mean, max, 95th percentile

---

## Interpreting Results

### What Ratio Means
**ratio = defect_mae / bulk_mae**

- **~1.0:** Errors balanced (model struggles equally everywhere)
- **>1.0:** Model worse at defects (common for all models)
- **Higher ratio:** Model finds defects harder (NOT better!)

### Statistical Significance
From `statistical_tests.csv`:
- **p < 0.05:** Difference is real
- **p > 0.05:** Could be random noise

If TailHuber vs MSE has p>0.05 → they're equivalent!

### 3-Region Insights
- **Low bulk errors** (>5 Å): Model respects locality ✓
- **High bulk errors** (>5 Å): Fundamental failure ✗
- **High core/bulk ratio:** Model recognizes defects are harder

---

## Paper Narratives (Choose Based on Data)

### Narrative A: Stability Comparison
> "All simple losses (MSE, RMSE, TailHuber) perform equivalently (p>0.05). Higher-order losses (RMCE, RMQE) fail catastrophically. **Recommendation:** Use proven MSE or theoretically-motivated TailHuber."

### Narrative B: Subtle Advantages (if 3-region shows it)
> "While mean errors are similar, TailHuber exhibits better bulk behavior (X vs Y eV/Å beyond 5 Å cutoff) and tighter error distributions."

### Narrative C: Methodology Contribution
> "We present a physics-aware testing framework (3-region split) that reveals subtle differences missed by standard metrics. Our negative results on RMCE/RMQE save the field from pursuing unstable approaches."

---

## Configuration

Edit `run_full_analysis.sh` lines 8-25 to customize:

```bash
DATA_PATH="../../data/good_atoms_objects_fixed.xyz"
N_STRUCTURES=20      # Reduce for quick tests
DEVICE="cpu"         # or "cuda"
CUTOFF=5.0           # Allegro cutoff
ENV_STANDARD="forge_allegro_paper_env"
ENV_CUSTOM="custom_allegro_env"
```

---

## Manual Commands (if not using run_full_analysis.sh)

### Get Training Curves
```bash
python get_wandb_training_curves.py \
  --project mnm-shortlab-mit/MLIP2025-Loss-Testing \
  --output training_history.csv
```

**Or use option 0 in the menu!**

---

## File Structure

```
loss_function_development/
├── run_full_analysis.sh              ← Main script (run this!)
├── models_config_standard.txt        ← Model paths (edit before running)
├── models_config_custom.txt          ← Model paths (edit before running)
├── README.md                         ← This file
├── GUIDE.md                          ← Detailed guide
│
├── Analysis Scripts:
│   ├── compare_loss_functions.py     ← Defect vs bulk errors
│   ├── analyze_3region_errors.py     ← Core/shell/bulk (5 Å cutoff)
│   ├── plot_ensemble_comparison.py   ← Error bars across seeds
│   └── plot_combined_analysis.py     ← Training + stats + publication fig
│
└── archive_old_approach/             ← Old "spatial sensitivity" approach (wrong!)
```

**Most users only need:** `run_full_analysis.sh` + edit model paths

---

## Summary

**For most users:** Just run `./run_full_analysis.sh` option 6

**The script handles:**
- Downloads training curves from wandb
- Auto-switches conda environments  
- Runs all 18 models
- Creates all plots and statistics
- Generates publication-ready figures

**You get:** Complete analysis showing accuracy + stability for all loss functions!

