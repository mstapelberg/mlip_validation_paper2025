# Config-Aware Testing Framework

A comprehensive evaluation framework for comparing MLIP models across different training configurations, with a focus on configuration-aware stress metrics and elastic tensor analysis.

## Overview

This framework evaluates machine learning interatomic potentials (MLIPs) on a diverse set of configurations including bulk crystals, surfaces, point defects, NEB, phonons, elasticity, and more. It provides:

- **Config-aware stress metrics**: Uses appropriate stress metrics for different configuration types
- **Multiple model comparison**: Generations, loss variants, and hyperparameter groups
- **Elastic tensor integration**: C44 values and Born stability analysis
- **Reproducible plots**: Generate figures from precomputed results without model files

## What We Calculate

### Per-Structure Metrics

For each structure in the dataset, we compute:

| Metric | Description | Units |
|--------|-------------|-------|
| `e_pa_abs_meV` | Energy error per atom | meV |
| `f_rmse` | Force RMSE | eV/Å |
| `p_abs_GPa` | Pressure error | GPa |
| `sigma_rmse_GPa` | Full stress tensor RMSE | GPa |
| `von_mises_abs_GPa` | Von Mises error | GPa |
| `config_stress_metric_GPa` | Config-aware stress metric | GPa |

### Configuration Classification

Each structure is classified by:
- **Section**: Bulk crystals, Intermetallics, Surfaces, Point defects, NEB, Phonon, Elastic, Liquids, Composition explore, Other
- **Purity**: Pure (single element) or Alloy (multi-element)

### Config-Aware Stress Metrics

Different configuration types use different stress metrics:

| Section | Stress Metric | Pressure Metric |
|---------|--------------|----------------|
| Bulk crystals | Full stress (s_mean) | ✗ Ignored |
| Intermetallics | Full stress (s_mean) | ✗ Ignored |
| Surfaces & γ | ✗ Ignored | ✗ Ignored |
| Point defects | ✗ Ignored | Hydrostatic pressure (p_mean) |
| NEB | ✗ Ignored | ✗ Ignored |
| Phonon | ✗ Ignored | ✗ Ignored |
| Elastic | Full stress (s_mean) | ✗ Ignored |
| Liquids & explore | ✗ Ignored | Hydrostatic pressure (p_mean) |
| Composition explore | ✗ Ignored | Hydrostatic pressure (p_mean) |
| Other | ✗ Ignored | ✗ Ignored |

This ensures stress predictions are evaluated using the most relevant metric for each configuration type.

## Key Functions

### Main Analysis Script

**`streamlined_compare.py`**: Main entry point for all analyses.

**Key Functions:**

- **`evaluate_ensemble()`**: Runs model predictions on structures, averages across multiple seeds
- **`compute_metrics_for_dataset()`**: Calculates error metrics for each structure
- **`add_config_stress_metrics()`**: Adds config-aware stress metric column
- **`evaluate_by_generation()`**: Compares models across training generations
- **`evaluate_by_loss_variant()`**: Compares different loss function variants
- **`evaluate_by_loss_groups()`**: Compares multiple hyperparameter groups
- **`significance_tests_loss()`**: Runs paired permutation tests comparing CATW vs others
- **`save_precomputed_results()`**: Saves evaluation results to JSON
- **`load_precomputed_results()`**: Loads precomputed results from JSON

### Plotting Functions

- **`save_metric_panel()`**: 2×2 panel showing E, F, P, S metrics
- **`save_stress_compare_bars()`**: Grouped bars comparing all-stress vs config-aware metrics
- **`save_four_metric_horizontal_by_loss_variant()`**: Horizontal bar chart with 4 metrics + elasticity
- **`save_four_metric_vertical_by_loss_variant()`**: Vertical bar chart with 4 metrics + elasticity
- **`save_grouped_bar()`**: Grouped bars across categories

## Outputs

The framework generates several types of outputs:

### CSV Files

- `eval_per_structure_by_generation.csv`: Per-structure metrics by generation
- `eval_per_structure_by_loss.csv`: Per-structure metrics by loss variant
- `eval_per_structure_by_loss_groups.csv`: Per-structure metrics by hyperparameter group
- `loss_significance.csv`: Statistical significance tests
- `loss_significance_by_group.csv`: Per-group significance tests

### Plots

**Generation Analysis:**
- `panel_generations_metrics.png`: 2×2 panel for generations
- `generations_stress_comparison.png`: Stress metric comparison

**Loss Variant Analysis:**
- `panel_loss_variants_metrics.png`: 2×2 panel for variants
- `loss_variants_stress_comparison.png`: Stress comparison
- `loss_variants_four_metric_horizontal.png`: 4-metric + C44 horizontal chart
- `loss_variants_four_metric_vertical.png`: 4-metric + C44 vertical chart

**Loss Group Analysis:**
- `loss_groups_energy.png`: Energy comparison across groups
- `loss_groups_forces.png`: Force comparison
- `loss_groups_pressure.png`: Pressure comparison
- `loss_groups_stress.png`: Stress comparison
- `loss_groups_stress_comparison_by_group.png`: Stress by group
- `loss_groups_stress_comparison_by_variant.png`: Stress by variant

### Elasticity Integration

When `--elasticity_tensors` is provided, the framework:

- Extracts C44 values for each variant
- Checks Born mechanical stability (C44 > 0, C11 - C12 > 0, C11 > 0)
- Displays unstable models with red labels and asterisks
- Shows DFT and experimental reference lines
- Uses log scale for better visualization

## Usage Examples

### Mode 1: Precomputed Data (Default - No Models Needed)

This is the recommended mode for reviewers and reproduction. It uses precomputed evaluation results.

```bash
# Basic usage (auto-detects precomputed_data.json if present)
python streamlined_compare.py --outdir results/

# Explicit precomputed data path
python streamlined_compare.py \
    --precomputed-data precomputed_data.json \
    --outdir results/ \
    --elasticity_tensors allegro_elastic_tensors_summary.json
```

**Outputs**: All plots and CSV files for analysis.

### Mode 2: Direct Model Evaluation (Requires Models)

This mode evaluates models on your dataset using the model files.

```bash
python streamlined_compare.py \
    --xyz ../../data/fixed_test_global.xyz \
    --outdir results/ \
    --device cuda \
    --elasticity_tensors allegro_elastic_tensors_summary.json
```

**Models Used** (defined in script):
- `MODELS_BY_GENERATION`: Gen 0, 7, 10 models
- `MODELS_BY_LOSS`: MSE, MSETW, CA, CATW variants
- `MODELS_BY_LOSS_GROUPS`: Multiple hyperparameter groups

### Mode 3: Save Precomputed Results for Distribution

Generate the precomputed data file that others can use:

```bash
python streamlined_compare.py \
    --xyz ../../data/fixed_test_global.xyz \
    --outdir results/ \
    --save-precomputed precomputed_data.json \
    --elasticity_tensors allegro_elastic_tensors_summary.json
```

This will:
1. Run all evaluations
2. Generate plots
3. Save `precomputed_data.json` for distribution (typically 2-5 MB)

## Running with Your Own Models

To evaluate your own models, you need to modify the model path dictionaries in `streamlined_compare.py`:

### Step 1: Define Your Models

Edit the model dictionaries (lines 77-117):

```python
MODELS_BY_GENERATION = {
    0: [
        "path/to/your/gen0_seed0.nequip.pt2",
        "path/to/your/gen0_seed1.nequip.pt2",
        # ... more seeds
    ],
    7: [
        "path/to/your/gen7_seed0.nequip.pt2",
        # ...
    ],
}

MODELS_BY_LOSS = {
    "MSE": ["path/to/mse_model.nequip.pt2"],
    "MSETW": ["path/to/msetw_model.nequip.pt2"],
    "CA": ["path/to/ca_model.nequip.pt2"],
    "CATW": ["path/to/catw_model.nequip.pt2"],
}

MODELS_BY_LOSS_GROUPS = {
    "lmax1_nlayers2": {
        "MSE": ["path/to/mse_lmax1.nequip.pt2"],
        # ... other variants
    },
}
```

### Step 2: Run Evaluation

```bash
python streamlined_compare.py \
    --xyz your_dataset.xyz \
    --outdir results/ \
    --device cuda \
    --elasticity_tensors your_elasticity_tensors.json
```

### Step 3: (Optional) Save Results for Distribution

```bash
python streamlined_compare.py \
    --xyz your_dataset.xyz \
    --outdir results/ \
    --save-precomputed precomputed_data.json \
    --elasticity_tensors your_elasticity_tensors.json
```

## Inspecting Precomputed Data

To see what's in your precomputed data file:

```bash
python inspect_precomputed_data.py precomputed_data.json
```

This shows:
- Number of records per DataFrame
- Available columns
- Unique grouping values (generations, variants, groups)
- Statistical summaries of metrics

## File Structure

```
config_aware_testing/
├── streamlined_compare.py          # Main analysis script
├── inspect_precomputed_data.py     # Inspect precomputed data
├── precomputed_data.json           # Precomputed results (2-5 MB)
├── allegro_elastic_tensors_summary.json  # Elasticity data
├── results/                         # Output directory
│   ├── *.csv                       # Per-structure metrics
│   ├── *.png                       # Generated plots
└── README.md                        # This file
```

## Requirements

```bash
pip install pandas numpy matplotlib ase nequip
```

## Elasticity Tensor JSON Format

The elasticity tensor file should have this structure:

```json
{
  "allegro-mse": {
    "backend": "allegro-mse",
    "cubic_constants_gpa": {
      "C11": 237.89,
      "C12": 145.19,
      "C44": 12.33
    },
    "replicates": {
      "kept": 5,
      "dropped": 0,
      "details": [...]
    }
  },
  "DFT": {
    "cubic_constants_gpa": {
      "C44": 22.0
    }
  },
  "PREDEXP": {
    "cubic_constants_gpa": {
      "C44": 20.5
    }
  }
}
```

## Notes

- **Precomputed data includes**: Per-structure metrics only, not raw predictions (keeps file size ~2-5 MB)
- **Caching**: Results are cached to avoid re-evaluation
- **Device support**: CUDA or CPU via `--device` argument
- **Elasticity integration**: Automatic log scale and stability markers when elasticity data provided
- **Statistical tests**: Paired permutation tests with FDR correction for CATW vs other comparisons

## Troubleshooting

**Missing precomputed_data.json**: The script will attempt to load it if present in current directory. If evaluation is needed, use `--xyz` mode.

**Model file not found**: Check that model paths in `MODELS_BY_*` dictionaries are correct.

**CUDA out of memory**: Use `--device cpu` or reduce batch size by limiting structures with `--max-structures`.

**Plot generation fails**: Ensure matplotlib backend supports your display (useful for HPC: `export MPLBACKEND=Agg`).

