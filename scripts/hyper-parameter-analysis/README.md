# HPO Analysis Scripts

This directory contains scripts for analyzing hyperparameter optimization (HPO) results, combining training error metrics with inference throughput benchmarks to find Pareto-optimal model configurations.

## Overview

You have two main analysis scripts:

1. **`main_analysis_adapted.py`**: Pareto analysis at a fixed supercell size
2. **`scaling_analysis.py`**: Throughput scaling analysis and extrapolation to larger systems

## Data Structure

### Training Data (`hpo_training_results.csv`)
- Contains cross-validation folds and random seed replicates for each HPO configuration
- Key columns:
  - `Name`: Format like `hpo_107_num_layers-2_l_max-2_..._fold2_seed2`
  - Hyperparameters: `model.r_max`, `model.l_max`, `model.num_layers`, etc.
  - Test metrics: `test0_epoch/forces_rmse`, `test0_epoch/per_atom_energy_rmse`, etc.
  - Validation metrics: `val0_epoch/forces_rmse`, etc.

### Inference Data (`data/wandb_data/hpo_inference_test_analysis/`)
- Directory structure: `benchmark_results_hpo_XXX_compiled.nequip/`
- Each contains: `benchmark_summary_all_supercells.json`
- Supercell sizes tested: 3×3×3 (54 atoms) through 16×16×16 (8192 atoms)
- Throughput metrics:
  - `timesteps_per_s_log`: MD timesteps per second
  - `katom_steps_per_s_log`: kilo-atom-steps per second (accounts for system size)

## Quick Start

### 1. Main Pareto Analysis

```bash
cd scripts/hyper-parameter-analysis

# Basic usage (default: 5 timesteps/s requirement)
python main_analysis_adapted.py

# Highlight a specific model
python main_analysis_adapted.py --selected-model 53

# Custom throughput requirement
python main_analysis_adapted.py --throughput-requirement 10.0

# Both options together
python main_analysis_adapted.py --selected-model 53 --throughput-requirement 10.0
```

**CLI Arguments:**
- `--selected-model N`: Highlight model hpo_N on the plot (e.g., `--selected-model 53`)
- `--throughput-requirement T`: Set minimum throughput in timesteps/s (default: 5.0)

**What it does:**
- Loads training data and aggregates across CV folds/seeds
- Computes robust error estimates (95% confidence intervals)
- Loads throughput data for a specified supercell size (default: 8×8×8)
- Finds Pareto-optimal configurations (minimize error, maximize throughput)
- Identifies best feasible configuration given throughput requirements

**Configuration:**

Most settings can be controlled via CLI arguments. For advanced configuration, edit these in the script:

```python
# In main_analysis_adapted.py

ERROR_COL = "test0_epoch/forces_rmse"       # Primary metric to optimize
ALTERNATIVE_ERROR_COL = "test0_epoch/weighted_sum"  # Alternative metric
TARGET_SUPERCELL = "16x16x16"               # Which benchmark size to use
THROUGHPUT_METRIC = "timesteps_per_s_log"   # Throughput metric
```

**Color Scheme:**
The plots use a custom color palette:
- `#2A33C3` (Blue) - All configurations
- `#A35D00` (Orange) - Best model & throughput threshold  
- `#0B7285` (Teal) - Pareto front (all)
- `#8F2D56` (Pink/Magenta) - Selected model
- `#6E8B00` (Olive green) - Feasible Pareto front

**Outputs:**
- `results/hpo_analysis/hpo_pareto_analysis.png`: Visualization
- `results/hpo_analysis/hpo_merged_results.csv`: All configurations
- `results/hpo_analysis/hpo_feasible_pareto.csv`: Pareto front
- `results/hpo_analysis/hpo_best_choice.csv`: Best single configuration

### 2. Scaling Analysis

```bash
python scaling_analysis.py
```

**What it does:**
- Analyzes throughput scaling across all tested supercell sizes
- Fits power-law scaling models for each configuration
- Extrapolates to target system size (default: 250,000 atoms)
- Visualizes scaling curves for top configurations
- Creates Pareto plot with extrapolated throughput

**Key Configuration Variables:**

```python
# In scaling_analysis.py

TARGET_ATOMS = 250000  # Extrapolation target
```

**Outputs:**
- `results/hpo_analysis/throughput_scaling_analysis.png`: Scaling curves
- `results/hpo_analysis/pareto_extrapolated_throughput.png`: Error vs extrapolated throughput
- `results/hpo_analysis/scaling_extrapolation_results.csv`: Extrapolation results

## Important Considerations

### ⚠️ Extrapolation Warning

**Current situation:**
- Largest tested system: 8,192 atoms (16×16×16 supercell)
- Target deployment: 250,000 atoms
- **Extrapolation factor: ~30×**

This is a significant extrapolation! Recommendations:

1. **Validate with larger benchmarks**: Test top candidates at intermediate sizes (e.g., 32×32×32 ≈ 65k atoms)
2. **Check scaling fit quality**: Only trust extrapolations with R² > 0.95
3. **Memory constraints**: GPU memory scaling is not linear - may hit limits
4. **Compilation effects**: Torch compilation may behave differently at extreme scales

### Setting Throughput Requirements

You mentioned you need to determine `REQUIRED_T`. To help with this:

1. **Physical constraints**: What's your target simulation time?
   - Example: 1 ns trajectory = 1,000,000 timesteps (at 1 fs/step)
   - If you want this in 1 day (86,400 s), need: 1,000,000/86,400 ≈ 12 timesteps/s

2. **Practical constraints**: Available compute time, number of simulations needed

3. **Use scaling analysis**: Check what throughput you can expect at 250k atoms
   ```python
   # After running scaling_analysis.py
   results = pd.read_csv('results/hpo_analysis/scaling_extrapolation_results.csv')
   print(results['extrapolated_timesteps_per_s'].describe())
   ```

## Workflow Recommendation

1. **Run scaling analysis first** to understand performance at target scale:
   ```bash
   python scaling_analysis.py
   ```

2. **Identify realistic throughput requirement** from scaling results

3. **Update and run main analysis** with throughput requirement:
   ```python
   # In main_analysis_adapted.py
   REQUIRED_T = 10.0  # Example: based on scaling analysis
   ```
   ```bash
   python main_analysis_adapted.py
   ```

4. **Benchmark top candidates** at larger supercells to validate extrapolation

5. **Re-run analysis** with validated throughput data

## Customization

### Multi-objective Optimization

Currently optimizes forces RMSE vs throughput. To also consider energy/stress:

```python
# In main_analysis_adapted.py

# Modify to create composite error metric
merged["composite_error"] = (
    merged["test0_epoch/forces_rmse"] * 1.0 +      # Forces weight
    merged["test0_epoch/per_atom_energy_rmse"] * 0.5 +  # Energy weight  
    merged["test0_epoch/stress_rmse"] * 0.3        # Stress weight
)

# Then use in Pareto analysis
front_all = pareto_nondominated(merged, "composite_error", "thr_l95")
```

### Different Supercell Sizes

Available options for `TARGET_SUPERCELL`:
- `"3x3x3"`: 54 atoms
- `"4x4x4"`: 128 atoms
- `"5x5x5"`: 250 atoms
- `"6x6x6"`: 432 atoms
- `"7x7x7"`: 686 atoms
- `"8x8x8"`: 1024 atoms
- `"16x16x16"`: 8192 atoms

### Throughput Metric Choice

- `timesteps_per_s_log`: Best for fixed-size systems (what you have)
- `katom_steps_per_s_log`: Better for comparing across system sizes (efficiency metric)

For a fixed target of 250k atoms → use `timesteps_per_s_log`

## Questions?

Let me know if you need help with:
- Setting appropriate throughput requirements
- Modifying the analysis for different objectives
- Interpreting the scaling results
- Adding additional constraints (memory, etc.)

