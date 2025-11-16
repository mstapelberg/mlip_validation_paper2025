# MLIP Validation Paper 2025

Repository for machine learning interatomic potential validation and analysis.

## Environment Setup

This repository uses two conda environments:

### 1. `forge_allegro_paper_env` (Repository-wide)

General-purpose environment for running FORGE workflows and analysis scripts across the repository.

**Setup:**
```bash
cd /home/myless/Packages/mlip_validation_paper2025
./setup_forge_environment.sh
conda activate forge_allegro_paper_env
```

**Includes:**
- PyTorch 2.7.1 (CUDA support by default)
- FORGE workflow management toolkit
- NequIP & Allegro (latest versions)
- WandB

See `FORGE_ENVIRONMENT_SETUP.md` for detailed instructions.

### 2. `custom_allegro_env` (Loss Function Development)

Specific environment for custom loss function models (RMCE, RMQE) in the loss function development scripts.

**Setup:**
```bash
cd /home/myless/Packages/mlip_validation_paper2025/scripts/loss_function_development
./setup_environment.sh
conda activate custom_allegro_env
```

**Includes:**
- PyTorch 2.7.1 (CUDA support by default)
- Custom NequIP fork (feature/L3_and_L4_loss)
- Allegro v0.6.0
- WandB 0.19.0

See `scripts/loss_function_development/ENVIRONMENT_SETUP.md` for detailed instructions.

## Repository Structure

- `scripts/` - Analysis and workflow scripts
- `data/` - Datasets and model files
- `results/` - Analysis results and outputs

