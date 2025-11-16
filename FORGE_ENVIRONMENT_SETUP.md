# FORGE-Allegro-Paper Environment Setup

This guide helps you set up the conda environment needed for using FORGE (Flexible Optimizer for Rapid Generation and Exploration) workflow management toolkit across all scripts in this repository.

## Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Git installed (for installing packages from GitHub)
- CUDA-enabled GPU recommended (for cuequivariance packages)

## Quick Setup

### Recommended: Automated Setup Script

The easiest way to set up the environment is using the provided setup script from the repository root:

**For CUDA support (default - CUDA 12.8):**
```bash
cd /home/myless/Packages/mlip_validation_paper2025
chmod +x setup_forge_environment.sh
./setup_forge_environment.sh
```

**For CPU-only PyTorch:**
```bash
./setup_forge_environment.sh cpu
```

**For different CUDA versions:**
```bash
./setup_forge_environment.sh cu118  # CUDA 11.8
./setup_forge_environment.sh cu128  # CUDA 12.8
```

The script will:
1. Create the conda environment `forge_allegro_paper_env`
2. Install PyTorch 2.7.1 with the appropriate CUDA/CPU support
3. Install cuequivariance packages (CUDA only)
4. Install FORGE from `mstapelberg/forge` (feature/config-aware-stress branch)
5. Install NequIP and Allegro (latest versions, compatible with any version)
6. Install WandB and all dependencies

### Manual Setup (Alternative)

If you prefer to set up manually:

1. Create the base environment:
   ```bash
   cd /home/myless/Packages/mlip_validation_paper2025
   conda env create -f environment_forge.yml
   conda activate forge_allegro_paper_env
   ```

2. Install PyTorch (choose one):
   ```bash
   # CUDA 12.8 (default)
   pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
   
   # CPU-only
   pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
   ```

3. Install CUDA packages (if using CUDA):
   ```bash
   # For CUDA 12.x
   pip install cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu12
   
   # For CUDA 11.x
   pip install cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu11
   ```

4. Install other packages:
   ```bash
   pip install wandb
   pip install nequip
   pip install allegro
   ```

5. Install FORGE:
   ```bash
   git clone --branch feature/config-aware-stress https://github.com/mstapelberg/forge.git
   cd forge
   pip install .
   ```
   
   **Note:** For development, use `pip install -e .` instead and keep the cloned directory.

## Verify Installation

After activation, verify the installation:

```bash
conda activate forge_allegro_paper_env

# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check FORGE
python -c "import forge; print('FORGE imported successfully')"

# Check NequIP
python -c "import nequip; print('NequIP imported successfully')"

# Check Allegro
python -c "import allegro; print('Allegro imported successfully')"

# Check WandB
python -c "import wandb; print(f'WandB: {wandb.__version__}')"
```

## Package Versions

This environment includes:
- **PyTorch 2.7.1** with CUDA 12.8 support (default) or CPU-only
- **FORGE** from custom fork: `mstapelberg/forge` (feature/config-aware-stress branch)
- **NequIP** - latest version (compatible with any version)
- **Allegro** - latest version (compatible with any version)
- **WandB** - latest version
- **cuequivariance** packages (CUDA only) - required by FORGE
- All their dependencies (installed automatically)

## Usage

This environment is available at the repository root and can be used for running FORGE workflows across all scripts in this repository. FORGE is a workflow management and materials analysis toolkit that works with NequIP and Allegro models.

Refer to the [FORGE repository](https://github.com/mstapelberg/forge/tree/feature/config-aware-stress) for usage examples and documentation.

## Troubleshooting

### Common Warnings

**Celery metadata warning**: You may see a warning like:
```
WARNING: Ignoring version 4.0.2 of celery since it has invalid metadata
```
This is **safe to ignore** - it's a known issue with older celery versions and newer pip. The installation will complete successfully. If you encounter actual errors (not just warnings), you can work around it by downgrading pip:
```bash
pip install "pip<24.1"
pip install nequip-allegro
```

### If installation fails

1. **GitHub access issues**: Make sure you have internet access and can reach GitHub
2. **CUDA version mismatch**: If you have a different CUDA version, adjust the CUDA version argument:
   ```bash
   ./setup_forge_environment.sh cu118  # For CUDA 11.8
   ```
3. **cuequivariance installation fails**: These packages require CUDA. If you're on CPU-only, they will be skipped automatically.
4. **Dependency conflicts**: Try creating a fresh environment:
   ```bash
   conda deactivate
   conda env remove -n forge_allegro_paper_env
   ./setup_forge_environment.sh
   ```

### Updating the environment

If you need to update packages:

```bash
conda activate forge_allegro_paper_env
conda env update -f environment_forge.yml --prune

# Update FORGE to latest from branch
FORGE_DIR=$(mktemp -d)
git clone --branch feature/config-aware-stress https://github.com/mstapelberg/forge.git $FORGE_DIR
cd $FORGE_DIR
pip install . --upgrade
cd -
rm -rf $FORGE_DIR
```

## Notes

- **NequIP and Allegro versions**: This environment installs the latest versions from PyPI. FORGE is compatible with any version, so you can update these independently if needed.
- **MACE**: The FORGE repository mentions MACE installation, but it's optional. Install it separately if needed for your workflows.
- **Repository-wide**: This environment is located at the repository root and is intended for use across all scripts in this repository.

