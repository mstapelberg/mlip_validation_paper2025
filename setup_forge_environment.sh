#!/bin/bash
# Setup script for forge-allegro-paper environment
# This environment includes FORGE workflow management toolkit
# Located at repo root for use across all scripts

set -e  # Exit on error

ENV_NAME="forge_allegro_paper_env"
CUDA_VERSION="${1:-cu129}"  # Default to CUDA 12.9, can pass 'cpu', 'cu118', 'cu128', or 'cu129' as argument
PYTORCH_VERSION="${2:-2.8.0}"  # Default to 2.8.0, can pass version as second argument (e.g., 2.7.0)

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Creating conda environment for FORGE: $ENV_NAME"
echo "This environment includes:"
echo "  - PyTorch $PYTORCH_VERSION"
echo "  - CUDA: $CUDA_VERSION"
echo "  - FORGE (feature/config-aware-stress branch)"
echo "  - NequIP (any version - will install latest)"
echo "  - Allegro (any version - will install latest)"
echo "  - WandB"
echo ""
echo "Note: For Tesla V100 (compute capability 7.0), consider using CUDA 11.8 (cu118) or PyTorch 2.8.0+"
echo ""

# Create base conda environment
conda env create -f "$SCRIPT_DIR/environment_forge.yml" -n $ENV_NAME || {
    echo "Environment already exists. Updating..."
    conda env update -f "$SCRIPT_DIR/environment_forge.yml" -n $ENV_NAME --prune
}

# Activate environment
echo "Activating environment and installing packages..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install PyTorch with appropriate CUDA support
if [ "$CUDA_VERSION" = "cpu" ]; then
    echo "Installing PyTorch $PYTORCH_VERSION (CPU-only)..."
    pip install torch==$PYTORCH_VERSION --index-url https://download.pytorch.org/whl/cpu
    echo "Note: cuequivariance packages require CUDA. Skipping CUDA-specific packages for CPU installation."
else
    echo "Installing PyTorch $PYTORCH_VERSION with CUDA $CUDA_VERSION..."
    pip install torch==$PYTORCH_VERSION --index-url https://download.pytorch.org/whl/$CUDA_VERSION
    
    # Install cuequivariance packages (required by FORGE for CUDA)
    echo "Installing cuequivariance packages for CUDA..."
    if [[ "$CUDA_VERSION" == cu12* ]]; then
        pip install cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu12
    elif [[ "$CUDA_VERSION" == cu11* ]]; then
        pip install cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu11
    else
        echo "Warning: Unknown CUDA version $CUDA_VERSION. Attempting cu12 packages..."
        pip install cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu12
    fi
fi

# Install WandB
echo "Installing WandB..."
pip install wandb

# Install NequIP (any version - latest from PyPI)
echo "Installing NequIP (latest version)..."
pip install nequip

# Install Allegro (any version - latest from PyPI)
echo "Installing Allegro (latest version)..."
echo "Note: You may see a warning about celery 4.0.2 metadata - this is safe to ignore if installation completes."
pip install allegro

# Clone and install FORGE from the feature branch (editable install)
echo "Installing FORGE from feature/config-aware-stress branch..."
FORGE_DIR="$SCRIPT_DIR/data/forge"
if [ -d "$FORGE_DIR" ]; then
    echo "FORGE source directory exists, updating..."
    cd $FORGE_DIR
    git fetch origin
    git checkout feature/config-aware-stress
    git pull origin feature/config-aware-stress
else
    echo "Cloning FORGE repository to $FORGE_DIR..."
    mkdir -p "$SCRIPT_DIR/data"
    git clone --branch feature/config-aware-stress https://github.com/mstapelberg/forge.git $FORGE_DIR
    cd $FORGE_DIR
fi
pip install -e .
cd -

echo ""
echo "✓ Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To verify installation, run:"
echo "  python -c \"import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA arch list: {torch.cuda.get_arch_list()}')\""
echo "  python -c \"import forge; print('FORGE imported successfully')\""
echo "  python -c \"import nequip; print('NequIP imported successfully')\""
echo "  python -c \"import allegro; print('Allegro imported successfully')\""
echo "  python -c \"import wandb; print(f'WandB: {wandb.__version__}')\""
echo ""
echo "Usage:"
echo "  ./setup_forge_environment.sh [CUDA_VERSION] [PYTORCH_VERSION]"
echo "  Examples:"
echo "    ./setup_forge_environment.sh cu129 2.8.0    # CUDA 12.9, PyTorch 2.8.0 (default)"
echo "    ./setup_forge_environment.sh cu118 2.8.0    # CUDA 11.8, PyTorch 2.8.0 (recommended for V100)"
echo "    ./setup_forge_environment.sh cu128 2.8.0    # CUDA 12.8, PyTorch 2.8.0"
echo "    ./setup_forge_environment.sh cu128 2.7.0    # CUDA 12.8, PyTorch 2.7.0"
echo ""
echo "This environment is available for use across all scripts in this repository."

