#!/bin/bash
# Setup script for custom_allegro_env environment
# This environment includes custom versions of NequIP and Allegro
# Located in scripts/loss_function_development

set -e  # Exit on error

ENV_NAME="custom_allegro_env"
CUDA_VERSION="${1:-cu129}"  # Default to CUDA 12.9, can pass 'cpu' as argument
PYTORCH_VERSION="${2:-2.8.0}"  # Default to 2.8.0, can pass version as second argument

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Creating conda environment for custom Allegro: $ENV_NAME"
echo "This environment includes:"
echo "  - PyTorch $PYTORCH_VERSION"
echo "  - CUDA: $CUDA_VERSION"
echo "  - NequIP (feature/L3_and_L4_loss branch from mstapelberg/nequip)"
echo "  - Allegro (v0.6.0 from mir-group/allegro)"
echo "  - pandas"
echo "  - matplotlib"
echo ""

# Create base conda environment
conda env create -f "$SCRIPT_DIR/environment_custom_allegro.yml" -n $ENV_NAME || {
    echo "Environment already exists. Updating..."
    conda env update -f "$SCRIPT_DIR/environment_custom_allegro.yml" -n $ENV_NAME --prune
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
    
    # Install cuequivariance packages (required by NequIP/Allegro for CUDA)
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

# Setup cleanup function for temporary directories
NEQUIP_DIR=""
ALLEGRO_DIR=""
cleanup() {
    if [ -n "$NEQUIP_DIR" ] && [ -d "$NEQUIP_DIR" ]; then
        rm -rf "$NEQUIP_DIR"
    fi
    if [ -n "$ALLEGRO_DIR" ] && [ -d "$ALLEGRO_DIR" ]; then
        rm -rf "$ALLEGRO_DIR"
    fi
}
trap cleanup EXIT

# Clone and install NequIP from feature branch
echo "Installing NequIP from feature/L3_and_L4_loss branch..."
NEQUIP_DIR=$(mktemp -d)
git clone --branch feature/L3_and_L4_loss https://github.com/mstapelberg/nequip.git $NEQUIP_DIR
cd $NEQUIP_DIR
pip install .
cd -

# Clone and install Allegro v0.6.0
echo "Installing Allegro v0.6.0..."
ALLEGRO_DIR=$(mktemp -d)
git clone --branch v0.6.0 https://github.com/mir-group/allegro.git $ALLEGRO_DIR
cd $ALLEGRO_DIR
pip install .
cd -

echo ""
echo "✓ Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To verify installation, run:"
echo "  python -c \"import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA arch list: {torch.cuda.get_arch_list()}')\""
echo "  python -c \"import nequip; print('NequIP imported successfully')\""
echo "  python -c \"import allegro; print('Allegro imported successfully')\""
echo "  python -c \"import pandas; import matplotlib; print('pandas and matplotlib imported successfully')\""
echo ""
echo "Usage:"
echo "  ./setup_custom_allegro_environment.sh [CUDA_VERSION] [PYTORCH_VERSION]"
echo "  Examples:"
echo "    ./setup_custom_allegro_environment.sh cu129 2.8.0    # CUDA 12.9, PyTorch 2.8.0 (default)"
echo "    ./setup_custom_allegro_environment.sh cpu 2.8.0      # CPU-only, PyTorch 2.8.0"
echo ""

