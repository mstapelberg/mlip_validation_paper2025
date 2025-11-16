#!/bin/bash
# Full Analysis Pipeline for Loss Function Comparison
# Run all 18 models (6 loss types × 3 seeds) and create ensemble plots

set -e  # Exit on error

# Configuration
DATA_PATH="../../data/fixed_test_global.xyz"  # Full test set with vac*, neb*, sia* from gen <= 7
N_STRUCTURES=750  # Use more structures for better statistics (will be filtered to vac*/neb*/sia* gen<=7)
DEVICE="cuda"

# Output directories
OUT_STANDARD="../../results/loss_function_development/loss_comparison_standard"
OUT_CUSTOM="../../results/loss_function_development/loss_comparison_custom"
OUT_COMBINED="../../results/loss_function_development/loss_comparison_all"
OUT_ENSEMBLE="../../results/loss_function_development/loss_comparison_ensemble"
OUT_3REGION="../../results/loss_function_development/3region_analysis"
OUT_COMBINED_ANALYSIS="../../results/loss_function_development/combined_analysis"

# Allegro cutoff (5 Å for standard Allegro models)
CUTOFF=5.0

# Conda environment names
ENV_STANDARD="forge_allegro_paper_env"
ENV_CUSTOM="custom_allegro_env"

echo "=========================================="
echo "LOSS FUNCTION COMPARISON PIPELINE"
echo "=========================================="
echo ""
echo "Select workflow:"
echo ""
echo "Setup:"
echo "  0. Get Training Data - Download training curves from wandb (run first!)"
echo ""
echo "Basic Analysis:"
echo "  1. forge_allegro    - Standard models (MSE, RMSE, StratHuber, TailHuber)"
echo "  2. custom_allegro   - Custom models (RMCE, RMQE)"
echo "  3. Combine & Plot   - Combine results from steps 1 & 2, create ensemble plots"
echo ""
echo "Deep Analysis (requires completed basic analysis):"
echo "  4. 3-Region Split   - Physics-aware core/shell/bulk analysis (5 Å cutoff)"
echo "  5. Combined Stats   - Training curves + statistical tests + publication figure"
echo ""
echo "Complete Pipeline:"
echo "  6. Run All          - Steps 0, 1, 2, 3, 4, 5 in sequence (fully automated!)"
echo ""
echo "Configuration:"
echo "  Data: $DATA_PATH"
echo "  Structures: $N_STRUCTURES"
echo "  Device: $DEVICE"
echo "  Cutoff: $CUTOFF Å"
echo ""
read -p "Enter choice (0-6): " -n 1 -r
echo
WORKFLOW=$REPLY

if [[ ! $WORKFLOW =~ ^[0123456]$ ]]; then
    echo "Invalid choice. Aborted."
    exit 1
fi

# Parse model configs into command-line arguments
parse_config() {
    local config_file=$1
    local models_arg=""
    
    while IFS=':' read -r name path || [ -n "$name" ]; do
        # Skip comments and empty lines
        [[ "$name" =~ ^#.*$ ]] && continue
        [[ -z "$name" ]] && continue
        
        # Trim whitespace
        name=$(echo "$name" | xargs)
        path=$(echo "$path" | xargs)
        
        models_arg="$models_arg \"$name:$path\""
    done < "$config_file"
    
    echo "$models_arg"
}

# ==========================================
# Workflow Functions
# ==========================================

get_training_data() {
    echo ""
    echo "=========================================="
    echo "DOWNLOADING TRAINING CURVES FROM WANDB"
    echo "=========================================="
    echo ""
    echo "This downloads training history (loss, validation metrics) from wandb."
    echo "Required for option 5 (training stability analysis)."
    echo ""
    
    # Check if training_history.csv already exists
    if [[ -f "training_history.csv" ]]; then
        echo "WARNING: training_history.csv already exists!"
        echo "   File size: $(du -h training_history.csv | cut -f1)"
        read -p "Re-download? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Keeping existing file. Skipped."
            echo ""
            echo "File ready: training_history.csv"
            echo "This will be used by option 5 (Combined Stats) for training analysis."
            # Function completes normally - don't return early
        else
            echo "Re-downloading..."
            # Continue to download section below
            # Prompt for wandb project (with default)
            read -p "Enter wandb project [mnm-shortlab-mit/MLIP2025-Loss-Testing]: " WANDB_PROJECT
            
            # Use default if empty
            if [[ -z "$WANDB_PROJECT" ]]; then
                WANDB_PROJECT="mnm-shortlab-mit/MLIP2025-Loss-Testing"
                echo "Using default project: $WANDB_PROJECT"
            fi
            
            echo ""
            echo "Downloading training history from: $WANDB_PROJECT"
            echo "This may take a few minutes..."
            echo ""
            
            # Download all runs (disable exit on error for this command)
            set +e
            python get_wandb_training_curves.py \
                --project "$WANDB_PROJECT" \
                --output training_history.csv
            PYTHON_EXIT=$?
            set -e
            
            if [[ $PYTHON_EXIT -ne 0 ]]; then
                echo ""
                echo "ERROR: Failed to download training history!"
                echo "Exit code: $PYTHON_EXIT"
                return 1
            fi
            
            echo ""
            echo "Training history saved to: training_history.csv"
            echo "  File size: $(du -h training_history.csv | cut -f1)"
        fi
    else
        # File doesn't exist, proceed with download
        # Prompt for wandb project (with default)
        read -p "Enter wandb project [mnm-shortlab-mit/MLIP2025-Loss-Testing]: " WANDB_PROJECT
        
        # Use default if empty
        if [[ -z "$WANDB_PROJECT" ]]; then
            WANDB_PROJECT="mnm-shortlab-mit/MLIP2025-Loss-Testing"
            echo "Using default project: $WANDB_PROJECT"
        fi
        
        echo ""
        echo "Downloading training history from: $WANDB_PROJECT"
        echo "This may take a few minutes..."
        echo ""
        
        # Download all runs (disable exit on error for this command)
        set +e
        python get_wandb_training_curves.py \
            --project "$WANDB_PROJECT" \
            --output training_history.csv
        PYTHON_EXIT=$?
        set -e
        
        if [[ $PYTHON_EXIT -ne 0 ]]; then
            echo ""
            echo "ERROR: Failed to download training history!"
            echo "Exit code: $PYTHON_EXIT"
            return 1
        fi
        
        echo ""
        echo "Training history saved to: training_history.csv"
        echo "  File size: $(du -h training_history.csv | cut -f1)"
    fi
    
    echo ""
    echo "This file will be used by option 5 (Combined Stats) to show:"
    echo "  - Training convergence"
    echo "  - Validation curves"
    echo "  - Loss stability across seeds"
    echo "  - Final convergence values"
    
    # Ensure function returns successfully
    return 0
}

run_standard_models() {
    echo ""
    echo "=========================================="
    echo "ANALYZING STANDARD MODELS (forge_allegro)"
    echo "=========================================="
    echo "Environment: forge-allegro (standard nequip)"
    echo "Models: MSE, RMSE, StratHuber, TailHuber"
    echo ""
    
    # Parse config
    MODELS_STANDARD=$(parse_config models_config_standard.txt)
    
    # Run analysis
    eval python compare_loss_functions.py \
        --data_path "$DATA_PATH" \
        --models $MODELS_STANDARD \
        --n_structures $N_STRUCTURES \
        --device $DEVICE \
        --output_dir "$OUT_STANDARD" \
        --environment standard
    
    echo ""
    echo "Standard models complete!"
}

run_custom_models() {
    echo ""
    echo "=========================================="
    echo "ANALYZING CUSTOM MODELS (custom_allegro)"
    echo "=========================================="
    echo ""
    echo "️  IMPORTANT: Custom models require a different conda environment!"
    echo ""
    echo "Please activate your custom NequIP environment now."
    echo "For example:"
    echo "  conda activate <your_custom_nequip_env_name>"
    echo ""
    read -p "Press Enter AFTER you've activated the custom environment (or Ctrl+C to cancel)..."
    echo ""
    
    # Check if we're in the right environment (optional check)
    if ! command -v python &> /dev/null; then
        echo "WARNING: python command not found. Make sure your environment is activated."
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 1
        fi
    fi
    
    # Parse config
    MODELS_CUSTOM=$(parse_config models_config_custom.txt)
    
    # Run analysis
    eval python compare_loss_functions.py \
        --data_path "$DATA_PATH" \
        --models $MODELS_CUSTOM \
        --n_structures $N_STRUCTURES \
        --device $DEVICE \
        --output_dir "$OUT_CUSTOM" \
        --environment custom
    
    echo ""
    echo "Custom models complete!"
}

combine_and_plot() {
    echo ""
    echo "=========================================="
    echo "COMBINING RESULTS"
    echo "=========================================="
    echo ""
    
    python combine_results.py \
        --dirs "$OUT_STANDARD" "$OUT_CUSTOM" \
        --output "$OUT_COMBINED"
    
    echo ""
    echo "Results combined!"
    
    echo ""
    echo "=========================================="
    echo "CREATING ENSEMBLE PLOTS"
    echo "=========================================="
    echo ""
    
    python plot_ensemble_comparison.py \
        --detailed_results "$OUT_COMBINED/detailed_results.csv" \
        --output "$OUT_ENSEMBLE"
    
    echo ""
    echo "Ensemble plots created!"
}

run_3region_analysis() {
    echo ""
    echo "=========================================="
    echo "3-REGION ANALYSIS (Core/Shell/Bulk)"
    echo "=========================================="
    echo ""
    echo "Physics-motivated split based on $CUTOFF Å Allegro cutoff:"
    echo "  Core:  8 nearest neighbors (direct vacancy interaction)"
    echo "  Shell: Within $CUTOFF Å (indirect perturbation)"
    echo "  Bulk:  Beyond $CUTOFF Å (should be unaffected by locality)"
    echo ""
    
    # Check if we have results to analyze
    if [[ ! -f "$OUT_COMBINED/detailed_results.csv" ]]; then
        echo "️  Error: Combined results not found!"
        echo "   Please run option 3 (Combine & Plot) first."
        exit 1
    fi
    
    # Parse all models from combined config files
    echo "Collecting models from configs..."
    MODELS_ALL=$(parse_config models_config_standard.txt)
    MODELS_CUSTOM=$(parse_config models_config_custom.txt)
    MODELS_ALL="$MODELS_ALL $MODELS_CUSTOM"
    
    # Run 3-region analysis
    eval python analyze_3region_errors.py \
        --data_path "$DATA_PATH" \
        --models $MODELS_ALL \
        --cutoff $CUTOFF \
        --n_structures $N_STRUCTURES \
        --device $DEVICE \
        --output_dir "$OUT_3REGION"
    
    echo ""
    echo "3-region analysis complete!"
    echo "  Results: $OUT_3REGION/3region_comparison.png"
    echo "           $OUT_3REGION/3region_summary.csv"
}

run_combined_analysis() {
    echo ""
    echo "=========================================="
    echo "COMBINED DEEP ANALYSIS"
    echo "=========================================="
    echo ""
    echo "Creating comprehensive analysis with:"
    echo "  - Training curves (convergence, stability)"
    echo "  - Error distributions (mean, max, 95th percentile)"
    echo "  - Statistical significance tests"
    echo "  - Publication-quality combined figure"
    echo ""
    
    # Check if we have ensemble results
    if [[ ! -f "$OUT_ENSEMBLE/ensemble_summary.csv" ]]; then
        echo "ERROR: Ensemble summary not found!"
        echo "   Please run option 3 (Combine & Plot) first."
        exit 1
    fi
    
    # Check for training history
    if [[ ! -f "training_history.csv" ]]; then
        echo "WARNING: training_history.csv not found in current directory."
        echo "   Training curves will be skipped."
        read -p "Continue without training analysis? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 1
        fi
        TRAINING_ARG=""
    else
        TRAINING_ARG="--training_history training_history.csv"
    fi
    
    # Run combined analysis
    python plot_combined_analysis.py \
        --ensemble_summary "$OUT_ENSEMBLE/ensemble_summary.csv" \
        --detailed_results "$OUT_COMBINED/detailed_results.csv" \
        $TRAINING_ARG \
        --output "$OUT_COMBINED_ANALYSIS"
    
    echo ""
    echo "Combined analysis complete!"
    echo "  Results: $OUT_COMBINED_ANALYSIS/publication_figure.png"
    echo "           $OUT_COMBINED_ANALYSIS/error_distributions.png"
    echo "           $OUT_COMBINED_ANALYSIS/statistical_tests.csv"
}

# ==========================================
# Execute Selected Workflow
# ==========================================

if [[ $WORKFLOW == "0" ]]; then
    # Option 0: Get training data from wandb
    get_training_data
    # Don't show "ANALYSIS COMPLETE" for option 0 - it's just data prep
    SKIP_COMPLETION_MSG=true

elif [[ $WORKFLOW == "1" ]]; then
    # Option 1: Standard models only
    run_standard_models
    
elif [[ $WORKFLOW == "2" ]]; then
    # Option 2: Custom models only
    run_custom_models
    
elif [[ $WORKFLOW == "3" ]]; then
    # Option 3: Combine and plot results
    echo ""
    echo "=========================================="
    echo "COMBINING & PLOTTING RESULTS"
    echo "=========================================="
    echo ""
    echo "This will combine results from:"
    echo "  - Standard models: $OUT_STANDARD"
    echo "  - Custom models:   $OUT_CUSTOM"
    echo ""
    
    # Check if required directories exist
    if [[ ! -d "$OUT_STANDARD" ]]; then
        echo "️  Warning: Standard models directory not found: $OUT_STANDARD"
        echo "   Please run option 1 (forge_allegro) first."
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 1
        fi
    fi
    
    if [[ ! -d "$OUT_CUSTOM" ]]; then
        echo "️  Warning: Custom models directory not found: $OUT_CUSTOM"
        echo "   Please run option 2 (custom_allegro) first."
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 1
        fi
    fi
    
    combine_and_plot

elif [[ $WORKFLOW == "4" ]]; then
    # Option 4: 3-region analysis
    run_3region_analysis

elif [[ $WORKFLOW == "5" ]]; then
    # Option 5: Combined deep analysis
    run_combined_analysis

elif [[ $WORKFLOW == "6" ]]; then
    # Option 6: Run everything with automatic environment switching
    echo ""
    echo "=========================================="
    echo "COMPLETE PIPELINE (AUTOMATIC)"
    echo "=========================================="
    echo ""
    echo "This will run the complete analysis pipeline with automatic environment switching:"
    echo "  0. Download training curves from wandb"
    echo "  1. Standard models → $ENV_STANDARD"
    echo "  2. Custom models → $ENV_CUSTOM"
    echo "  3. Combine & ensemble plots → $ENV_STANDARD"
    echo "  4. 3-region analysis → $ENV_STANDARD"
    echo "  5. Combined deep analysis (with training curves) → $ENV_STANDARD"
    echo ""
    echo "️  NOTE: This requires conda environments to be set up!"
    echo "   Standard: $ENV_STANDARD"
    echo "   Custom:   $ENV_CUSTOM"
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
    
    # Step 0: Get training data (if not already present)
    echo ""
    echo "=========================================="
    echo "STEP 0: TRAINING DATA"
    echo "=========================================="
    echo ""
    
    if [[ ! -f "training_history.csv" ]]; then
        echo "training_history.csv not found. Downloading from wandb..."
        echo ""
        read -p "Enter wandb project [mnm-shortlab-mit/MLIP2025-Loss-Testing]: " WANDB_PROJECT
        
        # Use default if empty
        if [[ -z "$WANDB_PROJECT" ]]; then
            WANDB_PROJECT="mnm-shortlab-mit/MLIP2025-Loss-Testing"
            echo "Using default project: $WANDB_PROJECT"
        fi
        
        python get_wandb_training_curves.py \
            --project "$WANDB_PROJECT" \
            --output training_history.csv
        echo "Step 0 complete!"
    else
        echo "training_history.csv already exists ($(du -h training_history.csv | cut -f1))"
        echo "Step 0 skipped (using existing file)"
    fi
    
    # Step 1: Standard models
    echo ""
    echo "=========================================="
    echo "STEP 1: STANDARD MODELS"
    echo "=========================================="
    echo "Environment: $ENV_STANDARD"
    echo ""
    
    MODELS_STANDARD=$(parse_config models_config_standard.txt)
    
    # Use explicit PYTHONUNBUFFERED for unbuffered output
    export PYTHONUNBUFFERED=1
    echo "Running: compare_loss_functions.py with $(echo $MODELS_STANDARD | wc -w) models"
    eval conda run -n $ENV_STANDARD python -u compare_loss_functions.py \
        --data_path "$DATA_PATH" \
        --models $MODELS_STANDARD \
        --n_structures $N_STRUCTURES \
        --device $DEVICE \
        --output_dir "$OUT_STANDARD" 2>&1 | tee /tmp/step1_output.log
    
    echo "Step 1 complete!"
    
    # Step 2: Custom models (automatic environment switch)
    echo ""
    echo "=========================================="
    echo "STEP 2: CUSTOM MODELS"
    echo "=========================================="
    echo "Environment: $ENV_CUSTOM (auto-switching...)"
    echo ""
    
    MODELS_CUSTOM=$(parse_config models_config_custom.txt)
    
    # Use explicit PYTHONUNBUFFERED for unbuffered output
    export PYTHONUNBUFFERED=1
    echo "Running: compare_loss_functions.py with $(echo $MODELS_CUSTOM | wc -w) models"
    eval conda run -n $ENV_CUSTOM python -u compare_loss_functions.py \
        --data_path "$DATA_PATH" \
        --models $MODELS_CUSTOM \
        --n_structures $N_STRUCTURES \
        --device $DEVICE \
        --output_dir "$OUT_CUSTOM" 2>&1 | tee /tmp/step2_output.log
    
    echo "Step 2 complete!"
    
    # Step 3: Combine and plot (back to standard env)
    echo ""
    echo "=========================================="
    echo "STEP 3: COMBINE & ENSEMBLE"
    echo "=========================================="
    echo "Environment: $ENV_STANDARD"
    echo ""
    
    conda run -n $ENV_STANDARD python combine_results.py \
        --dirs "$OUT_STANDARD" "$OUT_CUSTOM" \
        --output "$OUT_COMBINED"
    
    conda run -n $ENV_STANDARD python plot_ensemble_comparison.py \
        --detailed_results "$OUT_COMBINED/detailed_results.csv" \
        --output "$OUT_ENSEMBLE"
    
    echo "Step 3 complete!"
    
    # Step 4: 3-region analysis
    echo ""
    echo "=========================================="
    echo "STEP 4: 3-REGION ANALYSIS"
    echo "=========================================="
    echo ""
    
    MODELS_ALL="$MODELS_STANDARD $MODELS_CUSTOM"
    
    eval conda run -n $ENV_STANDARD python analyze_3region_errors.py \
        --data_path "$DATA_PATH" \
        --models $MODELS_ALL \
        --cutoff $CUTOFF \
        --n_structures $N_STRUCTURES \
        --device $DEVICE \
        --output_dir "$OUT_3REGION"
    
    echo "Step 4 complete!"
    
    # Step 5: Combined deep analysis
    echo ""
    echo "=========================================="
    echo "STEP 5: DEEP STATISTICAL ANALYSIS"
    echo "=========================================="
    echo ""
    
    TRAINING_ARG=""
    if [[ -f "training_history.csv" ]]; then
        TRAINING_ARG="--training_history training_history.csv"
    else
        echo "WARNING: training_history.csv not found, skipping training curves"
    fi
    
    conda run -n $ENV_STANDARD python plot_combined_analysis.py \
        --ensemble_summary "$OUT_ENSEMBLE/ensemble_summary.csv" \
        --detailed_results "$OUT_COMBINED/detailed_results.csv" \
        $TRAINING_ARG \
        --output "$OUT_COMBINED_ANALYSIS"
    
    echo "Step 5 complete!"
    
    # Done!
    echo ""
    echo "=========================================="
    echo "COMPLETE PIPELINE FINISHED!"
    echo "=========================================="
    echo ""
    echo "All results saved to:"
    echo "  Basic:       $OUT_ENSEMBLE"
    echo "  3-Region:    $OUT_3REGION"
    echo "  Deep Stats:  $OUT_COMBINED_ANALYSIS"
    echo ""
    echo "Key files to check:"
    echo "  $OUT_ENSEMBLE/ensemble_summary.csv"
    echo "  $OUT_3REGION/3region_comparison.png"
    echo "  $OUT_COMBINED_ANALYSIS/publication_figure.png"
    echo "  $OUT_COMBINED_ANALYSIS/statistical_tests.csv"
    echo ""
fi

# ==========================================
# Done!
# ==========================================
if [[ $WORKFLOW != "6" && "$SKIP_COMPLETION_MSG" != "true" ]]; then
    echo ""
    echo "=========================================="
    echo "ANALYSIS COMPLETE!"
    echo "=========================================="
    echo ""
    echo "Results locations:"
    if [[ $WORKFLOW == "0" ]]; then
        echo "  Training data: training_history.csv"
        echo ""
        echo "Next steps:"
        echo "  1. Run option 1 (forge_allegro)"
        echo "  2. Run option 2 (custom_allegro)"
        echo "  3. Run option 3 (Combine & Plot)"
        echo "  4. Run option 4 (3-Region)"
        echo "  5. Run option 5 (Combined Stats) - will use this training data!"
        
    elif [[ $WORKFLOW == "1" ]]; then
        echo "  Standard models:  $OUT_STANDARD"
        echo ""
        echo "Next steps:"
        echo "  1. Run option 2 (custom_allegro) in custom environment"
        echo "  2. Run option 3 (Combine & Plot)"
        echo "  3. Run option 4 (3-Region) for physics-aware analysis"
        echo "  4. Run option 5 (Combined Stats) for publication figures"
        
    elif [[ $WORKFLOW == "2" ]]; then
        echo "  Custom models:    $OUT_CUSTOM"
        echo ""
        echo "Next steps:"
        echo "  1. Return to standard environment"
        echo "  2. Run option 3 (Combine & Plot)"
        echo "  3. Run option 4 (3-Region) for physics-aware analysis"
        echo "  4. Run option 5 (Combined Stats) for publication figures"
        
    elif [[ $WORKFLOW == "3" ]]; then
        echo "  Combined:         $OUT_COMBINED"
        echo "  Ensemble plots:   $OUT_ENSEMBLE"
        echo ""
        echo "Key outputs:"
        echo "  $OUT_ENSEMBLE/ensemble_comparison.png"
        echo "  $OUT_ENSEMBLE/reproducibility_comparison.png"
        echo "  $OUT_ENSEMBLE/ensemble_summary.csv"
        echo ""
        echo "Next steps (optional deep analysis):"
        echo "  4. Run option 4 (3-Region) for core/shell/bulk split"
        echo "  5. Run option 5 (Combined Stats) for statistical tests"
        
    elif [[ $WORKFLOW == "4" ]]; then
        echo "  3-Region analysis: $OUT_3REGION"
        echo ""
        echo "Key outputs:"
        echo "  $OUT_3REGION/3region_comparison.png"
        echo "  $OUT_3REGION/3region_summary.csv"
        echo "  $OUT_3REGION/3region_detailed_results.csv"
        echo ""
        echo "Check if bulk errors are low (atoms > 5 Å shouldn't feel vacancy!)"
        
    elif [[ $WORKFLOW == "5" ]]; then
        echo "  Combined analysis: $OUT_COMBINED_ANALYSIS"
        echo ""
        echo "Key outputs:"
        echo "  $OUT_COMBINED_ANALYSIS/publication_figure.png"
        echo "  $OUT_COMBINED_ANALYSIS/error_distributions.png"
        echo "  $OUT_COMBINED_ANALYSIS/statistical_tests.csv"
        if [[ -f "training_history.csv" ]]; then
            echo "  $OUT_COMBINED_ANALYSIS/training_analysis.png"
        fi
        echo ""
        echo "Review statistical_tests.csv for p-values!"
    fi
    echo ""
fi

