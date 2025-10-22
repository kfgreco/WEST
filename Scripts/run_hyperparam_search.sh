#!/bin/bash
# ==============================================================
# WEST: Hyperparameter Search Runner
# --------------------------------------------------------------
# Main entry point to launch hyperparameter search experiments.
# Generates configurations, creates submission scripts, and
# optionally dispatches jobs to a SLURM cluster.
# ==============================================================

set -e  # Exit immediately on error

echo "WEST Hyperparameter Search"
echo "=========================="
echo ""

# Configuration
NUM_EXPERIMENTS=${1:-20}  # Number of experiments (default: 20)
SEARCH_SEED=${2:-42}      # Random seed for reproducibility (default: 42)

echo "Configuration:"
echo "  Number of experiments: ${NUM_EXPERIMENTS}"
echo "  Random seed: ${SEARCH_SEED}"
echo ""

# Step 1: Directory setup
echo "Step 1: Setting up directory structure..."
mkdir -p HyperparamSearch/{configs,logs,models}
echo "  Directories created: HyperparamSearch/configs, HyperparamSearch/logs, HyperparamSearch/models"
echo ""

# Step 2: Generate experiments
echo "Step 2: Generating hyperparameter configurations..."
python3 Scripts/hyperparameter_search.py \
    --num_experiments "${NUM_EXPERIMENTS}" \
    --seed "${SEARCH_SEED}"

if [ $? -ne 0 ]; then
    echo "Error: hyperparameter_search.py failed."
    exit 1
fi

echo "  Experiment configurations generated successfully."
echo ""

# Step 3: Submit experiments
if command -v sbatch &> /dev/null; then
    echo "Step 3: Submitting experiments to SLURM..."

    if [ -f "HyperparamSearch/submit_all.sh" ]; then
        bash HyperparamSearch/submit_all.sh
        echo ""
        echo "All experiments submitted to SLURM."
        echo "Monitor jobs with: squeue -u $USER"
        echo "Collect results with: python Scripts/collect_results.py"
    else
        echo "Error: submit_all.sh not found in HyperparamSearch/."
        exit 1
    fi
else
    echo "Step 3: SLURM not available. Manual execution required."
    echo ""
    echo "To run experiments manually:"
    echo "  cd HyperparamSearch/configs/"
    echo "  bash train_hp_XXX.sh (for each experiment)"
    echo ""
    echo "To collect results after completion:"
    echo "  python Scripts/collect_results.py"
fi

echo ""
echo "WEST Hyperparameter Search Complete"
echo "-----------------------------------"
echo "Generated directories:"
echo "  Configs: HyperparamSearch/configs/"
echo "  Logs:    HyperparamSearch/logs/"
echo "  Models:  HyperparamSearch/models/"
echo ""
echo "To summarize results:"
echo "  python Scripts/collect_results.py"
