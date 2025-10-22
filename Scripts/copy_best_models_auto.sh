#!/bin/bash
# ==============================================================
# WEST: Best Model Selection and Copy Script
# --------------------------------------------------------------
# Identifies the top-performing WEST models from the hyperparameter
# search logs and copies them (and their configs) into dedicated
# experiment directories for downstream analysis or retraining.
# ==============================================================

set -e  # Exit immediately on error

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BASE_DIR=".../Transformer"
HP_DIR="$BASE_DIR/HyperparamSearch"
LOGS_DIR="$HP_DIR/logs"
EXPERIMENTS_DIR="$BASE_DIR/Experiments"
SUMMARY_SCRIPT="$BASE_DIR/Scripts/summarize_fold_aucs_sorted_and_overall.py"

ROUND_DATE="10142025"   # Update this as needed
ROUND_NAME="round1"

# ----------------------------------------------------------------------
# Summarize Hyperparameter Results
# ----------------------------------------------------------------------
echo "Collecting and summarizing WEST hyperparameter search results..."
cd "$LOGS_DIR"

if [ ! -f "$SUMMARY_SCRIPT" ]; then
    echo "Error: Summary script not found at $SUMMARY_SCRIPT"
    exit 1
fi

cp "$SUMMARY_SCRIPT" .
python summarize_fold_aucs_sorted_and_overall.py > summary_output.txt

# ----------------------------------------------------------------------
# Extract Best Hyperparameter IDs
# ----------------------------------------------------------------------
echo "Extracting top-performing models from summary..."

FOLD1_HP=$(grep -A 30 "Sorted by Best Fold 1 AUC" summary_output.txt | grep hp_ | sed -n 3p | awk '{print $1}' | cut -d_ -f1,2)
FOLD2_HP=$(grep -A 30 "Sorted by Best Fold 2 AUC" summary_output.txt | grep hp_ | sed -n 3p | awk '{print $1}' | cut -d_ -f1,2)
FOLDALL_HP=$(grep -A 30 "Sorted by Best Overall AUC" summary_output.txt | grep hp_ | sed -n 1p | awk '{print $1}' | cut -d_ -f1,2)

if [ -z "$FOLD1_HP" ] || [ -z "$FOLD2_HP" ] || [ -z "$FOLDALL_HP" ]; then
    echo "Error: Unable to extract one or more model IDs from summary_output.txt"
    exit 1
fi

echo "Selected models:"
echo "  Fold 1:    $FOLD1_HP"
echo "  Fold 2:    $FOLD2_HP"
echo "  Overall:   $FOLDALL_HP"

# ----------------------------------------------------------------------
# Create Experiment Directories
# ----------------------------------------------------------------------
mkdir -p "$EXPERIMENTS_DIR/${ROUND_DATE}_${ROUND_NAME}_fold1"
mkdir -p "$EXPERIMENTS_DIR/${ROUND_DATE}_${ROUND_NAME}_fold2"
mkdir -p "$EXPERIMENTS_DIR/${ROUND_DATE}_${ROUND_NAME}_foldall"

# ----------------------------------------------------------------------
# Copy Best Models and Configs
# ----------------------------------------------------------------------
echo "Copying best WEST models and configuration files..."

# Fold 1
cp "$HP_DIR/$FOLD1_HP/best_fold1_patient_transformer_single.pt" "$EXPERIMENTS_DIR/${ROUND_DATE}_${ROUND_NAME}_fold1/best_patient_transformer_single.pt"
cp "$HP_DIR/$FOLD1_HP/config.json" "$EXPERIMENTS_DIR/${ROUND_DATE}_${ROUND_NAME}_fold1/"
cp "$HP_DIR/$FOLD1_HP/training_log.txt" "$EXPERIMENTS_DIR/${ROUND_DATE}_${ROUND_NAME}_fold1/"

# Fold 2
cp "$HP_DIR/$FOLD2_HP/best_fold2_patient_transformer_single.pt" "$EXPERIMENTS_DIR/${ROUND_DATE}_${ROUND_NAME}_fold2/best_patient_transformer_single.pt"
cp "$HP_DIR/$FOLD2_HP/config.json" "$EXPERIMENTS_DIR/${ROUND_DATE}_${ROUND_NAME}_fold2/"
cp "$HP_DIR/$FOLD2_HP/training_log.txt" "$EXPERIMENTS_DIR/${ROUND_DATE}_${ROUND_NAME}_fold2/"

# Overall
cp "$HP_DIR/$FOLDALL_HP/best_overall_patient_transformer_single.pt" "$EXPERIMENTS_DIR/${ROUND_DATE}_${ROUND_NAME}_foldall/best_patient_transformer_single.pt"
cp "$HP_DIR/$FOLDALL_HP/config.json" "$EXPERIMENTS_DIR/${ROUND_DATE}_${ROUND_NAME}_foldall/"
cp "$HP_DIR/$FOLDALL_HP/training_log.txt" "$EXPERIMENTS_DIR/${ROUND_DATE}_${ROUND_NAME}_foldall/"

# ----------------------------------------------------------------------
# Completion Message
# ----------------------------------------------------------------------
echo ""
echo "WEST model selection complete."
echo "Copied models and configuration files to:"
echo "  $EXPERIMENTS_DIR/${ROUND_DATE}_${ROUND_NAME}_fold1/"
echo "  $EXPERIMENTS_DIR/${ROUND_DATE}_${ROUND_NAME}_fold2/"
echo "  $EXPERIMENTS_DIR/${ROUND_DATE}_${ROUND_NAME}_foldall/"
