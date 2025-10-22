#!/bin/bash
# ==============================================================
# WEST: Next-Round Training Script
# --------------------------------------------------------------
# Submits a WEST model training job for a specific fold
# (fold1, fold2, or foldall) during the next training round.
#
# This script is automatically called by run_next_round.sh.
# ==============================================================

#SBATCH --job-name=train_next_%A_%a
#SBATCH --partition=gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/train_next_%x_%j.log

set -e  # Exit immediately on any error

# ----------------------------------------------------------------------
# Environment Setup
# ----------------------------------------------------------------------
source ~/.bashrc
conda activate deeplearning

export NUM_WORKERS=1

# ----------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------
FOLD=$1
NEXT_ROUND=$2
BASE_DIR=".../Transformer"
EXP_DIR="${BASE_DIR}/Experiments"
DATA_DIR="${BASE_DIR}/Data"
PYTHON=python

if [[ -z "$FOLD" || -z "$NEXT_ROUND" ]]; then
  echo "Usage: sbatch train_nextround_fold.sbatch <fold1|fold2|foldall> <next_round_number>"
  exit 1
fi

PREV_ROUND=$((NEXT_ROUND - 1))

# ----------------------------------------------------------------------
# Detect the date used in the previous round’s experiments
# ----------------------------------------------------------------------
EVAL_DIR="/n/data1/hsph/biostat/celehs/lab/kimgreco/Transformer/Evaluation"
DATE=$(find "${EVAL_DIR}" -maxdepth 1 -type d -name "AUC_*_round${PREV_ROUND}_fold1" \
  | head -1 | grep -oP 'AUC_\K[0-9]{8}' || true)

# Fallback to today’s date if not found
if [ -z "$DATE" ]; then
  DATE=$(date +"%m%d%Y")
  echo "No matching evaluation date found for round${PREV_ROUND}; using current date ${DATE}"
fi

echo "=============================================================="
echo "WEST Next-Round Training Job"
echo "Fold:           ${FOLD}"
echo "Next Round:     ${NEXT_ROUND}"
echo "Previous Round: ${PREV_ROUND}"
echo "Date:           ${DATE}"
echo "=============================================================="

# ----------------------------------------------------------------------
# Select Fold-Specific Training Script
# ----------------------------------------------------------------------
case $FOLD in
  fold1) PYTHON_SCRIPT="Scripts/train_fold1.py" ;;
  fold2) PYTHON_SCRIPT="Scripts/train_fold2.py" ;;
  foldall) PYTHON_SCRIPT="Scripts/train_foldall.py" ;;
  *)
    echo "Error: Invalid fold name '${FOLD}'. Choose from fold1, fold2, or foldall."
    exit 1
    ;;
esac

# ----------------------------------------------------------------------
# Paths and Configurations
# ----------------------------------------------------------------------
CONFIG="${EXP_DIR}/${DATE}_round${PREV_ROUND}_${FOLD}/config.json"
SAVE_PATH="${EXP_DIR}/${DATE}_round${NEXT_ROUND}_${FOLD}"
LABELS="${DATA_DIR}/patient_summary_KOMAP_round${PREV_ROUND}_${FOLD}.csv"

mkdir -p "$SAVE_PATH"
cd "$BASE_DIR"

echo "Working directory: $PWD"
echo "Using config:       $CONFIG"
echo "Saving to:          $SAVE_PATH"
echo "Labels file:        $LABELS"
echo "--------------------------------------------------------------"

# ----------------------------------------------------------------------
# Convert Path to Module Name
# ----------------------------------------------------------------------
MODULE_NAME=${PYTHON_SCRIPT%.py}
MODULE_NAME=${MODULE_NAME//\//.}

# ----------------------------------------------------------------------
# Launch Training
# ----------------------------------------------------------------------
$PYTHON -m "$MODULE_NAME" \
  --batch_size "$(jq .batch_size "$CONFIG")" \
  --data_path Data \
  --learning_rate "$(jq .learning_rate "$CONFIG")" \
  --num_epochs "$(jq .num_epochs "$CONFIG")" \
  --hidden_dim "$(jq .hidden_dim "$CONFIG")" \
  --num_heads "$(jq .num_heads "$CONFIG")" \
  --num_layers "$(jq .num_layers "$CONFIG")" \
  --dropout "$(jq .dropout "$CONFIG")" \
  --warmup_steps "$(jq .warmup_steps "$CONFIG")" \
  --weight_decay "$(jq .weight_decay "$CONFIG")" \
  --max_grad_norm "$(jq .max_grad_norm "$CONFIG")" \
  --early_stopping_patience "$(jq .early_stopping_patience "$CONFIG")" \
  --temperature "$(jq .temperature "$CONFIG")" \
  --contrastive_weight "$(jq .contrastive_weight "$CONFIG")" \
  --pos_ratio "$(jq .pos_ratio "$CONFIG")" \
  --save_dir "$SAVE_PATH" \
  --summary_file_name "$LABELS" \
  --use_ema \
  --use_augmentation \
  --max_seq_len "$(jq .max_seq_len "$CONFIG")" \
  --output_type "mean" \
  --label_column "KOMAP_calibrated" \
  --model_name "patient_transformer_single.pt" \
  | tee "${SAVE_PATH}/training_log.txt"

echo "--------------------------------------------------------------"
echo "Training complete for Fold: ${FOLD} (Round ${NEXT_ROUND})"
echo "Results saved in: ${SAVE_PATH}"
echo "=============================================================="
