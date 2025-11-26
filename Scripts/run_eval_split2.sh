#!/bin/bash
# ==============================================================
# WEST: Training-Set Evaluation Script (Next-Round Prep)
# --------------------------------------------------------------
# Runs evaluation on the training set for a given round and fold.
# Uses the best threshold from prior validation evaluation logs.
#
# Usage:
#   bash run_eval_split2.sh <round_number> <fold>
# Example:
#   bash run_eval_split2.sh 1 fold1
# ==============================================================

set -e  # Exit on first error

# ----------------------------------------------------------------------
# Argument Check
# ----------------------------------------------------------------------
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: bash run_eval_split2.sh <round_number> <fold>"
  echo "  where <fold> is one of: fold1, fold2, foldall"
  exit 1
fi

ROUND=$1
FOLD=$2
PREV_ROUND=$((ROUND - 1))
DATE="10142025" 

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_PATH="Data"
EVAL_SCRIPT="-m Scripts.eval"
SPLIT="training"
DEVICE="cuda"
TOP_K=70

CONFIG="Experiments/${DATE}_round1_${FOLD}/config.json"
MODEL_PATH="Experiments/${DATE}_round${ROUND}_${FOLD}/best_patient_transformer_single.pt"
OUTPUT_DIR="Evaluation/AUC_${DATE}_round${ROUND}_${FOLD}"
SUMMARY_FILE_NAME="patient_summary_KOMAP_round${PREV_ROUND}.csv"
LOG_FILE="logs/eval_${FOLD}.log"

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$OUTPUT_DIR"

# ----------------------------------------------------------------------
# Extract Best Threshold from Validation Logs
# ----------------------------------------------------------------------
if [ -f "$LOG_FILE" ]; then
  THRESH=$(grep "Using best threshold" "$LOG_FILE" | tail -1 | awk '{print $4}')
else
  echo "Warning: Log file not found ($LOG_FILE). Using default threshold -1."
  THRESH=-1
fi

# ----------------------------------------------------------------------
# Fold Flag Mapping
# ----------------------------------------------------------------------
case $FOLD in
  fold1)
    KFOLD_FLAG="--kfold_filter 1"
    ;;
  fold2)
    KFOLD_FLAG="--kfold_filter 2"
    ;;
  foldall)
    KFOLD_FLAG=""
    ;;
  *)
    echo "Error: Invalid fold '$FOLD'. Choose from: fold1, fold2, foldall."
    exit 1
    ;;
esac

# ----------------------------------------------------------------------
# Run Evaluation
# ----------------------------------------------------------------------
echo "=============================================================="
echo "WEST Training-Set Evaluation"
echo "Round:          ${ROUND}"
echo "Fold:           ${FOLD}"
echo "Previous Round: ${PREV_ROUND}"
echo "Model Path:     ${MODEL_PATH}"
echo "Summary File:   ${SUMMARY_FILE_NAME}"
echo "Threshold:      ${THRESH}"
echo "=============================================================="

python ${EVAL_SCRIPT} \
  --data_path "${DATA_PATH}" \
  --model_path "${MODEL_PATH}" \
  --sub_set "${SPLIT}" \
  --batch_size "$(jq -r '.batch_size' "${CONFIG}")" \
  --hidden_dim "$(jq -r '.hidden_dim' "${CONFIG}")" \
  --num_heads "$(jq -r '.num_heads' "${CONFIG}")" \
  --num_layers "$(jq -r '.num_layers' "${CONFIG}")" \
  --dropout "$(jq -r '.dropout' "${CONFIG}")" \
  --max_seq_len "$(jq -r '.max_seq_len' "${CONFIG}")" \
  --top_k "${TOP_K}" \
  --device "${DEVICE}" \
  --output_dir "${OUTPUT_DIR}" \
  --pred_threshold "${THRESH}" \
  --summary_file_name "${SUMMARY_FILE_NAME}" \
  ${KFOLD_FLAG}

echo "--------------------------------------------------------------"
echo "Evaluation complete for Round ${ROUND}, ${FOLD}"
echo "Results saved to: ${OUTPUT_DIR}"
echo "--------------------------------------------------------------"

