#!/bin/bash
# ==============================================================
# WEST: Next-Round Pipeline Runner
# --------------------------------------------------------------
# Automates the transition to the next WEST training round.
# Performs evaluation on training data, updates silver labels,
# and launches training jobs for the next round automatically.
# ==============================================================

set -e  # Exit immediately on error
set -x  # Enable debugging (remove once verified stable)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BASE_DIR=".../Transformer"
LOG_DIR="${BASE_DIR}/logs"
EVAL_DIR="${BASE_DIR}/Evaluation"
DATA_DIR="${BASE_DIR}/Data"
EXP_DIR="${BASE_DIR}/Experiments"
PYTHON=python

mkdir -p "${LOG_DIR}"

# ----------------------------------------------------------------------
# Determine round numbers for label update and training
# ----------------------------------------------------------------------
# Detect the latest completed evaluation round
LATEST_EVAL_ROUND=$(ls "${EVAL_DIR}" 2>/dev/null | grep -oP 'round\K[0-9]+' | sort -n | tail -1)

if [ -z "$LATEST_EVAL_ROUND" ]; then
  echo "No previous evaluations found. Assuming base Round 0."
  LATEST_EVAL_ROUND=0
fi

# Label update (use latest eval results to update next-round summaries)
LABEL_PREV_ROUND=$((LATEST_EVAL_ROUND - 1))
LABEL_NEXT_ROUND=$LATEST_EVAL_ROUND
if [ $LABEL_PREV_ROUND -lt 0 ]; then
  LABEL_PREV_ROUND=0
fi

# Training (uses newly generated summaries from label update)
TRAIN_PREV_ROUND=$LABEL_NEXT_ROUND
TRAIN_NEXT_ROUND=$((LABEL_NEXT_ROUND + 1))

# ----------------------------------------------------------------------
# Determine evaluation date for label update
# ----------------------------------------------------------------------
DATE=$(find "${EVAL_DIR}" -maxdepth 1 -type d -name "AUC_*_round${LABEL_NEXT_ROUND}_fold1" \
  | head -1 | grep -oP 'AUC_\K[0-9]{8}' || true)

# Fallback: use most recent evaluation folder if none match
if [ -z "$DATE" ]; then
  DATE=$(find "${EVAL_DIR}" -maxdepth 1 -type d -name "AUC_*_fold1" \
    | sort | tail -1 | grep -oP 'AUC_\K[0-9]{8}' || true)
  echo "No evaluation folder found for round${LABEL_NEXT_ROUND}; using latest available date ${DATE}"
fi

# Final fallback: use today's date
if [ -z "$DATE" ]; then
  DATE=$(date +"%m%d%Y")
  echo "No previous evaluation date found; using current date ${DATE}"
fi

echo "=============================================================="
echo "Starting WEST Next-Round Pipeline"
echo "Label Update Round: ${LABEL_NEXT_ROUND}"
echo "Training Round:     ${TRAIN_NEXT_ROUND}"
echo "Evaluation Date:    ${DATE}"
echo "=============================================================="

# ----------------------------------------------------------------------
# Step 1: Run evaluation on training data (using best thresholds)
# ----------------------------------------------------------------------
echo "Running evaluation on training set for all folds..."
for FOLD in fold1 fold2 foldall; do
  bash Scripts/run_eval_split2.sh "${TRAIN_PREV_ROUND}" "${FOLD}"
done

# ----------------------------------------------------------------------
# Step 2: Update silver labels per fold
# ----------------------------------------------------------------------
echo "Updating silver labels based on Round ${LABEL_NEXT_ROUND} training evaluations..."
for FOLD in fold1 fold2 foldall; do
  EVAL_RESULTS="${EVAL_DIR}/AUC_${DATE}_round${LABEL_NEXT_ROUND}_${FOLD}/evaluation_results_with_clusters_training.csv"
  INPUT_SUMMARY="${DATA_DIR}/patient_summary_KOMAP_round${LABEL_PREV_ROUND}.csv"
  OUTPUT_SUMMARY="${DATA_DIR}/patient_summary_KOMAP_round${LABEL_NEXT_ROUND}_${FOLD}.csv"
  OUTPUT_LOG="${DATA_DIR}/patient_summary_KOMAP_labels_changed_round${LABEL_NEXT_ROUND}_${FOLD}.csv"

  echo "Processing ${FOLD}..."
  ${PYTHON} Scripts/update_silver_label_custom.py \
    --summary_file "$INPUT_SUMMARY" \
    --eval_results "$EVAL_RESULTS" \
    --output_file "$OUTPUT_SUMMARY" \
    --output_log "$OUTPUT_LOG"
done

# ----------------------------------------------------------------------
# Step 3: Launch training for the next round
# ----------------------------------------------------------------------
echo "Submitting training jobs for Round ${TRAIN_NEXT_ROUND}..."
for FOLD in fold1 fold2 foldall; do
  sbatch Scripts/train_next_round_fold.sbatch "$FOLD" "$TRAIN_NEXT_ROUND"
done

echo "Training jobs for Round ${TRAIN_NEXT_ROUND} submitted."

# ----------------------------------------------------------------------
# Completion
# ----------------------------------------------------------------------
echo "=============================================================="
echo "WEST Next-Round Pipeline Complete"
echo "Label updates completed for Round ${LABEL_NEXT_ROUND}."
echo "Training jobs submitted for Round ${TRAIN_NEXT_ROUND}."
echo "Logs available in: ${LOG_DIR}"
echo "=============================================================="

