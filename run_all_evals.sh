#!/bin/bash
# ==============================================================
# WEST: Evaluation Runner
# --------------------------------------------------------------
# Runs model evaluation for all selected WEST experiments (fold1,
# fold2, and overall). Each evaluation uses parameters stored in
# the corresponding config.json file.
# ==============================================================

set -e  # Exit on error

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
ROUND=1
PREV_ROUND=0
DATE="10142025"
DEVICE="cuda"
SPLIT="validation"
TOP_K=70
PRED_THRESHOLD=-1
EVAL_SCRIPT="-m Scripts.eval"
DATA_PATH="Data"
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

# ----------------------------------------------------------------------
# Evaluation Function
# ----------------------------------------------------------------------
run_eval() {
  FOLDER=$1
  KFOLD_FLAG=$2
  LOGFILE=$3

  CONFIG_PATH="Experiments/${FOLDER}/config.json"
  MODEL_PATH="Experiments/${FOLDER}/best_patient_transformer_single.pt"
  OUTPUT_DIR="Evaluation/AUC_${FOLDER}"

  SUMMARY_FILE_NAME=$(jq -r '.summary_file_name' "$CONFIG_PATH")

  echo "--------------------------------------------------------------" | tee "$LOGFILE"
  echo "Running evaluation for: ${FOLDER}" | tee -a "$LOGFILE"
  echo "Model path:  $MODEL_PATH" | tee -a "$LOGFILE"
  echo "Output dir:  $OUTPUT_DIR" | tee -a "$LOGFILE"
  echo "--------------------------------------------------------------" | tee -a "$LOGFILE"

  python $EVAL_SCRIPT \
    --data_path "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --sub_set "$SPLIT" \
    --batch_size "$(jq -r '.batch_size' "$CONFIG_PATH")" \
    --hidden_dim "$(jq -r '.hidden_dim' "$CONFIG_PATH")" \
    --num_heads "$(jq -r '.num_heads' "$CONFIG_PATH")" \
    --num_layers "$(jq -r '.num_layers' "$CONFIG_PATH")" \
    --dropout "$(jq -r '.dropout' "$CONFIG_PATH")" \
    --max_seq_len "$(jq -r '.max_seq_len' "$CONFIG_PATH")" \
    --top_k "$TOP_K" \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    --pred_threshold "$PRED_THRESHOLD" \
    --summary_file_name "$SUMMARY_FILE_NAME" \
    $KFOLD_FLAG | tee -a "$LOGFILE"

  echo "Evaluation completed for: ${FOLDER}" | tee -a "$LOGFILE"
  echo "" | tee -a "$LOGFILE"
}

# ----------------------------------------------------------------------
# Run Evaluations for All Folds
# ----------------------------------------------------------------------
run_eval "${DATE}_round1_fold1" "--kfold_filter 1" "$LOG_DIR/eval_fold1.log"
run_eval "${DATE}_round1_fold2" "--kfold_filter 2" "$LOG_DIR/eval_fold2.log"
run_eval "${DATE}_round1_foldall" ""              "$LOG_DIR/eval_foldall.log"

echo "=============================================================="
echo "WEST evaluations complete."
echo "Logs saved to: $LOG_DIR/"
echo "=============================================================="
