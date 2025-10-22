#!/usr/bin/env python3
# ==============================================================
# WEST: Silver Label Update Script
# --------------------------------------------------------------
# Updates silver labels in the patient summary file based on
# model predictions from evaluation results.
#
# Typical usage:
#   python Scripts/update_silver_label_custom.py \
#       --summary_file Data/patient_summary_KOMAP_round1.csv \
#       --eval_results Evaluation/AUC_07102025_round1_fold1/evaluation_results_with_clusters_training.csv \
#       --output_file Data/patient_summary_KOMAP_round2_fold1.csv \
#       --output_log Data/patient_summary_KOMAP_labels_changed_round2_fold1.csv
# ==============================================================

import argparse
import os
import pandas as pd
from tqdm import tqdm


# ----------------------------------------------------------------------
# Argument Parsing
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Update WEST silver labels using model predictions.")
    parser.add_argument("--summary_file", type=str, required=True,
                        help="Path to the input patient summary CSV.")
    parser.add_argument("--eval_results", type=str, required=True,
                        help="Path to the evaluation results CSV containing predictions.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save the updated patient summary CSV.")
    parser.add_argument("--output_log", type=str, required=True,
                        help="Path to save a log of changed labels.")
    return parser.parse_args()


# ----------------------------------------------------------------------
# Silver Label Update Logic
# ----------------------------------------------------------------------
def update_silver_labels(summary_file, eval_results, output_file, output_log):
    print("==============================================================")
    print("WEST Silver Label Updater")
    print("--------------------------------------------------------------")
    print(f"Input summary:  {summary_file}")
    print(f"Eval results:   {eval_results}")
    print(f"Output summary: {output_file}")
    print(f"Change log:     {output_log}")
    print("==============================================================")

    # -----------------------------
    # Load input files
    # -----------------------------
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    if not os.path.exists(eval_results):
        raise FileNotFoundError(f"Eval results file not found: {eval_results}")

    print("Reading input files...")
    patient_summary = pd.read_csv(summary_file)
    eval_df = pd.read_csv(eval_results)

    # Normalize ID columns
    patient_summary["ID"] = patient_summary["ID"].astype(str)
    eval_df["patient_id"] = eval_df["patient_id"].astype(str)

    # -----------------------------
    # Filter target patients
    # -----------------------------
    print("Filtering target patients (gold=0 & training=1)...")
    mask = (patient_summary["gold"] == 0) & (patient_summary["training"] == 1)
    target_patients = patient_summary.loc[mask]
    print(f"Found {len(target_patients)} eligible patients for label update.")

    # -----------------------------
    # Prepare prediction mappings
    # -----------------------------
    pred_dict_binary = dict(zip(eval_df["patient_id"], eval_df["binary_prediction"]))
    pred_dict_prob = dict(zip(eval_df["patient_id"], eval_df["prediction"]))
    print(f"Loaded {len(pred_dict_binary)} binary predictions.")

    # -----------------------------
    # Apply updates
    # -----------------------------
    label_changes = []
    flip_0_to_1, flip_1_to_0, unchanged, update_count = 0, 0, 0, 0

    print("Updating labels based on model predictions...")
    for idx in tqdm(target_patients.index, desc="Updating", ncols=80):
        pid = str(patient_summary.loc[idx, "ID"])

        if pid in pred_dict_binary and pid in pred_dict_prob:
            new_label = pred_dict_binary[pid]
            new_prob = pred_dict_prob[pid]

            old_label = patient_summary.loc[idx, "FINALPAH"]
            old_prob = patient_summary.loc[idx, "KOMAP_calibrated"]

            label_changes.append({
                "patient_id": pid,
                "old_label": old_label,
                "new_label": new_label,
                "old_prob": old_prob,
                "new_prob": new_prob,
            })

            if old_label == 0 and new_label == 1:
                flip_0_to_1 += 1
            elif old_label == 1 and new_label == 0:
                flip_1_to_0 += 1
            else:
                unchanged += 1

            # Update main summary
            patient_summary.loc[idx, "FINALPAH_silver"] = new_label
            patient_summary.loc[idx, "FINALPAH"] = new_label
            patient_summary.loc[idx, "KOMAP_calibrated"] = new_prob
            update_count += 1

    # -----------------------------
    # Save results
    # -----------------------------
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_log), exist_ok=True)

    print(f"\nSaving label changes to: {output_log}")
    pd.DataFrame(label_changes).to_csv(output_log, index=False)

    print(f"Saving updated summary to: {output_file}")
    patient_summary.to_csv(output_file, index=False)

    # -----------------------------
    # Final summary
    # -----------------------------
    print("--------------------------------------------------------------")
    print(f"Update complete. Total records modified: {update_count}")
    print(f"  Flipped 0 → 1: {flip_0_to_1}")
    print(f"  Flipped 1 → 0: {flip_1_to_0}")
    print(f"  Unchanged:    {unchanged}")
    print("==============================================================")


# ----------------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    update_silver_labels(args.summary_file, args.eval_results, args.output_file, args.output_log)
