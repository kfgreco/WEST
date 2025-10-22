#!/usr/bin/env python3
# ==============================================================
# WEST: Hyperparameter Search Log Summarizer
# --------------------------------------------------------------
# Parses WEST training output logs, extracts Fold 1, Fold 2, and
# Overall validation AUCs, and produces sorted summaries by each
# criterion for easy model selection.
# ==============================================================

import os


# ----------------------------------------------------------------------
# AUC Extraction Utilities
# ----------------------------------------------------------------------

def extract_all_aucs_and_overall(file_path):
    """Extract all Fold 1, Fold 2, and Overall AUCs from a log file."""
    fold1_aucs, fold2_aucs, overall_aucs = [], [], []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if "Validation AUC (Fold 1):" in line:
                try:
                    fold1_aucs.append(float(line.split(":")[-1]))
                except ValueError:
                    continue
            elif "Validation AUC (Fold 2):" in line:
                try:
                    fold2_aucs.append(float(line.split(":")[-1]))
                except ValueError:
                    continue
            elif "Validation AUC (All):" in line:
                try:
                    overall_aucs.append(float(line.split(":")[-1]))
                except ValueError:
                    continue

    return fold1_aucs, fold2_aucs, overall_aucs


# ----------------------------------------------------------------------
# Summary and Sorting
# ----------------------------------------------------------------------

def summarize_sorted_and_overall(directory):
    """Summarize WEST training runs sorted by Fold 1, Fold 2, and Overall AUC."""
    results = []

    for filename in os.listdir(directory):
        if not filename.endswith(".out"):
            continue

        file_path = os.path.join(directory, filename)
        fold1_aucs, fold2_aucs, overall_aucs = extract_all_aucs_and_overall(file_path)

        min_len = min(len(fold1_aucs), len(fold2_aucs), len(overall_aucs))
        if min_len == 0:
            continue

        fold1_aucs, fold2_aucs, overall_aucs = (
            fold1_aucs[:min_len],
            fold2_aucs[:min_len],
            overall_aucs[:min_len],
        )

        best_f1 = max(fold1_aucs)
        best_f1_epoch = fold1_aucs.index(best_f1)
        corr_f2 = fold2_aucs[best_f1_epoch]

        best_f2 = max(fold2_aucs)
        best_f2_epoch = fold2_aucs.index(best_f2)
        corr_f1 = fold1_aucs[best_f2_epoch]

        best_overall = max(overall_aucs)

        results.append((filename, best_f1, corr_f2, best_f2, corr_f1, best_overall))

    if not results:
        print("No .out log files with valid AUC values found in directory.")
        return

    # ------------------------------------------------------------------
    # Table 1: Sorted by Best Fold 1 AUC
    # ------------------------------------------------------------------
    print("\nSorted by Best Fold 1 AUC")
    print(f"{'File':<35} {'Best F1':<10} {'Corr F2':<10}")
    print("-" * 60)
    for fname, best_f1, corr_f2, *_ in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{fname:<35} {best_f1:<10.4f} {corr_f2:<10.4f}")

    # ------------------------------------------------------------------
    # Table 2: Sorted by Best Fold 2 AUC
    # ------------------------------------------------------------------
    print("\nSorted by Best Fold 2 AUC")
    print(f"{'File':<35} {'Best F2':<10} {'Corr F1':<10}")
    print("-" * 60)
    for fname, _, _, best_f2, corr_f1, _ in sorted(results, key=lambda x: x[3], reverse=True):
        print(f"{fname:<35} {best_f2:<10.4f} {corr_f1:<10.4f}")

    # ------------------------------------------------------------------
    # Table 3: Sorted by Best Overall AUC
    # ------------------------------------------------------------------
    print("\nSorted by Best Overall AUC")
    print(f"{'File':<35} {'Best Overall AUC'}")
    print("-" * 50)
    for fname, *_, best_overall in sorted(results, key=lambda x: x[5], reverse=True):
        print(f"{fname:<35} {best_overall:<.4f}")


# ----------------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    log_directory = "."  # Default: current directory
    summarize_sorted_and_overall(log_directory)

