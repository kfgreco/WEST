#!/usr/bin/env python3
# ==============================================================
# WEST: Hyperparameter Search Results Collector
# --------------------------------------------------------------
# Collects logs and configurations from all WEST hyperparameter
# search experiments, extracts metrics, summarizes results, and
# generates a detailed analysis report.
# ==============================================================

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path


# ----------------------------------------------------------------------
# Log Parsing Utilities
# ----------------------------------------------------------------------

def extract_metrics_from_log(log_file):
    """Extract training metrics and AUC values from a log file."""
    metrics = {
        "experiment_id": None,
        "best_val_auc": None,
        "best_val_auc_fold1": None,
        "best_val_auc_fold2": None,
        "final_epoch": None,
        "training_time": None,
        "early_stopped": False,
        "converged": False,
        "error": None
    }

    if not os.path.exists(log_file):
        metrics["error"] = "Log file not found"
        return metrics

    try:
        with open(log_file, "r") as f:
            content = f.read()

        exp_match = re.search(r"hp_\d{3}", log_file)
        if exp_match:
            metrics["experiment_id"] = exp_match.group()

        # Best validation AUC (overall)
        auc_matches = re.findall(r"Best validation AUC: (\d+\.\d+)", content)
        if auc_matches:
            metrics["best_val_auc"] = float(auc_matches[-1])

        # Fold-specific AUCs
        fold1 = re.findall(r"Validation AUC \(Fold 1\): (\d+\.\d+)", content)
        if fold1:
            metrics["best_val_auc_fold1"] = float(fold1[-1])

        fold2 = re.findall(r"Validation AUC \(Fold 2\): (\d+\.\d+)", content)
        if fold2:
            metrics["best_val_auc_fold2"] = float(fold2[-1])

        # Final epoch
        epochs = re.findall(r"Epoch (\d+)/\d+", content)
        if epochs:
            metrics["final_epoch"] = int(epochs[-1])

        # Early stopping flag
        if "Early stopping triggered" in content:
            metrics["early_stopped"] = True

        # Estimate total training time
        times = re.findall(r"Total epoch time: (\d+\.\d+)s", content)
        if times:
            avg_time = np.mean([float(t) for t in times])
            metrics["training_time"] = avg_time * metrics.get("final_epoch", 1)

        # Convergence indicator
        if metrics["best_val_auc"] is not None:
            metrics["converged"] = True

    except Exception as e:
        metrics["error"] = str(e)

    return metrics


# ----------------------------------------------------------------------
# Configuration Loading
# ----------------------------------------------------------------------

def load_experiment_config(config_file):
    """Load experiment configuration from JSON file."""
    try:
        with open(config_file, "r") as f:
            return json.load(f)
    except Exception:
        return {}


# ----------------------------------------------------------------------
# Results Aggregation
# ----------------------------------------------------------------------

def collect_all_results(search_dir="HyperparamSearch"):
    """Aggregate configurations and logs from all WEST experiments."""
    results = []

    if not os.path.exists(search_dir):
        print(f"Search directory not found: {search_dir}")
        return pd.DataFrame()

    exp_dirs = [d for d in os.listdir(search_dir) if d.startswith("hp_")]
    if not exp_dirs:
        print(f"No experiment directories found in {search_dir}")
        return pd.DataFrame()

    print(f"Found {len(exp_dirs)} experiment directories.")

    for exp_dir in sorted(exp_dirs):
        exp_path = os.path.join(search_dir, exp_dir)
        config_file = os.path.join(exp_path, "config.json")
        log_file = os.path.join(exp_path, "training_log.txt")

        config = load_experiment_config(config_file)
        metrics = extract_metrics_from_log(log_file)

        result = {
            "experiment_id": exp_dir,
            "config_file": config_file,
            "log_file": log_file,
            **config,
            **metrics
        }

        results.append(result)

        status = "OK" if metrics["converged"] else "FAIL"
        auc = metrics["best_val_auc"] if metrics["best_val_auc"] else "N/A"
        print(f"[{status}] {exp_dir}: AUC = {auc}")

    return pd.DataFrame(results)


# ----------------------------------------------------------------------
# Results Analysis and Summary
# ----------------------------------------------------------------------

def analyze_results(df):
    """Compute summary statistics and print analysis overview."""
    if df.empty:
        print("No results available for analysis.")
        return

    print("\n============================================================")
    print("WEST Hyperparameter Search Results Analysis")
    print("============================================================")

    total = len(df)
    successful = len(df[df["converged"] == True])
    failed = total - successful

    print("\nSummary Statistics:")
    print(f"  Total experiments: {total}")
    print(f"  Successful experiments: {successful}")
    print(f"  Failed experiments: {failed}")
    print(f"  Success rate: {successful / total * 100:.1f}%")

    if successful == 0:
        print("No successful experiments to analyze.")
        return

    success_df = df[df["converged"] == True].copy()

    # AUC summary
    auc_stats = success_df["best_val_auc"].describe()
    print("\nAUC Performance:")
    print(f"  Best AUC: {auc_stats['max']:.4f}")
    print(f"  Worst AUC: {auc_stats['min']:.4f}")
    print(f"  Mean AUC: {auc_stats['mean']:.4f}")
    print(f"  Std AUC:  {auc_stats['std']:.4f}")

    # Identify best experiment
    best_exp = success_df.loc[success_df["best_val_auc"].idxmax()]
    print("\nBest Experiment:")
    print(f"  ID: {best_exp['experiment_id']}")
    print(f"  AUC: {best_exp['best_val_auc']:.4f}")
    print(f"  Batch Size: {best_exp['batch_size']}")
    print(f"  Learning Rate: {best_exp['learning_rate']}")
    print(f"  Hidden Dim: {best_exp['hidden_dim']}")
    print(f"  Num Layers: {best_exp['num_layers']}")
    print(f"  Dropout: {best_exp['dropout']}")
    print(f"  Final Epoch: {best_exp['final_epoch']}")

    # Top 5 experiments
    top5 = success_df.nlargest(5, "best_val_auc")
    print("\nTop 5 Experiments:")
    for i, (_, exp) in enumerate(top5.iterrows(), 1):
        print(f"  {i}. {exp['experiment_id']}: AUC = {exp['best_val_auc']:.4f}")

    # Hyperparameter importance summary
    print("\nHyperparameter Analysis:")
    params = ["batch_size", "learning_rate", "hidden_dim", "num_layers", "dropout"]
    for param in params:
        if param in success_df.columns:
            stats = success_df.groupby(param)["best_val_auc"].agg(["mean", "count", "std"]).round(4)
            print(f"\n  {param}:")
            for val, row in stats.iterrows():
                print(f"    {val}: mean={row['mean']:.4f}, count={row['count']}, std={row['std']:.4f}")

    # Early stopping summary
    stopped = success_df["early_stopped"].sum()
    print("\nEarly Stopping:")
    print(f"  Experiments stopped early: {stopped}/{successful}")
    print(f"  Rate: {stopped / successful * 100:.1f}%")

    # Training time statistics
    if "training_time" in success_df.columns and success_df["training_time"].notna().any():
        times = success_df["training_time"].describe()
        print("\nTraining Time (seconds):")
        print(f"  Mean: {times['mean']:.1f}")
        print(f"  Min:  {times['min']:.1f}")
        print(f"  Max:  {times['max']:.1f}")


# ----------------------------------------------------------------------
# Output Utilities
# ----------------------------------------------------------------------

def save_results(df, output_file="HyperparamSearch/results_summary.csv"):
    """Save summarized results to CSV."""
    if df.empty:
        print("No results to save.")
        return

    columns = [
        "experiment_id", "best_val_auc", "best_val_auc_fold1", "best_val_auc_fold2",
        "batch_size", "learning_rate", "num_epochs", "hidden_dim", "num_layers", "dropout",
        "final_epoch", "early_stopped", "converged", "training_time", "error"
    ]
    existing = [c for c in columns if c in df.columns]
    summary_df = df[existing].copy()

    if "best_val_auc" in summary_df.columns:
        summary_df = summary_df.sort_values("best_val_auc", ascending=False)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    summary_df.to_csv(output_file, index=False)

    print(f"\nResults saved to: {output_file}")
    print(f"Shape: {summary_df.shape}")


def generate_report(df, output_file="HyperparamSearch/hyperparameter_search_report.txt"):
    """Generate a plain-text summary report."""
    if df.empty:
        print("No results to report.")
        return

    with open(output_file, "w") as f:
        f.write("WEST Hyperparameter Search Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        total = len(df)
        successful = len(df[df["converged"] == True])

        f.write("Summary:\n")
        f.write(f"  Total experiments: {total}\n")
        f.write(f"  Successful experiments: {successful}\n")
        f.write(f"  Success rate: {successful / total * 100:.1f}%\n\n")

        if successful > 0:
            success_df = df[df["converged"] == True].copy()
            best_exp = success_df.loc[success_df["best_val_auc"].idxmax()]

            f.write("Best Experiment:\n")
            f.write(f"  ID: {best_exp['experiment_id']}\n")
            f.write(f"  AUC: {best_exp['best_val_auc']:.4f}\n")
            f.write(f"  Hyperparameters:\n")
            f.write(f"    Batch Size: {best_exp['batch_size']}\n")
            f.write(f"    Learning Rate: {best_exp['learning_rate']}\n")
            f.write(f"    Hidden Dim: {best_exp['hidden_dim']}\n")
            f.write(f"    Num Layers: {best_exp['num_layers']}\n")
            f.write(f"    Dropout: {best_exp['dropout']}\n")
            f.write(f"    Final Epoch: {best_exp['final_epoch']}\n\n")

            top10 = success_df.nlargest(10, "best_val_auc")
            f.write("Top 10 Experiments:\n")
            for i, (_, exp) in enumerate(top10.iterrows(), 1):
                f.write(f"  {i:2d}. {exp['experiment_id']}: AUC = {exp['best_val_auc']:.4f}\n")

        failed_df = df[df["converged"] != True]
        if len(failed_df) > 0:
            f.write(f"\nFailed Experiments ({len(failed_df)}):\n")
            for _, exp in failed_df.iterrows():
                error = exp.get("error", "Unknown error")
                f.write(f"  {exp['experiment_id']}: {error}\n")

    print(f"Report saved to: {output_file}")


# ----------------------------------------------------------------------
# Main Entry Point
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect and analyze WEST hyperparameter search results.")
    parser.add_argument("--search_dir", type=str, default="HyperparamSearch")
    parser.add_argument("--output_dir", type=str, default="HyperparamSearch")
    args = parser.parse_args()

    print("WEST Hyperparameter Search Results Collector")
    print("============================================================")

    df = collect_all_results(args.search_dir)
    if df.empty:
        print("No results found.")
        return

    analyze_results(df)

    output_file = os.path.join(args.output_dir, "results_summary.csv")
    save_results(df, output_file)

    report_file = os.path.join(args.output_dir, "hyperparameter_search_report.txt")
    generate_report(df, report_file)

    print("\nAnalysis complete.")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
