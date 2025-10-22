#!/usr/bin/env python3
# ==============================================================
# WEST: Model Evaluation Script
# --------------------------------------------------------------
# Evaluates a trained WEST model on validation or training data.
# Computes AUC and PR metrics, determines optimal thresholds,
# performs K-means feature clustering, and saves results.
# ==============================================================

import argparse
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from .model_v2 import PatientTransformer
from .patient_dataset import PatientDataset


# ----------------------------------------------------------------------
# Argument Parsing
# ----------------------------------------------------------------------

def parse_args():
    """Parse command line arguments for WEST evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate WEST Model")

    # Data parameters
    parser.add_argument("--data_path", type=str, default="./Patients",
                        help="Path to input data files")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--sub_set", type=str, default="validation",
                        help="Subset to evaluate (validation or training)")
    parser.add_argument("--summary_file_name", type=str,
                        default="patient_summary_KOMAP.csv",
                        help="Summary file name")

    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pred_threshold", type=float, default=-1,
                        help="Prediction threshold (-1 to auto-select)")

    # Model loading
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved WEST model checkpoint")

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    # Sequence parameters
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=50)

    # Output directory
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save results")

    # Fold filter
    parser.add_argument("--kfold_filter", type=int, default=None,
                        help="Restrict evaluation to a specific kfold_2 value (1 or 2)")

    return parser.parse_args()


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def evaluate_model(args):
    """Evaluate a trained WEST model and generate metrics and cluster analyses."""

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Load embeddings and mapping
    # ------------------------------------------------------------------
    datax = pd.read_csv("./Input/MUGS_Codified_Python_ARCH_JULY16.csv")
    mapping = pd.read_csv("./Input/MUGS_Code_Mapping.csv")
    code_embeddings = torch.tensor(datax.to_numpy(), dtype=torch.float32)

    # ------------------------------------------------------------------
    # Dataset and DataLoader
    # ------------------------------------------------------------------
    val_dataset = PatientDataset(
        data_dir=args.data_path,
        summary_file_name=args.summary_file_name,
        code_embeddings=code_embeddings,
        code_mapping=mapping,
        max_seq_len=args.max_seq_len,
        top_k=args.max_seq_len,
        training=args.sub_set == "training",
        use_augmentation=False,
        gold_repeat=1,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    model = PatientTransformer(
        d_model=args.hidden_dim,
        nhead=args.num_heads,
        dropout=args.dropout,
        num_layers=args.num_layers,
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    if "ema_state" in checkpoint and checkpoint["ema_state"] is not None:
        print("Loading EMA model parameters...")
        model.load_state_dict(checkpoint["ema_state"])
    else:
        print("Loading standard model parameters...")
        model.load_state_dict(checkpoint["model_state_dict"])

    if "epoch" in checkpoint and "best_val_auc" in checkpoint:
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, "
              f"best validation AUC: {checkpoint['best_val_auc']:.4f}")
    else:
        print("Checkpoint metadata not found.")

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    model.eval()
    val_predictions, val_labels, feature_vectors, patient_ids = [], [], [], []

    print("\nStarting evaluation...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            code_embeddings = batch["code_embeddings"].to(device)
            counts = batch["counts"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            batch_patient_ids = batch["patient_id"]

            outputs, features = model(code_embeddings, counts, attention_mask)
            val_predictions.extend(outputs.cpu().numpy())
            val_labels.extend(batch["gold_label"].cpu().numpy())
            feature_vectors.extend(features.cpu().numpy())
            patient_ids.extend(batch_patient_ids)

    val_predictions = np.array(val_predictions).flatten()
    val_labels = np.array(val_labels).flatten()
    feature_vectors = np.array(feature_vectors)
    patient_ids = [str(pid) for pid in patient_ids]

    # ------------------------------------------------------------------
    # Create combined DataFrame
    # ------------------------------------------------------------------
    features_df = pd.DataFrame(
        data=feature_vectors,
        index=patient_ids,
        columns=[f"feature_{i}" for i in range(feature_vectors.shape[1])],
    )
    features_df.index.name = "patient_id"
    features_df["true_label"] = val_labels
    features_df["prediction"] = val_predictions

    # Add fold information
    summary_df = pd.read_csv(os.path.join(args.data_path, args.summary_file_name))
    summary_df["ID"] = summary_df["ID"].astype(str)
    summary_df.set_index("ID", inplace=True)
    features_df["kfold_2"] = features_df.index.map(
        lambda pid: summary_df.loc[pid]["kfold_2"] if pid in summary_df.index else np.nan
    )

    # ------------------------------------------------------------------
    # Filter by fold if specified
    # ------------------------------------------------------------------
    if args.kfold_filter is not None:
        eval_df = features_df[features_df["kfold_2"] == args.kfold_filter].copy()
        print(f"Evaluating only kfold_2 == {args.kfold_filter} (n={len(eval_df)})")
    else:
        eval_df = features_df.copy()

    val_predictions = eval_df["prediction"].values
    val_labels = eval_df["true_label"].values

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    if args.sub_set.lower() == "validation":
        auc_roc = roc_auc_score(val_labels, val_predictions)
        average_precision = average_precision_score(val_labels, val_predictions)

        precision, recall, thresholds = precision_recall_curve(val_labels, val_predictions)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_f1_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_f1_idx]
        best_f1 = f1_scores[best_f1_idx]
        best_precision = precision[best_f1_idx]
        best_recall = recall[best_f1_idx]

        print("\nEvaluation Results:")
        print(f"ROC AUC:            {auc_roc:.4f}")
        print(f"Average Precision:  {average_precision:.4f}")
        print("Best Threshold (by F1):")
        print(f"  Threshold:        {best_threshold:.4f}")
        print(f"  F1 Score:         {best_f1:.4f}")
        print(f"  Precision:        {best_precision:.4f}")
        print(f"  Recall:           {best_recall:.4f}")

        # Apply threshold
        if args.pred_threshold > 0:
            threshold = args.pred_threshold
            print(f"Using user-specified threshold: {threshold}")
        else:
            threshold = best_threshold
            print(f"Using best threshold: {threshold}")

        features_df["binary_prediction"] = (
            features_df["prediction"] >= threshold
        ).astype(int)

        # Print AUCs per fold
        for fold in [1, 2]:
            mask = features_df["kfold_2"] == fold
            if mask.sum() > 0:
                auc = roc_auc_score(features_df.loc[mask, "true_label"],
                                    features_df.loc[mask, "prediction"])
                print(f"AUC for kfold_2 == {fold}: {auc:.4f}")
    else:
        print("Training subset detected — skipping AUC and thresholding.")
        features_df["binary_prediction"] = (
            features_df["prediction"] >= args.pred_threshold
        ).astype(int)

    # ------------------------------------------------------------------
    # Feature clustering analysis
    # ------------------------------------------------------------------
    feature_cols = [c for c in features_df.columns if c.startswith("feature_")]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df[feature_cols])

    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)
    features_df["cluster"] = cluster_labels

    cluster_stats = (
        features_df.groupby("cluster")
        .agg({
            "prediction": ["mean", "std", "count"],
            "true_label": ["mean", "sum"],
            "binary_prediction": ["mean", "sum"],
        })
        .round(4)
    )

    print("\nCluster Analysis Summary:")
    print(cluster_stats)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    subset_suffix = args.sub_set.lower()
    features_df.to_csv(os.path.join(args.output_dir,
                                    f"evaluation_results_with_clusters_{subset_suffix}.csv"))
    cluster_stats.to_csv(os.path.join(args.output_dir,
                                      f"cluster_statistics_{subset_suffix}.csv"))

    print(f"\nEvaluation complete. Results saved to: {args.output_dir}")


# ----------------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args)

