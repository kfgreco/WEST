#!/usr/bin/env python3
# ==============================================================
# WEST: Patient Dataset Utilities
# --------------------------------------------------------------
# Implements dataset loaders for WEST model training and evaluation.
# Provides both patient-level and visit-level dataset structures,
# along with time-based augmentations and count normalization.
# ==============================================================

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import time
import functools


# ----------------------------------------------------------------------
# Profiling Decorator
# ----------------------------------------------------------------------

def profile_decorator(func):
    """Measure and print execution time for debugging."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper


# ----------------------------------------------------------------------
# Patient-Level Dataset
# ----------------------------------------------------------------------

class PatientDataset(Dataset):
    """
    WEST patient-level dataset class.

    Loads individual patient data and prepares sequential features
    (codes, counts, times, and embeddings). Supports data augmentation,
    resampling, normalization, and positive/negative balance control.
    """

    def __init__(
        self,
        data_dir,
        code_embeddings,
        code_mapping,
        summary_file_name="patient_summary.csv",
        max_seq_len=50,
        top_k=50,
        training=True,
        pos_ratio=None,
        gold_repeat=10,
        use_augmentation=True,
        augmentation_method="truncate_v3",
        remove_gold_negative=False,
        gold_only=False,
        label_column="FINALPAH",
    ):
        self.data_dir = data_dir
        self.code_embeddings = code_embeddings
        self.code_mapping = code_mapping
        self.max_seq_len = max_seq_len
        self.top_k = top_k
        self.training = training
        self.pos_ratio = pos_ratio
        self.use_augmentation = use_augmentation
        self.augmentation_method = augmentation_method
        self.label_column = label_column

        # ------------------------------------------------------------------
        # Load patient summary and initialize subsets
        # ------------------------------------------------------------------
        summary_file = os.path.join(data_dir, summary_file_name)
        self.patient_summary = pd.read_csv(summary_file)
        self.patient_summary.set_index("ID", inplace=True)

        subset = "Training" if training else "Validation"
        self.subset = subset
        self.patient_ids = [
            f.split(".")[0]
            for f in os.listdir(os.path.join(data_dir, subset))
            if f.endswith(".csv")
        ]

        self.patient_summary.index = self.patient_summary.index.astype(str)
        self.patient_ids = [str(pid) for pid in self.patient_ids]
        total_samples = len(self.patient_ids)

        # ------------------------------------------------------------------
        # Gold vs. Silver Patient Breakdown
        # ------------------------------------------------------------------
        gold_patients = [
            pid
            for pid in self.patient_ids
            if not pd.isna(self.patient_summary.loc[pid]["gold"])
            and self.patient_summary.loc[pid]["gold"] == 1
        ]
        silver_patients = [pid for pid in self.patient_ids if pid not in gold_patients]

        if gold_only:
            print("Using only gold standard samples...")
            print(f"All samples before filtering: {len(self.patient_ids)}")
            self.patient_ids = gold_patients
            print(f"Remaining gold samples: {len(self.patient_ids)}")
            silver_patients = []

        gold_total = len(gold_patients)
        gold_pos = sum(
            self.patient_summary.loc[pid]["FINALPAH"] == 1 for pid in gold_patients
        )
        gold_neg = gold_total - gold_pos

        silver_total = len(silver_patients)
        silver_pos = sum(
            self.patient_summary.loc[pid]["FINALPAH"] == 1 for pid in silver_patients
        )
        silver_neg = silver_total - silver_pos

        # ------------------------------------------------------------------
        # Print Dataset Statistics
        # ------------------------------------------------------------------
        print(f"\n{'='*50}")
        print(f"Dataset Statistics ({subset}):")
        print(f"{'='*50}")
        print(f"Total samples: {total_samples}")
        print(
            f"\nGold standard patients ({gold_total} samples, {gold_total/total_samples*100:.2f}%):"
        )
        print(f"Positive: {gold_pos} ({gold_pos/gold_total*100:.2f}% of gold)")
        print(f"Negative: {gold_neg} ({gold_neg/gold_total*100:.2f}% of gold)")
        if silver_total > 0:
            print(
                f"\nSilver standard patients ({silver_total} samples, {silver_total/total_samples*100:.2f}%):"
            )
            print(f"Positive: {silver_pos} ({silver_pos/silver_total*100:.2f}% of silver)")
            print(f"Negative: {silver_neg} ({silver_neg/silver_total*100:.2f}% of silver)")
        else:
            print("No silver standard patients found.")
        print(f"\nOverall label distribution:")
        print(
            f"Total positive: {gold_pos + silver_pos} ({(gold_pos + silver_pos)/total_samples*100:.2f}%)"
        )
        print(
            f"Total negative: {gold_neg + silver_neg} ({(gold_neg + silver_neg)/total_samples*100:.2f}%)"
        )
        print(f"{'='*50}\n")

        # ------------------------------------------------------------------
        # Gold sample repetition, filtering, and balancing
        # ------------------------------------------------------------------
        if self.training and gold_repeat > 1:
            print("Repeating gold standard positive samples...")
            gold_ids = []
            for pid in self.patient_ids:
                if (
                    not pd.isna(self.patient_summary.loc[pid]["gold"])
                    and self.patient_summary.loc[pid]["gold"] == 1
                ):
                    gold_ids.extend([pid] * (gold_repeat - 1))
            self.patient_ids.extend(gold_ids)
            print(f"Gold standard positive samples repeated {gold_repeat} times")

        if remove_gold_negative:
            print("Removing gold negative samples...")
            print(f"All samples: {len(self.patient_ids)}")
            self.patient_ids = [
                pid
                for pid in self.patient_ids
                if not (
                    self.patient_summary.loc[pid]["gold"] == 1
                    and self.patient_summary.loc[pid]["FINALPAH"] == 0
                )
            ]
            print(f"Remaining samples: {len(self.patient_ids)}")

        if self.training and self.pos_ratio is not None:
            print("Preparing positive and negative sample indices...")
            self.pos_indices = []
            self.neg_indices = []
            for i, pid in enumerate(self.patient_ids):
                if self.patient_summary.loc[pid]["FINALPAH"] == 1:
                    self.pos_indices.append(i)
                else:
                    self.neg_indices.append(i)
            print(f"Positive samples: {len(self.pos_indices)}")
            print(f"Negative samples: {len(self.neg_indices)}")

        # ------------------------------------------------------------------
        # Load Similarity and Statistics Files
        # ------------------------------------------------------------------
        similarity_file = os.path.join(data_dir, "code_similarities.csv")
        self.code_similarities = pd.read_csv(similarity_file)
        print("Preloading patient data...")
        self.patient_data_cache = {}
        subset_dir = os.path.join(data_dir, subset)
        for pid in self.patient_ids:
            df = pd.read_csv(os.path.join(subset_dir, f"{pid}.csv"))
            df["TIME"] = df["TIME"].astype(int)
            self.patient_data_cache[pid] = df

        print("Creating similarity lookup...")
        self.similarity_dict = dict(
            zip(
                self.code_similarities["code"],
                self.code_similarities["similarity"],
            )
        )

        stats_file = os.path.join(data_dir, "code_count_statistics.csv")
        self.code_stats = pd.read_csv(stats_file)
        self.code_stats.set_index("CODE", inplace=True)
        self.code_stats_dict = {
            "mean": self.code_stats["mean_count"].to_dict(),
            "std": self.code_stats["std"].to_dict(),
        }

    # ------------------------------------------------------------------
    # Dataset Interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.patient_ids)

    # ------------------------------------------------------------------
    # Augmentation Utilities
    # ------------------------------------------------------------------

    def time_truncate_augmentation(self, df, max_gap=30):
        """Randomly truncate timeline within max_gap of latest time."""
        times = sorted(df["TIME"].unique())
        max_time = max(times)
        valid_times = [t for t in times if max_time - t <= max_gap]
        if len(valid_times) <= 1:
            return df
        truncate_time = random.choice(valid_times)
        return df[df["TIME"] <= truncate_time].copy()

    def time_truncate_augmentation_v2(self, df):
        """Alternative time truncation augmentation."""
        target_code = "PheCode:415.2"
        times = sorted(df["TIME"].unique())
        if len(times) <= 1:
            return df
        truncate_time = random.choice(times)
        augmented_df = df[df["TIME"] <= truncate_time].copy()
        if target_code not in augmented_df["CODE"].values:
            return df
        return augmented_df

    def time_truncate_augmentation_v3(self, df):
        """Random contiguous time window augmentation."""
        target_code = "PheCode:415.2"
        times = sorted(df["TIME"].unique())
        time_num = len(times)
        if time_num <= 1:
            return df
        random_time_num = random.randint(1, time_num)
        random_start_time = random.randint(0, time_num - random_time_num)
        random_end_time = random_start_time + random_time_num - 1
        start_time = times[random_start_time]
        end_time = times[random_end_time]
        augmented_df = df[(df["TIME"] >= start_time) & (df["TIME"] <= end_time)].copy()
        if target_code not in augmented_df["CODE"].values:
            return df
        return augmented_df

    def random_time_points_augmentation(self, df):
        """Randomly select subset of time points."""
        target_code = "PheCode:415.2"
        times = sorted(df["TIME"].unique())
        num_times = len(times)
        if num_times <= 1:
            return df
        num_points_to_keep = random.randint(1, num_times)
        selected_times = sorted(random.sample(times, num_points_to_keep))
        augmented_df = df[df["TIME"].isin(selected_times)].copy()
        if target_code not in augmented_df["CODE"].values:
            return df
        return augmented_df

    # ------------------------------------------------------------------
    # Data Preparation
    # ------------------------------------------------------------------

    def get_patient_data(self, patient_id):
        """Load and optionally augment data for a patient."""
        df = self.patient_data_cache[patient_id].copy()
        if self.training and self.use_augmentation:
            if self.augmentation_method == "truncate":
                df = self.time_truncate_augmentation(df, max_gap=30)
            elif self.augmentation_method == "random":
                df = self.random_time_points_augmentation(df)
            elif self.augmentation_method == "truncate_v2":
                df = self.time_truncate_augmentation_v2(df)
            elif self.augmentation_method == "truncate_v3":
                df = self.time_truncate_augmentation_v3(df)

        target_code = "PheCode:415.2"
        target_mask = df["CODE"] == target_code
        target_df = df[target_mask]
        other_df = df[~target_mask]
        if len(target_df) == 0:
            return None

        target_count = target_df["COUNT"].sum()
        other_grouped = (
            other_df.groupby("CODE").agg({"COUNT": "sum", "TIME": "first"}).reset_index()
        )
        similarities = [
            (code, self.similarity_dict.get(code, 0))
            for code in other_grouped["CODE"].values
        ]
        sorted_codes = sorted(similarities, key=lambda x: x[1], reverse=True)[
            : self.top_k
        ]

        codes = [target_code] + [code for code, _ in sorted_codes]
        counts = [target_count] + [
            other_grouped[other_grouped["CODE"] == code]["COUNT"].values[0]
            for code, _ in sorted_codes
        ]
        times = [target_df["TIME"].iloc[0]] + [
            other_grouped[other_grouped["CODE"] == code]["TIME"].values[0]
            for code, _ in sorted_codes
        ]
        return codes, counts, times

    # ------------------------------------------------------------------
    # Sampling and Tensor Conversion
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        """Return a patient sample, with dynamic resampling if enabled."""
        if self.training and self.pos_ratio is not None:
            if random.random() < self.pos_ratio:
                idx = random.choice(self.pos_indices)
            else:
                idx = random.choice(self.neg_indices)

        while True:
            patient_id = self.patient_ids[idx]
            result = self.get_patient_data(patient_id)
            if result is not None:
                codes, counts, times = result
                break
            idx = random.randint(0, len(self.patient_ids) - 1)

        indices = []
        for code in codes:
            idv = self.code_mapping[self.code_mapping["CODE"] == code]["Index"].values
            indices.append(idv[0] if len(idv) > 0 else 0)

        code_indices = torch.tensor(indices, dtype=torch.long)
        patient_code_embeddings = self.code_embeddings[code_indices]
        counts = torch.tensor(counts, dtype=torch.float)
        times = torch.tensor(times, dtype=torch.float)

        seq_len = len(code_indices)
        if seq_len > self.max_seq_len:
            code_indices = code_indices[: self.max_seq_len]
            patient_code_embeddings = patient_code_embeddings[: self.max_seq_len]
            counts = counts[: self.max_seq_len]
            times = times[: self.max_seq_len]
            attention_mask = torch.ones(self.max_seq_len, dtype=torch.bool)
        else:
            padding_length = self.max_seq_len - seq_len
            code_indices = torch.cat(
                [code_indices, torch.zeros(padding_length, dtype=torch.long)]
            )
            patient_code_embeddings = torch.cat(
                [
                    patient_code_embeddings,
                    torch.zeros(padding_length, patient_code_embeddings.size(1)),
                ]
            )
            counts = torch.cat([counts, torch.zeros(padding_length)])
            times = torch.cat([times, torch.zeros(padding_length)])
            attention_mask = torch.cat(
                [
                    torch.ones(seq_len, dtype=torch.bool),
                    torch.zeros(padding_length, dtype=torch.bool),
                ]
            )

        patient_info = self.patient_summary.loc[patient_id]
        if not pd.isna(patient_info["FINALPAH_gold"]):
            label = torch.tensor(patient_info["FINALPAH_gold"], dtype=torch.float)
        else:
            label = torch.tensor(patient_info[self.label_column], dtype=torch.float)
        gold_label = torch.tensor(
            patient_info["FINALPAH_gold"]
            if not pd.isna(patient_info["FINALPAH_gold"])
            else -1,
            dtype=torch.long,
        )

        return {
            "patient_id": patient_id,
            "code_indices": code_indices,
            "code_embeddings": patient_code_embeddings,
            "counts": counts,
            "times": times,
            "attention_mask": attention_mask,
            "label": label,
            "gold_label": gold_label,
        }


# ----------------------------------------------------------------------
# Time-Based (Visit-Level) Dataset
# ----------------------------------------------------------------------

class TimeBasedPatientDataset(Dataset):
    """
    WEST visit-level dataset class.

    Each sample represents a single visit (time point) for a patient.
    Useful for time-based modeling or visit-level supervision.
    """

    def __init__(
        self,
        data_dir,
        code_embeddings,
        code_mapping,
        max_codes_per_visit=50,
        top_k=50,
        training=True,
        pos_ratio=None,
        summary_file_name="patient_summary_KOMAP.csv",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.code_embeddings = code_embeddings
        self.code_mapping = code_mapping
        self.max_codes_per_visit = max_codes_per_visit
        self.top_k = top_k
        self.training = training
        self.pos_ratio = pos_ratio

        summary_file = os.path.join(data_dir, summary_file_name)
        self.patient_summary = pd.read_csv(summary_file)
        self.patient_summary.set_index("ID", inplace=True)
        self.patient_summary.index = self.patient_summary.index.astype(str)

        subset = "Training" if training else "Validation"
        self.subset = subset
        subset_dir = os.path.join(data_dir, subset)
        patient_ids = [f.split(".")[0] for f in os.listdir(subset_dir) if f.endswith(".csv")]

        print("Preparing visit-level samples...")
        self.samples = []
        self.patient_data_cache = {}
        target_code = "PheCode:415.2"
        total_visits = 0
        filtered_visits = 0

        for pid in patient_ids:
            df = pd.read_csv(os.path.join(subset_dir, f"{pid}.csv"))
            df["TIME"] = df["TIME"].astype(int)
            self.patient_data_cache[pid] = df
            unique_times = sorted(df["TIME"].unique())
            total_visits += len(unique_times)
            for time in unique_times:
                visit_df = df[df["TIME"] == time]
                if target_code in visit_df["CODE"].values:
                    self.samples.append((pid, time))
                else:
                    filtered_visits += 1

        similarity_file = os.path.join(data_dir, "code_similarities.csv")
        self.code_similarities = pd.read_csv(similarity_file)
        self.similarity_dict = dict(
            zip(
                self.code_similarities["code"],
                self.code_similarities["similarity"],
            )
        )

        print(f"\n{'='*50}")
        print(f"Time-Based Dataset Statistics ({subset}):")
        print(f"Number of patients: {len(patient_ids)}")
        print(f"Total visits before filtering: {total_visits}")
        print(f"Filtered visits (no {target_code}): {filtered_visits}")
        print(f"Final number of visits: {len(self.samples)}")
        print(f"Average visits per patient: {len(self.samples)/len(patient_ids):.2f}")
        print(f"{'='*50}\n")

    def __len__(self):
        return len(self.samples)

    def get_visit_data(self, patient_id, time):
        """Retrieve data for a specific visit."""
        df = self.patient_data_cache[patient_id]
        visit_df = df[df["TIME"] == time].copy()
        target_code = "PheCode:415.2"
        target_mask = visit
