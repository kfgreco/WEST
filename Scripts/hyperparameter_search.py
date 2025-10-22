#!/usr/bin/env python3
# ==============================================================
# WEST: Hyperparameter Search Generator
# --------------------------------------------------------------
# Defines the WEST hyperparameter search space, generates random
# experiment configurations, and creates training and SLURM
# submission scripts for model training.
# ==============================================================

import os
import json
import random
import itertools
import argparse
from datetime import datetime


# ----------------------------------------------------------------------
# Hyperparameter Definitions
# ----------------------------------------------------------------------

def get_hyperparameter_space():
    """Define the hyperparameter search space."""
    return {
        "batch_size": [64, 128, 256],
        "learning_rate": [5e-4, 1e-3, 2e-3],
        "num_epochs": [30, 50],
        "hidden_dim": [32, 64, 128],
        "num_layers": [2, 3, 4],
        "dropout": [0.3, 0.7],
    }


def get_fixed_parameters():
    """Define fixed parameters that remain constant across experiments."""
    return {
        "data_path": "./Data",
        "num_heads": 2,
        "warmup_steps": 50,
        "weight_decay": 0.07,
        "max_grad_norm": 10.0,
        "print_every": 1,
        "early_stopping_patience": 15,
        "temperature": 0.07,
        "contrastive_weight": 0.0,
        "pos_ratio": 0.5,
        "use_ema": True,
        "use_augmentation": True,
        "max_seq_len": 70,
        "output_type": "mean",
        "remove_gold_negative": False,
        "model_name": "patient_transformer_single.pt",
        "label_column": "KOMAP_calibrated",
        "summary_file_name": "patient_summary_KOMAP_round0.csv",
        "round": 1,
    }


# ----------------------------------------------------------------------
# Experiment Configuration Generation
# ----------------------------------------------------------------------

def generate_random_combination(param_space, fixed_params):
    """Sample one random combination of hyperparameters."""
    combination = fixed_params.copy()
    for param, values in param_space.items():
        combination[param] = random.choice(values)

    # Ensure hidden_dim is compatible with num_heads
    if combination["hidden_dim"] % combination["num_heads"] != 0:
        combination["hidden_dim"] = (
            combination["hidden_dim"] // combination["num_heads"]
        ) * combination["num_heads"]

    return combination


def create_experiment_config(combination, experiment_id, output_dir):
    """Save configuration for a single experiment."""
    config = combination.copy()
    config["experiment_id"] = experiment_id
    config["save_dir"] = f"./HyperparamSearch/{experiment_id}"
    config["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(config["save_dir"], exist_ok=True)
    config_path = os.path.join(output_dir, f"{experiment_id}.json")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return config_path


def create_training_script(experiment_id, config_file):
    """Generate a shell script that launches WEST training for this experiment."""
    script_content = f"""#!/bin/bash
# Auto-generated training script for WEST experiment {experiment_id}

CONFIG_FILE="{config_file}"
EXPERIMENT_ID="{experiment_id}"

echo "Starting WEST experiment: $EXPERIMENT_ID"
echo "Using config file: $CONFIG_FILE"

mkdir -p "./HyperparamSearch/$EXPERIMENT_ID"
cp "$CONFIG_FILE" "./HyperparamSearch/$EXPERIMENT_ID/config.json"

python -m Scripts.train_v2_hyperparameters \\
    --config="$CONFIG_FILE" \\
    2>&1 | tee "./HyperparamSearch/$EXPERIMENT_ID/training_log.txt"

echo "Experiment $EXPERIMENT_ID completed."
"""
    script_path = f"HyperparamSearch/configs/train_{experiment_id}.sh"
    with open(script_path, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    return script_path


def create_slurm_job(experiment_id, script_file):
    """Create a SLURM job submission script for this experiment."""
    slurm_content = f"""#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --job-name=hp_{experiment_id}
#SBATCH --output=HyperparamSearch/logs/{experiment_id}_%j.out
#SBATCH --error=HyperparamSearch/logs/{experiment_id}_%j.err

conda activate deeplearning
cd .../Transformer
bash {script_file}
"""
    slurm_path = f"HyperparamSearch/configs/{experiment_id}.sbatch"
    with open(slurm_path, "w") as f:
        f.write(slurm_content)
    return slurm_path


# ----------------------------------------------------------------------
# Batch Submission Utility
# ----------------------------------------------------------------------

def create_batch_submit_script(experiments):
    """Create a single script to submit all SLURM jobs."""
    lines = [
        "#!/bin/bash",
        "# Auto-generated batch submission script for WEST hyperparameter search",
        "echo 'Submitting all WEST hyperparameter search experiments...'",
        "EXPERIMENT_IDS=()",
        "",
    ]

    for exp in experiments:
        lines += [
            f"echo 'Submitting {exp['experiment_id']}...'",
            f"JOB_ID=$(sbatch {exp['slurm_file']} | awk '{{print $4}}')",
            f"EXPERIMENT_IDS+=(\"$JOB_ID\")",
            f"echo '   └── Job ID: '$JOB_ID",
            "",
        ]

    submit_file = "HyperparamSearch/submit_all.sh"
    with open(submit_file, "w") as f:
        f.write("\n".join(lines))
    os.chmod(submit_file, 0o755)
    return submit_file


# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------

def generate_search_experiments(num_experiments=20, seed=42):
    """Generate experiment configurations, scripts, and SLURM jobs."""
    random.seed(seed)
    param_space = get_hyperparameter_space()
    fixed_params = get_fixed_parameters()

    config_dir = "HyperparamSearch/configs"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs("HyperparamSearch/logs", exist_ok=True)

    experiments = []
    for i in range(num_experiments):
        exp_id = f"hp_{i+1:03d}"
        combo = generate_random_combination(param_space, fixed_params)
        config_path = create_experiment_config(combo, exp_id, config_dir)
        train_script = create_training_script(exp_id, config_path)
        slurm_file = create_slurm_job(exp_id, train_script)

        experiments.append({
            "experiment_id": exp_id,
            "config_file": config_path,
            "slurm_file": slurm_file,
        })

    with open("HyperparamSearch/experiment_summary.json", "w") as f:
        json.dump(experiments, f, indent=2)

    return experiments


def main():
    parser = argparse.ArgumentParser(description="Generate WEST hyperparameter search experiments.")
    parser.add_argument("--num_experiments", type=int, default=20,
                        help="Number of experiments to generate.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    print("WEST Hyperparameter Search Generator")
    print("====================================")
    print(f"Number of experiments: {args.num_experiments}")
    print(f"Random seed: {args.seed}")
    print("")

    experiments = generate_search_experiments(args.num_experiments, args.seed)
    submit_script = create_batch_submit_script(experiments)

    print(f"Generated {len(experiments)} WEST experiments.")
    print(f"Batch submission script: {submit_script}")
    print("")
    print("Next steps:")
    print("  1. Review generated configs in HyperparamSearch/configs/")
    print("  2. Submit all experiments with: bash run_hyperparam_search.sh [num_experiments] [seed]")
    print("  3. Monitor jobs with: squeue -u $USER")
    print("  4. Collect results with: python Scripts/collect_results.py")


if __name__ == "__main__":
    main()
