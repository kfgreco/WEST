# WEST: WEakly Supervised Transformer for Rare Disease Phenotyping and Subphenotyping from Electronic Health Records

WEST is a framework for data-efficient computational phenotyping and subphenotyping using electronic health records (EHRs). It combines a limited set of expert-validated ("gold-standard") labels with large-scale probabilistic ("silver-standard") labels, which are iteratively refined across training rounds. The provided pipeline automates hyperparameter optimization, model evaluation, cross-validation–based model selection, silver-label refinement, and retraining, enabling fully automated multi-round weakly supervised learning for EHR phenotyping.

---

## Repository Structure

Your project directory should be named `Transformer` and organized as follows:

```
Transformer/
├── Evaluation/
├── Experiments/
├── HyperparamSearch/
├── Input/
│   ├── Embeddings.csv
│   └── Mapping.csv
├── Logs/
├── Scripts/
│   ├── collect_results.py
│   ├── copy_best_models_auto.sh
│   ├── eval.py
│   ├── hyperparameter_search.py
│   ├── loss.py
│   ├── model_v2.py
│   ├── patient_dataset.py
│   ├── run_eval_split2.sh
│   ├── run_hyperparam_search.sh
│   ├── summarize_fold_aucs_sorted_and_overall.py
│   ├── train_fold1.py
│   ├── train_fold2.py
│   ├── train_foldall.py
│   ├── train_next_round_fold.sbatch
│   ├── train_v2_hyperparameters.py
│   └── update_silver_label_custom.py
├── run_all_evals.sh
└── run_next_round.sh
```

---

## Environment Setup

```bash
conda create -n deeplearning python=3.12
conda activate deeplearning
pip install torch scikit-learn pandas tqdm jq
```

---

## Pipeline Overview

| Stage                        | Script                                                                                   | Description                                                                                                                |
| ---------------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **1. Hyperparameter Search** | `Scripts/run_hyperparam_search.sh`                                                       | Samples random configurations, trains models, and performs cross-validation across folds to assess performance.            |
| **2. Model Selection**       | `Scripts/summarize_fold_aucs_sorted_and_overall.py` + `Scripts/copy_best_models_auto.sh` | Summarizes fold-level and overall AUCs, ranking models by validation AUC and selecting top models for downstream training. |
| **3. Evaluation**            | `run_all_evals.sh`                                                                       | Evaluates selected models on validation and training subsets; saves results in `Evaluation/`.                              |
| **4. Silver-Label Update**   | `run_next_round.sh`                                                                      | Updates silver labels using training-set predictions from the latest models, generating refined patient summaries.         |
| **5. Iterative Training**    | `Scripts/train_next_round_fold.sbatch`                                                   | Trains WEST models for each fold using newly updated silver labels.                                                        |

---

## Input Files

The model requires precomputed embeddings and mapping files located in the `Input/` directory:

```
Transformer/Input/Embeddings.csv
Transformer/Input/Mapping.csv
```

Model training scripts expect these paths:

```python
datax = pd.read_csv('.../Transformer/Input/Embeddings.csv')
mapping = pd.read_csv('.../Transformer/Input/Mapping.csv')
```

---

## Usage Guide

### 1. Hyperparameter Search

Run hyperparameter tuning and model training:

```bash
cd Transformer
bash Scripts/run_hyperparam_search.sh [number of configurations] [seed]
```

---

### 2. Summarize and Select Best Models (Cross-Validation)

Summarize fold-level and overall AUCs from hyperparameter search:

```bash
cd Transformer/HyperparamSearch/logs
cp ../../Scripts/summarize_fold_aucs_sorted_and_overall.py .
python summarize_fold_aucs_sorted_and_overall.py
bash ../../Scripts/copy_best_models_auto.sh
```

This step performs cross-validation-based model selection and copies the top-performing models into:

```
Experiments/<DATE>_round1_fold1/
Experiments/<DATE>_round1_fold2/
Experiments/<DATE>_round1_foldall/
```

---

### 3. Evaluate First-Round Models

Run evaluation on all selected models:

```bash
cd Transformer
chmod +x run_all_evals.sh
./run_all_evals.sh
```

Results are saved in:

```
Evaluation/AUC_<DATE>_round1_fold*/
```

---

### 4. Update Silver Labels and Train the Next Round

Use the automated next-round pipeline to:

1. Evaluate training sets from previous round
2. Update silver labels
3. Launch next-round training jobs

```bash
cd Transformer
chmod +x run_next_round.sh
./run_next_round.sh
```

This step:

* Reads evaluation results from Round 1 (`Evaluation/AUC_<DATE>_round1_*`)
* Updates silver labels to create `patient_summary_KOMAP_round1_*` CSVs
* Launches Round 2 model training using `Scripts/train_next_round_fold.sbatch`

---

### 5. Evaluate Next Round Models

Once training completes, rerun evaluation:

```bash
cd Transformer
./run_all_evals.sh
```

Results are stored in:

```
Evaluation/AUC_<DATE>_round2_fold*/
```

---

## Citation

If you use WEST in your research, please cite:

> @article{greco2025weakly,
  title={A Weakly Supervised Transformer for Rare Disease Diagnosis and Subphenotyping from EHRs with Pulmonary Case Studies},
  author={Greco, Kimberly F and Yang, Zongxin and Li, Mengyan and Tong, Han and Sweet, Sara Morini and Geva, Alon and Mandl, Kenneth D and Raby, Benjamin A and Cai, Tianxi},
  journal={arXiv preprint arXiv:2507.02998},
  year={2025}
}
