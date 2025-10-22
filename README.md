# WEST: WEakly Supervised Transformer for Rare Disease Phenotyping and Subphenotyping from Electronic Health Records

WEST is a framework for data-efficient computational phenotyping and subphenotyping using electronic health records (EHRs). It combines a limited set of expert-validated ("gold-standard") labels with large-scale probabilistic ("silver-standard") labels, which are iteratively refined across training rounds. The provided pipeline automates hyperparameter optimization, model evaluation, cross-validation–based model selection, silver-label refinement, and retraining, enabling fully automated multi-round weakly supervised learning for EHR phenotyping.

---

## Repository Structure

Your project directory should be named `Transformer` and organized as follows:

```
Transformer/
├── Data/
│   ├── Training/
│   │   ├── [ID].csv
│   │   └── ...
│   ├── Validation/
│   │   ├── [ID].csv
│   │   └── ...
│   ├── code_count_statistics.csv
│   ├── code_similarities.csv
│   └── patient_summary_round0.csv
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

## Input and Data Files

The WEST model requires **precomputed embeddings and code mappings** located in the `Input/` directory, as well as **patient-level data and code summaries** located in the `Data/` directory.

### **1. Patient-Level EHR Files**

Located in:

```
Transformer/Data/Training/
Transformer/Data/Validation/
```

Each file corresponds to a single patient (`[ID].csv`), representing a chronological sequence of medical codes and encounters.

**Example (`111111.csv`):**

```
"ID","CODE","COUNT","TIME"
"111111","PheCode:381.2",1,23
"111111","PheCode:395",1,0
"111111","LOINC:18768-2",1,29
"111111","CCS:227",1,29
"111111","RXNORM:89905",1,2
```

**Variable descriptions:**

* **ID** — Unique patient identifier.
* **CODE** — Clinical concept code (e.g., PheCode, LOINC, CCS, RXNORM, etc.) observed during a visit.
* **COUNT** — Number of times the code appears within a single timepoint or encounter.
* **TIME** — Temporal ordering index or time interval associated with the event.

These patient-level files are read by the `PatientDataset` class to build input sequences for the transformer model.

---

### **2. `code_count_statistics.csv`**

Aggregated statistics summarizing code usage across the entire cohort.

**Variable descriptions:**

* **CODE** — Medical code identifier (e.g., PheCode, LOINC, CCS, RXNORM, etc.).
* **total_count** — Total occurrences of the code across all patients.
* **mean_count** — Average number of occurrences per patient.
* **variance** — Variance in the count across patients.
* **std** — Standard deviation of the count distribution.
* **non_zero_ratio** — Proportion of patients with at least one instance of this code.

---

### **3. `code_similarities.csv`**

Pairwise similarity matrix for medical codes, typically derived from embeddings or co-occurrence statistics.

**Variable descriptions:**

* **code** — Medical code identifier (e.g., PheCode, LOINC, CCS, RXNORM, etc.).
* **similarity** — Numeric similarity score (e.g., cosine similarity or correlation).

---

### **4. `patient_summary_round0.csv`**

The baseline patient summary file linking demographic or cohort-level metadata with labeling information.

**Variable descriptions:**

* **ID** — Patient identifier linking to the corresponding `[ID].csv` file.
* **FINALPAH / FINALPAH_gold / FINALPAH_silver** — Binary labels for the target phenotype (gold-standard, silver-standard, or combined).
* **gold** — 1 if a patient has a gold-standard expert label.
* **training** — 1 if a patient is included in the training set.
* **KOMAP_calibrated** — Silver probability used for weak supervision.
* **kfold_2** — Cross-validation fold assignment (e.g., 1 or 2).

---

### **5. `Embeddings.csv`**

Contains precomputed vector embeddings for each medical code.
Each row corresponds to one code’s continuous embedding representation.

**Variable descriptions:**

* Each column represents one embedding dimension.
* Each row corresponds to a code, in the same order as defined in `Mapping.csv`.

Used to initialize the transformer’s code embedding matrix.

---

### **6. `Mapping.csv`**

Maps embedding indices to their corresponding medical code identifiers and categories.

**Variable descriptions:**

* **CODE** — Code identifier (matching those found in the patient-level files).
* **INDEX** — Row index corresponding to the code’s position in `Embeddings.csv`.

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

## Implementation and Computational Resources

The WEST framework is designed to run efficiently on a modern GPU-enabled computing environment. Training and evaluation can be executed either interactively or through SLURM job submissions (recommended for large-scale experiments).

| Task                                   | Recommended GPU                  | Typical Memory | Wall-Time (per round)                     |
| -------------------------------------- | -------------------------------- | -------------- | ----------------------------------------- |
| Hyperparameter Search (15–20 configs)  | 1× NVIDIA A100 / V100 / RTX 6000 | 16 GB          | ~6–10 hours total (0.3–0.5 hr per config) |
| Single-Fold Model Training             | 1× NVIDIA A100 / V100 / RTX 6000 | 16 GB          | ~2–4 hours                                |
| Evaluation (Validation + Training Set) | 1× GPU or CPU                    | 8 GB           | ~30–60 minutes                            |
| Silver-Label Update                    | CPU                              | < 4 GB         | ~5 minutes                                |
| Next-Round Training (3 folds)          | 1× GPU per fold                  | 16 GB each     | ~6–8 hours total                          |

These values assume a cohort of ~15 K patients with ~500 unique clinical codes.

---

## Citation

If you use WEST in your research, please cite:

Greco, K. F., Yang, Z., Li, M., Tong, H., Sweet, S. M., Geva, A., Mandl, K. D., Raby, B. A., & Cai, T. (2025).  
*A Weakly Supervised Transformer for Rare Disease Diagnosis and Subphenotyping from EHRs with Pulmonary Case Studies.*  

**arXiv preprint** [arXiv:2507.02998](https://arxiv.org/abs/2507.02998)
