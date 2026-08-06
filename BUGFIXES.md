# Bug fixes

Five defects in the WEST pipeline, each of which either crashes on current library
versions or produces a wrong result without any indication in the logs. The changes are
deliberately minimal: no new command-line arguments, no new configuration keys, no changes
to model architecture, defaults, or anything that would alter published results.

| # | Defect | Symptom | Files |
|---|---|---|---|
| 1 | Training and evaluation read different embedding files | `FileNotFoundError`, or a model scored against embeddings it was never trained on | `eval.py`, `train_v2_hyperparameters.py`, `train_fold{1,2,all}.py` |
| 2 | `grep -oP` is a GNU extension | Wrong training round selected, silently, on macOS/BSD | `run_next_round.sh`, `train_next_round_fold.sbatch` |
| 3 | Early-stopping counter never accumulates | `--early_stopping_patience` has no effect | `train_v2_hyperparameters.py` |
| 4 | `--device` is parsed and ignored | `AssertionError: Torch not compiled with CUDA enabled` on CPU | `train_v2_hyperparameters.py` |
| 5 | Checkpoints unreadable under `torch >= 2.6` | `UnpicklingError` on every `eval.py` run | `eval.py`, `train_v2_hyperparameters.py` |

Total diff: 7 files, ~50 insertions, ~23 deletions.

---

## 1. Training and evaluation read different embedding files

`Scripts/train_v2_hyperparameters.py:203-204` and the three `train_fold*.py` scripts read:

```python
datax   = pd.read_csv('.../Transformer/Input/Embeddings.csv')
mapping = pd.read_csv('.../Transformer/Input/Mapping.csv')
```

while `Scripts/eval.py:90-91` reads:

```python
datax   = pd.read_csv("./Input/MUGS_Codified_Python_ARCH_JULY16.csv")
mapping = pd.read_csv("./Input/MUGS_Code_Mapping.csv")
```

**Why this matters.** These are different filenames. `code_proj` is a learned
`nn.Linear(500, d_model)`, so as long as both files are 500-dimensional the tensor shapes
match and nothing raises — the model is simply scored against a concept space it never saw.
The result is an AUC near 0.5 with a completely healthy-looking log. Separately, the
`.../Transformer/` prefix is a placeholder that resolves nowhere, so the training scripts as
published raise `FileNotFoundError` before reaching the model.

**Fix.** Both now read the paths the README documents (`Input/Embeddings.csv` and
`Input/Mapping.csv`), relative to the project root, matching the convention every other path
in the pipeline uses (`data_path: "./Data"`, `./HyperparamSearch/...`).

```diff
-    datax = pd.read_csv('.../Transformer/Input/Embeddings.csv')
-    mapping = pd.read_csv('.../Transformer/Input/Mapping.csv')
+    datax = pd.read_csv('./Input/Embeddings.csv')
+    mapping = pd.read_csv('./Input/Mapping.csv')
```

If `MUGS_Codified_Python_ARCH_JULY16.csv` was the intended file for both, changing the two
training scripts instead would be equally correct — the point is only that they must match.
Happy to flip it either way.

---

## 2. `grep -oP` selects the wrong round on macOS and BSD

`run_next_round.sh:29`:

```bash
LATEST_EVAL_ROUND=$(ls "${EVAL_DIR}" 2>/dev/null | grep -oP 'round\K[0-9]+' | sort -n | tail -1)
```

`-P` (PCRE) is a GNU extension. BSD `grep`, which is what macOS ships, does not have it.

**Why this matters — and why `set -e` does not save you.** The usage error goes to *stderr*,
so the command substitution captures an empty string. The pipeline's exit status is that of
its last command, `tail -1`, which succeeds — so `set -e` never fires. The script then prints
"No previous evaluations found. Assuming base Round 0." and proceeds to update silver labels
using the wrong round's evaluation results. The two `DATE=$(... || true)` sites swallow the
failure explicitly.

Reproduced on macOS 15 with stock `/usr/bin/grep` against a directory containing
`AUC_10142025_round1_*` and `AUC_11202025_round2_*`:

```
grep: invalid option -- P
LATEST_EVAL_ROUND=''      <- should be 2
DATE=''                   <- should be 11202025
exit status: reached the end without set -e firing
```

**Fix.** POSIX `sed`, which behaves identically on GNU and BSD:

```diff
-LATEST_EVAL_ROUND=$(ls "${EVAL_DIR}" 2>/dev/null | grep -oP 'round\K[0-9]+' | sort -n | tail -1)
+LATEST_EVAL_ROUND=$(ls "${EVAL_DIR}" 2>/dev/null \
+  | sed -n 's/.*round\([0-9][0-9]*\).*/\1/p' | sort -n | tail -1)
```

```diff
-  | head -1 | grep -oP 'AUC_\K[0-9]{8}' || true)
+  | head -1 | sed -n 's/.*AUC_\([0-9]\{8\}\).*/\1/p' || true)
```

Same output as the original on GNU systems:

```
LATEST_EVAL_ROUND='2'
round1 DATE='10142025'   round2 DATE='11202025'   round3 DATE=''   (fallback: 11202025)
```

Four sites: three in `run_next_round.sh`, one in `Scripts/train_next_round_fold.sbatch:50`.

---

## 3. Early stopping never fires

`Scripts/train_v2_hyperparameters.py:414-435`:

```python
def save_best_model(auc_value, best_auc, tag):
    if auc_value > best_auc:
        torch.save(...)
        return auc_value, 0
    return best_auc, patience_counter + 1      # reads the OUTER counter

best_val_auc_fold1, patience_counter = save_best_model(val_auc_fold1, ...)
best_val_auc_fold2, patience_counter = save_best_model(val_auc_fold2, ...)
best_val_auc_all,   patience_counter = save_best_model(val_auc_all,   ...)

if patience_counter >= args.early_stopping_patience:
    break
```

**Why this matters.** Each of the three calls reads `patience_counter` from the enclosing
scope — unchanged since the previous epoch — and returns either `0` or that value `+ 1`. The
third call's return value overwrites the first two. So the counter can only ever hold `0` or
`1`, and with the configured `early_stopping_patience` of 15
(`hyperparameter_search.py:43`) the stopping condition is unreachable. Every run trains its
full `num_epochs`.

The selected model is still the best checkpoint seen, so results are unaffected — the cost is
compute (each of ~20 search configurations burns its full 30 or 50 epochs) and the loss of
protection against late-training divergence.

**Fix.** `save_best_model` returns whether it improved and no longer touches the counter; the
counter is updated once per epoch against the metric used for selection.

```diff
-                    return auc_value, 0
-                return best_auc, patience_counter + 1
+                    return auc_value, True
+                return best_auc, False
 
-            best_val_auc_fold1, patience_counter = save_best_model(...)
-            best_val_auc_fold2, patience_counter = save_best_model(...)
-            best_val_auc_all, patience_counter = save_best_model(...)
+            best_val_auc_fold1, _ = save_best_model(...)
+            best_val_auc_fold2, _ = save_best_model(...)
+            best_val_auc_all, improved_overall = save_best_model(...)
+
+            if improved_overall:
+                patience_counter = 0
+            else:
+                patience_counter += 1
```

Verified with `--num_epochs 25 --early_stopping_patience 3` on a small synthetic cohort where
validation AUC saturates at epoch 1:

```
Epoch 1/25   AUC 1.0000   New best AUC for overall: 1.0000
Epoch 2/25   AUC 1.0000
Epoch 3/25   AUC 1.0000
Epoch 4/25   AUC 1.0000
Early stopping triggered after 4 epochs
```

Before the change this ran all 25 epochs.

`train_fold{1,2,all}.py` were checked and do **not** have this bug — their counter logic is
already correct, and they are left untouched.

---

## 4. `--device` is parsed and ignored

`Scripts/train_v2_hyperparameters.py:176`:

```python
device = torch.device(args.local_rank if args.local_rank != -1 else "cuda")
```

`--device` is declared at line 123 and never read. Passing `--device cpu` is accepted and
then ignored, and the run dies at the first `.to(device)`:

```
AssertionError: Torch not compiled with CUDA enabled
```

**Fix.**

```diff
-    device = torch.device(args.local_rank if args.local_rank != -1 else "cuda")
+    if args.local_rank != -1:
+        device = torch.device(args.local_rank)
+    else:
+        device = torch.device(args.device)
```

The default is still `"cuda"`, so GPU behaviour is unchanged. `eval.py` already honours
`--device` correctly and is untouched here.

---

## 5. Checkpoints cannot be loaded under `torch >= 2.6`

`torch` 2.6 changed `torch.load`'s `weights_only` default from `False` to `True`.
`best_val_auc` is stored as the `numpy.float64` that `roc_auc_score` returns, which the
restricted unpickler rejects:

```
_pickle.UnpicklingError: Weights only load failed.
  WeightsUnpickler error: Unsupported global: GLOBAL numpy._core.multiarray.scalar
  was not an allowed global by default.
```

**Why this matters.** Every `eval.py` run fails on any checkpoint the pipeline has ever
written. This is not hypothetical — `pip install torch` today gives 2.6+.

**Fix.** Both sides:

```diff
-                        "best_val_auc": auc_value,
+                        "best_val_auc": float(auc_value),
```

so newly written checkpoints load under either default, and:

```diff
-    checkpoint = torch.load(args.model_path, map_location=device)
+    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
```

so checkpoints already on disk still load. `weights_only=False` is appropriate here because
the checkpoint intentionally carries non-tensor metadata (epoch, best AUC, optimizer and
scheduler state) and is produced by this pipeline rather than fetched from elsewhere.

---

## How this was verified

There is no test suite, so verification was done by running the pipeline end-to-end against a
synthetic project: 240 patients, 31 concepts, 500-dimensional embeddings (matching the
hard-coded `nn.Linear(500, d_model)`), on CPU with `torch` 2.8, `transformers` 4.57.

```bash
python -m Scripts.train_v2_hyperparameters \
  --data_path ./Data --summary_file_name patient_summary_KOMAP_round0.csv \
  --hidden_dim 32 --num_heads 2 --num_layers 2 --max_seq_len 12 \
  --num_epochs 25 --early_stopping_patience 3 --pos_ratio 0.5 \
  --use_ema --use_augmentation --label_column KOMAP_calibrated \
  --device cpu --save_dir ./Experiments/t

python -m Scripts.eval --data_path ./Data \
  --summary_file_name patient_summary_KOMAP_round0.csv \
  --model_path ./Experiments/t/best_overall_patient_transformer_single.pt \
  --sub_set validation --hidden_dim 32 --num_heads 2 --num_layers 2 \
  --max_seq_len 12 --device cpu --output_dir ./Evaluation/t
```

Both complete; evaluation writes `evaluation_results_with_clusters_validation.csv` and
`cluster_statistics_validation.csv` as expected. Before these changes, training failed at the
embedding load, and evaluation failed at `torch.load`.

One undocumented constraint worth recording, encountered while building the fixture:
**`Data/Validation/` must contain only gold-labelled patients.** `PatientDataset` assigns
`gold_label = -1` to patients without a gold label, and validation AUC is computed against
`gold_label`, so a single silver patient in the validation split makes `roc_auc_score` see
three classes and raise `ValueError: multi_class must be in ('ovo', 'ovr')`. This matches the
described design — held-out gold patients form the validation set — but the failure mode is
opaque. A one-line assertion or a note in the README would help.

---

## Deliberately not changed

These are real issues but they are design decisions, would change model behaviour, or would
alter published results, so they are out of scope for a bug-fix PR. Listed in case they are
useful.

- **`nn.Linear(500, d_model)` is hard-coded** (`model_v2.py:169`). Embeddings must be exactly
  500-dimensional. This fails loudly with a shape error, so it is not a silent bug, but making
  it a parameter derived from the loaded file would remove a class of confusion.
- **Concept counts reach the model unnormalised.** `PatientDataset` loads
  `code_count_statistics.csv` into `code_stats_dict` (lines 215-221) and never reads it, while
  raw counts are projected and *added* to the semantic embedding. On a test cohort the largest
  count was 45x the typical embedding row norm, so the frequency component can dominate the
  meaning it is added to. Normalising would change results, so it needs your call.
- **`gold_repeat` is unreachable from any config.** It defaults to 10 in
  `PatientDataset.__init__`, is never passed by a training script, and is in neither
  `get_hyperparameter_space()` nor `get_fixed_parameters()`. Since the paper describes choosing
  it so the effective label distribution matches expected prevalence, exposing it seems
  worthwhile — but it is a feature, not a fix.
- **The `gold_repeat` log message says "positive samples"** (`patient_dataset.py:158`) while
  the code replicates all gold patients, positive and negative alike.
- **`eval.py` re-selects an F1-optimal threshold each round**, and that threshold binarises
  training-set predictions for the silver-label update — so which patients get relabelled can
  change between rounds independently of the model. `--pred_threshold` already exists to pin it.
- **`sigmoid` + `BCELoss`** rather than the fused, more numerically stable
  `BCEWithLogitsLoss`.
- **The contrastive loss requires both `use_ema` and `contrastive_weight > 0`**
  (lines 320 and 326) and `contrastive_weight` is fixed at `0.0`, so it never runs.
- **`top_k` is overwritten by `max_seq_len`** in every caller, making `--top_k` inert.
- **Three of the four augmentation methods are unreachable** because `augmentation_method` is
  never set, so the `"truncate_v3"` default always wins.
- **A `ZeroDivisionError` in the dataset statistics block** (`patient_dataset.py:135`) if the
  cohort contains no gold patients.
- **`Index` vs `INDEX`**: the README documents the mapping column as `INDEX`,
  `patient_dataset.py:356` reads `"Index"`.
