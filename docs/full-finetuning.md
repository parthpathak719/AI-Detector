# Full MuRIL fine-tuning experiment

This is a new **all-layer** experiment. It is separate from the completed final-block experiment in [finetuning-results.md](finetuning-results.md). Do not treat that older experiment's numbers as results of this run.

The run was launched on 22 October 2026 on the cloud CPU. The committed protocol specifies the planned experiment; fold files record completed work. **No full-fine-tuning improvement or new production winner has been established yet.** Read the live local status command below to distinguish pending work from completed results.

## Scope and evaluation

- All MuRIL token/position/type embeddings and all 12 transformer blocks are trainable. The unused BERT pooler is excluded because inference uses masked-mean pooling. A temporary supervised head learns with the encoder; serving uses embeddings plus character n-grams.
- Three encoder learning rates: `2e-6`, `5e-6`, `1e-5`. Head learning rate `1e-4`; AdamW; weight decay `0.01` excluding biases and LayerNorm scales; batch size 16; 10% linear warmup then decay; clipping at 1.0; label smoothing 0.05.
- At most five epochs; early stopping after two epochs without inner validation macro-F1 improvement of more than 0.0001. The best inner-validation checkpoint is restored.
- **Five-fold StratifiedGroupKFold on the original training split only**, using seed 42. Frozen and full-fine-tuned candidates share identical outer folds. The existing validation and test splits are not pooled into CV.
- Within each outer training fold, a separate grouped split reserves approximately 20% for early stopping. The encoder sees only the remaining approximately 80% of that fold. After restoring its best checkpoint, the hybrid classifier fits all outer-training examples; the outer holdout has no role in encoder training or early stopping.
- Twelve predefined hybrid configurations per representation vary logistic-regression C, embedding weight, character n-gram range, and minimum document frequency. Vocabularies/scalers fit only outer-training data.
- This is 45 full-encoder training runs (three learning rates × five folds × three tasks), plus 15 frozen folds and final refits. It is a bounded search, not every possible setting. CPU runtime is many hours, potentially over a day.
- Selection maximizes mean outer-fold macro-F1; accuracy and fold variation are also recorded. Ties prefer frozen MuRIL. These CV scores are tuning scores, not unbiased final generalization estimates.
- If a full encoder wins, refit it on the complete original training split for the median selected best epoch count across its five folds. Calibrate the hybrid only on the original validation split. Evaluate the existing test split after saving the selection decision; do not change selection based on test accuracy.
- The same source-group IDs supplied by the original dataset define independence. They do not guarantee that every semantic near-duplicate or generator/topic dependence has been identified. Existing test data has been examined historically; new independent evaluation remains desirable. Task-level results do not measure the long-document cascade.

## Run on a Colab GPU

[Open the prepared notebook](https://colab.research.google.com/github/parthpathak719/AI-Detector/blob/main/notebooks/Full_MuRIL_Finetuning.ipynb), select **Runtime → Change runtime type → T4 GPU**, then **Run all** and authorize Google Drive. A connected Google account and an available Colab GPU are required; the cloud coding environment cannot allocate a GPU through the notebook link alone.

The notebook pins a source commit and uses uv to provision Python 3.12 independently of Colab’s system Python/ensurepip, then installs CUDA-enabled PyTorch 2.6 in a separate virtual environment. It imports the original supplied archive, recomputes all 15 frozen baseline folds under the same SciPy version, and runs the full GPU search independently of the ongoing CPU fine-tuning run. The notebook uses SciPy 1.14.1 as a compatibility workaround for the observed SciPy 1.15.3 L-BFGS segmentation fault during head initialization. A real-data optimizer preflight runs first. All 15 head initializations and three hybrid-fit/calibration checks passed locally; GPU execution of the workaround still requires Colab. CPU fine-tuned checkpoints are not mixed with GPU runs. No completed GPU result is claimed before execution.

Completed fold reports, logs and final models persist in `MyDrive/AI-Detector/full-muril-gpu-v2-scipy114`. Epoch checkpoints stay on local disk to prevent Drive Bin growth. Keep several GB free for final models and exports, in addition to any legacy checkpoints. After a Colab runtime reset, reconnect a GPU and run all cells again using the same run folder; the unfinished fold/refit may need to replay. Use only one Colab runtime per run folder. GPU availability/session duration are controlled by Colab; the notebook does not bypass their limits.

Once all comparisons, refits and application-loader parity checks pass, the notebook exports `results.zip` and `selected_models.zip` to Drive. The small results ZIP can be shared for review. The model ZIP contains the complete selected `artifacts/` directory, including full encoder checkpoints only where selected, and excludes private parity passages and optimizer states. Back up your local artifacts, extract the model ZIP, verify checksums and restart the app using the notebook's Windows instructions.

The Colab launch requires user authentication. The CPU run can continue/resume here while the notebook awaits a GPU session. Workflow tests and baseline provenance checks run in this environment; actual CUDA execution requires Colab and has not been verified here.

## Run and resume

From the repository root, with the supplied strict-cleaned datasets and pinned MuRIL download already available:

```powershell
# Windows: use the project's Python 3.12 virtual environment.
.\.venv\Scripts\python.exe -u -m scripts.full_finetune
.\.venv\Scripts\python.exe -m scripts.full_finetuning_status
```

On Linux use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`. The runner uses CUDA if available, otherwise CPU; the current cloud run has no GPU. No retraining is needed to use the existing app.

A process lock prevents concurrent runs. Completed folds are skipped on restart; interrupted training resumes from the last complete epoch, including optimizer and random state. The current incomplete epoch is replayed. A fixed protocol, source-code hash and dataset hashes prevent silently combining changed settings with old results. To run a different protocol, use separate report/cache directories in a separate checkout. Epoch checkpoints are several GB; allow substantial free disk and RAM.

Each first training batch audits nonzero gradients in the word embeddings and every transformer block. An automated small-model test verifies interruption/recovery. The completed run checks selected encoder embeddings and hybrid scores through the actual application loader before declaring success.

## Output and deployment

- `reports/full_finetuning/protocol.json`: predeclared settings and hashes.
- `reports/full_finetuning/task*/`: completed fold metrics, held-out prediction arrays, selection decision and eventual test report. Fold prediction indices refer to local dataset order; no document text is exported in these reports.
- `.cache/full_finetuning/status.json`: last worker update, including PID and timestamp; a saved timestamp alone does **not** prove the process is still alive.
- `.cache/full_finetuning/`: private resumable checkpoints and gradient audits.
- `.cache/full_finetuning/selected_artifacts/`: eventual serving bundles and full checkpoints, after refitting. The app supports both frozen and full-encoder artifacts.

The run **stages** selected models and verifies them. It does not automatically overwrite the working application or publish large checkpoints to Git. After completion, review the final results and serving checks, then install the selected bundles as a complete set and restart the app. A full MuRIL checkpoint is roughly 950 MB, so any selected full models need an appropriate download/release channel rather than ordinary Git blobs. Current production models remain the previously validated frozen hybrids until this step is completed.

## Drive storage and interrupted Colab runs

Older launchers wrote `latest.pt` (model plus Adam optimizer, about 2.64 GiB)
through the Drive mount every epoch. Replacing these files can leave the old
copies in Drive's Bin, so the old 15 GB estimate did not account for accumulated
trash. Identical filenames also occur in different folds; they are not
necessarily interchangeable snapshots.

The updated launcher keeps epoch checkpoints on Colab's **local disk**, and
writes completed fold reports and final selected artifacts to Drive. On a fresh
runtime it restores an available older Drive checkpoint for an unfinished fold,
without modifying that source. Otherwise only the unfinished fold/refit may need
to repeat. On the same runtime local epoch checkpoints can resume. This changes
storage only: the training source, folds, search and SciPy version stay unchanged.
Continue using **full-muril-gpu-v2-scipy114**, not a new run folder. The v1 folder
is the separate attempt before the SciPy workaround; do not merge its results.

Use `notebooks/Inspect_MuRIL_Run.ipynb` on a **CPU runtime** to download a small
inventory and reports ZIP, even when GPU quota is exhausted. It does not train,
load model files, or delete anything. Keep the active v2 folder intact. Review
Bin entries and permanently remove only obsolete training snapshots after
checking that the current resume files remain in the active folder. Do not
empty unrelated files from the Bin blindly; permanent deletion is irreversible.

Clearing Drive space does not restore GPU quota. Resume with the updated training
notebook when Colab grants GPU access again (no guaranteed reset time), or use
paid Colab compute if desired. Another GPU machine can run the same experiment
with the same run directory and dependencies after transferring the saved run.
Do not run two machines against one run folder simultaneously. Final model ZIPs
also duplicate selected weights, so export them once when needed.
