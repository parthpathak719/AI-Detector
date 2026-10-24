# Training notebook workflow

**Training is already complete; neither notebook is needed to run the web app.** Only the two training notebooks used in the Colab → Kaggle workflow are retained. The standalone inspection notebook was removed; its underlying inspection script remains available.

| Notebook | Purpose |
|---|---|
| [Full_MuRIL_Finetuning.ipynb](../notebooks/Full_MuRIL_Finetuning.ipynb) | Colab full-encoder training, Drive persistence, model/result export |
| [Resume_MuRIL_Kaggle.ipynb](../notebooks/Resume_MuRIL_Kaggle.ipynb) | Resume the transferred experiment on Kaggle; this workflow completed the run |

Both retain the original training source commit `b9cf2495bfccf3008676ce052f6539b313328d59`. Updating their instructions does not change the experiment's folds or numerical settings. Notebook outputs are cleared; the committed results are under `reports/full_finetuning_completed/`.

### Colab: original training stage

1. Open [the Colab notebook](https://colab.research.google.com/github/parthpathak719/AI-Detector/blob/main/notebooks/Full_MuRIL_Finetuning.ipynb) and save a copy if editing it.
2. Choose **Runtime → Change runtime type → T4 GPU**. Free GPU access depends on availability and quota; a 24-hour reset is not guaranteed.
3. Review `RUN_ROOT`. The existing experiment is `MyDrive/AI-Detector/full-muril-gpu-v2-scipy114`. Keep that folder for a compatible resume; use a new directory for a genuinely different experiment. Do not merge the earlier v1 failed attempt into it.
4. Run cells in order and authorize Drive. The notebook provisions isolated Python 3.12, PyTorch 2.6/cu124, and SciPy 1.14.1, imports the original supplied data ZIP, downloads MuRIL and runs an optimizer preflight. If the original Drive download link is unavailable, set `ARCHIVE` to your own copy of the same archive.
5. Run the search. It reuses completed compatible fold results, performs final fitting and serving-parity checks, then exports `results.zip`, `selected_models.zip` and `results.md`.

Frequent epoch snapshots remain on runtime-local storage to avoid repeated multi-GB writes accumulating in Drive Bin. Completed folds and final artifacts persist in Drive. A fresh runtime may repeat an unfinished fold. Use one runtime per run directory. A fresh complete search can take many hours; no fixed free-session completion time is promised.

### Kaggle: resume and completion stage

1. Keep the notebook and its input dataset **private**. The input bundle includes private text/model material.
2. Upload the previously prepared `muril-resume.zip` as a private Kaggle dataset. Kaggle may expose the ZIP or its extracted files; the notebook accepts either. The bundle must contain `transfer_manifest.json`, `run/reports/`, saved selected artifacts/checkpoints and `data/clean_strict/`. A `results.zip` alone is **not** a resume bundle: it lacks training data and model state.
3. If recreating the original Colab-to-Kaggle transfer, use [scripts/prepare_kaggle_transfer.py](../scripts/prepare_kaggle_transfer.py) in a **Colab CPU notebook**: paste its contents into a code cell, check the Drive path, and run it. This preserves the original files and creates a local download. It targets the exact interrupted stage used in this project and stops if its required files are absent; it is not a general arbitrary-run migration tool. Keep the original transfer bundle if repeating the historical resume.
4. In Kaggle, create a notebook, use **File → Import Notebook** to import `Resume_MuRIL_Kaggle.ipynb`, then **Add Input** to attach that one private resume dataset.
5. Enable **Internet** and **GPU T4 x2**. If unavailable, check account verification and remaining accelerator quota. The notebook deliberately uses one GPU; it is not distributed across both T4s.
6. Choose **Save Version → Save & Run All**. That starts a saved background version, even if the separate interactive editor says “Draft session off”. Follow that version's **Logs** tab; do not start a duplicate interactive run.
7. The notebook checks transfer hashes, training source and data, installs the original environment, reuses completed folds, and saves work under `/kaggle/working/muril-run`. It does not mount or write to Google Drive.
8. After completion, open the saved version's **Output** tab and download `muril-run/results.zip` and `muril-run/selected_models.zip`. Install selected models using the [README instructions](../README.md#install-the-selected-models). A printed `NOT COMPLETE` means training did not finish even if the notebook itself saved successfully. Preserve the entire saved output and logs in that case.

Kaggle working-directory files become durable only when the platform saves the output. Preserve the saved version before starting another attempt. A restart from the original input bundle only reuses that bundle's progress; it does not magically import newer unsaved work. Do not repeatedly export weights to Drive, and do not remove active checkpoints just to free space.

### Local scripts and evaluation records

The app does not need raw training data. To inspect/reproduce locally, import the original archive without changing its contents:

```bash
.venv/bin/python -m scripts.import_data '/path/to/ProjectPhase_Progress 2.zip'
.venv/bin/python -m scripts.check_embeddings
```

On Windows use `.\.venv\Scripts\python.exe` and a quoted Windows archive path. Full GPU experiments use the pinned notebook environment, not the app's current dependency lock. `scripts.train` is the earlier classifier-only search and `scripts.finetune` the earlier final-block search; neither is the completed all-layer experiment. See [the detailed full-training protocol](full-finetuning.md) before running `scripts.full_finetune`. Do not mix its current local results with the saved GPU run.

