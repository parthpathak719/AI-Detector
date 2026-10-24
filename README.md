# Origin — AI text detection & family attribution

A local web application using MuRIL embeddings, character n-grams and logistic regression. Paste text or upload TXT, Markdown, JSONL, searchable PDF, DOCX or legacy DOC files. The app compares human/AI patterns and, for AI-leaning text, identifies the closest of **Gemini, Muse and Sarvam**.

Supported language settings: English, Hindi, Tamil, Kannada, Hinglish, Tanglish and Kanglish. Choose the language explicitly. The interface shows an overall result; the downloadable PDF includes the full segment-by-segment analysis.

## Contents

- [Requirements](#requirements)
- [Windows setup](#windows-setup)
- [Linux setup](#linux-setup)
- [macOS setup](#macos-setup)
- [Daily use](#daily-use)
- [Install the selected models](#install-the-selected-models)
- [Update the application](#update-the-application)
- [Uploads and PDF reports](#uploads-and-pdf-reports)
- [Model results and limitations](#model-results-and-limitations)
- [Training notebooks](#training-notebooks)
- [Tests and troubleshooting](#tests-and-troubleshooting)

## Requirements

- **64-bit Python 3.12**, Git, and a modern browser. The pinned environment is not intended for Python 3.13.
- Internet for the first dependency/model download. Once setup is complete, inference works locally without an AI API, API key, database or GPU.
- Several GB of free disk and RAM. The base MuRIL download is about 950 MB; selected encoder weights add roughly 950 MB per task. Keep additional disk space for ZIPs, extraction and any backups. 16 GB RAM is a useful target for running the selected models, not a measured minimum.
- Windows x64, Linux x86-64, or Apple Silicon macOS using native arm64 Python. Automated application tests run on Linux; Windows/macOS instructions are provided but have not been executed on those operating systems in this environment.

Setup creates `.venv`, installs dependencies, downloads the pinned base encoder into `.cache/muril`, and verifies the model artifacts. It does **not** train models or replace your installed selected classifiers. The frontend and backend are served together; Node.js is not needed to run the app.

## Windows setup

1. Install [Python 3.12](https://www.python.org/downloads/) with the Python launcher, and [Git for Windows](https://git-scm.com/downloads/win).
2. Open **PowerShell**. Check the versions:

```powershell
py -3.12 --version
git --version
```

3. Clone into a drive with enough space. For example:

```powershell
Set-Location "E:\College Stuff\Final Year Project\Project Phase 2\Review 3"
git clone https://github.com/parthpathak719/AI-Detector.git
Set-Location AI-Detector
py -3.12 scripts/setup.py
```

If you already have the repository, do not clone again. Open its folder:

```powershell
Set-Location "E:\College Stuff\Final Year Project\Project Phase 2\Review 3\AI-Detector"
py -3.12 scripts/setup.py
```

4. Start the server:

```powershell
.\.venv\Scripts\python.exe -m uvicorn detector.app:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. Leave PowerShell running. No activation script, WSL or Bash is needed. If using Conda, its base Python version does not matter when you explicitly invoke `py -3.12` and the project's `.venv` executable.

If the launcher is unavailable, use the full path to a Python 3.12 interpreter to run `scripts/setup.py`; do not substitute a Python 3.13 interpreter.

## Linux setup

Install Git and Python 3.12 with its `venv` support. On Ubuntu 24.04:

```bash
sudo apt update
sudo apt install git python3.12 python3.12-venv
```

On another distribution use its package manager to install Python **3.12**, rather than assuming `python3` points to the right version.

```bash
python3.12 --version
git clone https://github.com/parthpathak719/AI-Detector.git
cd AI-Detector
python3.12 scripts/setup.py
.venv/bin/python -m uvicorn detector.app:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. `bash scripts/setup.sh` is also available if `python3 --version` reports 3.12. Do not use `sudo` for project setup or server execution.

## macOS setup

### Apple Silicon (M1 or newer)

Use a native Terminal session and arm64 Python, not an Intel interpreter under Rosetta. With [Homebrew](https://brew.sh/) installed:

```bash
brew install git python@3.12
"$(brew --prefix python@3.12)/bin/python3.12" --version
"$(brew --prefix python@3.12)/bin/python3.12" -c "import platform; print(platform.machine())"
git clone https://github.com/parthpathak719/AI-Detector.git
cd AI-Detector
"$(brew --prefix python@3.12)/bin/python3.12" scripts/setup.py
.venv/bin/python -m uvicorn detector.app:app --host 127.0.0.1 --port 8000
```

The architecture check should print `arm64`. Setup installs the macOS PyTorch wheel from PyPI, rather than the Windows/Linux CPU index. Application inference currently runs on CPU; it does not automatically use Apple's MPS accelerator. Open **http://127.0.0.1:8000**.

### Intel Macs

The pinned PyTorch 2.6 stack has no native macOS Intel wheel. Setup stops with an explanation. Do not downgrade PyTorch/NumPy arbitrarily: that would create a different model-serving environment.

Use a Linux x86-64 virtual machine (for example, Ubuntu 24.04 in UTM), then follow the Linux instructions and open the app in the VM's browser. Allocate sufficient RAM/disk to the VM. An Intel Mac can also access an app running on another computer; native Intel macOS inference is not currently supported by these pins.

## Daily use

1. Open a terminal in the repository folder.
2. Run the server command for your OS above. You do not need to rerun setup every time.
3. Wait for **Model ready** in the browser.
4. Paste text or choose a file, select its language, and click **Analyze content**.
5. Read the overall scores and any extraction notes. Click **Download PDF report** for all segment text and scores.
6. Stop the server with **Ctrl+C** in the terminal. Restart it after installing models or updating backend code.

Keep `--host 127.0.0.1` for local-only access. If port 8000 is already in use, replace it with `--port 8001` and open http://127.0.0.1:8001. Uploaded content is not saved by the application. Multipart uploads may use temporary storage, which is closed after each request. PDF generation formats the existing result in memory; it does not run the model again.

## Install the selected models

**A fresh clone includes the small frozen-baseline classifiers, not the final fine-tuned encoders.** The selected full models are distributed separately in `selected_models.zip`. Download that ZIP directly into your project folder or another drive with sufficient space. `git pull` does not download it.

A convenient installation uses a separate, Git-ignored directory so future pulls cannot conflict with your selected models. The ZIP contains an `artifacts/` directory. Stop the app first. Use a new destination name if `selected_artifacts` already contains a previous installation.

Windows PowerShell:

```powershell
Expand-Archive -LiteralPath ".\selected_models.zip" -DestinationPath ".\selected_artifacts"
$env:ARTIFACT_DIR = (Resolve-Path ".\selected_artifacts\artifacts").Path
.\.venv\Scripts\python.exe -m scripts.verify_artifacts
.\.venv\Scripts\python.exe -m uvicorn detector.app:app --host 127.0.0.1 --port 8000
```

Linux / macOS:

```bash
.venv/bin/python -m zipfile -e selected_models.zip selected_artifacts
export ARTIFACT_DIR="$PWD/selected_artifacts/artifacts"
.venv/bin/python -m scripts.verify_artifacts
.venv/bin/python -m uvicorn detector.app:app --host 127.0.0.1 --port 8000
```

Set `ARTIFACT_DIR` again in each new terminal before verification or starting the server. It must point to the folder containing the manifest and task `.joblib` files, not to its parent. The result metadata reports the actual encoder variant; verify it shows the installed full variant rather than `frozen`.

If you already installed the selected models directly into `artifacts/`, that installation still works without `ARTIFACT_DIR`. Keep it; there is no need to redownload or retrain. Back it up before any replacement. Only load trusted bundles: joblib model files can execute code while loading. `scripts.verify_artifacts` checks the supplied artifact manifest, not the trustworthiness of an unknown source.

Keep `results.zip` as the completed evaluation record. It contains fold results and configuration; the app does not need it. `selected_models.zip` is useful for reinstalling models. Neither archive should be committed to Git. Weights, optimizer snapshots, archives, caches and local output folders are ignored, but existing tracked files remain tracked.

## Update the application

Stop the server, open the repository folder, and inspect local modifications:

```bash
git status
git pull
```

Plain `git pull` fetches remote commits and integrates them according to your Git configuration. `git pull --ff-only` does the same update when your branch is simply behind, but refuses if both sides have different new commits. The latter prevents an unexpected merge; it is not a different download mechanism. If Git reports conflicts or divergent branches, preserve your changes and inspect them before choosing a merge or rebase. Do not use a hard reset to discard installed models.

Install changed dependencies and restart:

Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
# Set ARTIFACT_DIR here if using a separate selected-model installation.
.\.venv\Scripts\python.exe -m uvicorn detector.app:app --host 127.0.0.1 --port 8000
```

Linux / macOS:

```bash
.venv/bin/python -m pip install -r requirements.lock
# Set ARTIFACT_DIR here if using a separate selected-model installation.
.venv/bin/python -m uvicorn detector.app:app --host 127.0.0.1 --port 8000
```

Refresh with **Ctrl+Shift+R** on Windows/Linux or **Cmd+Shift+R** in Chrome on macOS. Safari users can reload without cached content from its developer controls. Pulling code does not overwrite a separately installed model directory.

### If Git refuses to pull because `artifacts/` was modified

This can happen with the earlier installation instructions, which replaced tracked baseline files with selected models. Do not delete them. One safe migration is to copy the **entire** installed artifact directory to an ignored location, verify that copy, and only then restore the tracked baseline files:

Windows (use a fresh destination):

```powershell
New-Item -ItemType Directory -Path .\selected_artifacts
Copy-Item -LiteralPath .\artifacts -Destination .\selected_artifacts\artifacts -Recurse
$env:ARTIFACT_DIR = (Resolve-Path .\selected_artifacts\artifacts).Path
.\.venv\Scripts\python.exe -m scripts.verify_artifacts
```

Linux / macOS (use a fresh destination):

```bash
mkdir selected_artifacts
cp -a artifacts selected_artifacts/artifacts
export ARTIFACT_DIR="$PWD/selected_artifacts/artifacts"
.venv/bin/python -m scripts.verify_artifacts
```

**Only after verification succeeds** and the copied folder contains your complete installed models:

```bash
git restore --source=HEAD -- artifacts
git pull
```

Continue using the copied directory through `ARTIFACT_DIR`. The restore affects tracked files only; do not run `git clean` on your model folder. For unrelated source edits, commit or back them up separately before pulling.

## Uploads and PDF reports

- Limits: **100,000 extracted characters**, **20,000 cleaned words**, **10 MB per upload**, and **300 PDF pages**.
- TXT and Markdown: UTF-8 text; an optional UTF-8 BOM is accepted.
- JSONL: one JSON string or object with a nonempty string `text` or `content` field per nonblank line. `text` takes precedence if both exist. Metadata such as IDs and labels is excluded. Records are joined in file order into **one document**, not separately classified records. Invalid lines produce an error with a line number.

```jsonl
{"id":"example_1","text":"The first passage goes here."}
{"id":"example_2","content":"The second passage goes here."}
```

- PDF: searchable text only. Run OCR externally for scans. Tables, columns, equations and superscripts may not extract exactly as printed.
- DOCX: main body and tables; images, headers, footers and footnotes are excluded. Legacy DOC supports Word 97–2003 piece tables; unsupported/encrypted documents should be saved as DOCX.
- The PDF report has an overall summary, all individual segments, and analysis/extraction notes. It preserves source punctuation and numbers where extraction permits; model scores use normalized input. Embedded fonts cover English, Hindi, Tamil and Kannada. Unsupported glyphs are replaced and noted.
- Reports include your source text. They are ordinary exports, not independently authenticated authorship certificates. Store/share them accordingly.

## Model results and limitations

All embedding parameters and 12 transformer blocks were trained (236,965,632 encoder parameters; unused pooler excluded). Three learning rates × five source-grouped folds × three tasks produced 45 full fine-tuning runs, compared with 15 frozen baseline folds and 12 hybrid configurations per representation. Early stopping used separate inner groups. Selection used mean outer-fold macro-F1; test scores did not choose the winners.

| Task | Frozen CV accuracy | Selected full CV accuracy | Frozen test accuracy | Selected full test accuracy |
|---|---:|---:|---:|---:|
| Human vs AI | 92.40% | 93.01% | 93.04% | 93.87% |
| Family attribution | 85.37% | 86.56% | 84.59% | 86.43% |
| Code-mixed detection | 95.63% | 95.71% | 95.35% | 95.74% |

These are separate short-text task results, not document or end-to-end cascade accuracy. CV was used for tuning. The existing test data has been inspected historically, and small improvements do not establish statistical superiority. Code-mixed test accuracy increased by one example. The original presentation's 85.9% attribution result used a different evaluation protocol.

See [completed selection and final reports](reports/full_finetuning_completed/), [training protocol](docs/full-finetuning.md), and [model card](MODEL_CARD.md). Earlier classifier-only and final-block experiments remain explicitly historical; see [report index](reports/README.md).


### What “long-document accuracy has not been benchmarked” means

The recorded accuracy figures come from labelled short-text examples in the supplied dataset. We have **not** measured accuracy on a separate labelled collection of complete assignments, essays or PDFs.

The app processes long inputs in windows of at most 128 MuRIL tokens and combines scores using cleaned word counts. That enables long-input processing, but does not validate the combined result. Segments from the same document are related, so more segments are not automatically independent evidence of authorship. Multi-segment results therefore say **Authorship inconclusive** while showing the original model scores.

A proper document benchmark would need independent human/AI-labelled full documents, representative topics and generators, and a held-out evaluation of the complete extraction → segmentation → aggregation pipeline. Until then, short-text accuracy should not be quoted as full-document accuracy.

Strict normalization removes punctuation and numbers to match training, which can lose technical information. An AI-generated assignment can still receive a high human score. Family attribution runs only for AI-leaning overall results using eligible AI-leaning segments and cannot reliably identify generators outside its three trained families.

## Training notebooks

**Training is already complete; neither notebook is needed to run the web app.** Only the two training notebooks used in the Colab → Kaggle workflow are retained. The standalone inspection notebook was removed; its underlying inspection script remains available.

| Notebook | Purpose |
|---|---|
| [Full_MuRIL_Finetuning.ipynb](notebooks/Full_MuRIL_Finetuning.ipynb) | Colab full-encoder training, Drive persistence, model/result export |
| [Resume_MuRIL_Kaggle.ipynb](notebooks/Resume_MuRIL_Kaggle.ipynb) | Resume the transferred experiment on Kaggle; this workflow completed the run |

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
3. If recreating the original Colab-to-Kaggle transfer, use [scripts/prepare_kaggle_transfer.py](scripts/prepare_kaggle_transfer.py) in a **Colab CPU notebook**: paste its contents into a code cell, check the Drive path, and run it. This preserves the original files and creates a local download. It targets the exact interrupted stage used in this project and stops if its required files are absent; it is not a general arbitrary-run migration tool. Keep the original transfer bundle if repeating the historical resume.
4. In Kaggle, create a notebook, use **File → Import Notebook** to import `Resume_MuRIL_Kaggle.ipynb`, then **Add Input** to attach that one private resume dataset.
5. Enable **Internet** and **GPU T4 x2**. If unavailable, check account verification and remaining accelerator quota. The notebook deliberately uses one GPU; it is not distributed across both T4s.
6. Choose **Save Version → Save & Run All**. That starts a saved background version, even if the separate interactive editor says “Draft session off”. Follow that version's **Logs** tab; do not start a duplicate interactive run.
7. The notebook checks transfer hashes, training source and data, installs the original environment, reuses completed folds, and saves work under `/kaggle/working/muril-run`. It does not mount or write to Google Drive.
8. After completion, open the saved version's **Output** tab and download `muril-run/results.zip` and `muril-run/selected_models.zip`. Install selected models using the earlier section. A printed `NOT COMPLETE` means training did not finish even if the notebook itself saved successfully. Preserve the entire saved output and logs in that case.

Kaggle working-directory files become durable only when the platform saves the output. Preserve the saved version before starting another attempt. A restart from the original input bundle only reuses that bundle's progress; it does not magically import newer unsaved work. Do not repeatedly export weights to Drive, and do not remove active checkpoints just to free space.

### Local scripts and evaluation records

The app does not need raw training data. To inspect/reproduce locally, import the original archive without changing its contents:

```bash
.venv/bin/python -m scripts.import_data '/path/to/ProjectPhase_Progress 2.zip'
.venv/bin/python -m scripts.check_embeddings
```

On Windows use `.\.venv\Scripts\python.exe` and a quoted Windows archive path. Full GPU experiments use the pinned notebook environment, not the app's current dependency lock. `scripts.train` is the earlier classifier-only search and `scripts.finetune` the earlier final-block search; neither is the completed all-layer experiment. See [the detailed full-training protocol](docs/full-finetuning.md) before running `scripts.full_finetune`. Do not mix its current local results with the saved GPU run.

## Tests and troubleshooting

Windows:

```powershell
.\.venv\Scripts\python.exe -m scripts.verify_artifacts
.\.venv\Scripts\python.exe -m pytest -q
```

Linux / macOS:

```bash
.venv/bin/python -m scripts.verify_artifacts
.venv/bin/python -m pytest -q
```

Tests cover uploads, extraction, PDF reports, normalization, model routing, aggregation and checkpoint recovery. The test suite expects the tracked baseline fixtures; test a clean checkout without `ARTIFACT_DIR` when reproducing baseline assertions. Keep selected models in their separate directory. Optional end-to-end browser checks require a running server and Playwright:

```bash
.venv/bin/python -m pip install playwright
.venv/bin/python -m playwright install chromium
.venv/bin/python scripts/browser_smoke.py
```

Use the Windows `.venv` executable instead on Windows. Playwright is a development-only dependency.

| Symptom | What to do |
|---|---|
| Python version rejected | Install Python 3.12 and invoke it explicitly; don't reuse a 3.13 environment. |
| Invalid `pipefail` / WSL path errors on Windows | Use `py -3.12 scripts/setup.py` in PowerShell instead of Bash. |
| `.venv` belongs to another OS or is incomplete | Rename it to preserve it, then rerun setup. Never share one `.venv` across Windows/WSL/macOS. |
| `Model unavailable` | Check terminal errors, encoder download, `ARTIFACT_DIR` and artifact verification. |
| Missing base model/tokenizer | Rerun setup or `python -m scripts.download_muril` with the project's interpreter. |
| Artifact hash mismatch | Restore a complete trusted bundle. Do not mix manifests/classifiers/weights from different exports. |
| PDF has no text | OCR the scan externally or paste its text. Password-protected files must be unlocked first. |
| JSONL rejected | Use UTF-8, one valid JSON value per line, and a nonempty string `text` or `content` field. |
| Changes aren't visible | Restart the server and hard-refresh the browser. Verify you pulled the correct repository folder. |
| Slow analysis / memory pressure | Use smaller documents and close other heavy applications. CPU inference needs no GPU but is not instantaneous. |
| Port is busy | Stop the earlier server or use another port, such as 8001. |
| GPU unavailable in Colab/Kaggle | Check the platform's quota/availability; local web inference still works on CPU. |
| C: is full on Windows | Keep repository/ZIPs on E:. Base encoder stays in the repository. Pip and temporary downloads can still use C:; move their caches to a drive with room if needed. |

Optional cache placement before setup on Windows:

```powershell
New-Item -ItemType Directory -Force E:\AI-Detector-temp, E:\AI-Detector-pip-cache
$env:TEMP = "E:\AI-Detector-temp"
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = "E:\AI-Detector-pip-cache"
py -3.12 scripts/setup.py
```

`MURIL_MODEL_PATH` can point to an existing compatible base-model directory. `ARTIFACT_DIR` selects a complete artifact bundle. `TORCH_NUM_THREADS` changes CPU thread count (default 2). Set these in the same terminal before starting the server; the app does not load a `.env` file automatically.

## Repository layout

```text
detector/                          API, extraction, inference, PDF export, frontend
scripts/                           Setup, training, export and verification tools
notebooks/                         Colab training and Kaggle resume notebooks
artifacts/                         Small frozen baseline classifiers (tracked)
reports/full_finetuning_completed/  Completed GPU experiment summaries
docs/                              Model-training and evaluation details
tests/                             Pipeline, document, API and workflow tests
selected_artifacts/                Your separate installed models (ignored)
data/                              Local training data (ignored)
.cache/                            Base encoder and local caches (ignored)
```
