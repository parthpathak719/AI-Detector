# Origin — AI text detection & family attribution

A local web app using fine-tuned MuRIL embeddings and character n-grams to classify text as human- or AI-leaning. For AI-leaning text, it compares three model families: **Gemini, Muse and Sarvam**.

Supports English, Hindi, Tamil, Kannada, Hinglish, Tanglish and Kanglish. Paste text or upload **TXT, MD, JSONL, PDF, DOCX or DOC**, then download a PDF report with segment-by-segment analysis.

## Requirements

- Git and **64-bit Python 3.12**.
- Several GB of free disk and RAM; 16 GB RAM is recommended for the selected full models.
- Internet for initial setup. The base MuRIL download is approximately 950 MB; selected models require additional space.
- No GPU, API key or paid service is needed to run the app.

Setup creates `.venv`, installs dependencies and downloads the base encoder. **Training is not required.**

## Windows setup

Install [Python 3.12](https://www.python.org/downloads/) and [Git](https://git-scm.com/downloads). Open PowerShell in the directory where you want the project:

```powershell
git clone https://github.com/parthpathak719/AI-Detector.git
cd AI-Detector
py -3.12 scripts/setup.py
.\.venv\Scripts\python.exe -m uvicorn detector.app:app --host 127.0.0.1 --port 8000
```

Use PowerShell directly; Bash/WSL and environment activation are unnecessary. If you already have the project, open its folder instead of cloning again.

## Linux setup

Install Git, Python 3.12 and its `venv` package. On Ubuntu 24.04:

```bash
sudo apt update
sudo apt install git python3.12 python3.12-venv
git clone https://github.com/parthpathak719/AI-Detector.git
cd AI-Detector
python3.12 scripts/setup.py
.venv/bin/python -m uvicorn detector.app:app --host 127.0.0.1 --port 8000
```

## macOS setup

For **Apple Silicon**, install [Homebrew](https://brew.sh/), then run in a native Terminal session:

```bash
brew install git python@3.12
git clone https://github.com/parthpathak719/AI-Detector.git
cd AI-Detector
"$(brew --prefix python@3.12)/bin/python3.12" scripts/setup.py
.venv/bin/python -m uvicorn detector.app:app --host 127.0.0.1 --port 8000
```

The pinned PyTorch version does not support native Intel macOS. On an Intel Mac, use a Linux virtual machine and follow the Linux steps. Automated checks run on Linux; native Windows/macOS execution has not been verified here.

## Install the selected models

**A fresh clone uses the earlier frozen baseline.** To use the selected fine-tuned models, obtain `selected_models.zip` from the completed training run. Save it in your project folder, stop the app with **Ctrl+C**, and extract it into a separate folder:

Windows:

```powershell
Expand-Archive -LiteralPath .\selected_models.zip -DestinationPath .\selected_artifacts
$env:ARTIFACT_DIR = (Resolve-Path .\selected_artifacts\artifacts).Path
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

Set `ARTIFACT_DIR` again when opening a new terminal. If the selected models are already installed directly in `artifacts/`, they continue to work without that variable; no reinstall is needed. Only load trusted model bundles, and keep a backup before replacing them. The separate `results.zip` contains evaluation records and is not needed to run the app.

## Start and use the app

1. Open **http://127.0.0.1:8000** and wait for **Model ready**.
2. Paste text or upload a file, select the language, and click **Analyze content**.
3. Read the overall result and model-family scores, when available.
4. Click **Download PDF report** for the summary, full segment analysis and extraction notes.

Limits: **10 MB per upload**, **100,000 extracted characters**, **20,000 cleaned words**, and **300 PDF pages**. Scanned PDFs require OCR before upload. JSONL accepts one JSON string or an object with a string `text` or `content` field per line; records are combined into one document and metadata is ignored.

Content is processed locally and is not saved by the app. Reports include source text. Keep the terminal running while using the app; stop it with **Ctrl+C**. Next time, run only the server command for your OS (and set `ARTIFACT_DIR` if applicable), not the setup command.

## Update

Stop the server and run from the project folder:

```bash
git pull
```

Install updated dependencies:

```powershell
# Windows
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
```

```bash
# Linux / macOS
.venv/bin/python -m pip install -r requirements.lock
```

Restart the server using the earlier command and hard-refresh the browser: **Ctrl+Shift+R** on Windows/Linux or **Cmd+Shift+R** in Chrome on macOS. If Git reports local changes, preserve them before resolving the conflict—do not reset or delete installed models. Large model ZIPs are separate downloads and do not arrive through `git pull`.

## Results

We fine-tuned MuRIL's embeddings and all 12 transformer blocks, compared three learning rates using **five source-grouped folds**, and evaluated 12 hybrid classifier configurations per representation. Early stopping used separate inner validation groups. Selection used CV macro-F1; test scores did not choose the winners.

| Task | Frozen CV accuracy | Fine-tuned CV accuracy | Frozen test accuracy | Fine-tuned test accuracy |
|---|---:|---:|---:|---:|
| Human vs AI | 92.40% | 93.01% | 93.04% | **93.87%** |
| Family attribution | 85.37% | 86.56% | 84.59% | **86.43%** |
| Code-mixed detection | 95.63% | 95.71% | 95.35% | **95.74%** |

These are **short-text task results**, not full-document accuracy or end-to-end family-attribution accuracy. Long documents are processed in segments, but their combined accuracy has not yet been measured on an independent labelled document set. Scores are model signals, not proof of authorship; even high scores can be wrong. Small improvements do not establish statistical superiority.

For more detail: [completed results](reports/full_finetuning_completed/), [model card](MODEL_CARD.md), and [training protocol](docs/full-finetuning.md). The [Colab training notebook](notebooks/Full_MuRIL_Finetuning.ipynb) and [Kaggle resume notebook](notebooks/Resume_MuRIL_Kaggle.ipynb) are retained for reproducing the completed experiment; neither is needed to start the app.
