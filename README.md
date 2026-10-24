# Origin — AI text detection & family attribution

A research web application using MuRIL embeddings, character n-grams and logistic regression. Paste text or upload TXT, Markdown, JSONL, PDF, DOCX or legacy DOC files. Detection compares human/AI patterns; conditional attribution compares **Gemini, Muse and Sarvam** only.

Supports English, Hindi, Tamil, Kannada, Hinglish, Tanglish and Kanglish. Select the language explicitly. Scores are model signals, not proof of authorship. Technical assignments and unseen generators can receive strongly incorrect scores.

## Run locally

Use **Python 3.12, 64-bit**. Setup supports Windows and Linux. Automated checks run on Linux with CPU inference; no GPU, API key or paid API is required for the web app. Allow several GB of disk and RAM, especially with three full encoder checkpoints. Initial setup downloads the pinned public MuRIL encoder (~950 MB).

Windows PowerShell, from the project folder:

```powershell
py -3.12 scripts/setup.py
.\.venv\Scripts\python.exe -m uvicorn detector.app:app --host 127.0.0.1 --port 8000
```

Linux:

```bash
bash scripts/setup.sh
.venv/bin/python -m uvicorn detector.app:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. No Bash/WSL or virtual-environment activation is needed on Windows. Do not share a `.venv` between Windows and WSL. The app uses FastAPI and a plain HTML/CSS/JavaScript frontend served by one process.

### Update an existing installation

Stop the server first. Pull the changes, install dependencies, and restart:

```powershell
git pull --ff-only
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn detector.app:app --host 127.0.0.1 --port 8000
```

Refresh the browser with **Ctrl+Shift+R**. The result footer displays `document-v3` and the actual encoder variant. Updating application code does not install the separately exported winning models.

## Model artifacts and completed results

**The small classifiers tracked in `artifacts/` are the earlier frozen baseline.** They make a fresh checkout runnable. The completed full-encoder experiment selected fine-tuned MuRIL for all three tasks; its large encoder weights are distributed separately as `selected_models.zip`, not stored in Git.

All embedding parameters and 12 transformer blocks were trained (236,965,632 encoder parameters; unused pooler excluded). Three learning rates × five source-grouped folds × three tasks produced 45 full fine-tuning runs, compared with 15 frozen baseline folds and 12 hybrid configurations per representation. Early stopping used separate inner groups. Selection used mean outer-fold macro-F1; test scores did not choose the winners.

| Task | Frozen CV accuracy | Selected full CV accuracy | Frozen test accuracy | Selected full test accuracy |
|---|---:|---:|---:|---:|
| Human vs AI | 92.40% | 93.01% | 93.04% | 93.87% |
| Family attribution | 85.37% | 86.56% | 84.59% | 86.43% |
| Code-mixed detection | 95.63% | 95.71% | 95.35% | 95.74% |

These are separate short-text task results, not document or end-to-end cascade accuracy. CV was used for tuning. The existing test data has been inspected historically, and small improvements do not establish statistical superiority. Code-mixed test accuracy increased by one example. The original presentation's 85.9% attribution result used a different evaluation protocol.

See [completed selection and final reports](reports/full_finetuning_completed/), [training protocol](docs/full-finetuning.md), and [model card](MODEL_CARD.md). Earlier classifier-only and final-block experiments remain explicitly historical; see [report index](reports/README.md).

### Install the selected models on Windows

Save `selected_models.zip` directly inside your project folder, including on E: if C: is full. Stop the app, then:

```powershell
$backupName = "artifacts-backup-" + (Get-Date -Format "yyyyMMdd-HHmmss")
if (Test-Path ".\artifacts") {
    Rename-Item -LiteralPath ".\artifacts" -NewName $backupName
}
Expand-Archive -LiteralPath ".\selected_models.zip" -DestinationPath .
.\.venv\Scripts\python.exe -m scripts.verify_artifacts
.\.venv\Scripts\python.exe -m uvicorn detector.app:app --host 127.0.0.1 --port 8000
```

Keep the backup until inference works. Retain `results.zip` for the full fold predictions and research record; the app does not need it. Only load trusted, checksum-verified artifacts. Set `ARTIFACT_DIR` to another trusted local model directory if desired.

ZIP archives, full encoder weights, optimizer snapshots, local experiment outputs and backup folders are ignored by Git. Ignore rules do not untrack existing files: the small baseline joblib files and manifest remain tracked, and replacing them locally can appear in `git status`. Do not stage those replacements unintentionally or use `git restore artifacts` on your installed models.

## Document analysis and limitations

- Up to **100,000 characters / 20,000 cleaned words**, **10 MB** per upload, and **300 PDF pages**.
- Searchable PDF extraction uses spacing-aware text flow. Tables, diagrams, superscripts and subscripts can still flatten; scanned PDFs require external OCR. Word extraction covers the main body and tables, with format-specific omissions reported.
- Every cleaned word is processed in windows of at most 128 MuRIL tokens. Scores are averaged by cleaned word count; they are not the percentage of text written by AI.
- Segment cards show extracted **source text**, preserving punctuation, case and numbers. Source excerpts can overlap at a split compound word while model words are counted once.
- Segment cards expand individually; the unused full-text review/edit controls have been removed. Ordinary text is 16–17 px, with helper labels at least 14 px.
- JSONL accepts one JSON string or an object with a string `text` (or `content`) field per nonblank line. Records are combined in file order into **one document analysis**, not independent predictions. Labels and metadata are excluded. Invalid records report their line number.
- **Download PDF report** exports the summary, model scores, extraction notes and all source segments as a paginated PDF. Embedded Noto fonts support English, Hindi, Tamil and Kannada; unsupported characters are marked with a replacement symbol.
- Multi-segment reports say **Authorship inconclusive** because document-level accuracy has not been validated. Their underlying model scores remain visible and unchanged. Short-text headings describe model signals, not verified origin.
- Family attribution runs only for an AI-leaning overall model label, using AI-leaning segments with at least eight cleaned words. Unknown generators cannot be identified reliably by this three-family classifier.
- Strict normalization removes numbers and punctuation to match training. This can discard important information from equations and technical material. Improving extraction does not fix model domain failures; the application update does not retrain or alter model thresholds.

Inputs are processed locally and not retained by application code or sent to a generation API. Multipart processing may use temporary upload storage, which is closed after each request. PDF export sends the existing analysis to the local server for in-memory formatting; it does not run inference again. Exported PDFs include source text and are not independently authenticated; save and share them intentionally. Avoid exposing the research server publicly without deployment controls.

## Reproduce and verify

No training is needed to run the app. Raw data, MuRIL downloads and training caches stay outside Git. Import the original supplied archive:

```bash
.venv/bin/python -m scripts.import_data '/path/to/ProjectPhase_Progress 2.zip'
.venv/bin/python -m scripts.check_embeddings
```

`scripts.train` reproduces the earlier classifier-only search; `scripts.finetune` reproduces the historical final-block search. These are not the completed full-encoder experiment. Follow [full fine-tuning instructions](docs/full-finetuning.md) for the latter, including its pinned source version and SciPy 1.14.1 compatibility environment. Do not mix CPU/GPU experiment protocols or overwrite existing run folders with different settings.

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m scripts.verify_artifacts
```

Tests cover HTTP uploads, Unicode normalization, model routing, source-to-segment alignment, score aggregation, checkpoint handling and artifact integrity. Optional browser checks are in `scripts/browser_smoke.py`. On Windows, replace `.venv/bin/python` with `.\.venv\Scripts\python.exe`.

## Layout

```text
detector/                          API, extraction, inference and frontend
scripts/                           Setup, research, export and verification tools
artifacts/                         Small frozen baseline classifiers (tracked)
reports/full_finetuning_completed/  Final GPU experiment summaries
tests/                             Pipeline, document, API and workflow tests
data/                              Local research data (ignored)
.cache/                            Local encoders and training caches (ignored)
```

## Research provenance

Based on the user-supplied ProjectPhase_Progress 2 code/data and Phase 2 Review 2 presentation, Team 23UG010 (Parth Pathak, Pranav Achudha S, Siripurapu Jishwanth Sai Satwik). Feature fusion follows `src/models/common.py`; pooling follows `src/features/step5_features_v2.py`; strict inference normalization follows `preprocess/p8_clean_and_match.py`. Label-dependent training sentence trimming is not applied at inference. Raw corpora and MuRIL weights are not redistributed in this repository.
