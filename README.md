# Origin — AI text detection & family attribution

A working web application built from the **ProjectPhase_Progress 2** research project. Paste text or upload a `.txt`, `.md`, `.pdf`, `.docx` or `.doc` document. The application predicts human vs AI and, for AI predictions, the closest of **Gemini, Muse and Sarvam**.

Supports English, Hindi, Tamil, Kannada, Hinglish, Tanglish and Kanglish. Choose the language explicitly: script detection cannot reliably distinguish English from Romanized code-mixed text.

![Origin dark document-analysis interface](docs/interface.png)

## Run locally

Use **Python 3.12 (64-bit)**. Setup supports native Windows and Linux; application tests have been run on Linux, CPU only. Native Windows execution has not yet been verified on a Windows machine. Allow several GB of disk and RAM for dependencies and the encoder. No API keys, database, paid APIs or GPU required. The first setup downloads the public MuRIL encoder (~950 MB).

### Windows — PowerShell

Install Python 3.12 from python.org if `py -3.12 --version` cannot find it. From your existing checkout:

```powershell
git pull
py -3.12 scripts/setup.py
.\.venv\Scripts\python.exe -m uvicorn detector.app:app --host 127.0.0.1 --port 8000
```

No Bash, WSL, virtual-environment activation or PowerShell execution-policy change is needed. Setup creates a dedicated `.venv`; it does not install packages into an active Conda environment such as `(tf)`. Paths containing spaces are supported. If you already have a Python 3.12 Conda environment but no `py` launcher, use `python scripts/setup.py` after confirming `python --version` reports 3.12.

Do not share `.venv` between WSL and Windows. If setup reports an incompatible existing environment, rename that directory and retry; setup will not delete it. The WSL `Failed to translate` messages concern old Windows PATH entries and are unrelated to the native setup. A Bash `pipefail` error can result from CRLF line endings; `.gitattributes` now keeps shell scripts in LF format.

### Linux

```bash
git clone https://github.com/parthpathak719/AI-Detector.git
cd AI-Detector
bash scripts/setup.sh
.venv/bin/python -m uvicorn detector.app:app --host 127.0.0.1 --port 8000
```

Open port 8000 in your own browser. For a remote development server, bind to `0.0.0.0` and use your platform's port forwarding. This repository is a research prototype, not a public hosting deployment. Add authentication, TLS, rate limiting and an appropriate retention policy before exposing it publicly.

The three trained classifiers are included in `artifacts/`. Setup downloads the **pinned** `google/muril-base-cased` revision, checks the classifier hashes, and retains everything locally. No model downloads happen during requests. Only load trusted joblib artifacts; never upload arbitrary pickle files.

### What the interface does

- Paste up to **100,000 characters / 20,000 words**, or upload a document of up to **10 MB** (the same extracted-text limits apply).
- Routes monolingual inputs to Task 1 and code-mixed inputs to Task 3.
- Runs Task 2 family attribution only for an AI-leaning overall prediction, using AI-leaning segments with at least 8 cleaned words.
- Shows calibrated segment scores, word-weighted document scores, segment detail and JSON report export. Flags close predictions. The 65% detection and 60% attribution uncertainty flags are **heuristics**, not validated abstention thresholds.
- Splits long documents at word boundaries into independent windows of at most 128 MuRIL tokens. Every cleaned word is included exactly once. Minimum input remains 8 words, or 4 for code-mixed text.
- Does not retain submitted content or send it to a generation API. Extraction and inference use memory; the multipart framework may use temporary upload storage, closed after each request. Access logs contain request paths, not request bodies.

Text/Markdown must use UTF-8. PDF extraction supports searchable text (up to 300 pages); scanned PDFs require OCR first. DOCX extraction covers main-body paragraphs and tables. Legacy DOC support covers unencrypted Word 97–2003 compound documents; older or unsupported variants should be saved as DOCX. Extraction is native Python, with no Word or LibreOffice installation required. Images, embedded objects, headers and footnotes are not analyzed. PDF pages without text are reported rather than silently treated as human.

**Document scores are not percentages of AI-authored words.** The detector averages segment probabilities weighted by cleaned word count. Attribution averages only eligible AI-leaning segments, weighted by word count × AI score. The overall label uses the larger aggregate score. Segment results may disagree with one another. The existing classifiers were trained on short passages; aggregate document scores have not been separately calibrated or benchmarked for long documents or mixed authorship. The short-text metrics below must not be advertised as long-document accuracy.

The dark interface supports drag-and-drop upload, a visible elapsed timer during long requests, per-segment inspection and optional JSON report download. No input or report is saved server-side. CPU inference on long documents may take minutes on slower machines.

## Model and measured results

The production pipeline uses **frozen 768-dimensional MuRIL masked-mean embeddings + character `char_wb` TF-IDF + logistic regression**. It excludes the 16 handcrafted features. Normalization follows the study's strict Unicode cleaning; label-dependent AI sentence trimming remains a training-data operation and is never applied at inference.

We also fine-tuned **MuRIL's final transformer block (7,087,872 parameters)**, with the embeddings and first 11 blocks frozen. One and two epochs were compared against the frozen encoder on the same three source-grouped training folds. **The frozen encoder won all three comparisons**, so all serving models remain frozen MuRIL + character n-grams.

| Task | Frozen CV macro-F1 | Best fine-tuned CV macro-F1 | Selected test accuracy |
|---|---:|---:|---:|
| Monolingual human vs AI | 92.23% | 92.06% | 93.04% |
| Gemini / Muse / Sarvam | 84.71% | 84.62% | 84.25% |
| Code-mixed human vs AI | 95.37% | 95.29% | 95.35% |

Selection used **training-only grouped CV**, not test scores. This partial fine-tune did not demonstrate an improvement; it does not establish that every possible full-encoder fine-tune would fail. See [the fine-tuning experiment](docs/finetuning-results.md) and [complete fold results](reports/finetuning/).

For comparison, the earlier **classifier-only** search produced these test results (these are historical, not a new fine-tuned-encoder result):

| Task | Test examples | Tuned accuracy | Untuned baseline accuracy | Tuned macro-F1 |
|---|---:|---:|---:|---:|
| Human vs AI, monolingual | 718 | 93.04% | 93.31% | 93.04% |
| Gemini / Muse / Sarvam | 597 | 84.25% | 84.76% | 84.20% |
| Human vs AI, code-mixed | 258 | 95.35% | 94.96% | 95.35% |

These are **separate task metrics on the existing strict-cleaned test splits**, not end-to-end application accuracy or fresh independent data. Tuning improved code-mixed test accuracy by one example, and slightly reduced the other two task scores. We retain the configurations chosen by training-only cross-validation rather than selecting models after looking at test results. There is no claim of a statistically established overall improvement.

Read [the model card](MODEL_CARD.md) for methodology, limitations and the cross-task split issue. Detailed search traces, confusion matrices, calibration metrics and per-language results are in [reports/](reports/).

### Why attribution differs from the presentation's 85.9%

In the **first application experiment**, MuRIL's weights were not fine-tuned. A 12-candidate search tuned classifier `C`, embedding weight, character n-gram range and `min_df`. Validation data then fitted a score calibrator.

| Task 2 setting | Baseline | Selected by train-only grouped CV |
|---|---|---|
| Logistic-regression C | 10 | 10 |
| Embedding weight | 1.0 | 1.0 |
| Character n-gram range | 2–4 | 3–5 |
| Minimum document frequency | 3 | 3 |
| Training CV macro-F1 | 84.21% | 84.71% |
| Existing test-split accuracy | 84.76% | 84.25% |

The selected setting improved the selection metric but did not generalize better to this test split: it made three additional errors out of 597. Your presentation's **85.9%** is a five-fold grouped-CV result using a different protocol, not this test-split measurement. These numbers do not establish an improvement over your original study. The subsequent final-block fine-tuning experiment is reported separately above; it retained the frozen encoder because the fine-tuned candidates did not improve the selection metric.

## Full-encoder experiment in progress

[Open the full fine-tuning notebook in Google Colab](https://colab.research.google.com/github/parthpathak719/AI-Detector/blob/main/notebooks/Full_MuRIL_Finetuning.ipynb). Select a GPU runtime, click **Run all**, and authorize Drive. The notebook preserves checkpoints in Drive, runs the complete five-fold experiment, and exports verified selected models. It requires your Google sign-in; preparing the notebook does not start a Colab GPU automatically.

A new all-layer MuRIL experiment uses **five grouped folds**, three encoder learning rates, early stopping inside each training fold, and twelve hybrid configurations. It is separate from the completed partial fine-tuning results above. **No completed full-fine-tuning gain is claimed yet.** See [the protocol, progress commands, and resume instructions](docs/full-finetuning.md). Production models stay unchanged while it runs.

## Reproduce training

The datasets and cached embeddings remain local and are excluded from Git. Use your original `ProjectPhase_Progress 2.zip`:

```bash
.venv/bin/python -m scripts.import_data '/path/to/ProjectPhase_Progress 2.zip'
.venv/bin/python -m scripts.check_embeddings
.venv/bin/python -m scripts.train
.venv/bin/python -m scripts.verify_artifacts
```

Training uses 12 predefined configurations with three-fold **StratifiedGroupKFold on training data only**. The TF-IDF vocabulary and embedding scaler are refitted inside each fold. Source/paired IDs remain together. The winner is trained on the training split; logistic score calibration uses only validation data. The original test split is used only after selection. Every run checks label/ID alignment, finite embeddings, exact duplicate text and source-group separation within each task. A dataset-plus-embedding SHA-256 is recorded in each report.

The commands above reproduce the earlier classifier-only search. To reproduce the **new MuRIL fine-tuning comparison and current model selection**, run:

```bash
.venv/bin/python -m scripts.finetune
.venv/bin/python -m scripts.summarize_finetuning
.venv/bin/python -m scripts.check_serving
```

This updates the final MuRIL block and a temporary supervised head within each CV training fold, while caching frozen lower-layer outputs. It selects the best frozen/fine-tuned hybrid by mean macro-F1, refits on train, calibrates on val and only then reports test performance. It took about 24 minutes on the cloud CPU and uses several GB of cache storage; runtime varies by machine. Earlier serving artifacts are preserved in `.cache/finetune/previous_artifacts/`. Full-encoder fine-tuning was not attempted. No training is needed just to run the application.

## Verify

```bash
.venv/bin/python -m pytest -q
curl --fail http://127.0.0.1:8000/api/health
curl --fail http://127.0.0.1:8000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text":"I went back to the little bookshop near our old house yesterday and found a novel I remembered reading after school.","language":"en"}'
```

Integration tests use the **real encoder and trained classifiers**. Unit tests additionally exercise conditional attribution and uncertain results using explicitly mocked scores. Browser verification covers desktop/mobile layouts, pasting, upload parity, error states, and console errors.

On Windows, replace `.venv/bin/python` in training and verification commands with `.\.venv\Scripts\python.exe`. Run API checks with `curl.exe` rather than PowerShell's `curl` alias.

## Layout

```text
detector/            API, normalization, inference, static browser interface
scripts/             Setup, data import, training, integrity and parity checks
artifacts/           Three trained hybrid classifiers and checksum manifest
reports/             Actual measured results; no submitted user text
tests/               Real HTTP integration and inference behavior tests
data/                Local imported research data (gitignored)
.cache/muril/        Pinned encoder and tokenizer (gitignored)
```

The app uses FastAPI and a dependency-free HTML/CSS/JavaScript frontend. It serves both from one process. Set `MURIL_MODEL_PATH` or `ARTIFACT_DIR` to trusted local directories if needed; `TORCH_NUM_THREADS` defaults to 2. Do not run many workers on a small machine: each loads another encoder.

## Research provenance

Based on the user-supplied ProjectPhase_Progress 2 code/data and Phase 2 Review 2 presentation, Team 23UG010 (Parth Pathak, Pranav Achudha S, Siripurapu Jishwanth Sai Satwik). Feature fusion follows `src/models/common.py`; pooling follows `src/features/step5_features_v2.py`; inference normalization follows `preprocess/p8_clean_and_match.py`. Original files are retained separately in the onboarding workspace. Uploaded legacy pickle models were not executed.

The archive identifies the generator families; exact generator versions and dataset redistribution rights have not been independently verified. This repository does not redistribute the raw corpora or the MuRIL weights. Consult their original licenses before redistribution.
