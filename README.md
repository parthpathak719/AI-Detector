# Origin — AI text detection & family attribution

A working web application built from the **ProjectPhase_Progress 2** research project. Paste text or upload a UTF-8 `.txt` / `.md` file. The application predicts human vs AI and, for AI predictions, the closest of **Gemini, Muse and Sarvam**.

Supports English, Hindi, Tamil, Kannada, Hinglish, Tanglish and Kanglish. Choose the language explicitly: script detection cannot reliably distinguish English from Romanized code-mixed text.

## Run locally

Tested with Python 3.12 on Linux, CPU only. Allow about 2 GB disk for dependencies/encoder and several GB RAM. No API keys, database, paid APIs or GPU required. The first setup downloads the public MuRIL encoder (~950 MB).

```bash
git clone https://github.com/parthpathak719/AI-Detector.git
cd AI-Detector
bash scripts/setup.sh
.venv/bin/python -m uvicorn detector.app:app --host 127.0.0.1 --port 8000
```

Open port 8000 in your own browser. For a remote development server, bind to `0.0.0.0` and use your platform's port forwarding. This repository is a research prototype, not a public hosting deployment. Add authentication, TLS, rate limiting and an appropriate retention policy before exposing it publicly.

The three trained classifiers are included in `artifacts/`. Setup downloads the **pinned** `google/muril-base-cased` revision, checks the classifier hashes, and retains everything locally. No model downloads happen during requests. Only load trusted joblib artifacts; never upload arbitrary pickle files.

### What the interface does

- Paste text or upload a small UTF-8 `.txt` / `.md` file (48 KB maximum).
- Routes monolingual inputs to Task 1 and code-mixed inputs to Task 3.
- Runs Task 2 family attribution only for an AI prediction with at least 8 cleaned words.
- Shows calibrated model scores and flags close predictions. The 65% detection and 60% attribution uncertainty flags are **heuristics**, not validated abstention thresholds.
- Enforces the short-text scope: 8–150 cleaned words, or 4–150 for code-mixed input, and at most 128 MuRIL tokens. It rejects longer passages instead of silently truncating them.
- Processes submitted text in memory. It does not save passages or send them to a generation API. Access logs contain request paths, not request bodies.

PDF, DOCX and full-document classification are not implemented. Save a short excerpt as plain text.

## Model and measured results

The production pipeline uses **frozen 768-dimensional MuRIL masked-mean embeddings + character `char_wb` TF-IDF + logistic regression**. It excludes the 16 handcrafted features. Normalization follows the study's strict Unicode cleaning; label-dependent AI sentence trimming remains a training-data operation and is never applied at inference.

| Task | Test examples | Tuned accuracy | Untuned baseline accuracy | Tuned macro-F1 |
|---|---:|---:|---:|---:|
| Human vs AI, monolingual | 718 | 93.04% | 93.31% | 93.04% |
| Gemini / Muse / Sarvam | 597 | 84.25% | 84.76% | 84.20% |
| Human vs AI, code-mixed | 258 | 95.35% | 94.96% | 95.35% |

These are **separate task metrics on the existing strict-cleaned test splits**, not end-to-end application accuracy or fresh independent data. Tuning improved code-mixed test accuracy by one example, and slightly reduced the other two task scores. We retain the configurations chosen by training-only cross-validation rather than selecting models after looking at test results. There is no claim of a statistically established overall improvement.

Read [the model card](MODEL_CARD.md) for methodology, limitations and the cross-task split issue. Detailed search traces, confusion matrices, calibration metrics and per-language results are in [reports/](reports/).

## Reproduce training

The datasets and cached embeddings remain local and are excluded from Git. Use your original `ProjectPhase_Progress 2.zip`:

```bash
.venv/bin/python -m scripts.import_data '/path/to/ProjectPhase_Progress 2.zip'
.venv/bin/python -m scripts.check_embeddings
.venv/bin/python -m scripts.train
.venv/bin/python -m scripts.verify_artifacts
```

Training uses 12 predefined configurations with three-fold **StratifiedGroupKFold on training data only**. The TF-IDF vocabulary and embedding scaler are refitted inside each fold. Source/paired IDs remain together. The winner is trained on the training split; logistic score calibration uses only validation data. The original test split is used only after selection. Every run checks label/ID alignment, finite embeddings, exact duplicate text and source-group separation within each task. A dataset-plus-embedding SHA-256 is recorded in each report.

MuRIL is deliberately frozen. With this dataset size, a CPU-only setup, and no fresh validation corpus, fine-tuning the entire encoder is not justified as an assumed improvement. Test it as a separate controlled experiment, with nested grouped validation and the same test boundary, preferably on GPU.

## Verify

```bash
.venv/bin/python -m pytest -q
curl --fail http://127.0.0.1:8000/api/health
curl --fail http://127.0.0.1:8000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text":"I went back to the little bookshop near our old house yesterday and found a novel I remembered reading after school.","language":"en"}'
```

Integration tests use the **real encoder and trained classifiers**. Unit tests additionally exercise conditional attribution and uncertain results using explicitly mocked scores. Browser verification covers desktop/mobile layouts, pasting, upload parity, error states, and console errors.

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
