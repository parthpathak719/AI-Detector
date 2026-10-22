# MuRIL fine-tuning experiment

This experiment updates actual MuRIL weights: the final transformer block (encoder layer 11) and a temporary supervised linear head. Token embeddings and encoder layers 0–10 remain frozen. The temporary head is discarded; the served classifier still uses MuRIL embeddings plus character n-grams.

The same three source-grouped folds within the original training split compare frozen representations with one and two fine-tuning epochs. Each representation is tested with three fixed hybrid configurations. Selection maximizes mean macro-F1, with ties favoring fewer fine-tuning epochs. The selected representation and classifier are refitted on train, calibrated on validation, and evaluated on test only afterward.

| Task | Best frozen CV F1 | Best fine-tuned CV F1 | Selected encoder | Selected test accuracy | Best frozen test accuracy |
|---|---:|---:|---|---:|---:|
| Monolingual human/AI | 92.23% | 92.06% | Frozen | 93.04% | 93.04% |
| Family attribution | 84.71% | 84.62% | Frozen | 84.25% | 84.25% |
| Code-mixed human/AI | 95.37% | 95.29% | Frozen | 95.35% | 95.35% |

The serving model for each task is the CV-selected winner, even if another candidate later has a higher test accuracy. Choosing a winner after inspecting the test split would contaminate the evaluation. Small differences here do not establish a statistically significant gain.

## Fine-tuning settings

- Optimizer: AdamW; final-block learning rate 1e-5, supervised-head learning rate 1e-4, weight decay 0.01.
- Batch size 16, gradient clipping at 1.0, at most two epochs, seed 42 (fold seed = 42 + fold index).
- Trainable encoder parameters: 7,087,872. CPU execution; no full-encoder fine-tune was attempted.
- The temporary head is initialized from a logistic regression on frozen embeddings fitted only on the fold training subset.
- Outputs of the frozen lower encoder are cached in float32. Cache reconstruction is checked against the original MuRIL embeddings before training.
- TF-IDF vocabulary, scaling, supervised head and final transformer block are all fitted using only the training side of each fold.

## Reproduce

From the repository root after importing the supplied strict-cleaned data:

```bash
.venv/bin/python -m scripts.finetune
.venv/bin/python -m scripts.summarize_finetuning
```

On Windows use `.\.venv\Scripts\python.exe` instead of `.venv/bin/python`. This CPU experiment may take tens of minutes and several GB of cache storage. It replaces the serving artifacts with CV-selected winners; earlier artifacts are preserved locally in `.cache/finetune/previous_artifacts/`. Never rerun searches to optimize observed test scores.

## Interpretation

The presentation's 85.9% attribution result is five-fold grouped CV under the historical study protocol. The first application version's 84.25% was a single existing-test-split result. They are not directly comparable. In that first application experiment the same-test frozen baseline scored 84.76%, so classifier-only tuning did not improve attribution test accuracy. The table above is a new, matched comparison of frozen and partially fine-tuned representations.

These remain separate short-text task metrics, not document-level or end-to-end cascade accuracy. Cross-task source overlaps and the historical reuse of the existing test data remain limitations. Full fold results, losses, per-language scores and confusion matrices are in `reports/finetuning/`.
