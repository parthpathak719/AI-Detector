# Origin model card

## Intended use

Exploratory human/AI classification of short English, Hindi, Tamil and Kannada text, plus Romanized Hinglish, Tanglish and Kanglish. Conditional attribution compares the three labels in the supplied dataset: Gemini, Muse, Sarvam. It does not authenticate authorship, detect an arbitrary model's identity, or validate misconduct allegations.

## Training source and representation

User-supplied strict-cleaned data: 4,902 monolingual detection examples, 4,137 attribution examples and 1,712 code-mixed detection examples, including train/validation/test. Tasks reuse some records. Human sources in the study include CC-100, L3Cube and Dravidian code-mixed corpora; AI data uses topic-paired generation.

Encoder: `google/muril-base-cased`, revision `afd9f36c7923d54e97903922ff1b260d091d202f`, frozen, masked mean pooling including attended special tokens, max length 128, 768 dimensions. Fresh vectors were checked against cached vectors for one training example of each language in each task. The maximum absolute error was below 1e-6. All training rows were unchanged by the inference cleaner.

Classifier: character word-boundary TF-IDF (up to 150,000 features), concatenated with standardized embeddings weighted by `weight / sqrt(768)`, then logistic regression. No 16-feature block. The initial 12-candidate classifier search varied C, dense weight, n-gram range and minimum document frequency. A subsequent experiment compared three fixed hybrid settings with frozen, one-epoch and two-epoch final-block fine-tuned MuRIL representations. All three tasks selected frozen MuRIL. See the JSON reports for both traces.

## Evaluation boundaries

1. Compare frozen and final-block fine-tuned MuRIL on three grouped folds inside **train only**, maximizing mean macro-F1. Fit the fine-tuned block, supervised head, TF-IDF and scaling separately on each fold's training portion. Ties favor fewer fine-tuning epochs.
2. Fit the selected classifier on train only. Fit regularized logistic calibration on **val only**.
3. Evaluate once on the existing test split, and report the fixed baseline alongside it. Do not select the baseline or retune after seeing test results.

Within each task, train/validation/test source groups do not overlap. However, the historical study split each task independently. For example, 199 Task 1 test source groups also occur in Task 2 training; 43 Task 3 test source groups also occur in Task 2 training. Therefore the task test scores must **not** be multiplied or presented as a leakage-free cascade evaluation. A proper end-to-end benchmark needs a single shared source-group split across all tasks and retraining of all three classifiers. The app's functional cascade is tested, but no end-to-end predictive accuracy is claimed.

The original study selected hyperparameters on validation before pooling train/val/test for its five-fold CV. That is not fully nested CV, so its CV headline numbers are not directly comparable with this application's train-only tuning and held-out-split results. Furthermore, the original test data has already been analyzed historically; new independently collected data would be a stronger final assessment.

## Final-block fine-tuning findings

The experiment updated 7,087,872 MuRIL parameters in encoder layer 11, plus a temporary linear classification head. Embeddings and layers 0–10 stayed frozen. It used AdamW (encoder LR 1e-5, head LR 1e-4), batch size 16, weight decay 0.01 and gradient clipping at 1.0, for one/two epochs. Frozen lower-layer states were cached in float32 and their parity verified. This is actual partial encoder fine-tuning, not just tuning logistic regression; it is not a full-encoder fine-tune.

| Task | Best frozen selection CV macro-F1 | Best fine-tuned selection CV macro-F1 | Serving choice |
|---|---:|---:|---|
| Monolingual detection | 92.23% | 92.06% | Frozen |
| Attribution | 84.71% | 84.62% | Frozen |
| Code-mixed detection | 95.37% | 95.29% | Frozen |

The fine-tuned candidates did not improve the selection metric. These small differences do not establish statistical superiority. The production classifiers were refitted with the CV-selected settings, calibrated on the original validation split and evaluated on test only afterward. Current test accuracies remain 93.04%, 84.25% and 95.35%. Full results and the reproduction protocol are in [docs/finetuning-results.md](docs/finetuning-results.md).

## Earlier classifier-only findings

- Tuned test accuracy: Task 1 **93.04%**, Task 2 **84.25%**, Task 3 **95.35%**.
- Baseline: **93.31%**, **84.76%**, **94.96%**, respectively.
- Tuning has not demonstrated a broad gain. Task 3 improved by one test example. Do not describe this as a statistically significant improvement.
- Attribution is weakest in Hindi: **69.05% over 42 examples**. This small sample creates substantial uncertainty.
- Code-mixed Tanglish accuracy is **88.89% over 72 examples**, versus 99.29% for Hinglish over 140 examples. These language-specific small-sample differences matter.
- Calibration is not a guarantee of real-world correctness. It reduced test log loss on Tasks 2 and 3 but slightly increased it on Task 1 (0.1872 uncalibrated vs 0.1899 calibrated). It was retained uniformly as a predeclared serving protocol rather than chosen on test scores.

## Limitations and product behavior

- Source/domain confounds may remain after cleaning. Cleaning alone does not establish that all residual signal is an intrinsic writing-style fingerprint.
- Source grouping and exact deduplication do not establish the absence of all semantic/near-duplicate leakage.
- Family attribution is a **closed set** task. Unknown families will be forced toward one known family; there is no validated unknown-generator detector.
- The original unseen-generator study found weak transfer for Muse. This new training run has not repeated that leave-one-generator-out evaluation; it makes no improved unseen-generator claim.
- The 65% detector / 60% family uncertainty markers are interface heuristics, not optimized selective-classification thresholds or confidence intervals.
- Scores reflect balanced, cleaned research data and can be misleading under different real-world class proportions, prompts, domains, model versions or paraphrasing.
- The original feature extractor truncates at 128 tokens. Reported test metrics use those original cached features, including any truncated examples. The app now partitions long text into non-overlapping windows of at most 128 tokens, without discarding words. The task metrics do not measure this document aggregation workflow.
- Training-only AI sentence trimming and label-balanced length filtering cannot be replicated on unknown input. Inference applies only label-independent strict cleaning.
- Texts below 8 words have no supported attribution result even when the code-mixed detector accepts them. Long documents are supported through segment aggregation, and searchable PDF/DOCX/Word 97–2003 DOC text can be extracted. Mixed-authorship accuracy, OCR, and token-level attribution are not supported or validated.
- Full-encoder MuRIL fine-tuning was not performed. Final-block fine-tuning was evaluated but was not selected for serving; all deployed encoders remain frozen.

## Document aggregation (v2)

Each cleaned word appears once, in a contiguous segment that fits the original encoder window. Segment probabilities are averaged using cleaned word counts. This prevents a tiny final segment from having the same weight as a full segment. The resulting score is a model-score aggregate, **not a calibrated document probability or fraction of AI-authored content**. Overall family attribution runs only when the aggregate detector predicts AI, and uses AI-leaning segments with at least eight words; family probabilities are weighted by segment word count × AI score. Mixed human/AI segments are displayed individually. No new accuracy claim is made for long documents.

Limits: 100,000 raw/extracted characters, 20,000 cleaned words, 10 MB upload, 300 PDF pages, and at most 1,024 analysis segments. These are resource limits, not changes to the classifier or its trained token window. All classifier files and previous evaluation results remain unchanged.

## Next useful experiment

Create one shared source-group partition across tasks, reserve a new evaluation corpus, evaluate the complete cascade and unseen-generator detection, then evaluate broader encoder fine-tuning against the frozen hybrid using a new, fixed evaluation protocol. Expand Hindi attribution and Tanglish/Kanglish human data before enlarging the hyperparameter search.
