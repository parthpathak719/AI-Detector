# Research report index

- **`full_finetuning_completed/`**: final GPU experiment completed on 24 October 2026 after resuming from Colab on Kaggle. Contains the protocol, all candidate selection summaries, three final test reports, application-loader parity checks, and result table. All 60 fold prediction files from the supplied `results.zip` were checked against the original dataset, fold indices, saved metrics, and selected winners. Keep that private/local ZIP for the complete prediction arrays; this directory stores the compact final record, not model weights.
- **`full_finetuning/`**: historical CPU preparation, frozen folds and audit records. These support earlier provenance checks and are not the completed GPU experiment. Do not mix them into the completed result directory.
- **`finetuning/`**: historical three-fold final-transformer-block experiment. Frozen representations won that search; it did not train the full encoder.
- **`task*.json`**: historical classifier-only search reports.
- **`embedding_parity.json`**: original cached/fresh embedding checks.
- **`compatibility/`**: SciPy workaround preflight results.

Scores are task-specific, not document-authorship accuracy or leakage-free cascade accuracy. There are no uploaded user documents or full encoder checkpoints in this report tree.
