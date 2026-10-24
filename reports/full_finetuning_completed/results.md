# Full MuRIL five-fold experiment

Models were selected by mean five-fold source-grouped CV macro-F1 within the training split. Early stopping used separate inner groups. Test accuracy was measured after selection; it did not choose the model.

| Task | Frozen CV F1 | Full-tuned CV F1 | Selected | Selected CV accuracy | Selected test accuracy |
|---|---:|---:|---|---:|---:|
| task1_binary | 92.40% | 93.01% | lr_1e-05 | 93.01% | 93.87% |
| task2_attribution | 85.29% | 86.50% | lr_1e-05 | 86.56% | 86.43% |
| task3_codemix | 95.62% | 95.70% | lr_5e-06 | 95.71% | 95.74% |

CV scores above were used for tuning and are not unbiased final estimates. Small differences do not establish statistical superiority. The existing test set has been examined historically. These are separate short-text task metrics.

The selected_models.zip archive contains an artifacts/ directory for the application. It excludes private parity examples and training checkpoints. Back up the current artifacts directory before replacing it, then run scripts.verify_artifacts and restart the application.
