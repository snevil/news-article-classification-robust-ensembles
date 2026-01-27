## Transformer Models – Development Split (Best Configurations)

The following table reports the best no-seed transformer configurations
selected on the development split. For each model, the maximum input length,
the epoch achieving the highest macro-F1 score, and the corresponding score
are reported.

| Model    | Max Length | Best Epoch | Macro-F1 (DEV) |
|----------|------------|------------|----------------|
| DeBERTa  | 256        | 2          | 0.743 |
| DeBERTa  | 512        | 2          | 0.747 |
| RoBERTa  | 256        | 3          | 0.749 |
| RoBERTa  | 512        | 3          | **0.756** |

---

## Submission Files Overview

This section lists all submission files used for overlap analysis,
ensembling, and final leaderboard submissions. Models are grouped by
architecture and maximum sequence length.

### DeBERTa – Max Length 512

| Model ID | Seed | File |
|---------|------|------|
| deberta_42 | 42 | `deberta_processed/deberta_seed_42_processed/submission_seed42.csv` |
| deberta_1337 | 1337 | `deberta_processed/deberta_seed_1337_processed/submission_seed_daberta_processed1337.csv` |
| deberta_2024 | 2024 | `deberta_processed/deberta_seed_2024_processed/submission_seed_daberta_processed2024.csv` |
| deberta_noseed | – | `deberta_processed/submission_deberta_processed_noseed_MAXLEN_512.csv` |

---

### DeBERTa – Max Length 256

| Model ID | Seed | File |
|---------|------|------|
| deberta256_42 | 42 | `deberta_MAXLEN256/deberta_seed42_MAXLEN256/submission_roberta_processed_seed42_MAXLEN256.csv` |
| deberta256_1337 | 1337 | `deberta_MAXLEN256/deberta_seed1337_MAXLEN256/submission_roberta_processed_seed1337_MAXLEN256.csv` |
| deberta256_2024 | 2024 | `deberta_MAXLEN256/deberta_seed2024_MAXLEN256/submission_roberta_processed_seed2024_MAXLEN256.csv` |
| deberta256_noseed | – | `deberta_MAXLEN256/submission_deberta_processed_noseed_MAXLEN_256.csv` |

---

### RoBERTa – Max Length 512

| Model ID | Seed | File |
|---------|------|------|
| roberta_42 | 42 | `roberta_processed/roberta_processed_seed_42/submission_seed_roberta_processed42.csv` |
| roberta_1337 | 1337 | `roberta_processed/roberta_processed_seed_1337/submission_seed_roberta_processed1337.csv` |
| roberta_2024 | 2024 | `roberta_processed/roberta_processed_seed_2024/submission_seed_roberta_processed2024.csv` |
| roberta_noseed | – | `roberta_processed/submission_roberta_processed_noseed_MAXLEN_512.csv` |

---

### RoBERTa – Max Length 256

| Model ID | Seed | File |
|---------|------|------|
| roberta256_42 | 42 | `roberta_MAXLEN256/roberta_seed42_MAXLEN256/submission_roberta_processed_seed42_MAXLEN256.csv` |
| roberta256_1337 | 1337 | `roberta_MAXLEN256/roberta_seed1337_MAXLEN256/submission_roberta_processed_seed1337_MAXLEN256.csv` |
| roberta256_2024 | 2024 | `roberta_MAXLEN256/roberta_seed2024_MAXLEN256/submission_roberta_processed_seed2024_MAXLEN256.csv` |
| roberta256_noseed | – | `roberta_MAXLEN256/submission_roberta_processed_noseed_MAXLEN_256.csv` |

---

**Notes**
- No-seed models are used as reference baselines.
- Multi-seed variants are employed to analyze stability and prediction overlap.
- All files are compatible with ensemble, temperature scaling, and agreement analysis pipelines.
