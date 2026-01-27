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


## Hard-Case Analysis and Model Agreement

To analyze the performance limits of transformer models and the effectiveness of
ensembling, we conduct a dedicated agreement and hard-case analysis implemented
in `HARD_CASE.ipynb`. The analysis covers all transformer-based submissions,
including no-seed baselines and multi-seed variants across architectures and
input lengths.

### Methodology (`HARD_CASE.ipynb`)
The notebook performs:
- **Global prediction overlap**: pairwise agreement matrices measuring the
  fraction of identical predictions between models.
- **Intra-model stability (seed analysis)**: agreement across different random
  seeds for the same architecture and input length.
- **Cross-architecture comparison**: agreement between DeBERTa and RoBERTa
  variants.
- **Disagreement-based hard-case detection**: samples with high disagreement
  relative to the majority vote are flagged as hard cases.
- **Class-wise analysis of hard cases** to identify systematic ambiguity.

### Quantitative Findings

#### Intra-model Agreement (Stability Across Seeds)

| Model         | Mean Agreement | Std     | # Seeds |
|---------------|----------------|---------|---------|
| DeBERTa-512   | 0.8968         | 0.0011  | 3       |
| DeBERTa-256   | 0.9073         | 0.0009  | 3       |
| RoBERTa-512   | 0.9114         | 0.0004  | 3       |
| RoBERTa-256   | 0.9112         | 0.0001  | 3       |

**Observation.**  
All configurations show very high intra-seed agreement with negligible variance,
indicating that differences are driven by model capacity and input length rather
than random initialization.

#### Cross-Architecture Agreement
Average agreement between DeBERTa and RoBERTa models lies in the **0.88–0.91**
range (depending on input length). No-seed models exhibit slightly lower
agreement, consistent with the absence of implicit regularization from seed
averaging.

**Observation.**  
Despite architectural differences, models converge to highly similar predictions,
suggesting reliance on largely overlapping signals.

#### Class Distribution of Hard Cases

| Class          | Count | Percentage |
|----------------|-------|------------|
| Entertainment  | 366   | 27.35%     |
| General        | 352   | 26.31%     |
| International  | 277   | 20.70%     |
| Business       | 158   | 11.81%     |
| Technology     | 83    | 6.20%      |
| Health         | 58    | 4.33%      |
| Sports         | 44    | 3.29%      |

**Observation.**  
Hard cases are concentrated in semantically overlapping categories
(Entertainment, General, International), while specialized domains
(Sports, Health) are rarely ambiguous, consistent with duplicated or
republished articles carrying conflicting labels.

### Key Insights
- **Upper-bound effect**: a non-negligible fraction of errors is shared across
  all models, indicating intrinsic dataset ambiguity.
- **Limited ensemble headroom**: high cross-model agreement explains diminishing
  returns from complex ensembling.
- **Early-token dominance**: longer context windows do not reduce hard-case
  disagreement, suggesting that most discriminative information is contained in
  titles and early article segments.
- **Selective ensembling rationale**: results motivate lightweight strategies
  (e.g., weighted averaging, entropy-based routing) over deep stacking.