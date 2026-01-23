# Two-Stage Model

This file implements the final two-stage classification model.  
The theoretical motivation for this design choice is discussed in detail in the main `README.md` and is empirically supported by the analyses in:

- `experiments/baseline_dev_pred_analyis_Features_Importance.ipynb`
- `experiments/why_hot_encoding_source.ipynb`
- `experiments/timestamp_is_already_codified.ipynb`
- `experiments/parameter-two-stage-finding-ipynb.ipynb` 
- `src/TwoStage/optunna_two_stage_hyperparameter_tuning.ipynb` 
--- 

## Model Overview

The two-stage model explicitly separates deterministic and ambiguous cases, reflecting the intrinsic heterogeneity observed in the dataset.

| Stage | Description |
|------|-------------|
| **Stage 1 – Rule-based filtering** | Identifies documents containing highly deterministic lexical or editorial patterns (e.g. RSS markers, publisher-specific tokens, HTML layout artifacts). These patterns exhibit near-perfect class purity and zero entropy, meaning that their presence alone is sufficient to assign a label with extremely high confidence. |
| **Stage 2 – Probabilistic classification** | Documents that do not trigger any high-purity rule are passed to a standard linear classifier based on TF-IDF representations and additional features. This stage handles genuinely ambiguous cases, where no single token or structural marker is sufficient to determine the label. |

This decomposition avoids averaging deterministic and non-deterministic signals within a single model and allows each regime to be handled with the appropriate inductive bias.

---

## Why the Two-Stage Model Is Efficient

| Aspect | Explanation |
|-------|-------------|
| **Early resolution** | A substantial fraction of documents can be classified immediately by Stage 1, without invoking the full probabilistic pipeline. |
| **Robust fallback** | When the rule-based stage is inconclusive, the decision is automatically deferred to Stage 2, preventing overconfident or forced assignments. |
| **Cleaner decision boundary** | By removing near-deterministic examples, Stage 2 operates on a reduced and more homogeneous subset of the data, where probabilistic reasoning is actually required. |

---

## Two-Stage Parameters and Optimization

All parameters were optimized on the development set only using Bayesian Optimization (Optuna).

### Stage 1 – Rule Mining Parameters

| Parameter | Value | Meaning |
|----------|-------|---------|
| `MIN_RULE_SUPPORT` | 34 | Minimum number of documents in which a token or pattern must appear to be considered a valid rule. This prevents spurious or dataset-specific patterns from being treated as deterministic. |
| `MIN_RULE_PURITY` | 0.9263 | Minimum conditional class probability required for a pattern to be considered deterministic. A rule is accepted only if it assigns at least 92.6% of its occurrences to the same class, ensuring low entropy and high reliability. |

These parameters control the strictness of the deterministic regime and balance coverage against precision in Stage 1.

### Stage 2 – Probabilistic Classifier Parameters

| Parameter | Value | Meaning |
|----------|-------|---------|
| `WORD_NG_MAX` | 2 | Maximum word n-gram length used in TF-IDF features. Captures short lexical patterns without introducing excessive sparsity. |
| `CHAR_NG_MAX` | 5 | Maximum character n-gram length (word-boundary based). Useful for modeling editorial templates, markup fragments, and stylistic patterns. |
| `MIN_DF` | 2 | Minimum document frequency for TF-IDF features, filtering out extremely rare and noisy tokens. |
| `MAX_DF` | 0.8783 | Maximum document frequency threshold, removing overly common tokens that carry little discriminative information. |
| `C_VALUE` | 0.6449 | Inverse regularization strength of the logistic regression classifier. Controls the trade-off between model flexibility and generalization. |

---

## Design Choices and Ablations

### Source Encoding

Source information is encoded using one-hot encoding, rather than numerical source-derived statistics.

This choice is justified empirically in `experiments/why_hot_encoding_source.ipynb`, which shows that:

- source identity alone provides a meaningful but coarse discriminative signal,
- numerical features derived from source statistics (entropy, support, dominant prior) degrade performance,
- combining source-derived features with one-hot encoding does not yield improvements.

### Timestamp Handling

Timestamp features are intentionally excluded from the final model.

As shown in `experiments/timestamp_is_already_codified.ipynb`, the temporal information (year-level signal) is already implicitly encoded in the textual content itself. Adding explicit timestamp features does not provide additional predictive power and risks introducing leakage or redundant signals.

---

## Results

| Model | Macro F1 (Leaderboard) |
|------|--------------------------|
| Best single-stage baseline (Optuna) | 0.728 |
| Two-stage model (Optuna) | 0.731 |

The two-stage model improves upon the best single-stage baseline, confirming that explicitly separating deterministic and ambiguous regimes yields measurable gains beyond standard feature tuning.

---
## Two-Stage Model – Hyperparameter Strategy

- `experiments/parameter-two-stage-finding-ipynb.ipynb`  
  Explores the Two-Stage parameter space to identify stable and robust regions in terms of rule purity, rule support, and linear regularization.

- `src/TwoStage/two_stage_model_optuna_best_parameters.ipynb`  
  Applies **Bayesian hyperparameter optimization (Optuna)** within the identified region to maximize Macro F1 on the development set.

This two-step process combines **interpretability-driven analysis** with **principled hyperparameter tuning**, ensuring robustness and avoiding ad-hoc optimization.
---

## Runtime and Hardware

| Component | Specification |
|----------|----------------|
| Runtime | 8m 03s |
| Machine | AMD64 |
| Processor | AMD64 Family 23 Model 104 Stepping 1, AuthenticAMD |
| CPU | 8 physical cores / 16 logical cores |
| Max frequency | 1801.0 MHz |
| RAM | 31.33 GB |
| GPU | No CUDA GPU detected |

