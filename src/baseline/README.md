# Baseline Model – Linear Classification with TF-IDF  
*(Optuna-tuned Logistic Regression)*

This file implements the final **single-stage baseline model** for the news article classification task.  
The goal of this model is to establish a strong, well-calibrated linear reference point before introducing hierarchical or neural architectures.

All results reported here are obtained using **parameter tuning via Bayesian Optimization (Optuna)** on the development set only.

## File Structure

- `baseline-parameter-studing.ipynb`  
  Performs a **controlled, hardcoded analysis of feature configurations**, used to understand the contribution of individual feature groups and to constrain the hyperparameter search space.

- `baseline-submission.ipynb`  
  Trains the final baseline model using the selected configuration and produces the **official submission file**.


---

## Model Overview

The baseline model is a standard linear classifier based on:

- TF-IDF representations of the text  
  - Word n-grams  
  - Character n-grams  
- One-hot encoded `source`  
- Minimal numeric features (lengths, ratios)

A **Logistic Regression** classifier is trained on this representation.

Despite its simplicity, this model already captures:

- coarse topical structure,  
- stylistic and editorial cues,  
- publisher-level regularities.

It provides a strong reference point for measuring the benefit of more complex architectures.

---

## Parameter Optimization

Before running Bayesian optimization, a controlled **hardcoded parameter study** was conducted in  
`baseline-parameter-studing.ipynb` to analyze the stability of the linear model across feature configurations.

The analysis reveals a **clear and stable performance jump** for the following configuration regime:

- `word_ng = 2`
- `char_ng = 5`
- `C ∈ [0.6, 1.0]`
- `max_df ∈ [0.80, 0.92]`

Top configurations consistently cluster around the same region:

| C | max_df | word_ng | char_ng | Macro F1 |
|---|--------|---------|---------|----------|
| 1.0 | 0.85 | 2 | 5 | 0.7193 |
| 1.0 | 0.90 | 2 | 5 | 0.7191 |
| 0.6 | 0.90 | 2 | 5 | 0.7190 |
| 0.6 | 0.85 | 2 | 5 | 0.7188 |

The improvement is **net, reproducible, and robust**, with negligible variance across nearby values.


This study shows that:
- performance gains are driven primarily by **n-gram structure**, not fine-grained regularization,
- the model reaches a **stable plateau** once the correct representational regime is selected,
- further gains cannot be obtained by linear feature tuning alone.


WORD_NG_MAX = 2
CHAR_NG_MAX = 5
MIN_DF      = 2
MAX_DF      = 0.85
C_VALUE     = 1


The tuning process explores:

- word and character n-gram ranges,  
- TF-IDF pruning thresholds (`min_df`, `max_df`),  
- regularization strength (`C`) of the Logistic Regression.

The objective function is **Macro F1**, ensuring balanced performance across all classes.

This setup avoids manual heuristics and yields a reproducible, statistically grounded baseline.

---

## Design Choices

### Source Encoding

The `source` feature is encoded using **one-hot encoding**.  
This choice is empirically motivated and analyzed in:

- `experiments/why_hot_encoding_source.ipynb`

The analysis shows that:

- publisher identity provides a meaningful but coarse discriminative signal,  
- numerical features derived from source statistics (entropy, support, dominant prior) degrade performance,  
- combining such statistics with one-hot encoding does not yield improvements.

For this reason, `source` is treated as a categorical variable and integrated via one-hot encoding in the linear pipeline.

---

### Timestamp Handling

Explicit timestamp features are intentionally **dropped** from the baseline model.

As demonstrated in:

- `experiments/timestamp_is_already_codified.ipynb`

the temporal signal (e.g. year-level information) is already implicitly encoded in the textual content itself through:

- vocabulary drift,  
- publication style,  
- topic distribution changes over time.

Adding explicit timestamp features does not provide additional predictive power and introduces instability and potential leakage.  
Therefore, the baseline model relies exclusively on textual and editorial signals.

---

## Results

| Model | Macro F1 (Leaderboard) |
|------|--------------------------|
| Baseline Linear Model (Optuna) | 0.728 |

This score represents the effective ceiling of a *single-stage linear model* on this dataset.

Further improvements beyond this point require either:

- explicit structural decomposition (e.g. multi-stage pipelines), or  
- contextual semantic representations (e.g. transformers).

---

## Runtime and Hardware

| Component | Specification |
|----------|----------------|
| Runtime | ~6–8 minutes |
| Machine | AMD64 |
| Processor | AMD64 Family 23 Model 104 Stepping 1, AuthenticAMD |
| CPU | 8 physical cores / 16 logical cores |
| RAM | 31.33 GB |
| GPU | No CUDA GPU detected |

---

## Role in the Project

This baseline serves as:

- the reference point for all subsequent modeling choices,  
- the lower bound for any architectural improvement,  
- a diagnostic tool for understanding dataset structure and class overlap.

Every multi-stage or neural model in this project is evaluated relative to this Optuna-tuned baseline.
