# Data Science and Machine Learning Lab  
## News Article Classification – Full Modeling Pipeline

This repository documents the full experimental pipeline developed for the *Data Science and Machine Learning Lab* project at Politecnico di Torino.

The work progresses from a lightweight and fully reproducible preprocessing stage to increasingly expressive modeling strategies, with the explicit goal of identifying the structural limits of linear models and motivating the transition to contextual neural representations.

The project is structured as a *progressive exploration of model capacity*, rather than a pure hyperparameter tuning exercise.

---
## Project Structure

The repository is organized around three main areas: data, modeling pipelines, and experimentation. Each folder has a readme file explaining the entire process step by step.

- **`data/`**  
  Contains all dataset variants:
  - `raw/` – original dataset as provided  
  - `processed/` – cleaned and feature-engineered datasets  
  - `submission/` – final CSV files submitted to the leaderboard  

- **`src/`**  
  Hosts the canonical modeling pipelines:
  - `EDA.ipynb` – Exploratory Data Analysis  
  - `preprocessing.ipynb` – base preprocessing pipeline  
  - `preprocessing_and_FE/` – feature engineering and dataset construction  
  - `baseline/` – baseline linear models  
  - `TwoStage/` – Two-Stage linear pipeline (rules + Logistic Regression)   
  - `roberta_deberta_ensemble/` – contextual model experiments DeBERTa and RoBERTa  

- **`experiments/`**  
  Contains targeted analytical notebooks whose purpose is to validate modeling assumptions and justify architectural and feature-engineering choices through controlled empirical studies on the development set.
## Project Overview

The dataset consists of online news articles described by:

- `title`  
- `article` (HTML-heavy body text)  
- `source`  
- `timestamp`  
- `label` (7 classes)
- `PageRank`
 
The task is multi-class classification evaluated using **Macro F1**, which strongly penalizes systematic class confusion and makes the problem particularly sensitive to overlapping semantic content.

---

## Key Modeling Results (Leaderboard)

| Model                               | Macro F1 |
|-------------------------------------|----------|
| Baseline Logistic Regression        | 0.716    |
| Tuned Logistic Regression           | 0.728   |
| Two-Stage Linear Model              | 0.731    |

The results show a clear plateau around **0.72–0.73** for linear models, followed by a regime change when contextual neural representations are introduced.

---

## Preprocessing Strategy

The preprocessing stage is intentionally minimal and reproducible, designed to remove structural noise while preserving semantic content.

### Text Processing
 
- HTML entities polished  
- Lowercasing and whitespace normalization  
- Concatenation of `title` and `article` into a single text field  

**Motivation**

The combination *(title + article)* consistently outperforms either field alone:

- Titles contain highly compressed, discriminative signals  
- Articles provide broader contextual semantics  

Feature engineering experiments confirmed that titles are particularly effective when combined with word and character n-grams.

Regex is used only to clean residual special characters; no aggressive token rewriting is applied.

---

## Timestamp Analysis

- Invalid timestamps converted to missing values  
- Simple temporal features tested (year, month, availability flag)

**Findings**

- Timestamp carries weak intrinsic semantic information already embedded in the text (publication style, vocabulary drift).
- Explicit timestamp features frequently overfit due to:
  - missing values,
  - republished articles,
  - label ambiguity across time.

Cross-validation showed unstable gains → timestamps were excluded from the final linear models.

---

## Duplicate and Republishing Effects

EDA revealed near-duplicate articles republished:

- with different timestamps,
- sometimes with different labels.

Duplicates were **not removed** because:

- their distribution is consistent between development and evaluation sets,
- removing them risks leakage or distribution shift.

Empirically, duplicates affect only a small subset of classes and do not provide a deterministic signal.

---

## Exploratory Data Analysis Highlights

- The **Technology** class shows near-deterministic patterns driven more by editorial structure (HTML tags such as `h4`, `div`) than lexical content.
- The **Health** class is instead driven by strong domain-specific vocabulary.
- Classes *Politics*, *International*, and *Business* exhibit significant semantic overlap, limiting perfect separability.
- Conditional entropy analysis confirms that some labels are intrinsically ambiguous, defining an upper bound on achievable Macro F1.

---

## Use of the `source` Feature

The `source` attribute (news publisher) is included in the modeling pipeline using **One-Hot Encoding**. Its inclusion is intentional and empirically motivated.

### Motivation

EDA revealed that the news source carries strong structural information only partially encoded in the text:

- Many publishers exhibit highly consistent editorial policies in terms of:
  - topic coverage,
  - writing style,
  - HTML structure and formatting.
- Several sources show a skewed label distribution, with some classes appearing disproportionately often.
- In multiple cases, the publisher name appears **explicitly inside the article text**, both:
  - in *inner links* (e.g. attribution blocks, embedded references),
  - in *outside links* and HTML artefacts.

This creates a non-trivial interaction between `source` and textual features.  
The source is therefore not an auxiliary variable, but a *latent generator* of the observed text distribution.

### Why One-Hot Encoding

`source` is categorical and non-ordinal. One-Hot Encoding:

- preserves the non-ordinal nature of publishers,
- allows the classifier to learn publisher-specific biases,
- avoids artificial similarity between unrelated sources,
- prevents information loss that would occur with hashing or embeddings under limited data.

### Interaction with the Model

Including `source` enables the model to:

- disambiguate articles with weak or ambiguous lexical signals,
- exploit stable publisher-level regularities,
- reduce variance in classes with strong editorial separation (e.g. Technology, Entertainment).

Empirically, removing `source` leads to:

- consistent degradation in Macro F1,
- increased confusion between semantically overlapping classes.

This confirms that `source` acts as a **low-noise, high-signal feature** when combined with TF-IDF and character n-grams.

### Methodological Justification

- `source` is part of the provided dataset,
- it does not introduce leakage,
- its distribution is consistent across splits.

Rather than being a shortcut, `source` serves as a proxy for *editorial context*, which is otherwise difficult to recover from text alone.

---

## Linear Modeling Pipeline

### Feature Engineering

Extensive experiments were conducted:

- clustering features,  
- a-priori rules,  
- advanced tokenization,  
- timestamp interactions.

Most degraded performance due to overfitting or redundancy.  
The best-performing configuration remains close to the raw text representation, with limited numeric features.

---
### Baseline Model

- Logistic Regression  
- Word and character TF-IDF  
- `source` as categorical feature  

**Result:** ~0.718 Macro F1

---


## Two-Stage Linear Model (Key Contribution)

The Two-Stage Pipeline addresses a fundamental limitation of linear models:

> Highly informative tokens are diluted when averaged with weak signals.

### Stage 1 – Rule Mining

High-purity lexical rules are mined directly from the training data.

A token becomes a rule if it satisfies:

- MIN_RULE_SUPPORT = 20
- MIN_RULE_PURITY = 0.984162581589456  

These thresholds ensure:

- sufficient statistical support,  
- near-deterministic class association.

The extracted rules capture:

- HTML artefacts,  
- source-specific editorial markers,  
- recurring deterministic patterns.

These tokens correspond to low-entropy signals that a linear model would otherwise dilute when mixed with weaker evidence.

---

### Stage 2 – Linear Classifier

Samples not covered by any rule are classified using a tuned Logistic Regression with:

- WORD_NG_MAX = 2  
- CHAR_NG_MAX = 5  
- MIN_DF = 2  
- MAX_DF = 0.9077050155274585 
- C_VALUE = 0.8424574689304396  

#### Motivation

- Word n-grams up to bigrams capture short semantic patterns.  
- Character n-grams capture stylistic and markup signals.  
- The regularization parameter C balances variance reduction and signal preservation.

---

## Why the Two-Stage Model Works

- Stage 1 captures low-entropy cases with near-zero uncertainty.  
- Stage 2 focuses exclusively on genuinely ambiguous samples.

This separation prevents deterministic information from being diluted by weak or noisy signals.

**Result:** ~0.731 Macro F1  
This value represents the ceiling of linear separability for this task.
There are multiple maximum points that can be obtained by combining those parameters.
---



This section reports the best-performing transformer configurations selected on
the development split, followed by an analysis of prediction stability across
random seeds, sequence lengths, and architectures.

---

## Transition to Neural Models Best Transformer Configurations (Development Split)

| Model    | Max Length | Best Epoch | Macro-F1 (DEV) |
|----------|------------|------------|----------------|
| DeBERTa  | 256        | 2          | 0.743 |
| DeBERTa  | 512        | 2          | 0.747 |
| RoBERTa  | 256        | 3          | 0.749 |
| RoBERTa  | 512        | 3          | **0.756** |

**Interpretation.**  
RoBERTa with longer input length yields the strongest single-model performance.
Increasing the maximum sequence length provides consistent gains for both
architectures, suggesting that relevant information is mostly contained within
the first few hundred tokens, but benefits from moderate context expansion.

---

## Overlap and Disagreement Analysis

Prediction overlap is defined as the fraction of samples receiving identical
labels across different models. This analysis evaluates stability across random
seeds (intra-model) and diversity across architectures.

---

### Intra-Model Agreement (Seed Stability)

| Model | Max Length | Mean Agreement | Std | #Seeds |
|------|------------|----------------|-----|--------|
| DeBERTa | 512 | 0.8968 | 0.0011 | 3 |
| DeBERTa | 256 | 0.9073 | 0.0009 | 3 |
| RoBERTa | 512 | 0.9114 | 0.0004 | 3 |
| RoBERTa | 256 | 0.9112 | 0.0001 | 3 |

**Interpretation.**  
All models exhibit very high intra-seed agreement with extremely low variance.
This indicates strong optimization stability and confirms that random
initialization contributes marginally to prediction differences.

---

### Cross-Architecture Overlap

| Comparison | Overlap Range |
|-----------|---------------|
| DeBERTa ↔ RoBERTa (512) | 0.85 – 0.88 |
| DeBERTa ↔ RoBERTa (256) | 0.84 – 0.88 |

**Interpretation.**  
Despite comparable macro-F1 scores, RoBERTa and DeBERTa show systematic
prediction differences. This architectural diversity provides a clear rationale
for cross-model ensembling.

---

### Pairwise Disagreement (Summary)

| Comparison Type | Disagreement Range |
|-----------------|-------------------|
| Same architecture, different seed | 0.09 – 0.11 |
| Same architecture, different length | 0.11 – 0.13 |
| Cross-architecture | 0.13 – 0.15 |

Disagreement increases when moving from seed-level variation to architectural
variation, indicating that diversity is structural rather than noise-driven.

---

### Hard-Case Distribution

A limited subset of samples exhibits high disagreement across models. These
cases are concentrated in semantically overlapping categories.

| Class | Percentage |
|------|------------|
| Entertainment | 27.35% |
| General | 26.31% |
| International | 20.70% |
| Business | 11.81% |
| Technology | 6.20% |
| Health | 4.33% |
| Sports | 3.29% |

---

### Summary

The analysis shows that:
- Transformer predictions are highly stable across random seeds.
- Architectural differences induce meaningful diversity.
- Ambiguity is concentrated in a small subset of samples.

These findings motivate the use of weighted cross-architecture ensembles and
selective hard-case routing rather than aggressive retraining or deep stacking.

## Conclusions

- Linear models plateau around **0.728–0.731** due to semantic overlap and label ambiguity.  
- Two-stage strategies extract the maximum possible signal from linear representations.  
- Contextual neural models are required to go beyond this limit.  
- The final score reflects near-saturation of available information, not leakage or overfitting.

This project demonstrates how careful EDA, principled modeling choices, and controlled increases in model expressiveness lead to interpretable and reproducible performance gains.
