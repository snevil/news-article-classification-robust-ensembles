# Data Science and Machine Learning Lab  
## News Article Classification – Full Modeling Pipeline

This repository documents the full experimental pipeline developed for the *Data Science and Machine Learning Lab* project at Politecnico di Torino.

The work progresses from a lightweight and fully reproducible preprocessing stage to increasingly expressive modeling strategies, with the explicit goal of identifying the structural limits of linear models and motivating the transition to contextual neural representations.

The project is structured as a *progressive exploration of model capacity*, rather than a pure hyperparameter tuning exercise.

---
## Project Structure

The repository is organized to clearly separate data, preprocessing, modeling pipelines, and experimental work.

.
├── data/
│   ├── raw/                  # Original dataset (as provided)
│   ├── processed/            # Cleaned and feature-engineered datasets
│   └── submission/           # Final CSV files submitted to the leaderboard
│
├── src/
│   ├── EDA.ipynb             # Exploratory Data Analysis
│   ├── preprocessing.ipynb   # Base preprocessing pipeline
│   ├── preprocessing_and_FE/ # Feature engineering and dataset construction
│   │
│   ├── baseline/             # Baseline linear models
│   ├── TwoStage/             # Two-Stage linear pipeline (rules + LR)
│   ├── Three_Stage_with_bert/# Three-Stage model with BERT embeddings
│   └── deberta_prediction/   # Contextual model experiments (DeBERTa)
│
└── experiments/              # Exploratory notebooks and ablation studies

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
| Two-Stage Linear Model              | 0.731    
| Three-Stage Model (BERT embeddings)| 0.733    |
| Final Model (deBerta fine-tuned)    | 0.745    |

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

- MIN_RULE_SUPPORT = 34  
- MIN_RULE_PURITY = 0.926328564964768  

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
- MAX_DF = 0.8782583211530898  
- C_VALUE = 0.64491922705094  

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

---

## Transition to Neural Models

### Three-Stage Model with BERT Embeddings

Adding sentence-level embeddings as dense features improves performance to ~0.733, confirming that semantic representations add information, but remain constrained by linear decision boundaries.

### Final Model – deBERTa Fine-Tuning

- RoBERTa large  
- 512-token context  
- Early stopping and proper validation  

**Final Result:** **0.745 Macro F1**

This jump is not incremental:

- it reflects a change of hypothesis space,  
- not better tuning,  
- the score saturates rapidly, indicating proximity to the dataset’s intrinsic ceiling (~0.75).

---

## Conclusions

- Linear models plateau around **0.728–0.731** due to semantic overlap and label ambiguity.  
- Two-stage strategies extract the maximum possible signal from linear representations.  
- Contextual neural models are required to go beyond this limit.  
- The final score reflects near-saturation of available information, not leakage or overfitting.

This project demonstrates how careful EDA, principled modeling choices, and controlled increases in model expressiveness lead to interpretable and reproducible performance gains.
