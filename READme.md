# Data Science and Machine Learning Lab  
## News Article Classification – Full Modeling Pipeline

This repository documents the complete experimental pipeline developed for the
*Data Science and Machine Learning Lab* project at Politecnico di Torino.

The work progresses from a lightweight and fully reproducible preprocessing
stage to increasingly expressive modeling strategies, with the explicit goal of
identifying the structural limits of linear models and motivating the transition
to contextual neural representations.

The project is structured as a **progressive exploration of model capacity**,
rather than a pure hyperparameter tuning exercise.

---

## Project Structure

The repository is organized around three main areas: data, modeling pipelines,
and experimentation. Each folder includes documentation describing its role in
the overall pipeline.

- **`data/`**  
  Contains all dataset variants:
  - `raw/` – original dataset as provided  
  - `processed/` – cleaned and feature-engineered datasets  
  - `submission/` – final CSV files submitted to the leaderboard  
  - `figures/` – figures used in analysis and reporting  

- **`src/`**  
  Canonical modeling pipelines:
  - `EDA.ipynb` – Exploratory Data Analysis  
  - `preprocessing.ipynb` – base preprocessing pipeline  
  - `preprocessing_and_FE/` – feature engineering and dataset construction  
  - `baseline/` – baseline linear models  
  - `TwoStage/` – Two-Stage linear pipeline (rules + Logistic Regression)  
  - `roberta_deberta_ensemble/` – contextual model experiments (RoBERTa, DeBERTa)

- **`experiments/`**  
  Targeted analytical notebooks used to validate modeling assumptions and justify
  architectural and feature-engineering choices through controlled empirical
  studies on the development set.

---

## Project Overview

The dataset consists of online news articles described by:

- `title`  
- `article` (HTML-heavy body text)  
- `source`  
- `timestamp`  
- `label` (7 classes)  
- `PageRank`

The task is a **multi-class classification problem** evaluated using
**Macro-F1**, which strongly penalizes systematic class confusion and makes the
task particularly sensitive to semantic overlap and class imbalance.

---

## Key Modeling Results (Leaderboard)

| Model                                   | Macro F1 |
|----------------------------------------|----------|
| Baseline Logistic Regression            | 0.716    |
| Tuned Logistic Regression               | 0.728    |
| Two-Stage Linear Model                  | 0.731    |
| Majority Vote (RoBERTa + DeBERTa, seeds)| 0.760    |
| **Pruned Majority Vote (final)**        | **0.761**|

The results show a clear performance plateau around **0.72–0.73** for linear
models, followed by a regime change when contextual neural representations are
introduced.

---

## Preprocessing Strategy

The preprocessing stage is intentionally minimal and fully reproducible. Its
goal is to remove structural noise while preserving semantic content.

### Text Processing

- HTML entities normalized  
- Lowercasing and whitespace normalization  
- Concatenation of `title` and `article` into a single text field  

**Motivation**

The combination *(title + article)* consistently outperforms either field alone:

- Titles provide highly compressed, discriminative signals  
- Articles supply broader semantic context  

No aggressive token rewriting or heuristic normalization is applied.

---

## Timestamp Analysis

- Invalid timestamps converted to missing values  
- Simple temporal features tested (year, month, availability flag)

**Findings**

Temporal information is largely implicit in lexical and editorial patterns.
Explicit timestamp features show unstable gains and often overfit due to missing
values and republished content. As a result, timestamps are excluded from final
models.

---

## Duplicate and Republishing Effects

EDA reveals near-duplicate articles that are republished:

- with different timestamps,  
- occasionally with different labels.

Duplicates are **not removed**, since:

- their distribution is consistent across splits,  
- removal risks introducing distribution shift.

Empirically, duplicates do not provide a deterministic shortcut signal.

---

## Use of the `source` Feature

The `source` attribute (news publisher) is included via **One-Hot Encoding**.

### Motivation

EDA shows that publishers exhibit strong and stable editorial regularities:

- topic specialization,  
- stylistic conventions,  
- recurring HTML structure.

The source therefore acts as a proxy for **editorial context**, which is only
partially recoverable from text alone.

Including `source` consistently improves Macro-F1 and reduces confusion between
semantically overlapping classes.

---

## Linear Modeling Pipeline

### Baseline

- Logistic Regression  
- Word and character TF-IDF  
- `source` as categorical feature  

**Result:** ~0.718 Macro-F1

---

## Two-Stage Linear Model (Key Contribution)

The Two-Stage pipeline addresses a fundamental limitation of linear models:
deterministic signals are diluted when averaged with weak evidence.

### Stage 1 – Rule Mining

High-purity lexical rules are extracted using support and purity constraints,
capturing near-deterministic patterns such as HTML artefacts and publisher
markers.

### Stage 2 – Linear Classifier

Remaining samples are classified with a tuned Logistic Regression using word and
character n-grams.

**Result:** ~0.731 Macro-F1  
This value represents the effective ceiling of linear separability for this
task.

---

## Transition to Neural Models

### Best Transformer Configurations (Development Split)

| Model    | Max Length | Best Epoch | Macro-F1 (DEV) |
|----------|------------|------------|----------------|
| DeBERTa  | 256        | 2          | 0.743 |
| DeBERTa  | 512        | 2          | 0.747 |
| RoBERTa  | 256        | 3          | 0.749 |
| RoBERTa  | 512        | 3          | **0.756** |

Transformer models converge rapidly and exhibit high stability across random
seeds.

---

## Ensemble and Agreement Analysis

Extensive overlap and disagreement analysis shows that:

- Predictions are **highly stable across seeds**,  
- Architectural differences induce **meaningful diversity**,  
- Ambiguity is concentrated in a small subset of samples.

---

## Final Ensemble Strategy

Final predictions are obtained via a **pruned majority-vote ensemble** over
multiple transformer instances trained with different architectures, sequence
lengths, and random seeds.

Model pruning is guided by:

- development-set performance,  
- prediction agreement analysis,  
- removal of consistently weaker or redundant predictors.

### Why Majority Voting

Majority voting operates at the **decision level** and does not rely on
calibrated confidence scores. This choice is deliberate.

Prior work shows that under **distribution shift, class imbalance, and
miscalibration**, probability-based ensembles (e.g. weighted logits, softmax
averaging) can degrade, while decision-level aggregation remains comparatively
stable.

The empirical results in this project confirm this behavior: more complex
logit-based ensembles do not consistently outperform majority voting on the
evaluation set.

**Final result:**  
**Macro-F1 = 0.761**, the best-performing configuration submitted.

---

## Conclusions

- Linear models saturate around **0.728–0.731** due to intrinsic ambiguity.  
- Two-stage strategies extract the maximum possible signal from linear features.  
- Contextual neural models are required to surpass this limit.  
- Majority-vote ensembling provides the most robust aggregation under realistic
  evaluation conditions.

This project demonstrates how careful EDA, principled modeling choices, and
controlled increases in model expressiveness lead to interpretable, reproducible,
and competitive results.
