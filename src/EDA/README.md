# Exploratory Data Analysis (EDA)

This exploratory data analysis aims to characterize the structure, biases, and
potential sources of noise in the dataset, and to motivate the modeling choices
adopted in subsequent stages. The analysis is conducted on the development
split unless otherwise specified.

---

## Dataset Overview

The development set consists of approximately 80k news articles, each described
by:
- a textual component (`title`, `article`),
- categorical metadata (`source`),
- numerical metadata (`page_rank`, `timestamp`),
- a single target label among 7 topical classes.

The evaluation set follows the same structure, except for the absence of labels.

---

## Label Distribution

The class distribution in the development set is moderately imbalanced.
Some classes (e.g., labels 0 and 5) are substantially more frequent than others
(e.g., label 6), motivating the use of macro-averaged evaluation metrics and
class-balanced losses in downstream models.

This imbalance is consistent across multiple exploratory views and is treated
as an intrinsic property of the dataset rather than a preprocessing artifact.

---

## Missingness and Metadata Quality

A basic missingness analysis reveals that:
- the `timestamp` field contains a high fraction of missing or invalid values
  (≈35%),
- all other fields exhibit negligible missingness.
Further inspection shows that timestamps exhibit coarse granularity and strong
editorial regularities, suggesting that temporal information is largely
*implicitly encoded in the article text itself*. An auxiliary experiment
confirms that publication time can be predicted from raw textual content with
non-trivial accuracy, indicating that temporal cues are already captured by
lexical and editorial patterns. For this reason, fine-grained and explicit
timestamp features are not emphasized in subsequent models (see
`experiments/timestamp_is_already_codified.ipynb`).

---

## Exact Duplicates and Label Noise

A non-negligible number of exact duplicates is observed:
- ≈10.5% duplicated articles,
- ≈9.6% duplicated titles,
- ≈6.9% duplicated title–article pairs.

Importantly, duplicated articles are **not consistently labeled**:
more than 57% of duplicated article groups are associated with **multiple
distinct labels**.

This indicates that identical content can receive different topical annotations,
likely due to editorial framing or source-dependent categorization. Such behavior
constitutes a clear source of label noise and motivates:
- robust modeling choices,
- conservative hyperparameter tuning,
- and the analysis of hard or ambiguous examples in later stages.

---

## Duplicates vs Global Label Distribution

Comparing label distributions between duplicated articles and the full dataset
reveals significant deviations. Certain classes are over-represented among
duplicates, confirming that duplication is not label-neutral and may amplify
class-specific noise patterns.

---

## Source Distribution and Long-Tail Behavior

The `source` feature exhibits a pronounced long-tail distribution:
- a small number of sources accounts for a large fraction of the dataset,
- the majority of sources appear infrequently.

This heavy-tailed behavior suggests strong editorial biases and motivates:
- explicit modeling of the `source` feature,
- the use of regularization to avoid source overfitting.

Given the strong editorial biases observed across sources, the `source` feature
is explicitly modeled using one-hot encoding. A dedicated experiment confirms
that source identity provides complementary signal beyond textual features and
improves linear and hybrid models (see
`experiments/why_hot_encoding_source.ipynb`).

---

## PageRank Analysis

The `page_rank` feature is highly skewed, with most articles concentrated at the
maximum value. Nevertheless:
- its distribution varies across labels,
- a simple effect-size analysis (η² ≈ 0.29) indicates a non-trivial association
  between page rank and topic.

While not decisive on its own, `page_rank` provides complementary signal and is
therefore retained as a numerical feature in linear and hybrid models.

---

## Summary and Implications

Overall, the EDA highlights that the dataset is characterized by:
- moderate class imbalance,
- substantial duplication with inconsistent labeling,
- strong source-dependent biases,
- and partially informative but noisy metadata.

These findings motivate:
- the use of macro-averaged evaluation metrics,
- class-balanced learning,
- conservative hyperparameter exploration,
- and modeling strategies designed to handle ambiguity and label noise rather
  than aggressively overfitting the training data.
