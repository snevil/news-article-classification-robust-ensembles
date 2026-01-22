# Experiments

This folder contains targeted analytical notebooks whose purpose is not to train final models, but to **understand the structure of the dataset** and to **justify key architectural and feature-engineering choices** in a principled way.

Each experiment isolates a specific hypothesis and validates it empirically on the development split only.  
The results in this folder directly motivate the design of the baseline, two-stage, and three-stage pipelines.
--- 

## `two_stage_parameter_finding.ipynb`

This notebook explores the hyperparameters of the **Two-Stage architecture** to understand *when* rule-based intervention is beneficial, rather than to optimize leaderboard score.

The grid search evaluates combinations of:
- **C**: regularization strength of the TF-IDF linear classifier  
- **min rule purity**: minimum precision required for a rule to be applied  
- **min rule support**: minimum number of occurrences required for a rule  

Each configuration is evaluated **only on the development set**, reporting:
- **Macro F1**
- **Rule coverage** (fraction of documents classified by the rule-based stage)

### Key Observations

- Best configurations consistently lie in the region:
  - `min rule purity ≥ 0.90`
  - `min rule support ∈ [20, 40]`
  - `C ∈ [0.5, 1.0]`
- Macro F1 remains stable (≈ 0.720–0.722) across a wide parameter range, indicating robustness rather than sensitivity.
- Increasing `C` beyond 1.0 provides no measurable benefit.

### Rule Coverage and Robustness

- Rule coverage is intentionally limited (≈ 15–20%), capturing only highly deterministic documents.
- Documents not matching any rule are automatically handled by the TF-IDF baseline.
- If evaluation documents contain unseen tokens or patterns, the model **falls back entirely to the baseline**, ensuring safe behavior under distribution shift.

### Conclusion

The Two-Stage model improves performance by isolating deterministic cases while preserving the generalization properties of the baseline classifier.  
Its conservative rule application and fallback mechanism prevent information leakage and ensure stable performance beyond the development distribution.

---

## `baseline_dev_pred_analyis_Features_Importance.ipynb`

This notebook runs the Optuna-tuned **baseline linear model** and performs a detailed feature-importance analysis.

The goal is to understand *how* the model makes decisions and *which* signals dominate.

### Key Findings

- A small subset of lexical tokens, n-grams, and editorial markers dominates the decision function.
- Many of these features exhibit **near-perfect purity** (often 1.0) and **zero entropy** across hundreds or thousands of documents.
- The presence of a single such pattern is often sufficient to determine the class with extremely high confidence, independently of surrounding context.

This reveals a strong **structural heterogeneity** in the dataset:

1. Documents that are *deterministic*, due to highly specific lexical or editorial patterns.  
2. Documents that are *genuinely ambiguous*, requiring probabilistic reasoning over weak semantic signals.

A single-stage classifier necessarily averages these two regimes, treating deterministic and noisy features in the same way.

This observation directly motivates the **Two-Stage architecture**:

- Stage 1 handles deterministic cases explicitly via high-purity rules.  
- Stage 2 focuses only on the remaining ambiguous documents.

The two-stage model therefore reflects the *intrinsic structure of the dataset*, rather than introducing ad-hoc complexity.

---

## `timestamp_is_already_codified.ipynb`

This notebook analyzes the `timestamp` feature and explains why it is **dropped from all final models**, starting from the baseline.

### Observations

- In the development set, the timestamp shows:
  - `missing_frac ≈ 0.3469`
  - `missing_count ≈ 27,750`
- The dataset is unevenly distributed across years (2004–2008), with a strong dominance of 2007.

Using only textual features and `source`, the **publication year can be predicted with high accuracy**.  
This demonstrates that temporal information is **implicitly encoded in the text itself** via:

- vocabulary drift,  
- publication style,  
- topic evolution over time.

Finer-grained temporal labels (month, week, day) were evaluated but found to be:

- noisy,  
- incomplete,  
- weakly discriminative.

### Conclusion

The explicit timestamp feature is:

- largely redundant,  
- unstable due to missing values,  
- a potential source of leakage.

For this reason, all final models rely exclusively on **textual and editorial signals**, and `timestamp` is intentionally excluded.

---

## `why_hot_encoding_source.ipynb`

This notebook evaluates how `source` should be represented.

Three controlled setups are tested on the development split:

1. **One-hot encoded source identity**  
   - Macro F1: **0.396**  
   - Accuracy: **0.454**  
   - Macro Recall: **0.395**  
   Several classes exhibit strong recall purely due to editorial bias, showing that source alone provides a meaningful but coarse signal.

2. **Numerical features derived from source statistics**  
   (support, entropy, dominant class prior)  
   - Macro F1: **0.225**  
   - Accuracy: **0.314**  
   - Macro Recall: **0.277**  
   Performance collapses, indicating that aggregation destroys class-specific structure and introduces noise.

3. **One-hot source + source-derived features**  
   - Macro F1: **0.395**  
   - Accuracy: **0.453**  
   No improvement over one-hot alone.

### Conclusion

Source information is best modeled as a **categorical identity via one-hot encoding**.

- Source identity provides a non-trivial discriminative signal.
- Numerical summaries of source behavior do not add independent information.
- Combining both is redundant.

This experiment justifies the consistent use of **one-hot encoding for `source`** in all pipelines.

---

## Role of the `experiments/` Folder

These notebooks:

- do not aim to maximize leaderboard score,  
- isolate and validate individual modeling hypotheses,  
- provide empirical grounding for every major architectural decision.

They ensure that the final pipelines are not heuristic, but **data-driven and structurally motivated**.
