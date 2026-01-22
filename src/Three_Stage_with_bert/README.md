# Three-Stage Model (Linear + Rules + BERT)  
**Leaderboard Macro F1: 0.733**

This notebook implements the three-stage extension of the final pipeline.  
All theoretical motivations and design choices are already discussed in detail in `src/TwoStage/README.md`.  
Here we only document *what is implemented* and *how it is integrated*.

---

## What Is Implemented

The three-stage model extends the two-stage architecture by adding a contextual semantic resolver based on BERT.

### Stage 1 – Rule-Based Filtering  
Deterministic lexical and editorial patterns (high support, high purity) are used to directly assign labels with near-zero entropy.

### Stage 2 – Linear Probabilistic Model  
Remaining samples are classified using a linear model with:

- TF-IDF (word + character n-grams)  
- One-hot encoded `source`  
- Minimal numeric features  

### Stage 3 – BERT Semantic Resolution  
Only samples that remain ambiguous after Stage 2 are passed to a BERT-based classifier, which operates on plain text and captures long-range semantic dependencies not accessible to linear models.

---

## Why a Three-Stage Design

The three-stage structure enforces a progressive increase in model capacity:

- deterministic cases are resolved early and cheaply,  
- semi-ambiguous cases are handled by a calibrated linear model,  
- only genuinely hard cases reach the contextual transformer.

This avoids:

- diluting deterministic signals inside high-capacity models,  
- wasting transformer capacity on trivial samples,  
- introducing unnecessary variance at earlier stages.

---

## Integration Strategy

- BERT is **not** used as a standalone classifier.  
- It is used selectively, as a **semantic fallback**.  
- Outputs are integrated into the existing linear pipeline without retraining earlier stages.  
- The final prediction is composed deterministically based on stage priority.

---

## Scope of This Notebook

- No new theoretical analysis is introduced here.  
- No feature engineering decisions are revisited.  
- This notebook focuses strictly on pipeline integration and execution.  

All assumptions, thresholds, and parameter choices are inherited from the two-stage model and documented in `src/TwoStage/README.md`.

---

## Summary

This notebook demonstrates how a contextual transformer (BERT) can be integrated into an already optimized linear + rule-based system, forming a hierarchical classifier where model complexity is applied only when necessary.
