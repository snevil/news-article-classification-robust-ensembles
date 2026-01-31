# Source Code Structure

This directory contains the full modeling pipeline, organized in a strictly progressive and modular manner.  
Each folder corresponds to a well-defined experimental phase and is **fully documented through a dedicated README file inside the folder itself**.

All modules can be inspected independently, both conceptually and technically.

---

## - EDA

**Purpose**  
Exploratory analysis of the raw dataset to identify structural patterns, biases, redundancies, and non-informative signals.

**Contents**
- Distribution analysis of labels and metadata  
- Missing value inspection  
- Redundancy and duplication checks  
- Preliminary sanity baselines  

**Outcome**  
Guides preprocessing choices and motivates all subsequent modeling decisions.

---

## - Preprocessing_and_FE

**Purpose**  
Data cleaning and feature engineering informed directly by EDA findings.

**Contents**
- Text normalization and HTML cleaning  
- Metadata filtering and feature selection  
- Removal of non-informative or biased attributes  
- Construction of model-ready inputs  

**Outcome**  
A consistent, reproducible input representation shared by all downstream models.

---

## - Two_Stage

**Purpose**  
Implementation of a structured two-stage classification approach designed to explicitly capture editorial regularities.

**Stage 1**
- Deterministic or near-deterministic routing based on high-confidence patterns  

**Stage 2**
- Probabilistic classification applied only to genuinely ambiguous samples  

**Outcome**  
Reduces noise and isolates hard cases before applying high-capacity models, while empirically validating EDA hypotheses.

---

## - roberta_deberta_ensamble

**Purpose**  
Final performance-oriented modeling stage focused on robustness, agreement analysis, and estimation of the effective performance ceiling.

**Contents**
- Multi-model prediction agreement analysis  
- Intra-seed stability analysis  
- Cross-architecture comparison (RoBERTa vs DeBERTa)  
- Hard-case identification and characterization  
- Logit-level ensemble strategies  
- Entropy-based and architecture-aware routing  

**Key Design Choice**  
This module prioritizes **architecture diversity and selective routing** over aggressive stacking or calibration, based on empirical agreement and hard-case analysis.

**Outcome**  
A data-driven ensemble strategy that approaches the intrinsic performance limit of the task while remaining interpretable and robust.

---

## Notes

- Each folder contains a dedicated README describing its methodology, assumptions, and outputs in detail.
- Diagnostic artifacts are stored outside `src/` to avoid coupling analysis outputs with implementation code.
- The folder order reflects the logical progression of the project rather than execution order.
