# RoBERTa Experiment – Full Pipeline  
*(DEV-split training, no-seed evaluation + EVAL predictions)*

This notebook implements a full end-to-end training pipeline based on **RoBERTa** for the news article classification task.  
The experiment is designed both as a strong contextual baseline and as a building block for future ensembling with other transformer models (e.g. DeBERTa).

---

## Why RoBERTa

RoBERTa is chosen for methodological and architectural reasons, not for convenience.

### What RoBERTa Captures Well

RoBERTa is particularly effective at modeling:

- Long-range contextual dependencies across full articles  
- Subtle semantic distinctions between news topics (e.g. *Politics* vs *International*)  
- Contextual disambiguation of recurring lexical patterns that are ambiguous in isolation  
- Implicit stylistic cues not easily captured by TF-IDF or linear models  

Unlike bag-of-words approaches, RoBERTa learns **context-dependent representations**, meaning the same token can contribute differently depending on surrounding content.

### Why RoBERTa over BERT

RoBERTa improves over BERT by design:

| Aspect | BERT | RoBERTa |
|---|---|---|
| Training data | Limited | Much larger |
| Masking | Static | Dynamic |
| Next Sentence Prediction | Yes | Removed |
| Optimization | Conservative | More aggressive |

Removing Next Sentence Prediction and using dynamic masking makes RoBERTa better suited for long, single-document classification, which matches this dataset.

### Why Not DeBERTa at This Stage

DeBERTa is not excluded, but **postponed intentionally**.

DeBERTa introduces:

- Disentangled attention (separating content and positional embeddings)  
- Relative position encoding  

These often yield higher peak performance, but:

- are more sensitive to randomness,  
- exhibit higher variance across seeds,  
- benefit more from multi-seed averaging or ensembling.

For this phase, RoBERTa is preferred because it offers:

- a strong and stable single-model baseline,  
- lower variance in no-seed / single-run settings,  
- cleaner diagnostics when analyzing errors and convergence.

DeBERTa is instead used as a complementary model in later ensemble experiments.

---

## Training Setup

- **Dataset:** development split only  
- **Preprocessing:** cleaned and normalized article text (see notebook for details)  
- **Tokenizer:** RoBERTa tokenizer  
- **Loss:** Cross-entropy  
- **Metric:** Macro F1 (primary), validation loss (secondary)

This notebook runs **without fixing a random seed**.  
This is intentional:

- the goal is to measure *typical* model behavior, not best-case variance,  
- seed control is introduced later during ensemble construction.

---

## Epochs, Logs, and Convergence

Training logs report:

- Training loss  
- Validation loss  
- Macro F1 on validation  

Key points when reading the logs:

- Validation loss typically stabilizes early  
- Macro F1 may fluctuate between epochs due to class imbalance  
- Early epochs capture coarse topical structure  
- Later epochs refine class boundaries but may overfit  

The number of epochs is chosen to balance:

- sufficient convergence,  
- avoidance of overfitting,  
- reasonable runtime.

This notebook is **not optimized for early stopping**, as its purpose is to generate a reliable single-model signal for comparison and ensembling.

---

## Role in the Overall Project

This RoBERTa experiment serves as:

- a contextual semantic baseline,  
- a complementary signal to linear and two-stage models,  
- a candidate component for ensemble methods.

It is not intended to replace the Two-Stage architecture, but to:

- capture semantic structure where deterministic patterns are absent,  
- provide diversity when combined with models such as DeBERTa.

---

## Training Dynamics (DEV Split)

Training on the development split shows a stable and consistent convergence pattern:

- **Epoch 1** captures most of the coarse semantic structure, reaching a Macro F1 of **0.726** with relatively high training and validation loss.  
- **Epoch 2** yields a clear improvement in both optimization and generalization, with Macro F1 increasing to **0.739** and validation loss decreasing further.  
- **Epoch 3** provides the best trade-off, achieving a Macro F1 of **0.751**, while training loss continues to decrease and validation loss slightly increases, indicating the onset of mild overfitting.

Overall, the monotonic decrease in training loss (from **0.718 → 0.487**) combined with a non-monotonic validation loss suggests that RoBERTa learns meaningful semantic representations early, with later epochs refining class boundaries at the cost of reduced regularization.

For this reason, the final checkpoint is selected based on **validation Macro F1** rather than loss, as the primary objective is balanced multi-class performance rather than likelihood optimization.
