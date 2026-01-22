## Preprocessing and Feature Engineering

The preprocessing pipeline is intentionally **minimal and conservative**, with the goal of preserving the original structure of the dataset and avoiding the introduction of heuristic or label-dependent signals.

Textual content is standardized by constructing a single `text` field obtained from the concatenation of `title` and `article`, followed by lowercasing. No additional text normalization is applied: there is no stemming, lemmatization, stopword removal, or regex-based manipulation. Semantic interpretation is entirely delegated to the modeling stage.

Data cleaning is limited to defensive operations only: missing textual fields are filled with empty strings, invalid rows are removed, and categorical fields are normalized as strings. No a priori assumptions or transformations are introduced.

Feature engineering is restricted to a small set of **transparent, length-based numeric features**, capturing basic document structure (e.g. token count, character lengths, title-to-article ratio). These features are model-agnostic, interpretable, and do not encode semantic or class-specific information.

No handcrafted features or priors are derived from the `source` attribute. As shown in dedicated experiments, source information is best treated as a pure categorical identity and encoded downstream via one-hot encoding. Numerical summaries of source behavior are intentionally excluded.

The `timestamp` feature is also excluded from all final models. Empirical analysis shows that temporal information is already implicitly encoded in the text itself, while explicit timestamp features are incomplete, unstable, and potentially leaky.

Overall, preprocessing and feature engineering are deliberately kept simple, ensuring that downstream performance gains arise from modeling choices rather than preprocessing artifacts.
