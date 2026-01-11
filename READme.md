# News Article Classification – Preprocessing (v1)

This repository contains the first preprocessing step for the Data Science and Machine Learning Lab project at Politecnico di Torino.

## UPDATE: There is a new best model, I tried very time but I think the plateau is reached. It is very simple to explain, only a good EDA to support it and that's all. Tomorrow I wil do the final introduction, convalidate it with optuna and start to write the final report

## In SRC there are basical EDA stuff
# The best model is in v3Code/little_aggressive.ipybn 
The data one have to been fixed becouse i used too much feature engeneering but basically the final model is the raw one, plus some FE on Timestamp. I didn't fine some best model with all experiment with other FE data (clustering, a priori, tokenization etc). They downgrade the model. In the section it use basically the first dataset without other features. 
Right now I'm tring to get some extra. To push on 0.732. I think there are correlation with timestamp. 
 
## Overview
In this version (v1), we focus on preparing the development dataset for subsequent exploratory analysis and modeling. The preprocessing is intentionally lightweight and fully reproducible, avoiding unnecessary linguistic transformations.

## Preprocessing steps
The following operations are applied:

- **Text cleaning**
  - Removal of HTML tags and embedded hyperlinks using *BeautifulSoup*
  - Decoding of HTML entities
  - Lowercasing and normalization of whitespace
  - Concatenation of `title` and `article` into a single text field

- **Timestamp handling**
  - Conversion of invalid timestamps (e.g. `0000-00-00 00:00:00`) to missing values (the missing one are with median attribute)
  - Extraction of simple temporal features (`year`, `month`)
  - Addition of a binary flag indicating timestamp availability

- **Missing values**
  - Minimal handling of missing textual fields through empty-string imputation
  - No row removal is performed

## Output
The preprocessed development dataset is saved programmatically and used as input for the subsequent EDA and modeling steps.

## Notes
This preprocessing stage is designed to remove structural noise while preserving the original lexical and semantic content of the articles. No advanced NLP parsing or external data sources are used.
 
