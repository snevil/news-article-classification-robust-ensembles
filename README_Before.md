# News Article Classification – Preprocessing (v1)

This repository contains the first preprocessing step for the Data Science and Machine Learning Lab project at Politecnico di Torino.

## UPDATE: There is a new best model, I tried very time but I think the plateau is reached. It is very simple to explain, only a good EDA to support it and that's all. Tomorrow I wil do the final introduction, convalidate it with optuna and start to write the final report repo in v5/best_model_full_pipeline.ipynb

## In SRC there are basical EDA stuff
# The best model is in v3Code/little_aggressive.ipybn 
The data one have to been fixed becouse i used too much feature engeneering but basically the final model is the raw one, plus some FE on Timestamp. I didn't fine some best model with all experiment with other FE data (clustering, a priori, tokenization etc). They downgrade the model. In the section it use basically the first dataset without other features. 
Right now I'm tring to get some extra. To push on 0.732. I think there are correlation with timestamp. 
 
## Overview
In this version (v1), we focus on preparing the development dataset for subsequent exploratory analysis and modeling. The preprocessing is intentionally lightweight and fully reproducible, avoiding unnecessary linguistic transformations.

EDA: 
We can do EDA in this order: 
Check null (timestamp), looking that there are some valuation, but the deterministic impact is lower 
Check token, best words and coditional entropy 
Chek Title and source token 
We can say that Thecnology label has a quasi deterministic correlation. 
We can say there are duplicate in dataset, but we chose to not drop it (I sow there are the same distribution on eval so drop down them is slippery)
Source analysis (show them some website appears also in the text) 
Pipeline: 
We do some data clening to better mine the HREF tag 
After a simple FE trasformation to bettere perform with tokenization
We use source becouse is label deterministic

We use Logistic regression, becouse SVD cutting down all variability
BASELINE RESULT IN COMP 0.723

Explain the TWO Stage model: why is important and why it impact 
(It search with some strategy where the parameter is risky) 
Hyrparameter tuning (optuna bayesian search) 
Result 0.731 This is the roof of linear model search, we tryed to include TIMESTAP to interract but it overfit. Other strategy they overfit and likage the prediction. The cross fold analysys say that the result is stable e replicable. (some Lebal are overlapped so it cannot reach the full determination and some article are republished in oreder to TIMESTAMP and the label is not deterministich, so it is impossibile to predict) 

Best result for Tecnlogy class and Health class with avarega precision of 84% (so almost the top). The most crucial information of Tecnolgy was not the text but the editorial style(h4 and div tag) 
Healt by the words. 
0-5 are more rapresentative 
To motivate the Two Stage Pipeline 
Motivate the cutting edge strategy. 


3 Stage Bert embedded with 
Why finally use the embedded with Bert NPL model, and we reach 0.733 accuracy 













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
 
