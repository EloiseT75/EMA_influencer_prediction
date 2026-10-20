# README

This repository contains three main model folders — each corresponding to a different methodology explored by the team — as well as a shared ensemble notebook used to combine results.

- **`BERT_and_user_aggregation/`** — user-level aggregation combined with BERT OOF features and XGBoost  
- **`Stacking_of_several_models/`** — stacking-based modeling approach  
- **`Catboost_optimization/`** — optimized CatBoost pipeline

The repository also includes a notebook performing a hard-voting ensemble across the three best models.

---

# `hardvote.ipynb`

### Purpose
- Loads predictions from multiple models (e.g., CatBoost, stacking, user-level XGB).  
- Applies **majority voting (hard vote)** across model predictions.  
- Exports an additional ensemble submission.

### Output
- `submission_hardvote.csv`

---


#\BERT_and_user_aggregation

## `bert-with-proper-oos.ipynb`

### Purpose
- Creates the **train 75% (to train the BERTs) / train 25% (the out of sample for the aggregation)** split by user.  
- Trains the **BERT models** (text + description).  
- Generates **BERT OOF predictions** (required as features for the user-level XGB).  
- Generates **BERT predictions on the test set**.

### Outputs
- `bert_oof_text.csv`  
- `bert_oof_desc.csv`  
- `bert_test_text.csv`  
- `bert_test_desc.csv`  
- `oos_idx_users.npy`
- `train_idx_users.npy`

These files are later merged into the user-level dataset as model-based features.


## `ml-user-lvl-bert.ipynb`

### Purpose
- Aggregates all available information at the **user level**.  
- Merges BERT OOF features, tweet-level features, and user-level metadata.  
- Trains the **XGBoost user-level model**.  
- Produces the final submission.

### Output
- `submission_user_level_xgb.csv`

---

# Stacking_of_several_models — Pipeline Overview

This folder contains three main notebooks implementing a complete stacking-based machine-learning pipeline, combining tabular features, CamemBERT embeddings, and ensemble modeling.

## 1. File Structure

The stacking workflow uses the following notebooks:

- **5.1. preprocessing.ipynb**  
  Tabular feature extraction.
- **5.2. camembert.ipynb**  
  Semantic feature extraction using CamemBERT.
- **5.3. stacking algo.ipynb**  
  Model training, ensemble creation, and submission generation.

All three notebooks must be executed in the same folder as the input data.



## 2. How to Run the Pipeline

The notebooks must be run **in order**, since each step produces intermediate files (CSV, NPY) used by the next step.

### Prerequisites  
Install Python and the following packages:

- `pandas`, `numpy`, `scikit-learn`  
- `lightgbm`, `catboost`, `xgboost`  
- `transformers`, `torch`  
- `tensorflow`  
- `scipy`

### Execution Steps

1. **STEP 1 — Tabular Feature Engineering**  
   Run **5.1. preprocessing.ipynb**  
   → Produces all base tabular features.

2. **STEP 2 — CamemBERT Feature Extraction**  
   Run **5.2. camembert.ipynb**  
   → Generates semantic features from tweet text.

3. **STEP 3 — Stacking & Submission**  
   Run **5.3. stacking algo.ipynb**  
   → Trains ML models, performs ensemble stacking, and outputs submission files.

After completing Step 3, the directory will contain **13 prediction files**, including the final ensemble used in hard voting:  
`submission_ultimate_ensemble.csv`

---

# Catboost_optimization

This folder contains the CatBoost-based solution.


## Files
- **preprocess_v3.py** – builds the tabular feature matrix (tweet metadata, user metadata, TF-IDF/SVD semantic features).
- **catboost_v3.py** – trains CatBoost (with cross-validation) and generates the final submission CSV.

