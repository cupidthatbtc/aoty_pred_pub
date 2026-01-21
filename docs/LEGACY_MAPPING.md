# Legacy Mapping

Use docs/lineage/DATA_LINEAGE_DETAILED.md as the authoritative step map.

Suggested mapping to new modules
- Step 1-3 (loading, dedup, filtering): src/aoty_pred/data/ingest.py + cleaning.py
- Step 4-4b (feature engineering, reputation): src/aoty_pred/features/*
- Step 5 (split): src/aoty_pred/data/split.py
- Step 6 (imputation): src/aoty_pred/data/cleaning.py
- Step 7-8 (encoding and PCA): src/aoty_pred/features/genre.py + pca.py
- Step 9 (CV): src/aoty_pred/evaluation/cv.py
- Step 10-13 (modeling and outputs): src/aoty_pred/models/bayes + reporting