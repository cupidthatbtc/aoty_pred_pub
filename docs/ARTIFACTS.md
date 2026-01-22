# Artifacts

Data
- data/raw/all_albums_full.csv (raw input)
- data/processed/regression_ready.csv (modeling dataset)
- data/features/feature_matrix.parquet (combined features)
- data/features/features_manifest.json (feature metadata)
- data/splits/split_manifest.json (train/val/test splits)
- data/processed/imputation_log.csv (imputation provenance)

Runs
- runs/<run_id>/config.yaml
- runs/<run_id>/dataset_hash.txt
- runs/<run_id>/trace.nc
- runs/<run_id>/diagnostics.json

Reports
- reports/tables/*.csv
- reports/figures/*.png
- docs/MODEL_CARD.md
