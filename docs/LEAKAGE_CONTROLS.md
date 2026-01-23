# Leakage Controls

Mandatory rules
- No test data used for imputation, scaling, or feature selection.
- Group splits must prevent artist overlap across train/val/test.
- Within-artist temporal holdouts must use only prior albums for features.
- Leave-one-out artist features must exclude the target album.
- Category vocabularies must be fit on train only.
- CV folds must be group-aware and nested when tuning.

Artifacts to store
- data/splits/split_manifest.json
- data/splits/split_manifest_within_artist.json
- data/processed/imputation_log.csv
- runs/<run_id>/config.yaml
- runs/<run_id>/dataset_hash.txt
