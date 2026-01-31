# Extensibility Guide

Goal
- Add, remove, or modify features without touching model code or pipelines.

Feature block pattern
1. Implement a block that follows `FeatureBlock` in `src/aoty_pred/features/base.py`.
2. Register it in `src/aoty_pred/features/registry.py` with a unique name.
3. Add it to `features.blocks` in config with parameters.
4. Add unit tests for fit/transform behavior and leakage checks.
5. Rebuild cached features via the feature builder pipeline.

Required behavior
- Fit uses train data only.
- Transform works on any split (train/val/test) without re-fitting.
- Block declares any dependencies in `requires`.
- Block logs its inputs/outputs for lineage.

Recommended file layout
- Feature implementation: `src/aoty_pred/features/<name>.py`
- Registry wiring: `src/aoty_pred/features/registry.py`
- Composition helper: `src/aoty_pred/features/pipeline.py`
- Tests: `tests/unit/test_feature_<name>.py`

Default block files
- core_numeric: `src/aoty_pred/features/core.py`
- temporal: `src/aoty_pred/features/temporal.py`
- artist_reputation: `src/aoty_pred/features/artist.py`
- genre_pca: `src/aoty_pred/features/genre.py`
- descriptor_pca: `src/aoty_pred/features/descriptor_pca.py`
- album_type: `src/aoty_pred/features/album_type.py`

Config usage (example)
```yaml
features:
  blocks:
    - name: core_numeric
      params: {}
    - name: genre_pca
      params:
        n_components: 30
```

Feature cache
- Build: `aoty-pipeline stage features`
- Outputs: `data/features/feature_matrix.parquet` and `data/features/features_manifest.json`

Robustness checks
- Add a small fixture dataset in `tests/fixtures`.
- Confirm no target leakage by comparing train-only stats to full-data stats.
- Validate outputs are stable under seed changes.
