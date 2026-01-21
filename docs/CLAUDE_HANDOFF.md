# Claude Handoff

Purpose
- Implement a publication-ready Bayesian prediction pipeline for next-album artist scores.
- Keep this repository clean-room and separate from the legacy codebase.

Non-goals
- Do not edit legacy code.
- Do not rely on hidden caches or ad-hoc notebooks for results.

Inputs
- Raw CSV: all_albums_full.csv (see docs/DATA_CONTRACT.md)
- Lineage reference: docs/lineage/DATA_LINEAGE_DETAILED.md
- Schema snapshot: docs/RAW_SCHEMA_SNAPSHOT.md

Implementation order (recommended)
1. Config system
   - Implement config loading and validation in src/aoty_pred/config.
   - Add overrides via environment variables where needed.
2. Ingestion and validation
   - Implement raw schema checks and basic sanity tests.
   - Rename raw headers to canonical names (see RAW_TO_CANONICAL).
   - Record dataset hash and row counts for lineage.
3. Cleaning and filtering
   - Apply min ratings threshold.
   - Drop missing user score and critical numeric fields.
   - Log exclusion reasons per row.
4. Splitting and leakage guards
   - Primary: within-artist temporal holdout (last album per artist).
   - Secondary: artist-group split; no artist overlap across splits.
   - Store split manifests in data/splits.
5. Imputation and feature building
   - Train-only statistics for imputation and scaling.
   - Track imputation source for each field.
6. Feature block assembly
   - Build features via feature blocks and save to data/features.
   - Write a manifest with feature names and block metadata.
7. Modeling (Bayesian)
   - Implement baseline hierarchical model.
   - Implement dynamic slope model for artist trajectories.
   - Prior predictive checks before sampling.
8. Diagnostics and evaluation
   - R-hat <= 1.01, ESS thresholds, divergence checks.
   - PPC and calibration metrics.
9. Prediction and reporting
   - Generate next-album predictions with credible intervals.
   - Produce tables and figures for publication.
10. Sensitivity analysis
   - Threshold sensitivity, prior sensitivity, feature ablation.

Definition of done (publication-ready)
- All pipeline stages reproducible via CLI or scripts.
- Full lineage recorded with dataset hash and counts.
- Leakage controls enforced and tested.
- Diagnostics and calibration meet thresholds.
- Sensitivity analyses completed and documented.
- Tables and figures generated from code, not manually edited.

Preflight (defaults)
- Use the defaults in docs/DECISIONS_TO_LOCK.md unless overridden.
- Config merging is supported via multiple `-c` arguments (later files win).
- Dev environment: see docs/DEV_SETUP.md (miniforge + tests).

Key files to implement
- src/aoty_pred/pipelines/prepare_dataset.py
- src/aoty_pred/pipelines/build_features.py
- src/aoty_pred/pipelines/train_bayes.py
- src/aoty_pred/pipelines/predict_next.py
- src/aoty_pred/pipelines/sensitivity.py
- src/aoty_pred/pipelines/publication.py
- src/aoty_pred/models/bayes/*.py
- src/aoty_pred/evaluation/*.py
- src/aoty_pred/reporting/*.py

Testing requirements
- Unit tests for config, splits, imputation, and feature builders.
- Integration test for end-to-end dataset preparation.
- E2E test on a small fixture dataset.
