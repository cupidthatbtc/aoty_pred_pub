# Decisions To Lock

Purpose
- Make all publication-critical choices explicit so implementation is deterministic.
- Use these defaults unless overridden via config.

Change control
- All defaults must be reflected in `configs/base.yaml`.
- Change values in config (or per-run overrides), not in code.

Dataset and scope (defaults)
- Raw dataset path: `C:\Users\jcwen\Downloads\aoty full artist\all_albums_full.csv`
- Primary prediction target: next album `User Score` (0–100 scale) for each artist
- Inclusion criteria: `min_ratings = 30` and required fields present
- Required raw fields: Artist, Album, Year, Release Date, Genres, User Score, User Ratings, Tracks, Runtime (min), Avg Track Runtime (min), Album Type, All Artists
- Optional raw fields: Critic Score, Critic Reviews, Avg Track Score, Descriptors, Label, Album URL
- Exclusion rules: missing User Score, invalid Year, or missing required numeric fields after repair
- Canonical naming: map raw headers to snake_case (see docs/DATA_CONTRACT.md)
- Mapping reference: `src/aoty_pred/data/cleaning.py` (`RAW_TO_CANONICAL`)

Splits and leakage (defaults)
- Primary evaluation split: within-artist temporal holdout (last album per artist with >=2 albums)
- Secondary evaluation split: artist-group split (no artist overlap)
- Group split fractions: train/val/test = 0.65/0.15/0.20
- Group split seed: 42
- CV: 5-fold group-aware CV on training data
- Time-based holdout: enabled as a robustness check

Imputation and preprocessing (defaults)
- Imputation hierarchy: artist -> genre -> decade -> global (train-only stats)
- Minimum counts per level: artist >= 2, genre >= 5, decade >= 20
- Numeric repair rules (after canonical naming):
  - User_Score outside [0, 100] => drop
  - User_Ratings <= 0 => drop
  - Num_Tracks <= 0 => set NaN, then impute or drop if required
  - Runtime_Min <= 0 or Avg_Runtime <= 0 => set NaN, then impute or drop if required
  - Critic/Track scores outside [0, 100] => set NaN, then impute
  - Year outside [1950, current_year+1] => drop
- Category normalization:
  - lower-case, trim whitespace
  - replace '&' with 'and'
  - remove punctuation
  - collapse whitespace
  - replace spaces with underscores
- Standardization:
  - Continuous predictors z-scored using train-only stats
  - Target standardized for modeling, then inverse-transformed for outputs
- PCA:
  - Core numeric features: no PCA (keep interpretable)
  - Genres: PCA to 30 components (fit on train only)
  - Descriptors: PCA to 30 components (fit on train only)

Feature scope (defaults)
- Include genre features: yes
- Include artist random effects: yes
- Include temporal features (year, time_since_debut, album_seq): yes
- Include album type dummies: yes (Album as reference)
- Artist reputation: leave-one-out weighted mean with shrinkage (k=10) toward global mean
- Feature blocks order:
  - core_numeric
  - temporal
  - artist_reputation
  - genre_pca
  - descriptor_pca
  - album_type
- Each block lives in its own module under `src/aoty_pred/features/`.

Bayesian modeling (defaults)
- Baseline model: hierarchical with global + artist + genre effects
- Dynamic slope model: enabled; min_albums = 2 for slope estimation
- Likelihood: Normal on standardized target
- Priors (standardized scale):
  - Intercept: Normal(0, 1)
  - Slopes: Normal(0, 1)
  - Group SDs: HalfNormal(1)
  - Residual sigma: HalfNormal(1)
- Sampler: NUTS (NumPyro/JAX)
  - num_warmup = 2000, num_samples = 2000, num_chains = 4
  - target_accept_prob = 0.90, max_tree_depth = 12
- Diagnostics thresholds:
  - R-hat <= 1.01
  - bulk_ess >= 400, tail_ess >= 200
  - divergences = 0

Evaluation and reporting (defaults)
- Primary metrics: R2, RMSE, MAE
- Calibration checks: 80% and 95% interval coverage (target within ±3%)
- Model comparison: WAIC/LOO on training data; predictive metrics on held-out
- Output artifacts:
  - predictions.csv (mean + 80/95% intervals)
  - diagnostics.json
  - tables and figures
  - model card and reproducibility report
- Feature caching:
  - save combined matrix to data/features/feature_matrix.parquet
  - save manifest to data/features/features_manifest.json

Sensitivity analyses (defaults)
- Threshold sensitivity: min ratings (10/20/30/40)
- Feature ablations: remove genre, remove artist, remove temporal
- Prior sensitivity: slope_sd (0.5 vs 1.0)
- Dynamic slope sensitivity: min_albums (2 vs 3)
- Split sensitivity: within-artist vs artist-group split

Reproducibility (defaults)
- Dataset hash recorded per run
- Full config saved per run
- Dependency lock required before publication
