# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2025-01-18)

**Core value:** Predict artist's next album score with calibrated uncertainty estimates
**Current focus:** Phase 14 Interactive Visualization

## Current Position

Phase: 14 of 14 (Interactive Visualization)
Plan: 2 of 3 in current phase
Status: In progress
Last activity: 2026-01-20 - Completed 14-02-PLAN.md (Export & Dashboard Assembly)

Progress: [#########################] ~100% (33/33 plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 33
- Average duration: ~8.5min
- Total execution time: ~4.72 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-data-foundation | 2 | ~27min | ~13.5min |
| 02-split-infrastructure | 2 | ~19min | ~9.5min |
| 03-feature-framework | 2 | ~30min | ~15min |
| 04-core-features | 3 | ~25min | ~8.3min |
| 05-model-basics | 2 | ~34min | ~17min |
| 06-fitting-prediction | 2 | ~26min | ~13min |
| 07-evaluation | 4 | ~34min | ~8.5min |
| 08-reporting | 3 | ~14min | ~4.7min |
| 09-pipeline-integration | 6 | ~33min | ~5.5min |
| 10-integration-gap-closure | 1 | ~8min | ~8min |
| 11-publication-interface-alignment | 1 | ~5min | ~5min |
| 12-errors-from-testing | 2 | ~8min | ~4min |
| 13-gpu-features-testing | 2 | ~27min | ~13.5min |
| 14-interactive-visualization | 2 | ~22min | ~11min |

**Recent Trend:**
- Last 5 plans: 13-01 (~12min), 13-02 (~15min), 14-01 (~12min), 14-02 (~10min)
- Trend: GPU benchmarking complete, visualization module coming together

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Init]: NumPyro + JAX chosen over PyMC for GPU performance (research confirmed)
- [Init]: 9-phase structure derived from requirements and research
- [01-01]: Default validate=False in load_raw_albums() (raw data has 5 null Albums)
- [01-01]: Use pandera.pandas import (future-proof)
- [01-02]: Drop Descriptors column (4.2% coverage, selection bias)
- [01-02]: Three-tier date risk classification (low/medium/high)
- [01-02]: Multi-threshold datasets at 5, 10, 25 min ratings
- [02-01]: Default 1 test album, 1 val album, 1 min train album per artist (min 3 total)
- [02-01]: Artist-disjoint uses 15% test, 15% val by default
- [02-01]: Deprecated old group_split stub with DeprecationWarning
- [02-02]: Same-date albums allowed in temporal validation (> not >=)
- [02-02]: Manifest filenames: split_{version}_{timestamp}_{hash_prefix}.json
- [03-01]: is_fitted uses getattr with default False (not hasattr) for cleaner property
- [03-01]: NotFittedError inherits both ValueError and AttributeError (sklearn convention)
- [03-01]: FittedVocabulary.encode returns unknown_idx for unseen values (not error)
- [03-02]: FeaturePipeline uses same is_fitted pattern as BaseFeatureBlock
- [03-02]: Deprecated run_feature_blocks kept for backwards compatibility
- [04-01]: TemporalBlock is stateless - validates columns, no learned statistics
- [04-01]: Album name used as tiebreaker for same-date album ordering (determinism)
- [04-01]: AlbumTypeBlock uses one-hot encoding with FittedVocabulary
- [04-01]: Missing Album_Type defaults to "Album" (most common type)
- [04-01]: Unknown album types in test data get all-zero encoding
- [04-02]: ArtistHistoryBlock uses leave-one-out for prior album stats
- [04-02]: Debut detection uses user_prior_mean.isna() (not count - count returns 0 not NaN)
- [04-02]: Global mean from training used for debut imputation
- [04-02]: Unknown artists get nan values (later imputed or zero-filled)
- [04-03]: GenreBlock uses min_genre_count=20 default for rare genre exclusion
- [04-03]: PCA skipped if vocab_size < n_components
- [04-03]: CollaborationBlock ordinal: solo=0, duo=1, small_group=2, ensemble=3
- [05-01]: Non-centered parameterization via LocScaleReparam(centered=0) for artist_effect
- [05-01]: Default sigma_artist_scale=0.5 encourages moderate pooling
- [05-01]: n_artists must be explicitly passed to model (not inferred)
- [05-01]: Model returns both artist_effect and artist_effect_decentered in samples
- [05-02]: max_seq passed as explicit parameter for JAX tracing compatibility
- [05-02]: Random walk implemented via numpyro.contrib.control_flow.scan
- [05-02]: TruncatedNormal(-0.99, 0.99) for AR coefficient to ensure stationarity
- [05-02]: Parameter prefixes (user_, critic_) enable fitting both models with distinct posteriors
- [06-01]: Check existing groups before add_groups (az.from_numpyro creates observed_data)
- [06-01]: Use jax.random.key() modern API over deprecated PRNGKey()
- [06-02]: batch_ndims=1 with Predictive since mcmc.get_samples() returns flattened samples
- [06-02]: y=None required in model_args for prediction mode
- [06-02]: predict_new_artist samples from Normal(mu_artist, sigma_artist) for unseen artists
- [07-01]: ESS threshold applied as total ESS (not per-chain) matching ArviZ convention
- [07-01]: allow_divergences parameter enables sensitivity analysis with non-zero divergences
- [07-01]: Use .sizes for xarray dict access (avoids FutureWarning on .dims)
- [07-02]: batch_ndims=1 required for log_likelihood since mcmc.get_samples() returns flattened samples
- [07-02]: Pareto-k threshold 0.7 used for high-k detection (ArviZ standard)
- [07-02]: Log-likelihood reshaped to (chain, draw, obs) for ArviZ compatibility
- [07-02]: Prior predictive uses y=None in model_args to generate predictions
- [07-03]: Equal-tailed intervals for coverage (not HPD)
- [07-03]: Equal-count binning for reliability diagrams (not equal-width)
- [07-03]: CRPS loop over observations for properscoring compatibility
- [07-04]: Three prior configs: default, diffuse (5x wider), informative (2x tighter)
- [07-04]: Zero-out ablated features (assumes standardized where zero = mean)
- [07-04]: Threshold sensitivity uses data_loader callable for decoupled data loading
- [08-02]: Wong (2011) colorblind-safe palette from Nature Methods
- [08-02]: Context manager pattern for style isolation (not global rcParams)
- [08-02]: Error bars for uncertainty in predictions plot (not fill_between)
- [08-03]: Hugging Face model card style adapted for Bayesian models
- [08-03]: Two-tier model cards: simple overview at root, technical in reports/
- [08-03]: Dual format output (Markdown + LaTeX) for docs and publications
- [08-01]: HDI column detection: dynamic parsing of ArviZ column names for varying decimal precision
- [08-01]: Adaptive precision uses uncertainties package with PDG-style 2 sig figs on uncertainty
- [08-01]: Convergence status reports specific failure reason (R-hat, ESS, or both)
- [09-01]: JAX PRNG handled separately via get_rng_key, not global state
- [09-01]: Console logging to stderr (not stdout) for pipeline compatibility
- [09-01]: Deprecated setup_logging kept for backwards compatibility
- [09-01]: EnvironmentError custom exception for strict mode failures
- [09-02]: GitStateModel Pydantic wrapper for GitState dataclass serialization
- [09-02]: pixi_lock_hash in EnvironmentInfo for exact environment reproducibility
- [09-02]: should_skip strict conditions: hash match, outputs exist, not force
- [09-02]: Kahn's algorithm for topological sort of stage dependencies
- [09-03]: Separate EnvironmentError in errors.py for consistent exit codes
- [09-03]: Windows file handler cleanup before directory move
- [09-03]: PermissionError catch for graceful degradation on Windows
- [09-04]: Comma-separated stages string for --stages flag (cleaner than Typer list)
- [09-04]: Legacy config-based commands marked hidden=True (backwards compatible)
- [09-04]: invoke_without_command=True for version flag handling
- [09-05]: StageContext defined in stages.py (not orchestrator.py)
- [09-05]: Factory pattern (make_stage_*) for lazy imports
- [09-05]: ConvergenceError handling at orchestrator level
- [09-06]: Use workspace instead of deprecated project in pixi.toml
- [09-06]: Merge dev dependencies into main (avoids unused feature warning)
- [09-06]: pixi.toml for environment management, pyproject.toml for package metadata
- [10-01]: New dataclass fields added at END to maintain backward compatibility
- [10-01]: Threshold fields in ConvergenceDiagnostics: rhat_threshold (float), ess_threshold (int)
- [10-01]: Posterior samples reshaped (chains, draws, n_obs) -> (n_samples, n_obs) for compute_coverage
- [10-01]: properscoring declared in both pyproject.toml and pixi.toml pypi-dependencies
- [11-01]: Pass idata only to update_model_card_with_results (typed object reconstruction from JSON deferred)
- [11-01]: Capture dual return (pdf_path, png_path) from figure saving functions
- [12-01]: Added jinja2 and gitpython as missing dependencies
- [12-02]: Use strict=True in PipelineConfig when testing error propagation paths
- [12-02]: Delete obsolete tests using deprecated APIs rather than rewrite
- [13-01]: WSL2 required for JAX CUDA on Windows (direct Windows CUDA not supported)
- [13-01]: Install cuda-toolkit-12-x only, never cuda or cuda-drivers in WSL2
- [13-01]: RTX 5090 (Blackwell sm_120) requires JAX >= 0.4.38
- [14-01]: Register Plotly templates on module import via register_themes()
- [14-01]: Use same Wong (2011) palette as reporting/figures.py for visual consistency
- [14-01]: Template parameter pattern: all chart functions accept template='aoty_light' default
- [14-02]: Kaleido v1+ Chrome detection via ensure_kaleido_chrome() with fallback
- [14-02]: Scale factor 2.0 default for raster exports (~300dpi at 4" width)
- [14-02]: DashboardData dataclass with optional fields for flexible data input
- [14-02]: CSS variables enable runtime theme switching via class toggle

### Pending Todos

None.

### Roadmap Evolution

- Phase 12 added: Errors from Testing
- Phase 13 added: GPU Features Testing (WSL + RTX 5090 validation)
- Phase 14 added: Interactive Visualization (web dashboard + SVG/PNG export)

### Blockers/Concerns

- **RESOLVED:** Raw CSV null Album values - now handled by cleaning pipeline exclusion filtering
- **RESOLVED:** JAX/NumPyro version conflict - upgraded to jax 0.8.2, numpyro 0.19.0
- **RESOLVED:** pixi.lock now exists - verify_environment returns is_reproducible=True (09-06)

## Phase 1 Deliverables

- data/processed/cleaned_all.parquet (130,023 rows)
- data/processed/user_score_minratings_5.parquet (76,362 rows)
- data/processed/user_score_minratings_10.parquet (61,942 rows)
- data/processed/user_score_minratings_25.parquet (44,340 rows)
- data/processed/critic_score.parquet (59,482 rows)
- data/audit/*.jsonl (277,966 exclusion records)

## Phase 2 Deliverables (Complete)

- src/aoty_pred/data/split.py - within_artist_temporal_split, artist_disjoint_split, assert_no_artist_overlap, validate_temporal_split
- src/aoty_pred/data/manifests.py - SplitManifest, save_manifest, load_manifest
- src/aoty_pred/utils/hashing.py - hash_dataframe
- src/aoty_pred/pipelines/create_splits.py - end-to-end split pipeline
- data/splits/within_artist_temporal/ - train/val/test parquets (41,094/7,562/7,562 rows) + manifest
- data/splits/artist_disjoint/ - train/val/test parquets (43,182/9,363/9,397 rows) + manifest
- data/splits/pipeline_summary.json - execution summary with hashes

## Phase 3 Deliverables (Complete)

- src/aoty_pred/features/errors.py - NotFittedError, FittedVocabulary, FittedStatistics
- src/aoty_pred/features/base.py - Enhanced BaseFeatureBlock with is_fitted, _check_is_fitted()
- src/aoty_pred/features/pipeline.py - FeaturePipeline with fit/transform separation
- tests/unit/test_feature_base.py - 14 tests for fit/transform enforcement
- tests/unit/test_feature_pipeline.py - 14 tests for pipeline leakage prevention

## Phase 4 Deliverables (Complete)

- src/aoty_pred/features/temporal.py - TemporalBlock with 5 temporal features (album_sequence, career_years, release_gap_days, release_year, date_risk_ordinal)
- src/aoty_pred/features/album_type.py - AlbumTypeBlock with one-hot encoding using FittedVocabulary
- src/aoty_pred/features/artist.py - ArtistReputationBlock, ArtistHistoryBlock with LOO
- src/aoty_pred/features/genre.py - GenreBlock with multi-hot + PCA
- src/aoty_pred/features/collaboration.py - CollaborationBlock ordinal encoding
- src/aoty_pred/features/registry.py - All blocks registered
- tests/unit/test_feature_*.py - 126 total feature tests

## Phase 5 Deliverables (Complete)

- src/aoty_pred/models/bayes/priors.py - PriorConfig dataclass with 9 hyperparameters (extended with sigma_rw_scale, rho_loc, rho_scale)
- src/aoty_pred/models/bayes/model.py - make_score_model factory, user_score_model, critic_score_model with time-varying effects and AR(1)
- src/aoty_pred/models/__init__.py - Models package init
- src/aoty_pred/models/bayes/__init__.py - Bayes module exports (make_score_model, user_score_model, critic_score_model, album_score_model)

## Phase 6 Deliverables (Complete)

- src/aoty_pred/models/bayes/fit.py - fit_model, MCMCConfig, FitResult, get_gpu_info
- src/aoty_pred/models/bayes/io.py - save_model, load_model, ModelManifest, ModelsManifest
- src/aoty_pred/models/bayes/predict.py - generate_posterior_predictive, predict_out_of_sample, predict_new_artist, PredictionResult

## Phase 7 Deliverables (Complete)

- src/aoty_pred/models/bayes/diagnostics.py - ConvergenceDiagnostics, check_convergence, get_divergence_info
- src/aoty_pred/evaluation/cv.py - compute_log_likelihood, add_log_likelihood_to_idata, LOOResult, compute_loo, compare_models, generate_prior_predictive
- src/aoty_pred/evaluation/calibration.py - CoverageResult, compute_coverage, compute_multi_coverage, ReliabilityData, compute_reliability_data
- src/aoty_pred/evaluation/metrics.py - CRPSResult, compute_crps, PointMetrics, compute_point_metrics, posterior_mean
- src/aoty_pred/evaluation/__init__.py - Package exports for calibration, cv, metrics
- src/aoty_pred/pipelines/sensitivity.py - SensitivityResult, PRIOR_CONFIGS, run_prior_sensitivity, run_threshold_sensitivity, run_feature_ablation, aggregate_sensitivity_results, create_coefficient_comparison_df
- tests/unit/test_diagnostics.py - 20 tests for convergence diagnostics
- tests/unit/test_cv.py - 14 tests for LOO-CV and predictive checks
- tests/unit/test_calibration.py - 22 tests for calibration module
- tests/unit/test_metrics.py - 24 tests for metrics module
- tests/unit/test_sensitivity.py - 20 tests for sensitivity analysis

## Phase 8 Deliverables (Complete)

- src/aoty_pred/reporting/tables.py - create_coefficient_table, create_diagnostics_table, create_comparison_table, export_table with adaptive precision
- src/aoty_pred/reporting/__init__.py - Package exports for tables module
- src/aoty_pred/reporting/figures.py - set_publication_style, save_trace_plot, save_posterior_plot, save_predictions_plot, save_reliability_plot, save_forest_plot, COLORBLIND_COLORS
- src/aoty_pred/reporting/model_card.py - ModelCardData, generate_model_card, write_model_card, create_default_model_card_data, update_model_card_with_results
- MODEL_CARD.md - Simple overview model card at project root
- tests/unit/test_reporting_tables.py - 40 tests for table generation
- tests/unit/test_reporting_figures.py - 27 tests for figure generation
- tests/unit/test_reporting_model_card.py - 27 tests for model card generation

## Phase 9 Deliverables (Complete)

- src/aoty_pred/utils/git_state.py - GitState, capture_git_state
- src/aoty_pred/utils/random.py - set_seeds, get_rng_key
- src/aoty_pred/utils/logging.py - setup_pipeline_logging, is_interactive
- src/aoty_pred/utils/environment.py - EnvironmentStatus, verify_environment, ensure_environment_locked
- src/aoty_pred/pipelines/manifest.py - RunManifest, EnvironmentInfo, GitStateModel, save/load_run_manifest
- src/aoty_pred/pipelines/stages.py - PipelineStage, StageContext, PIPELINE_STAGES, get_execution_order, get_stage
- src/aoty_pred/pipelines/errors.py - PipelineError, ConvergenceError, DataValidationError, StageError, EnvironmentError, StageSkipped
- src/aoty_pred/pipelines/orchestrator.py - PipelineOrchestrator, PipelineConfig, run_pipeline
- src/aoty_pred/pipelines/build_features.py - build_features with FeaturePipeline
- src/aoty_pred/pipelines/train_bayes.py - train_models with MCMC fitting
- src/aoty_pred/pipelines/evaluate.py - evaluate_models with LOO-CV and metrics
- src/aoty_pred/pipelines/publication.py - generate_publication_artifacts
- src/aoty_pred/cli.py - CLI entry point with run command and stage subcommands
- src/aoty_pred/pipelines/__init__.py - Package exports for pipeline components
- pyproject.toml - Entry point: aoty-pipeline = aoty_pred.cli:main
- pixi.toml - Pixi workspace configuration for environment management
- pixi.lock - Locked environment with exact dependency versions
- tests/unit/test_utils_git_state.py - 20 tests for git state capture
- tests/unit/test_utils_random.py - 17 tests for random seed management
- tests/unit/test_utils_environment.py - 22 tests for environment verification
- tests/unit/test_pipeline_manifest.py - 21 tests for manifest schema
- tests/unit/test_pipeline_stages.py - 28 tests for stage definitions
- tests/unit/test_pipeline_orchestrator.py - 19 tests for orchestrator
- tests/unit/test_cli_pipeline.py - 25 tests for CLI commands

## Phase 10 Deliverables (Complete)

- src/aoty_pred/evaluation/metrics.py - Extended PointMetrics with n_observations, mean_bias
- src/aoty_pred/models/bayes/diagnostics.py - Extended ConvergenceDiagnostics with rhat_threshold, ess_threshold
- src/aoty_pred/pipelines/train_bayes.py - Fixed field names (rhat_max, ess_bulk_min)
- src/aoty_pred/pipelines/evaluate.py - Fixed field names and compute_coverage signature
- pyproject.toml - Added properscoring>=0.1 dependency
- pixi.toml - Added properscoring>=0.1 to pypi-dependencies
- tests/unit/test_metrics.py - Updated with new field assertions
- tests/unit/test_diagnostics.py - Updated ConvergenceDiagnostics constructions
- tests/unit/test_sensitivity.py - Updated mock_convergence fixture

## Phase 11 Deliverables (Complete)

- src/aoty_pred/pipelines/publication.py - Fixed interface calls to figures.py and model_card.py

## Phase 12 Deliverables (Complete)

- pixi.toml - Added jinja2>=3.1 (conda-forge) and gitpython>=3.1 (pypi) dependencies
- pixi.lock - Regenerated with explicit dependency declarations
- tests/unit/test_pipeline_orchestrator.py - Fixed strict mode for error handling test
- tests/integration/test_build_features_pipeline.py - DELETED (obsolete API)
- All 475 tests pass

## Phase 13 Deliverables (Complete)

- scripts/verify_gpu.py - GPU verification script with 5 checks (404 lines)
- docs/GPU_SETUP.md - Complete WSL2 CUDA setup guide (243 lines)
- scripts/benchmark_gpu.py - GPU MCMC benchmark with ESS/second metrics
- reports/gpu_benchmark_results.json - Captured benchmark (RTX 5090, 0.1 ESS/s, 0 divergences)
- GPU verified: NVIDIA GeForce RTX 5090 Laptop GPU, 24463 MiB

## Phase 14 Deliverables (In Progress)

- pixi.toml - Added plotly>=6.1.1, kaleido>=1.0.0, fastapi>=0.115, uvicorn>=0.32
- src/aoty_pred/visualization/__init__.py - Module exports and theme registration
- src/aoty_pred/visualization/theme.py - COLORBLIND_COLORS and register_themes()
- src/aoty_pred/visualization/charts.py - Five interactive chart creation functions
- src/aoty_pred/visualization/export.py - Static export pipeline (SVG, PNG via Kaleido)
- src/aoty_pred/visualization/dashboard.py - Dashboard assembly functions
- src/aoty_pred/visualization/templates/ - Jinja2 templates (base, dashboard, header, sidebar)

## Session Continuity

Last session: 2026-01-20
Stopped at: Completed 13-02-PLAN.md - GPU Benchmarking
Resume file: None
