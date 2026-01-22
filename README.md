# AOTY Artist Prediction (Bayesian)

This repository is a clean-room rebuild focused on publication-quality Bayesian prediction of artist-level outcomes (next album score). It is intentionally separated from the legacy codebase to make leakage controls, data lineage, and reproducibility explicit.

Goals
- Leak-safe data pipeline and evaluation
- Clear data lineage from raw CSV to final outputs
- Bayesian models with robust diagnostics and sensitivity analysis
- Publication-ready artifacts (tables, figures, citations)

Quickstart (planned)
1. Set `AOTY_DATASET_PATH` to the raw CSV path.
2. Run the preparation pipeline to produce `data/processed` artifacts.
3. Train Bayesian models and produce predictions.
4. Generate publication artifacts and checks.

Config overrides
- Use multiple `-c` flags to layer configs (later files win).

Key entry points (to be implemented)
- `src/aoty_pred/pipelines/prepare_dataset.py`
- `src/aoty_pred/pipelines/train_bayes.py`
- `src/aoty_pred/pipelines/predict_next.py`
- `src/aoty_pred/pipelines/publication.py`

See `docs/PIPELINE_PLAN.md` for the detailed build plan and `docs/LEAKAGE_CONTROLS.md` for guardrails.

Start here
- `docs/CLAUDE_HANDOFF.md` (implementation order and file-by-file tasks)
- `docs/DECISIONS_TO_LOCK.md` (publication-critical defaults)
- `docs/PROJECT_STRUCTURE.md` (what each folder is for)
- `docs/DATA_CONTRACT.md` (raw schema and cleaned artifacts)
- `docs/RAW_SCHEMA_SNAPSHOT.md` (raw CSV header snapshot)
- `docs/PIPELINE_RUNBOOK.md` (how to run the end-to-end pipeline)
- `docs/EVALUATION_PROTOCOL.md` (metrics, diagnostics, and thresholds)
- `docs/EXTENSIBILITY.md` (how to add features safely)
- `docs/DEV_SETUP.md` (environment and test setup)

