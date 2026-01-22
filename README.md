# AOTY Artist Prediction (Bayesian)

A publication-quality Bayesian prediction pipeline for artist-level outcomes (next album score). This repository emphasizes leakage controls, data lineage, and reproducibility.

## Features

- Leak-safe data pipeline and evaluation
- Clear data lineage from raw CSV to final outputs
- Bayesian models with robust diagnostics and sensitivity analysis
- Publication-ready artifacts (tables, figures, citations)

## Quickstart

1. Set `AOTY_DATASET_PATH` to the raw CSV path.
2. Run the preparation pipeline to produce `data/processed` artifacts.
3. Train Bayesian models and produce predictions.
4. Generate publication artifacts and checks.

## Config Overrides

Use multiple `-c` flags to layer configs (later files win).

## Key Entry Points

- `src/aoty_pred/pipelines/prepare_dataset.py`
- `src/aoty_pred/pipelines/train_bayes.py`
- `src/aoty_pred/pipelines/predict_next.py`
- `src/aoty_pred/pipelines/publication.py`

## Documentation

- `docs/PIPELINE_PLAN.md` — Detailed build plan
- `docs/LEAKAGE_CONTROLS.md` — Guardrails and leakage prevention
- `docs/DECISIONS_TO_LOCK.md` — Publication-critical defaults
- `docs/PROJECT_STRUCTURE.md` — Directory and file layout
- `docs/DATA_CONTRACT.md` — Raw schema and cleaned artifacts
- `docs/RAW_SCHEMA_SNAPSHOT.md` — Raw CSV header snapshot
- `docs/PIPELINE_RUNBOOK.md` — End-to-end pipeline instructions
- `docs/EVALUATION_PROTOCOL.md` — Metrics, diagnostics, and thresholds
- `docs/EXTENSIBILITY.md` — Adding features safely
- `docs/DEV_SETUP.md` — Environment and test setup

## License

MIT License. See [LICENSE](LICENSE) for details.
