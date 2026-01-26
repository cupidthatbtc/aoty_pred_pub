# AOTY Artist Prediction (Bayesian)

![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![pixi](https://img.shields.io/badge/pixi-package%20manager-brightgreen)](https://pixi.sh)

A publication-quality Bayesian prediction pipeline for artist-level outcomes (next album score). This repository emphasizes leakage controls, data lineage, and reproducibility.

## Features

- Leak-safe data pipeline and evaluation
- Clear data lineage from raw CSV to final outputs
- Bayesian models with robust diagnostics and sensitivity analysis
- Publication-ready artifacts (tables, figures, citations)

## Installation

**Prerequisites:** Python >= 3.11

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Clone and install
git clone https://github.com/cupidthatbtc/aoty_pred_pub.git
cd aoty_pred_pub
pixi install
```

## Usage

Set the dataset path and run the pipeline:

```bash
# Set dataset path
export AOTY_DATASET_PATH="/path/to/aoty_data.csv"

# Check GPU memory before running
aoty-pipeline run --preflight-only

# Run full pipeline
aoty-pipeline run

# Quick exploratory run
aoty-pipeline run --num-chains 1 --num-samples 500

# Run specific stages
aoty-pipeline stage train --verbose
aoty-pipeline stage report
```

See `docs/CLI.md` for complete command reference.

## Key Entry Points

- `src/aoty_pred/pipelines/prepare_dataset.py`
- `src/aoty_pred/pipelines/train_bayes.py`
- `src/aoty_pred/pipelines/predict_next.py`
- `src/aoty_pred/pipelines/publication.py`

## Documentation

- `docs/CLI.md` — Complete CLI reference
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
