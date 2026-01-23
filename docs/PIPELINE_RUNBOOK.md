# Pipeline Runbook

Environment setup (example)
- Create venv
- Install dependencies

Commands (planned)
- Prepare dataset
  ```bash
  python -m aoty_pred.cli prepare -c configs/base.yaml
  ```
- Build feature matrix
  ```bash
  python -m aoty_pred.cli build-features -c configs/base.yaml
  ```
- Train Bayesian models
  ```bash
  python -m aoty_pred.cli train -c configs/base.yaml -c configs/publication.yaml
  ```
- Predict next album scores
  ```bash
  python -m aoty_pred.cli predict -c configs/base.yaml -c configs/publication.yaml
  ```
- Sensitivity analysis
  ```bash
  python -m aoty_pred.cli sensitivity -c configs/base.yaml
  ```
- Build publication artifacts
  ```bash
  python -m aoty_pred.cli publication -c configs/base.yaml -c configs/publication.yaml
  ```

Note
- Use multiple `-c` args to layer overrides (later files win).

Expected outputs
- data/processed/*
- data/features/*
- runs/<run_id>/*
- reports/tables/*
- reports/figures/*
