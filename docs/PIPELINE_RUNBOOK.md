# Pipeline Runbook

Environment setup (example)
- Create venv
- Install dependencies

Commands
- Load and clean data
  ```bash
  aoty-pipeline stage data
  ```
- Create train/test splits
  ```bash
  aoty-pipeline stage splits
  ```
- Build feature matrix
  ```bash
  aoty-pipeline stage features
  ```
- Train Bayesian model
  ```bash
  aoty-pipeline stage train
  ```
- Evaluate model on test data
  ```bash
  aoty-pipeline stage evaluate
  ```
- Generate next-album predictions
  ```bash
  aoty-pipeline stage predict
  ```
- Build publication artifacts
  ```bash
  aoty-pipeline stage report
  ```
- Or run everything end-to-end
  ```bash
  aoty-pipeline run
  ```

Note
- Use multiple `-c` args to layer overrides (later files win).

Expected outputs
- data/processed/*
- data/features/*
- runs/<run_id>/*
- outputs/evaluation/*
- outputs/predictions/*
- reports/tables/*
- reports/figures/*
