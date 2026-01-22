# Dev Setup (Miniforge)

This repo is designed to run in a clean environment.

Environment
1. Install Miniforge (Python 3.12).
2. Create and activate a fresh env:
   - conda create -n aoty_pred_pub python=3.12
   - conda activate aoty_pred_pub

Install dependencies
- pip install -e .[dev]

GPU (for model training)
- Use a GPU enabled JAX/NumPyro setup for MCMC training (see docs/GPU_SETUP.md).
- Tests and feature building are CPU only.

Run tests
- python -m pytest

Notes
- Integration tests need pyarrow (already in project dependencies).
