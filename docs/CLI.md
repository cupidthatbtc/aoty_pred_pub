# CLI Reference

Complete command-line interface documentation for `aoty-pipeline`.

## Installation

The CLI is installed as `aoty-pipeline` when the package is installed:

```bash
pip install -e .
# or with pixi
pixi install
```

## Quick Reference

```bash
aoty-pipeline --help              # Show all commands
aoty-pipeline run --help          # Full pipeline options
aoty-pipeline stage --help        # Individual stage commands
aoty-pipeline visualize --help    # Dashboard options
```

---

## Global Options

| Option | Short | Description |
|--------|-------|-------------|
| `--version` | `-V` | Show version and exit |
| `--help` | | Show help message and exit |

```bash
aoty-pipeline --version
aoty-pipeline --help
```

---

## Commands

### `run` — Execute Full Pipeline

Runs all pipeline stages in dependency order: data → splits → features → train → evaluate → report.

```bash
aoty-pipeline run [OPTIONS]
```

#### General Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--seed` | | `42` | Random seed for reproducibility |
| `--verbose` | `-v` | `false` | Enable DEBUG logging |
| `--dry-run` | | `false` | Show execution plan without running |
| `--strict` | | `false` | Fail on convergence warnings or missing pixi.lock |
| `--skip-existing` | | `false` | Skip stages with unchanged inputs |
| `--stages` | `-s` | all | Comma-separated stages (e.g., `data,splits,train`) |
| `--resume` | | | Resume failed run by run-id (e.g., `2026-01-19_143052`) |

#### Preflight Memory Check Options

| Option | Default | Description |
|--------|---------|-------------|
| `--preflight` | `false` | Quick memory check (~1s) with fixed estimates |
| `--preflight-full` | `false` | Mini-MCMC to measure actual peak memory (~30-60s) |
| `--preflight-only` | `false` | Run memory check and exit (0=pass, 1=fail, 2=warning) |
| `--force-run` | `false` | Override preflight failure and continue anyway |
| `--recalibrate` | `false` | Force fresh calibration even if cached calibration exists |

**Note:** `--preflight-full` takes precedence over `--preflight`. The `--preflight-only` flag controls whether to exit after the check:

| Flags | Behavior |
|-------|----------|
| `--preflight-only` | Quick check (~1s) and exit |
| `--preflight-only --preflight-full` | Full measured check (~30-60s) and exit |
| `--preflight` | Quick check, then run pipeline |
| `--preflight-full` | Full check, then run pipeline |

#### MCMC Configuration

| Option | Default | Range | Description |
|--------|---------|-------|-------------|
| `--num-chains` | `4` | ≥1 | Number of parallel MCMC chains |
| `--num-samples` | `1000` | ≥100 | Post-warmup samples per chain |
| `--num-warmup` | `1000` | ≥50 | Warmup iterations per chain |
| `--target-accept` | `0.8` | 0.5–0.999 | Target acceptance probability |
| `--max-albums` | `50` | ≥1 | Maximum albums per artist |

#### Convergence Thresholds

| Option | Default | Range | Description |
|--------|---------|-------|-------------|
| `--rhat-threshold` | `1.01` | 1.0–1.1 | Maximum acceptable R-hat |
| `--ess-threshold` | `400` | ≥100 | Minimum ESS per chain |
| `--allow-divergences` | `false` | | Don't fail on divergences |

#### Data Filtering

| Option | Default | Range | Description |
|--------|---------|-------|-------------|
| `--min-ratings` | `10` | ≥1 | Minimum user ratings per album |
| `--min-albums` | `2` | ≥1 | Minimum albums per artist for dynamic effects |

#### Feature Ablation

| Option | Default | Description |
|--------|---------|-------------|
| `--no-genre` | enabled | Disable genre features |
| `--no-artist` | enabled | Disable artist reputation features |
| `--no-temporal` | enabled | Disable temporal features |

#### Heteroscedastic Noise Configuration

| Option | Default | Range | Description |
|--------|---------|-------|-------------|
| `--n-exponent` | `0.0` | 0.0–1.0 | Scaling exponent (0=homoscedastic, 0.5=sqrt) |
| `--learn-n-exponent` | `false` | | Learn exponent from data using Beta prior |
| `--n-exponent-alpha` | `2.0` | ≥0.01 | Beta prior alpha parameter (advanced) |
| `--n-exponent-beta` | `4.0` | ≥0.01 | Beta prior beta parameter (advanced) |

#### Examples

```bash
# Default run
aoty-pipeline run

# High-accuracy run
aoty-pipeline run --num-chains 8 --num-samples 2000 --target-accept 0.95

# Fast exploratory run
aoty-pipeline run --num-chains 1 --num-samples 500 --num-warmup 500

# Feature ablation study
aoty-pipeline run --no-genre --no-temporal

# Relaxed convergence for testing
aoty-pipeline run --rhat-threshold 1.05 --allow-divergences

# Resume a failed run
aoty-pipeline run --resume 2026-01-19_143052

# Check memory before running
aoty-pipeline run --preflight

# Check memory only (CI/scripting)
aoty-pipeline run --preflight-only

# Force run despite preflight failure
aoty-pipeline run --preflight --force-run

# Full preflight with mini-MCMC measurement
aoty-pipeline run --preflight-full

# Run specific stages only
aoty-pipeline run --stages data,splits,features
```

---

### `stage` — Run Individual Stages

Run pipeline stages independently.

```bash
aoty-pipeline stage <STAGE> [OPTIONS]
```

#### Available Stages

| Stage | Description |
|-------|-------------|
| `data` | Load raw data, apply cleaning, create processed datasets |
| `splits` | Create train/validation/test splits |
| `features` | Build feature matrices from split data |
| `train` | Fit Bayesian models using NumPyro MCMC |
| `evaluate` | Compute diagnostics, calibration metrics, LOO-CV |
| `report` | Generate publication artifacts (figures, tables, model cards) |

#### Common Stage Options

All stages support:

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--seed` | | `42` | Random seed |
| `--verbose` | `-v` | `false` | Enable DEBUG logging |
| `--help` | | | Show stage-specific help |

#### Train Stage Additional Options

| Option | Default | Description |
|--------|---------|-------------|
| `--strict` | `false` | Fail on convergence warnings |
| `--rhat-threshold` | `1.01` | Maximum acceptable R-hat |
| `--ess-threshold` | `400` | Minimum ESS per chain |
| `--allow-divergences` | `false` | Don't fail on divergences |

#### Examples

```bash
# Run data preparation only
aoty-pipeline stage data --verbose

# Run training with relaxed thresholds
aoty-pipeline stage train --rhat-threshold 1.05 --allow-divergences

# Generate reports
aoty-pipeline stage report
```

---

### `visualize` — Interactive Dashboard

Launch interactive model visualization dashboard.

```bash
aoty-pipeline visualize [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--port` | `-p` | `8050` | Server port |
| `--host` | | `127.0.0.1` | Server host |
| `--no-browser` | | `false` | Don't auto-open browser |
| `--run` | `-r` | | Path to pipeline run directory |
| `--help` | | | Show help |

#### Examples

```bash
# Default dashboard
aoty-pipeline visualize

# Custom port
aoty-pipeline visualize --port 8080

# Don't open browser automatically
aoty-pipeline visualize --no-browser

# Load specific run
aoty-pipeline visualize --run reports/2026-01-19_143052
```

---

### `generate-diagrams` — Data Flow Diagrams

Generate publication-quality pipeline diagrams.

```bash
aoty-pipeline generate-diagrams [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `docs/figures` | Output directory |
| `--theme` | `-t` | `all` | Theme: `light`, `dark`, `transparent`, or `all` |
| `--level` | `-l` | `all` | Detail: `high`, `intermediate`, `detailed`, or `all` |
| `--help` | | | Show help |

#### Detail Levels

| Level | Nodes | Use Case |
|-------|-------|----------|
| `high` | ~10 | README, quick overview |
| `intermediate` | ~20 | Presentations |
| `detailed` | 30+ | Technical papers |

#### Examples

```bash
# Generate all 9 variants (3 levels × 3 themes)
aoty-pipeline generate-diagrams

# High-level only (3 themes)
aoty-pipeline generate-diagrams --level high

# Single specific diagram
aoty-pipeline generate-diagrams --theme light --level intermediate

# Custom output
aoty-pipeline generate-diagrams -o ./my_diagrams
```

---

### `export-figures` — Static Figure Export

Export visualization figures to static formats.

```bash
aoty-pipeline export-figures [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `reports/interactive` | Output directory |
| `--formats` | `-f` | `svg,png` | Comma-separated formats (svg,png,pdf) |
| `--width` | `-w` | `800` | Figure width in pixels |
| `--height` | | `600` | Figure height in pixels |
| `--scale` | `-s` | `2.0` | Scale factor (2.0 = ~300dpi) |
| `--run` | `-r` | | Path to pipeline run directory |
| `--help` | | | Show help |

#### Examples

```bash
# Default export
aoty-pipeline export-figures

# All formats, high resolution
aoty-pipeline export-figures --formats svg,png,pdf --scale 3.0

# Custom dimensions
aoty-pipeline export-figures --width 1200 --height 800

# Specific run
aoty-pipeline export-figures --run reports/2026-01-19_143052
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AOTY_DATASET_PATH` | Path to raw CSV dataset (required) |
| `WSL_DISTRO_NAME` | Auto-detected for WSL2 GPU memory adjustments |
| `WSL_INTEROP` | Auto-detected for WSL2 GPU memory adjustments |

### Setting AOTY_DATASET_PATH

```bash
# Unix/Linux/macOS
export AOTY_DATASET_PATH="/path/to/dataset.csv"

# Windows PowerShell
$env:AOTY_DATASET_PATH = "C:\path\to\dataset.csv"

# Windows CMD
set AOTY_DATASET_PATH=C:\path\to\dataset.csv
```

Or create a `.env` file (see `.env.example`):

```bash
AOTY_DATASET_PATH="path/to/your/dataset.csv"
```

---

## Exit Codes

| Code | Name | Description |
|------|------|-------------|
| `0` | Success | Pipeline completed successfully |
| `1` | General Error | Unspecified error / preflight FAIL |
| `2` | Convergence Error | MCMC convergence failure (R-hat, ESS, divergences) / preflight CANNOT_CHECK |
| `3` | Data Validation Error | Input data validation failure |
| `4` | Stage Error | General stage execution error |
| `5` | Environment Error | Environment verification failure (pixi.lock missing in strict mode) |
| `6` | GPU Memory Error | GPU memory check failure |

### Preflight-Specific Exit Codes

When using `--preflight-only`:

| Code | Status | Meaning |
|------|--------|---------|
| `0` | PASS | Sufficient GPU memory available |
| `1` | FAIL | Insufficient GPU memory |
| `2` | CANNOT_CHECK | Unable to check (no GPU, NVML unavailable, missing data) |

---

## Configuration Files

The CLI uses YAML configuration files that can be layered with multiple `-c` flags (later files override earlier ones). Environment variables in values are expanded.

### Default Configs

| File | Purpose |
|------|---------|
| `configs/base.yaml` | Base configuration |
| `configs/publication.yaml` | Publication-quality settings |
| `configs/datasets/aoty_full.yaml` | Full dataset configuration |

### Example base.yaml

```yaml
data:
  raw_csv: "${AOTY_DATASET_PATH}"
```

---

## Typical Workflows

### First-Time Setup

```bash
# 1. Set dataset path
export AOTY_DATASET_PATH="/path/to/aoty_data.csv"

# 2. Check GPU memory
aoty-pipeline run --preflight-only

# 3. Run full pipeline
aoty-pipeline run
```

### Development Iteration

```bash
# Quick test run
aoty-pipeline run --num-chains 1 --num-samples 500 --allow-divergences

# Run specific stages
aoty-pipeline stage train --verbose
aoty-pipeline stage evaluate
aoty-pipeline stage report
```

### Publication Run

```bash
# Full quality run
aoty-pipeline run --num-chains 8 --num-samples 2000 --target-accept 0.95 --strict

# Export figures
aoty-pipeline export-figures --formats svg,png,pdf --scale 3.0

# Generate diagrams
aoty-pipeline generate-diagrams
```

### Preflight Checks (CI/Scripting)

```bash
# Quick estimate check (~1s, formula-based)
aoty-pipeline run --preflight-only
echo "Exit code: $?"

# Accurate measured check (~30-60s, mini-MCMC)
aoty-pipeline run --preflight-full --preflight-only
echo "Exit code: $?"
# Note: --preflight-full takes precedence over --preflight when both are given
```

---

## See Also

- `docs/PIPELINE_RUNBOOK.md` — End-to-end pipeline instructions
- `docs/DEV_SETUP.md` — Environment and test setup
- `docs/CONFIG_SPEC.md` — Configuration file specification
