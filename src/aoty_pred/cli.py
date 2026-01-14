"""Command-line interface for AOTY prediction pipeline.

This module provides CLI entry points for running the full pipeline or
individual stages. The primary entry point is `aoty-pipeline run`, which
executes all stages in dependency order with progress tracking.

Usage:
    aoty-pipeline run --seed 42
    aoty-pipeline run --dry-run --verbose
    aoty-pipeline stage data --verbose
"""

from __future__ import annotations

from typing import Optional

import typer

# Legacy imports for backwards compatibility with config-based commands
from aoty_pred.pipelines import (
    build_features,
    prepare_dataset,
    publication,
    predict_next,
    sensitivity,
    train_bayes,
)

__version__ = "0.1.0"

app = typer.Typer(
    add_completion=False,
    help="AOTY Prediction Pipeline - reproducible ML workflow for album score prediction.",
    invoke_without_command=True,
)

# Stage subcommand group
stage_app = typer.Typer(help="Run individual pipeline stages")
app.add_typer(stage_app, name="stage")


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
    ),
) -> None:
    """AOTY Prediction Pipeline - reproducible ML workflow."""
    if version:
        typer.echo(f"aoty-pred version {__version__}")
        raise typer.Exit()
    # If no command provided, show help
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command("run")
def run(
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    skip_existing: bool = typer.Option(
        False,
        "--skip-existing",
        help="Skip stages with unchanged inputs",
    ),
    stages: Optional[str] = typer.Option(
        None,
        "--stages",
        "-s",
        help="Comma-separated list of stages to run (e.g., 'data,splits,train')",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show execution plan without running stages",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Fail on convergence warnings or missing pixi.lock",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable DEBUG logging",
    ),
    resume: Optional[str] = typer.Option(
        None,
        "--resume",
        help="Resume failed run by run-id (e.g., '2026-01-19_143052')",
    ),
) -> None:
    """Execute full pipeline from raw data to publication artifacts.

    Runs all pipeline stages in dependency order: data -> splits -> features ->
    train -> evaluate -> report. Creates a timestamped output directory with
    manifest for reproducibility.

    Examples:
        aoty-pipeline run
        aoty-pipeline run --seed 123 --verbose
        aoty-pipeline run --dry-run
        aoty-pipeline run --stages data,splits
        aoty-pipeline run --resume 2026-01-19_143052
    """
    from aoty_pred.pipelines.orchestrator import PipelineConfig, run_pipeline

    # Parse stages from comma-separated string
    stage_list: list[str] | None = None
    if stages:
        stage_list = [s.strip() for s in stages.split(",") if s.strip()]

    config = PipelineConfig(
        seed=seed,
        skip_existing=skip_existing,
        stages=stage_list,
        dry_run=dry_run,
        strict=strict,
        verbose=verbose,
        resume=resume,
    )

    exit_code = run_pipeline(config)
    raise typer.Exit(code=exit_code)


# Individual stage commands
@stage_app.command("data")
def stage_data(
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
) -> None:
    """Run data preparation stage only.

    Loads raw album data, applies cleaning transformations, and creates
    processed datasets at multiple rating thresholds.
    """
    from aoty_pred.pipelines.orchestrator import PipelineConfig, run_pipeline

    config = PipelineConfig(
        seed=seed,
        stages=["data"],
        verbose=verbose,
    )
    exit_code = run_pipeline(config)
    raise typer.Exit(code=exit_code)


@stage_app.command("splits")
def stage_splits(
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
) -> None:
    """Run split creation stage only.

    Creates train/validation/test splits using within-artist temporal
    and artist-disjoint strategies.
    """
    from aoty_pred.pipelines.orchestrator import PipelineConfig, run_pipeline

    config = PipelineConfig(
        seed=seed,
        stages=["splits"],
        verbose=verbose,
    )
    exit_code = run_pipeline(config)
    raise typer.Exit(code=exit_code)


@stage_app.command("features")
def stage_features(
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
) -> None:
    """Run feature building stage only.

    Builds feature matrices from split data using configured feature blocks.
    """
    from aoty_pred.pipelines.orchestrator import PipelineConfig, run_pipeline

    config = PipelineConfig(
        seed=seed,
        stages=["features"],
        verbose=verbose,
    )
    exit_code = run_pipeline(config)
    raise typer.Exit(code=exit_code)


@stage_app.command("train")
def stage_train(
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Fail on convergence warnings",
    ),
) -> None:
    """Run model training stage only.

    Fits Bayesian models on training data using NumPyro MCMC.
    """
    from aoty_pred.pipelines.orchestrator import PipelineConfig, run_pipeline

    config = PipelineConfig(
        seed=seed,
        stages=["train"],
        strict=strict,
        verbose=verbose,
    )
    exit_code = run_pipeline(config)
    raise typer.Exit(code=exit_code)


@stage_app.command("evaluate")
def stage_evaluate(
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
) -> None:
    """Run evaluation stage only.

    Computes model diagnostics, calibration metrics, and LOO-CV.
    """
    from aoty_pred.pipelines.orchestrator import PipelineConfig, run_pipeline

    config = PipelineConfig(
        seed=seed,
        stages=["evaluate"],
        verbose=verbose,
    )
    exit_code = run_pipeline(config)
    raise typer.Exit(code=exit_code)


@stage_app.command("report")
def stage_report(
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging"),
) -> None:
    """Run report generation stage only.

    Generates publication artifacts: figures, tables, and model cards.
    """
    from aoty_pred.pipelines.orchestrator import PipelineConfig, run_pipeline

    config = PipelineConfig(
        seed=seed,
        stages=["report"],
        verbose=verbose,
    )
    exit_code = run_pipeline(config)
    raise typer.Exit(code=exit_code)


# Legacy config-based commands (kept for backwards compatibility)
@app.command("prepare", hidden=True)
def prepare(
    config: list[str] = typer.Option(..., "--config", "-c"),
) -> None:
    """[Legacy] Prepare dataset using config files."""
    prepare_dataset.run(config)


@app.command("build-features", hidden=True)
def build_features_cmd(
    config: list[str] = typer.Option(..., "--config", "-c"),
) -> None:
    """[Legacy] Build features using config files."""
    build_features.run(config)


@app.command("train-legacy", hidden=True)
def train_legacy(
    config: list[str] = typer.Option(..., "--config", "-c"),
) -> None:
    """[Legacy] Train model using config files."""
    train_bayes.run(config)


@app.command("predict-legacy", hidden=True)
def predict_legacy(
    config: list[str] = typer.Option(..., "--config", "-c"),
) -> None:
    """[Legacy] Generate predictions using config files."""
    predict_next.run(config)


@app.command("sensitivity-legacy", hidden=True)
def sensitivity_cmd(
    config: list[str] = typer.Option(..., "--config", "-c"),
) -> None:
    """[Legacy] Run sensitivity analysis using config files."""
    sensitivity.run(config)


@app.command("publication-legacy", hidden=True)
def publication_cmd(
    config: list[str] = typer.Option(..., "--config", "-c"),
) -> None:
    """[Legacy] Generate publication artifacts using config files."""
    publication.run(config)


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
