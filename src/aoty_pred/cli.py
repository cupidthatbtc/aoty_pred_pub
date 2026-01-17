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


# Visualization commands
@app.command("visualize")
def visualize(
    port: int = typer.Option(8050, "--port", "-p", help="Server port"),
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't auto-open browser"),
    run_dir: Optional[str] = typer.Option(
        None, "--run", "-r", help="Path to pipeline run directory"
    ),
) -> None:
    """Launch interactive model visualization dashboard.

    Opens a local web server with interactive Plotly charts for
    exploring model results, diagnostics, and predictions.

    Examples:
        aoty-pipeline visualize
        aoty-pipeline visualize --port 8080
        aoty-pipeline visualize --no-browser
        aoty-pipeline visualize --run reports/2026-01-19_143052
    """
    from pathlib import Path

    from aoty_pred.visualization.server import run_server

    run_path = Path(run_dir) if run_dir else None
    run_server(port=port, host=host, open_browser=not no_browser, run_dir=run_path)


@app.command("generate-diagrams")
def generate_diagrams(
    output_dir: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for diagrams",
    ),
    theme: Optional[str] = typer.Option(
        "all",
        "--theme",
        "-t",
        help="Theme: light, dark, transparent, or all",
    ),
) -> None:
    """Generate data flow diagrams for documentation.

    Creates publication-quality SVG/PNG/PDF diagrams using Graphviz DOT format
    showing the data transformation pipeline from raw input to predictions.

    Examples:
        # Generate all 3 theme variants
        aoty-pipeline generate-diagrams

        # Generate only light theme
        aoty-pipeline generate-diagrams --theme light

        # Custom output directory
        aoty-pipeline generate-diagrams -o ./my_diagrams
    """
    from pathlib import Path

    from aoty_pred.visualization.diagrams import (
        create_aoty_pipeline_diagram,
        generate_all_diagrams,
    )

    # Default output directory
    if output_dir is None:
        output_path = Path("docs/figures")
    else:
        output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    # Handle "all" case
    if theme == "all":
        # Generate all 3 variants
        results = generate_all_diagrams(output_path)
        typer.echo(f"Generated {len(results)} diagram sets in {output_path}")
        for name, paths in results.items():
            typer.echo(f"  {name}: {[p.name for p in paths]}")
        return

    # Validate theme
    valid_themes = {"light", "dark", "transparent"}
    if theme not in valid_themes:
        typer.echo(
            f"Error: Unknown theme '{theme}'. Choose: light, dark, transparent, all",
            err=True,
        )
        raise typer.Exit(1)

    # Generate single theme
    name = f"aoty_pipeline_{theme}"
    diagram = create_aoty_pipeline_diagram(theme)  # type: ignore[arg-type]

    created_paths: list[Path] = []
    for fmt in ["svg", "png", "pdf"]:
        base_path = output_path / name
        diagram.format = fmt
        output = diagram.render(
            filename=str(base_path),
            directory=None,
            cleanup=True,
        )
        created_paths.append(Path(output))

    # Save .dot file
    dot_path = output_path / f"{name}.dot"
    dot_path.write_text(diagram.source, encoding="utf-8")
    created_paths.append(dot_path)

    typer.echo(f"Created: {name} ({len(created_paths)} files)")
    for p in created_paths:
        typer.echo(f"  {p.name}")


@app.command("export-figures")
def export_figures(
    output_dir: str = typer.Option(
        "reports/interactive", "--output", "-o", help="Output directory"
    ),
    formats: str = typer.Option(
        "svg,png", "--formats", "-f", help="Comma-separated formats (svg,png,pdf)"
    ),
    width: int = typer.Option(800, "--width", "-w", help="Figure width in pixels"),
    height: int = typer.Option(600, "--height", help="Figure height in pixels"),
    scale: float = typer.Option(
        2.0, "--scale", "-s", help="Scale factor for raster output (2.0 = ~300dpi)"
    ),
    run_dir: Optional[str] = typer.Option(
        None, "--run", "-r", help="Path to pipeline run directory"
    ),
) -> None:
    """Export all visualization figures to static formats.

    Generates publication-quality SVG and PNG files from the
    interactive dashboard figures.

    Examples:
        aoty-pipeline export-figures
        aoty-pipeline export-figures --output figs/ --formats svg,png,pdf
        aoty-pipeline export-figures --width 1200 --height 800 --scale 3.0
    """
    from pathlib import Path

    import plotly.graph_objects as go

    from aoty_pred.visualization.charts import (
        create_forest_plot,
        create_predictions_plot,
        create_reliability_plot,
    )
    from aoty_pred.visualization.export import ensure_kaleido_chrome, export_all_figures
    from aoty_pred.visualization.server import load_dashboard_data

    # Parse formats
    format_list = tuple(f.strip() for f in formats.split(",") if f.strip())

    # Ensure Kaleido Chrome is available for raster formats
    if any(fmt in ("png", "jpeg", "webp") for fmt in format_list):
        if not ensure_kaleido_chrome():
            typer.echo("Warning: Kaleido Chrome not available, PNG export may fail", err=True)

    # Load data
    run_path = Path(run_dir) if run_dir else None
    data = load_dashboard_data(run_path)

    # Create figures as go.Figure objects (not HTML strings)
    figures: dict[str, go.Figure] = {}

    if data.predictions is not None:
        pred = data.predictions
        required = ["y_true", "y_pred_mean", "y_pred_lower", "y_pred_upper"]
        if all(k in pred for k in required):
            figures["predictions"] = create_predictions_plot(
                pred["y_true"],
                pred["y_pred_mean"],
                pred["y_pred_lower"],
                pred["y_pred_upper"],
            )

    if data.coefficients is not None:
        figures["coefficients"] = create_forest_plot(data.coefficients)

    if data.reliability is not None:
        rel = data.reliability
        required = ["predicted_probs", "observed_freq", "counts"]
        if all(k in rel for k in required):
            figures["reliability"] = create_reliability_plot(
                rel["predicted_probs"],
                rel["observed_freq"],
                rel["counts"],
            )

    # Add trace/posterior plots if idata available
    if data.idata is not None:
        try:
            from aoty_pred.visualization.charts import create_trace_plot

            posterior = data.idata.posterior
            if hasattr(posterior, "data_vars"):
                var_names = list(posterior.data_vars)
                if var_names:
                    var_name = var_names[0]
                    samples = posterior[var_name].values
                    # Handle multi-dimensional samples
                    if samples.ndim > 2:
                        samples = samples.reshape(samples.shape[0], -1)[:, 0:100]
                    elif samples.ndim == 1:
                        samples = samples.reshape(1, -1)
                    figures["trace"] = create_trace_plot(samples, var_name)
        except Exception:
            pass  # Skip if idata format is unexpected

    if not figures:
        typer.echo("No data available for export. Run pipeline first.", err=True)
        raise typer.Exit(code=1)

    # Export
    output_path = Path(output_dir)
    results = export_all_figures(
        output_dir=output_path,
        figures=figures,
        formats=format_list,
        width=width,
        height=height,
        scale=scale,
    )

    typer.echo(f"Exported {len(results)} figures to {output_path}")
    for name, paths in results.items():
        typer.echo(f"  {name}: {', '.join(p.name for p in paths)}")


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
