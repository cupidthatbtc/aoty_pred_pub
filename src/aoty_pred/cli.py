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

import logging
from typing import Annotated, Optional

import typer

logger = logging.getLogger(__name__)

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
    preflight: bool = typer.Option(
        False,
        "--preflight",
        help="Run memory check before starting; abort if insufficient",
    ),
    preflight_only: bool = typer.Option(
        False,
        "--preflight-only",
        help="Run memory check and exit (0=pass, 1=fail, 2=warning/cannot-check)",
    ),
    force_run: bool = typer.Option(
        False,
        "--force-run",
        help="Override preflight failure and continue anyway (use with --preflight)",
    ),
    resume: Optional[str] = typer.Option(
        None,
        "--resume",
        help="Resume failed run by run-id (e.g., '2026-01-19_143052')",
    ),
    max_albums: Annotated[int, typer.Option(
        min=1,
        help="Maximum albums per artist for model training. Albums beyond this use the same artist effect.",
    )] = 50,
    # MCMC Configuration
    num_chains: Annotated[int, typer.Option(
        min=1,
        help="Number of parallel MCMC chains (default 4)",
    )] = 4,
    num_samples: Annotated[int, typer.Option(
        min=100,
        help="Post-warmup samples per chain (default 1000)",
    )] = 1000,
    num_warmup: Annotated[int, typer.Option(
        min=50,
        help="Warmup iterations per chain (default 1000)",
    )] = 1000,
    target_accept: Annotated[float, typer.Option(
        min=0.5,
        max=0.999,
        help="Target acceptance probability (default 0.8, increase to 0.9-0.95 if divergences)",
    )] = 0.8,
    # Convergence Thresholds
    rhat_threshold: Annotated[float, typer.Option(
        min=1.0,
        max=1.1,
        help="Maximum acceptable R-hat (default 1.01)",
    )] = 1.01,
    ess_threshold: Annotated[int, typer.Option(
        min=100,
        help="Minimum ESS per chain (default 400)",
    )] = 400,
    allow_divergences: bool = typer.Option(
        False,
        "--allow-divergences",
        help="Don't fail on divergences (for exploratory runs)",
    ),
    # Data Filtering
    min_ratings: Annotated[int, typer.Option(
        min=1,
        help="Minimum user ratings per album (default 10)",
    )] = 10,
    min_albums: Annotated[int, typer.Option(
        min=1,
        help="Minimum albums per artist for dynamic effects (default 2)",
    )] = 2,
    # Feature Ablation flags
    enable_genre: Annotated[bool, typer.Option(
        " /--no-genre",
        help="Disable genre features",
    )] = True,
    enable_artist: Annotated[bool, typer.Option(
        " /--no-artist",
        help="Disable artist reputation features",
    )] = True,
    enable_temporal: Annotated[bool, typer.Option(
        " /--no-temporal",
        help="Disable temporal features",
    )] = True,
    # Heteroscedastic noise configuration
    n_exponent: Annotated[float, typer.Option(
        min=0.0,
        max=1.0,
        help="Scaling exponent for review count noise adjustment (0.0=homoscedastic, 0.5=sqrt scaling)",
    )] = 0.0,
    learn_n_exponent: bool = typer.Option(
        False,
        "--learn-n-exponent",
        help="Learn exponent from data using Beta prior (ignores --n-exponent if set)",
    ),
    n_exponent_alpha: Annotated[float, typer.Option(
        min=0.01,
        help="Beta prior alpha parameter for learned exponent (advanced, default 2.0)",
    )] = 2.0,
    n_exponent_beta: Annotated[float, typer.Option(
        min=0.01,
        help="Beta prior beta parameter for learned exponent (advanced, default 4.0)",
    )] = 4.0,
) -> None:
    """Execute full pipeline from raw data to publication artifacts.

    Runs all pipeline stages in dependency order: data -> splits -> features ->
    train -> evaluate -> report. Creates a timestamped output directory with
    manifest for reproducibility.

    Examples:
        # Default run
        aoty-pipeline run

        # High-accuracy run with more chains and samples
        aoty-pipeline run --num-chains 8 --num-samples 2000 --target-accept 0.95

        # Fast exploratory run
        aoty-pipeline run --num-chains 1 --num-samples 500 --num-warmup 500

        # Feature ablation
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
        max_albums=max_albums,
        # MCMC config
        num_chains=num_chains,
        num_samples=num_samples,
        num_warmup=num_warmup,
        target_accept=target_accept,
        # Convergence thresholds
        rhat_threshold=rhat_threshold,
        ess_threshold=ess_threshold,
        allow_divergences=allow_divergences,
        # Data filtering
        min_ratings=min_ratings,
        min_albums_filter=min_albums,
        # Feature flags
        enable_genre=enable_genre,
        enable_artist=enable_artist,
        enable_temporal=enable_temporal,
        # Heteroscedastic noise
        n_exponent=n_exponent,
        learn_n_exponent=learn_n_exponent,
        n_exponent_alpha=n_exponent_alpha,
        n_exponent_beta=n_exponent_beta,
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
    rhat_threshold: Annotated[float, typer.Option(
        min=1.0,
        max=1.1,
        help="Maximum acceptable R-hat (default 1.01)",
    )] = 1.01,
    ess_threshold: Annotated[int, typer.Option(
        min=100,
        help="Minimum ESS per chain (default 400)",
    )] = 400,
    allow_divergences: bool = typer.Option(
        False,
        "--allow-divergences",
        help="Don't fail on divergences (for exploratory runs)",
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
        rhat_threshold=rhat_threshold,
        ess_threshold=ess_threshold,
        allow_divergences=allow_divergences,
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
    level: Optional[str] = typer.Option(
        "all",
        "--level",
        "-l",
        help="Detail level: high, intermediate, detailed, or all",
    ),
) -> None:
    """Generate data flow diagrams for documentation.

    Creates publication-quality SVG/PNG/PDF diagrams using Graphviz DOT format
    showing the data transformation pipeline from raw input to predictions.

    Three detail levels are available:
    - high: ~10 nodes, simplified overview for README
    - intermediate: ~20 nodes, balanced detail for presentations
    - detailed: 30+ nodes, full technical reference for papers

    Examples:
        # Generate all 9 diagram variants (3 levels x 3 themes)
        aoty-pipeline generate-diagrams

        # Generate only high-level diagrams (3 themes)
        aoty-pipeline generate-diagrams --level high

        # Generate only light theme intermediate diagram
        aoty-pipeline generate-diagrams --theme light --level intermediate

        # Custom output directory
        aoty-pipeline generate-diagrams -o ./my_diagrams
    """
    from pathlib import Path

    from aoty_pred.visualization.diagrams import (
        LEVEL_FUNCTIONS,
        DetailLevel,
        generate_all_diagrams,
    )

    # Default output directory
    if output_dir is None:
        output_path = Path("docs/figures")
    else:
        output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    # Validate level
    valid_levels = {"high", "intermediate", "detailed", "all"}
    if level not in valid_levels:
        typer.echo(
            f"Error: Unknown level '{level}'. Choose: high, intermediate, detailed, all",
            err=True,
        )
        raise typer.Exit(1)

    # Validate theme
    valid_themes = {"light", "dark", "transparent", "all"}
    if theme not in valid_themes:
        typer.echo(
            f"Error: Unknown theme '{theme}'. Choose: light, dark, transparent, all",
            err=True,
        )
        raise typer.Exit(1)

    # Determine which levels to generate
    if level == "all":
        levels_to_generate: list[DetailLevel] = ["high", "intermediate", "detailed"]
    else:
        levels_to_generate = [level]  # type: ignore[list-item]

    # Determine which themes to generate
    themes_to_generate: list[str] = []
    if theme == "all":
        themes_to_generate = ["light", "dark", "transparent"]
    else:
        themes_to_generate = [theme]

    # Handle "all themes" case via generate_all_diagrams
    if theme == "all":
        results = generate_all_diagrams(output_path, levels=levels_to_generate)
        typer.echo(f"Generated {len(results)} diagram sets in {output_path}")
        typer.echo(f"  Levels: {', '.join(levels_to_generate)}")
        typer.echo("  Themes: light, dark, transparent")
        for name, paths in results.items():
            typer.echo(f"  {name}: {[p.name for p in paths]}")
        return

    # Generate specific theme(s) for specified level(s)
    total_files = 0
    for lvl in levels_to_generate:
        diagram_func = LEVEL_FUNCTIONS[lvl]
        for thm in themes_to_generate:
            name = f"pipeline_{lvl}_{thm}"
            diagram = diagram_func(thm)  # type: ignore[arg-type]

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
            total_files += len(created_paths)

    typer.echo(f"\nTotal: {total_files} files in {output_path}")


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
        except Exception as e:  # Broad catch intentional: idata format varies widely
            logger.debug("trace_plot_skipped", reason="unexpected_idata_format", error=str(e))

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


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
