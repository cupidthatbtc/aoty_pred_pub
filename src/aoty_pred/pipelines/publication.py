"""Publication artifact generation pipeline.

Generates publication-ready tables, figures, and model cards from
evaluation results.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import structlog

from aoty_pred.models.bayes.io import load_manifest, load_model
from aoty_pred.reporting.figures import (
    get_trace_plot_vars,
    save_posterior_plot,
    save_trace_plot,
)
from aoty_pred.reporting.model_card import (
    create_default_model_card_data,
    update_model_card_with_results,
    write_model_card,
)
from aoty_pred.reporting.tables import (
    create_coefficient_table,
    create_diagnostics_table,
    export_table,
)

if TYPE_CHECKING:
    import arviz as az

    from aoty_pred.pipelines.stages import StageContext

log = structlog.get_logger()


def _get_coefficient_var_names(
    idata: az.InferenceData,
    prefix: str = "user_",
) -> list[str]:
    """Build var_names for coefficient table based on InferenceData contents.

    Returns a dynamic list including sigma_ref and n_exponent when present
    in the posterior, ensuring publication tables adapt to the model
    configuration without hardcoded lists.

    Parameters
    ----------
    idata : az.InferenceData
        Inference data containing posterior samples.
    prefix : str, default "user_"
        Parameter name prefix ("user_" or "critic_").

    Returns
    -------
    list[str]
        Variable names for coefficient table.
    """
    var_names = [
        f"{prefix}beta",
        f"{prefix}mu_artist",
        f"{prefix}sigma_artist",
    ]
    # Include sigma_ref if present (sampled parameter with full diagnostics)
    if f"{prefix}sigma_ref" in idata.posterior:
        var_names.append(f"{prefix}sigma_ref")
    # Always include sigma_obs (sampled or deterministic)
    var_names.append(f"{prefix}sigma_obs")
    # Include n_exponent if learned
    if f"{prefix}n_exponent" in idata.posterior:
        var_names.append(f"{prefix}n_exponent")
    return var_names


def generate_publication_artifacts(ctx: StageContext) -> dict:
    """Generate publication-ready artifacts.

    Creates tables, figures, and model documentation from the fitted
    model and evaluation results.

    Args:
        ctx: Stage context with run configuration.

    Returns:
        Dictionary with paths to generated artifacts.
    """
    log.info("publication_pipeline_start")

    # Set up output directories
    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"
    tables_dir = reports_dir / "tables"

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_dir = Path("models")
    manifest = load_manifest(model_dir)

    if manifest is None or "user_score" not in manifest.current:
        raise ValueError("No trained user_score model found")

    model_filename = manifest.current["user_score"]
    model_path = model_dir / model_filename

    log.info("loading_model", path=str(model_path))
    idata = load_model(model_path)

    # Load evaluation results
    eval_dir = Path("outputs/evaluation")
    try:
        with open(eval_dir / "metrics.json", "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        log.warning("could_not_load_metrics", error=str(e))
        metrics = {}

    artifacts = {"tables": [], "figures": [], "docs": []}

    # =========================================================================
    # Generate Tables
    # =========================================================================
    # Note: Each artifact uses broad exception handling intentionally.
    # This is best-effort generation: log failures but continue to generate
    # remaining artifacts. Failures in one artifact should not block others.
    # =========================================================================

    log.info("generating_tables")

    # Coefficient table
    try:
        coef_df = create_coefficient_table(
            idata,
            var_names=_get_coefficient_var_names(idata),
        )
        coef_path = tables_dir / "coefficients"
        export_table(coef_df, str(coef_path), caption="Model coefficient estimates")
        artifacts["tables"].append(str(coef_path) + ".csv")
        artifacts["tables"].append(str(coef_path) + ".tex")
        log.info("coefficient_table_saved", path=str(coef_path))
    except Exception:  # Broad catch: best-effort artifact generation
        log.exception("coefficient_table_failed")

    # Diagnostics table
    try:
        diag_df = create_diagnostics_table(idata)
        diag_path = tables_dir / "diagnostics"
        export_table(diag_df, str(diag_path), caption="Convergence diagnostics")
        artifacts["tables"].append(str(diag_path) + ".csv")
        artifacts["tables"].append(str(diag_path) + ".tex")
        log.info("diagnostics_table_saved", path=str(diag_path))
    except Exception:  # Broad catch: best-effort artifact generation
        log.exception("diagnostics_table_failed")

    # Metrics summary table
    try:
        metrics_df = pd.DataFrame(
            [
                {"Metric": "RMSE", "Value": metrics["point_metrics"]["rmse"]},
                {"Metric": "MAE", "Value": metrics["point_metrics"]["mae"]},
                {"Metric": "R-squared", "Value": metrics["point_metrics"]["r2"]},
                {"Metric": "Coverage (90%)", "Value": metrics["calibration"]["coverage_90"]},
                {"Metric": "Coverage (50%)", "Value": metrics["calibration"]["coverage_50"]},
            ]
        )
        metrics_path = tables_dir / "metrics_summary"
        export_table(metrics_df, str(metrics_path), caption="Model performance metrics")
        artifacts["tables"].append(str(metrics_path) + ".csv")
        artifacts["tables"].append(str(metrics_path) + ".tex")
        log.info("metrics_table_saved", path=str(metrics_path))
    except Exception:  # Broad catch: best-effort artifact generation
        log.exception("metrics_table_failed")

    # =========================================================================
    # Generate Figures
    # =========================================================================

    log.info("generating_figures")

    # Trace plots
    try:
        pdf_path, png_path = save_trace_plot(
            idata,
            var_names=get_trace_plot_vars(idata),
            output_dir=figures_dir,
            filename_base="trace_plot",
        )
        artifacts["figures"].append(str(pdf_path))
        artifacts["figures"].append(str(png_path))
        log.info("trace_plot_saved", pdf=str(pdf_path), png=str(png_path))
    except Exception:  # Broad catch: best-effort artifact generation
        log.exception("trace_plot_failed")

    # Posterior plots
    try:
        pdf_path, png_path = save_posterior_plot(
            idata,
            var_names=get_trace_plot_vars(idata),
            output_dir=figures_dir,
            filename_base="posterior_plot",
        )
        artifacts["figures"].append(str(pdf_path))
        artifacts["figures"].append(str(png_path))
        log.info("posterior_plot_saved", pdf=str(pdf_path), png=str(png_path))
    except Exception:  # Broad catch: best-effort artifact generation
        log.exception("posterior_plot_failed")

    # =========================================================================
    # Generate Model Card
    # =========================================================================

    log.info("generating_model_card")

    try:
        # Create model card data
        model_card_data = create_default_model_card_data()

        # Update with results
        # Note: Full update requires typed objects (ConvergenceDiagnostics, CoverageResult, etc.)
        # For now, just update the idata reference; detailed metrics remain as defaults
        # A future enhancement could reconstruct typed objects from the JSON files
        model_card_data = update_model_card_with_results(
            model_card_data,
            idata=idata,
        )

        # Write model card
        model_card_path = reports_dir / "MODEL_CARD.md"
        write_model_card(model_card_data, model_card_path)
        artifacts["docs"].append(str(model_card_path))

        # Also copy to project root
        root_card_path = Path("MODEL_CARD.md")
        shutil.copy(model_card_path, root_card_path)
        artifacts["docs"].append(str(root_card_path))

        log.info("model_card_saved", path=str(model_card_path))
    except Exception:  # Broad catch: best-effort artifact generation
        log.exception("model_card_failed")

    # =========================================================================
    # Copy artifacts to run directory if available
    # =========================================================================

    if ctx.run_dir and ctx.run_dir.exists():
        run_reports_dir = ctx.run_dir / "reports"
        run_reports_dir.mkdir(parents=True, exist_ok=True)

        # Copy figures
        for fig_path in artifacts["figures"]:
            if Path(fig_path).exists():
                dest = run_reports_dir / "figures" / Path(fig_path).name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(fig_path, dest)

        # Copy tables
        for table_path in artifacts["tables"]:
            if Path(table_path).exists():
                dest = run_reports_dir / "tables" / Path(table_path).name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(table_path, dest)

        log.info("artifacts_copied_to_run_dir", run_dir=str(ctx.run_dir))

    log.info(
        "publication_pipeline_complete",
        n_tables=len(artifacts["tables"]),
        n_figures=len(artifacts["figures"]),
        n_docs=len(artifacts["docs"]),
    )

    return artifacts
