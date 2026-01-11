"""Publication-quality figure generation for Bayesian model results.

This module provides functions for generating trace plots, posterior distributions,
prediction plots, and calibration diagrams suitable for journal publication.
All figures use colorblind-safe palettes and export to both PDF (vector) and
PNG (300dpi raster) formats.

Key features:
- Consistent publication styling via context manager
- Colorblind-safe color palette (Wong, 2011)
- Dual-format export (PDF + PNG)
- Automatic figure sizing based on content
- Proper memory management (figures closed after saving)

Usage:
    >>> from aoty_pred.reporting.figures import (
    ...     set_publication_style,
    ...     save_trace_plot,
    ...     save_predictions_plot,
    ... )
    >>> with set_publication_style():
    ...     pdf, png = save_trace_plot(idata, ["mu"], Path("figs"), "trace")
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Generator

    from aoty_pred.evaluation.calibration import ReliabilityData

__all__ = [
    "COLORBLIND_COLORS",
    "save_forest_plot",
    "save_posterior_plot",
    "save_predictions_plot",
    "save_reliability_plot",
    "save_trace_plot",
    "set_publication_style",
]

# Colorblind-safe palette from Wong (2011), Nature Methods
# https://www.nature.com/articles/nmeth.1618
COLORBLIND_COLORS = [
    "#0072B2",  # Blue
    "#E69F00",  # Orange
    "#009E73",  # Green
    "#CC79A7",  # Pink
    "#F0E442",  # Yellow
    "#56B4E9",  # Light blue
    "#D55E00",  # Red-orange
]


@contextmanager
def set_publication_style() -> Generator[None, None, None]:
    """Context manager for publication-quality figure styling.

    Uses plt.rc_context() to avoid global state pollution. All rcParams
    are restored after the context exits.

    Style settings:
    - Serif font family (Times New Roman with DejaVu Serif fallback)
    - Font sizes: 9pt body, 10pt titles, 8pt legends/ticks
    - DPI: 100 for screen, 300 for savefig
    - PDF/PS fonttype 42 for vector font embedding
    - Removed top/right spines for cleaner appearance
    - Colorblind-safe color cycle

    Examples
    --------
    >>> with set_publication_style():
    ...     fig, ax = plt.subplots()
    ...     ax.plot([1, 2, 3], [1, 4, 9])
    ...     fig.savefig("plot.pdf")
    ...     plt.close(fig)
    """
    style_params = {
        # Font settings
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        # Figure settings
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "figure.figsize": (6.5, 4),  # Single column width
        # Export settings for vector compatibility
        "pdf.fonttype": 42,  # TrueType fonts for Illustrator
        "ps.fonttype": 42,
        # Line settings
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        # Clean appearance
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Colorblind-safe color cycle
        "axes.prop_cycle": plt.cycler("color", COLORBLIND_COLORS),
    }

    with plt.rc_context(style_params):
        yield


def _ensure_output_dir(output_dir: Path) -> Path:
    """Ensure output directory exists, create if missing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_dual_format(
    fig: plt.Figure,
    output_dir: Path,
    filename_base: str,
) -> tuple[Path, Path]:
    """Save figure in both PDF and PNG formats.

    Parameters
    ----------
    fig : plt.Figure
        Figure to save.
    output_dir : Path
        Output directory.
    filename_base : str
        Base filename without extension.

    Returns
    -------
    tuple[Path, Path]
        Paths to (pdf_file, png_file).
    """
    output_dir = _ensure_output_dir(output_dir)

    pdf_path = output_dir / f"{filename_base}.pdf"
    png_path = output_dir / f"{filename_base}.png"

    fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
    fig.savefig(png_path, bbox_inches="tight", format="png", dpi=300)

    return pdf_path, png_path


def save_trace_plot(
    idata: az.InferenceData,
    var_names: list[str],
    output_dir: Path,
    filename_base: str,
    figsize: tuple[float, float] | None = None,
) -> tuple[Path, Path]:
    """Generate and save MCMC trace plot in PDF and PNG formats.

    Creates side-by-side trace plots (chain traces and posterior density)
    using ArviZ's plot_trace function.

    Parameters
    ----------
    idata : az.InferenceData
        Inference data containing posterior samples.
    var_names : list[str]
        Variable names to include in the trace plot.
    output_dir : Path
        Directory to save figures.
    filename_base : str
        Base filename without extension (e.g., "trace_user_score").
    figsize : tuple[float, float], optional
        Figure size in inches (width, height). If None, auto-sizes
        based on number of variables: (10, 2 * n_vars).

    Returns
    -------
    tuple[Path, Path]
        Paths to (pdf_file, png_file).

    Examples
    --------
    >>> pdf, png = save_trace_plot(
    ...     idata, ["mu", "sigma"], Path("figs"), "trace_main"
    ... )
    >>> print(f"Created: {pdf}")
    """
    with set_publication_style():
        # Auto-size figure based on number of variables
        if figsize is None:
            figsize = (10, 2 * len(var_names))

        # Create trace plot with ArviZ
        axes = az.plot_trace(
            idata,
            var_names=var_names,
            compact=True,
            divergences="bottom",
            figsize=figsize,
        )

        # Get figure from axes array
        fig = axes.ravel()[0].figure
        fig.tight_layout()

        # Save dual formats
        pdf_path, png_path = _save_dual_format(fig, output_dir, filename_base)

        # Clean up to avoid memory leaks
        plt.close(fig)

    return pdf_path, png_path


def save_posterior_plot(
    idata: az.InferenceData,
    var_names: list[str],
    output_dir: Path,
    filename_base: str,
    hdi_prob: float = 0.94,
    point_estimate: str = "mean",
    figsize: tuple[float, float] | None = None,
) -> tuple[Path, Path]:
    """Generate and save posterior distribution plot in PDF and PNG formats.

    Creates posterior density plots with HDI and point estimate annotations
    using ArviZ's plot_posterior function.

    Parameters
    ----------
    idata : az.InferenceData
        Inference data containing posterior samples.
    var_names : list[str]
        Variable names to include in the plot.
    output_dir : Path
        Directory to save figures.
    filename_base : str
        Base filename without extension.
    hdi_prob : float, default 0.94
        Probability for highest density interval.
    point_estimate : str, default "mean"
        Point estimate to display ("mean", "median", or "mode").
    figsize : tuple[float, float], optional
        Figure size in inches. If None, uses ArviZ default.

    Returns
    -------
    tuple[Path, Path]
        Paths to (pdf_file, png_file).

    Examples
    --------
    >>> pdf, png = save_posterior_plot(
    ...     idata, ["mu", "sigma"], Path("figs"), "posterior_main",
    ...     hdi_prob=0.95
    ... )
    """
    with set_publication_style():
        # Create posterior plot with ArviZ
        plot_kwargs = {
            "hdi_prob": hdi_prob,
            "point_estimate": point_estimate,
        }
        if figsize is not None:
            plot_kwargs["figsize"] = figsize

        axes = az.plot_posterior(
            idata,
            var_names=var_names,
            **plot_kwargs,
        )

        # Handle single variable case (returns single Axes, not array)
        if isinstance(axes, np.ndarray):
            fig = axes.ravel()[0].figure
        else:
            fig = axes.figure

        fig.tight_layout()

        # Save dual formats
        pdf_path, png_path = _save_dual_format(fig, output_dir, filename_base)

        # Clean up
        plt.close(fig)

    return pdf_path, png_path


def save_predictions_plot(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_lower: np.ndarray,
    y_pred_upper: np.ndarray,
    output_dir: Path,
    filename_base: str,
    ci_label: str = "94% CI",
    figsize: tuple[float, float] = (6, 6),
) -> tuple[Path, Path]:
    """Generate and save predicted vs actual scatter plot with uncertainty bands.

    Creates a scatter plot comparing predictions to actual values, with
    uncertainty intervals shown as error bars or bands.

    Parameters
    ----------
    y_true : np.ndarray
        True observed values, shape (n_obs,).
    y_pred_mean : np.ndarray
        Predicted values (posterior mean), shape (n_obs,).
    y_pred_lower : np.ndarray
        Lower bound of credible interval, shape (n_obs,).
    y_pred_upper : np.ndarray
        Upper bound of credible interval, shape (n_obs,).
    output_dir : Path
        Directory to save figures.
    filename_base : str
        Base filename without extension.
    ci_label : str, default "94% CI"
        Label for the credible interval in legend.
    figsize : tuple[float, float], default (6, 6)
        Figure size in inches (square for equal aspect).

    Returns
    -------
    tuple[Path, Path]
        Paths to (pdf_file, png_file).

    Examples
    --------
    >>> pdf, png = save_predictions_plot(
    ...     y_true=actual_scores,
    ...     y_pred_mean=predicted_mean,
    ...     y_pred_lower=ci_lower,
    ...     y_pred_upper=ci_upper,
    ...     output_dir=Path("figs"),
    ...     filename_base="predictions_test",
    ... )
    """
    y_true = np.asarray(y_true)
    y_pred_mean = np.asarray(y_pred_mean)
    y_pred_lower = np.asarray(y_pred_lower)
    y_pred_upper = np.asarray(y_pred_upper)

    with set_publication_style():
        fig, ax = plt.subplots(figsize=figsize)

        # Plot error bars for uncertainty
        ax.errorbar(
            y_pred_mean,
            y_true,
            xerr=[y_pred_mean - y_pred_lower, y_pred_upper - y_pred_mean],
            fmt="o",
            alpha=0.5,
            markersize=4,
            color=COLORBLIND_COLORS[0],
            ecolor=COLORBLIND_COLORS[5],  # Light blue for error bars
            elinewidth=0.5,
            capsize=0,
            label=ci_label,
        )

        # Add diagonal reference line (perfect prediction)
        all_values = np.concatenate([y_true, y_pred_mean, y_pred_lower, y_pred_upper])
        lims = [np.min(all_values) - 2, np.max(all_values) + 2]
        ax.plot(
            lims,
            lims,
            "k--",
            alpha=0.5,
            linewidth=1,
            label="Perfect prediction",
        )
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # Labels
        ax.set_xlabel("Predicted Score")
        ax.set_ylabel("Actual Score")
        ax.legend(loc="lower right")

        # Equal aspect ratio for square plot
        ax.set_aspect("equal", adjustable="box")

        fig.tight_layout()

        # Save dual formats
        pdf_path, png_path = _save_dual_format(fig, output_dir, filename_base)

        # Clean up
        plt.close(fig)

    return pdf_path, png_path


def save_reliability_plot(
    reliability_data: ReliabilityData,
    output_dir: Path,
    filename_base: str,
    figsize: tuple[float, float] = (6, 5),
) -> tuple[Path, Path]:
    """Generate and save reliability diagram (calibration plot).

    Creates a reliability diagram showing predicted probability vs observed
    frequency. A well-calibrated model shows points along the diagonal.

    Parameters
    ----------
    reliability_data : ReliabilityData
        Data from compute_reliability_data().
    output_dir : Path
        Directory to save figures.
    filename_base : str
        Base filename without extension.
    figsize : tuple[float, float], default (6, 5)
        Figure size in inches.

    Returns
    -------
    tuple[Path, Path]
        Paths to (pdf_file, png_file).

    Examples
    --------
    >>> from aoty_pred.evaluation.calibration import compute_reliability_data
    >>> rel_data = compute_reliability_data(y_true, y_samples)
    >>> pdf, png = save_reliability_plot(
    ...     rel_data, Path("figs"), "reliability_user"
    ... )
    """
    with set_publication_style():
        fig, ax = plt.subplots(figsize=figsize)

        # Plot reliability points with bin counts as marker size
        # Normalize counts for marker sizing
        counts = reliability_data.counts
        size_scale = 100 * counts / max(counts.max(), 1)  # Avoid division by zero

        ax.scatter(
            reliability_data.predicted_probs,
            reliability_data.observed_freq,
            s=size_scale + 20,  # Minimum size of 20
            alpha=0.7,
            color=COLORBLIND_COLORS[0],
            edgecolors=COLORBLIND_COLORS[0],
            linewidth=1,
            label="Observed",
        )

        # Connect points with line for visual clarity
        ax.plot(
            reliability_data.predicted_probs,
            reliability_data.observed_freq,
            "-",
            alpha=0.5,
            color=COLORBLIND_COLORS[0],
            linewidth=1,
        )

        # Add diagonal reference line (perfect calibration)
        ax.plot(
            [0, 1],
            [0, 1],
            "k--",
            alpha=0.5,
            linewidth=1,
            label="Perfect calibration",
        )

        # Add count annotations
        for i, (x, y, n) in enumerate(
            zip(
                reliability_data.predicted_probs,
                reliability_data.observed_freq,
                reliability_data.counts,
                strict=True,
            )
        ):
            if n > 0:
                ax.annotate(
                    f"n={n}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=7,
                    alpha=0.7,
                )

        # Labels
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Frequency")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower right")

        fig.tight_layout()

        # Save dual formats
        pdf_path, png_path = _save_dual_format(fig, output_dir, filename_base)

        # Clean up
        plt.close(fig)

    return pdf_path, png_path


def save_forest_plot(
    comparison_df: pd.DataFrame,
    output_dir: Path,
    filename_base: str,
    param_col: str = "param",
    variant_col: str = "variant",
    estimate_col: str = "mean",
    lower_col: str = "hdi_3%",
    upper_col: str = "hdi_97%",
    figsize: tuple[float, float] | None = None,
) -> tuple[Path, Path]:
    """Generate and save forest plot for coefficient comparison.

    Creates a horizontal error bar plot showing coefficients across
    different model variants or sensitivity analyses.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with columns for parameter name, variant name,
        estimate, and HDI bounds.
    output_dir : Path
        Directory to save figures.
    filename_base : str
        Base filename without extension.
    param_col : str, default "param"
        Column name for parameter identifier.
    variant_col : str, default "variant"
        Column name for model variant identifier.
    estimate_col : str, default "mean"
        Column name for point estimate.
    lower_col : str, default "hdi_3%"
        Column name for lower HDI bound.
    upper_col : str, default "hdi_97%"
        Column name for upper HDI bound.
    figsize : tuple[float, float], optional
        Figure size in inches. If None, auto-sizes based on
        number of parameters: (8, 0.5 * n_rows + 2).

    Returns
    -------
    tuple[Path, Path]
        Paths to (pdf_file, png_file).

    Examples
    --------
    >>> from aoty_pred.pipelines.sensitivity import create_coefficient_comparison_df
    >>> comparison_df = create_coefficient_comparison_df(results)
    >>> pdf, png = save_forest_plot(
    ...     comparison_df, Path("figs"), "coefficient_comparison"
    ... )
    """
    with set_publication_style():
        # Get unique parameters and variants
        params = comparison_df[param_col].unique()
        variants = comparison_df[variant_col].unique()

        n_params = len(params)
        n_variants = len(variants)

        # Auto-size figure
        if figsize is None:
            height = max(0.4 * n_params * n_variants + 2, 4)
            figsize = (8, height)

        fig, ax = plt.subplots(figsize=figsize)

        # Create y-positions for each parameter-variant combination
        y_positions = []
        y_labels = []
        y_pos = 0

        for param in params:
            param_data = comparison_df[comparison_df[param_col] == param]

            for i, (_, row) in enumerate(param_data.iterrows()):
                y_positions.append(y_pos)
                y_labels.append(f"{row[variant_col]}")

                # Plot error bar
                color = COLORBLIND_COLORS[i % len(COLORBLIND_COLORS)]
                ax.errorbar(
                    row[estimate_col],
                    y_pos,
                    xerr=[[row[estimate_col] - row[lower_col]],
                          [row[upper_col] - row[estimate_col]]],
                    fmt="o",
                    color=color,
                    markersize=6,
                    capsize=3,
                    capthick=1,
                    elinewidth=1.5,
                )
                y_pos += 1

            # Add parameter label
            ax.axhline(y=y_pos - 0.5, color="gray", linewidth=0.5, linestyle="-")
            ax.text(
                ax.get_xlim()[0] if ax.get_xlim()[0] != ax.get_xlim()[1] else -0.5,
                y_pos - (n_variants / 2) - 0.5,
                param,
                fontweight="bold",
                fontsize=9,
                ha="right",
                va="center",
            )
            y_pos += 0.5  # Gap between parameter groups

        # Add vertical reference line at zero
        ax.axvline(x=0, color="gray", linewidth=1, linestyle="--", alpha=0.7)

        # Labels
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Coefficient Value")
        ax.invert_yaxis()  # Top to bottom ordering

        # Update x-axis limits after all data plotted
        ax.autoscale(axis="x")

        fig.tight_layout()

        # Save dual formats
        pdf_path, png_path = _save_dual_format(fig, output_dir, filename_base)

        # Clean up
        plt.close(fig)

    return pdf_path, png_path
