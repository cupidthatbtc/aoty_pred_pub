"""Dashboard assembly functions for multi-chart visualization.

This module provides functions to assemble multiple interactive charts
into dashboard views. It integrates with the chart functions from
charts.py and provides data structures for dashboard state.

Usage:
    >>> from aoty_pred.visualization.dashboard import DashboardData, create_dashboard_figures
    >>> data = DashboardData(predictions={...}, coefficients=df)
    >>> figures = create_dashboard_figures(data)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from aoty_pred.visualization.charts import (
    create_forest_plot,
    create_predictions_plot,
    create_reliability_plot,
    create_trace_plot,
)

if TYPE_CHECKING:
    import arviz as az

__all__ = [
    "DashboardData",
    "create_artist_view",
    "create_coefficients_table",
    "create_dashboard_figures",
    "get_artist_list",
]


@dataclass
class DashboardData:
    """Data container for dashboard visualization.

    This dataclass holds all data needed to generate dashboard views.
    All fields are optional - views are only generated for available data.

    Attributes
    ----------
    idata : az.InferenceData | None
        ArviZ inference data object with posterior samples.
    predictions : dict | None
        Prediction results with keys:
        - y_true: array of actual values
        - y_pred_mean: array of predicted means
        - y_pred_lower: array of lower CI bounds
        - y_pred_upper: array of upper CI bounds
    coefficients : pd.DataFrame | None
        Coefficient summary table with columns:
        - param: parameter name
        - mean: posterior mean
        - hdi_3%: lower HDI bound
        - hdi_97%: upper HDI bound
    reliability : dict | None
        Calibration data with keys:
        - predicted_probs: array of bin centers
        - observed_freq: array of observed frequencies
        - counts: array of bin counts
    artist_data : pd.DataFrame | None
        Per-artist predictions for artist search view.
        Should have 'artist' column and prediction columns.
    """

    idata: Any | None = None  # az.InferenceData
    predictions: dict[str, np.ndarray] | None = None
    coefficients: pd.DataFrame | None = None
    reliability: dict[str, np.ndarray] | None = None
    artist_data: pd.DataFrame | None = field(default=None)


def create_dashboard_figures(
    data: DashboardData,
    theme: str = "aoty_light",
) -> dict[str, str]:
    """Generate all dashboard figures as HTML strings.

    Creates HTML representations of each available view based on
    the data provided. Only views with corresponding data are generated.

    Parameters
    ----------
    data : DashboardData
        Dashboard data container.
    theme : str, default "aoty_light"
        Plotly template name.

    Returns
    -------
    dict[str, str]
        Dictionary mapping view_id to Plotly HTML string.
        Possible keys: "trace", "predictions", "coefficients", "reliability"

    Notes
    -----
    Only the first figure includes Plotly.js to minimize total size.
    Subsequent figures use include_plotlyjs=False.

    Examples
    --------
    >>> data = DashboardData(predictions={...})
    >>> figures = create_dashboard_figures(data)
    >>> print(list(figures.keys()))
    ['predictions']
    """
    figures: dict[str, str] = {}
    first_figure = True

    # Trace plot (from idata posterior)
    if data.idata is not None:
        try:
            # Get first parameter from posterior
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

                    fig = create_trace_plot(samples, var_name, template=theme)
                    figures["trace"] = fig.to_html(
                        full_html=False,
                        include_plotlyjs=first_figure,
                    )
                    first_figure = False
        except Exception as e:
            logger.debug("Skipping trace plot due to unexpected idata format: %s", e)

    # Predictions scatter plot
    if data.predictions is not None:
        pred = data.predictions
        required = ["y_true", "y_pred_mean", "y_pred_lower", "y_pred_upper"]
        if all(k in pred for k in required):
            fig = create_predictions_plot(
                pred["y_true"],
                pred["y_pred_mean"],
                pred["y_pred_lower"],
                pred["y_pred_upper"],
                template=theme,
            )
            figures["predictions"] = fig.to_html(
                full_html=False,
                include_plotlyjs=first_figure,
            )
            first_figure = False

    # Coefficient forest plot
    if data.coefficients is not None:
        df = data.coefficients
        # Auto-detect column names
        estimate_col = _find_column(df, ["mean", "estimate", "coef"])
        lower_col = _find_column(df, ["hdi_3%", "hdi_2.5%", "lower", "ci_lower"])
        upper_col = _find_column(df, ["hdi_97%", "hdi_97.5%", "upper", "ci_upper"])
        label_col = _find_column(df, ["param", "parameter", "name", "index"])

        if all([estimate_col, lower_col, upper_col, label_col]):
            fig = create_forest_plot(
                df,
                estimate_col=estimate_col,
                lower_col=lower_col,
                upper_col=upper_col,
                label_col=label_col,
                template=theme,
            )
            figures["coefficients"] = fig.to_html(
                full_html=False,
                include_plotlyjs=first_figure,
            )
            first_figure = False

    # Reliability diagram
    if data.reliability is not None:
        rel = data.reliability
        required = ["predicted_probs", "observed_freq", "counts"]
        if all(k in rel for k in required):
            fig = create_reliability_plot(
                rel["predicted_probs"],
                rel["observed_freq"],
                rel["counts"],
                template=theme,
            )
            figures["reliability"] = fig.to_html(
                full_html=False,
                include_plotlyjs=first_figure,
            )
            first_figure = False

    return figures


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find first matching column name from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def create_artist_view(
    artist_name: str,
    artist_data: pd.DataFrame,
    theme: str = "aoty_light",
) -> str:
    """Generate artist-specific view showing prediction history.

    Creates a line chart showing album scores over time with
    prediction intervals for the specified artist.

    Parameters
    ----------
    artist_name : str
        Name of the artist to display.
    artist_data : pd.DataFrame
        DataFrame with artist predictions. Expected columns:
        - artist: artist name
        - date or year: time column
        - score or y_true: actual score
        - prediction or y_pred: predicted score (optional)
        - lower or y_pred_lower: lower bound (optional)
        - upper or y_pred_upper: upper bound (optional)
    theme : str, default "aoty_light"
        Plotly template name.

    Returns
    -------
    str
        Plotly HTML string for the artist view, or an informative
        message if the artist is not found.

    Examples
    --------
    >>> html = create_artist_view("Radiohead", artist_data)
    >>> print(html[:50])
    '<div id="...'
    """
    import plotly.graph_objects as go

    # Filter to artist
    artist_mask = artist_data["artist"].str.lower() == artist_name.lower()
    artist_df = artist_data[artist_mask].copy()

    if artist_df.empty:
        return f'<div class="not-found">Artist "{artist_name}" not found.</div>'

    # Auto-detect columns
    time_col = _find_column(artist_df, ["date", "year", "release_date", "time"])
    score_col = _find_column(artist_df, ["score", "y_true", "actual", "user_score"])
    pred_col = _find_column(artist_df, ["prediction", "y_pred", "y_pred_mean", "predicted"])
    lower_col = _find_column(artist_df, ["lower", "y_pred_lower", "ci_lower"])
    upper_col = _find_column(artist_df, ["upper", "y_pred_upper", "ci_upper"])

    # Sort by time if available
    if time_col:
        artist_df = artist_df.sort_values(time_col)
        x_data = artist_df[time_col]
    else:
        x_data = np.arange(len(artist_df))

    fig = go.Figure()

    # Add prediction interval if available
    if lower_col and upper_col and pred_col:
        fig.add_trace(
            go.Scatter(
                x=list(x_data) + list(x_data[::-1]),
                y=list(artist_df[upper_col]) + list(artist_df[lower_col][::-1]),
                fill="toself",
                fillcolor="rgba(0, 114, 178, 0.2)",
                line=dict(color="rgba(0, 114, 178, 0)"),
                name="94% CI",
                hoverinfo="skip",
            )
        )

    # Add actual scores
    if score_col:
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=artist_df[score_col],
                mode="lines+markers",
                name="Actual Score",
                line=dict(color="#0072B2", width=2),
                marker=dict(size=8),
                hovertemplate="Actual: %{y:.1f}<extra></extra>",
            )
        )

    # Add predictions
    if pred_col:
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=artist_df[pred_col],
                mode="lines+markers",
                name="Predicted",
                line=dict(color="#E69F00", dash="dash", width=2),
                marker=dict(size=6),
                hovertemplate="Predicted: %{y:.1f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Album Scores: {artist_name}",
        xaxis_title=time_col.title() if time_col else "Album Index",
        yaxis_title="Score",
        template=theme,
    )

    return fig.to_html(full_html=False, include_plotlyjs=True)


def get_artist_list(artist_data: pd.DataFrame) -> list[str]:
    """Return sorted list of unique artist names for autocomplete.

    Parameters
    ----------
    artist_data : pd.DataFrame
        DataFrame with 'artist' column.

    Returns
    -------
    list[str]
        Sorted list of unique artist names.

    Examples
    --------
    >>> df = pd.DataFrame({"artist": ["Radiohead", "Beatles", "Radiohead"]})
    >>> get_artist_list(df)
    ['Beatles', 'Radiohead']
    """
    if "artist" not in artist_data.columns:
        return []
    artists = artist_data["artist"].dropna().unique().tolist()
    return sorted(artists)


def create_coefficients_table(coefficients: pd.DataFrame) -> str:
    """Generate sortable HTML table of coefficients.

    Creates a simple HTML table with data attributes for JavaScript
    sorting. Numbers are formatted to 3 decimal places.

    Parameters
    ----------
    coefficients : pd.DataFrame
        Coefficient summary with columns for parameter name, mean,
        and HDI bounds.

    Returns
    -------
    str
        HTML table string.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "param": ["intercept", "slope"],
    ...     "mean": [0.5, 1.2],
    ...     "hdi_3%": [0.3, 0.9],
    ...     "hdi_97%": [0.7, 1.5]
    ... })
    >>> html = create_coefficients_table(df)
    """
    # Auto-detect columns
    label_col = _find_column(coefficients, ["param", "parameter", "name", "index"])
    estimate_col = _find_column(coefficients, ["mean", "estimate", "coef"])
    lower_col = _find_column(coefficients, ["hdi_3%", "hdi_2.5%", "lower", "ci_lower"])
    upper_col = _find_column(coefficients, ["hdi_97%", "hdi_97.5%", "upper", "ci_upper"])

    if not all([label_col, estimate_col, lower_col, upper_col]):
        return "<p>Unable to generate table: missing required columns.</p>"

    html_parts = [
        '<table class="coefficient-table" id="coef-table">',
        "<thead>",
        "<tr>",
        '    <th data-sort="string">Parameter</th>',
        '    <th data-sort="number">Mean</th>',
        '    <th data-sort="number">HDI Low</th>',
        '    <th data-sort="number">HDI High</th>',
        "</tr>",
        "</thead>",
        "<tbody>",
    ]

    for _, row in coefficients.iterrows():
        html_parts.append("<tr>")
        html_parts.append(f"    <td>{row[label_col]}</td>")
        html_parts.append(f'    <td data-value="{row[estimate_col]:.6f}">{row[estimate_col]:.3f}</td>')
        html_parts.append(f'    <td data-value="{row[lower_col]:.6f}">{row[lower_col]:.3f}</td>')
        html_parts.append(f'    <td data-value="{row[upper_col]:.6f}">{row[upper_col]:.3f}</td>')
        html_parts.append("</tr>")

    html_parts.extend([
        "</tbody>",
        "</table>",
    ])

    return "\n".join(html_parts)
