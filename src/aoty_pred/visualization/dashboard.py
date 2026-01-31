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

import html
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from aoty_pred.visualization.charts import (
    create_forest_plot,
    create_next_album_chart,
    create_predictions_plot,
    create_reliability_plot,
    create_trace_plot,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "DashboardData",
    "create_artist_view",
    "create_coefficients_table",
    "create_dashboard_figures",
    "create_next_album_tables",
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
    eval_metrics : dict | None
        Evaluation metrics from metrics.json (point_metrics, calibration, crps).
    known_predictions : pd.DataFrame | None
        Next-album predictions for known artists (next_album_known_artists.csv).
    new_predictions : pd.DataFrame | None
        Next-album predictions for new/unknown artists (next_album_new_artist.csv).
    """

    idata: Any | None = None  # az.InferenceData
    predictions: dict[str, np.ndarray] | None = None
    coefficients: pd.DataFrame | None = None
    reliability: dict[str, np.ndarray] | None = None
    artist_data: pd.DataFrame | None = field(default=None)
    eval_metrics: dict | None = None
    known_predictions: pd.DataFrame | None = None
    new_predictions: pd.DataFrame | None = None


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

    # Next-album prediction chart
    if data.known_predictions is not None and "pred_q50" in data.known_predictions.columns:
        fig = create_next_album_chart(data.known_predictions, template=theme)
        figures["next_albums"] = fig.to_html(
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
        return f'<div class="not-found">Artist "{html.escape(artist_name)}" not found.</div>'

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
        html_parts.append(f"    <td>{html.escape(str(row[label_col]))}</td>")
        html_parts.append(
            f'    <td data-value="{row[estimate_col]:.6f}">{row[estimate_col]:.3f}</td>'
        )
        html_parts.append(f'    <td data-value="{row[lower_col]:.6f}">{row[lower_col]:.3f}</td>')
        html_parts.append(f'    <td data-value="{row[upper_col]:.6f}">{row[upper_col]:.3f}</td>')
        html_parts.append("</tr>")

    html_parts.extend(
        [
            "</tbody>",
            "</table>",
        ]
    )

    return "\n".join(html_parts)


def create_next_album_tables(
    known_predictions: pd.DataFrame,
    new_predictions: pd.DataFrame,
) -> dict[str, str]:
    """Generate HTML tables for next-album predictions.

    Parameters
    ----------
    known_predictions : pd.DataFrame
        Known artist predictions with columns: artist, scenario, pred_mean,
        pred_std, pred_q05, pred_q25, pred_q50, pred_q75, pred_q95,
        last_score, n_training_albums.
    new_predictions : pd.DataFrame
        New artist predictions with columns: scenario, pred_mean, pred_std,
        pred_q05, pred_q25, pred_q50, pred_q75, pred_q95.

    Returns
    -------
    dict[str, str]
        Dictionary with keys 'known_table' and 'new_table' containing HTML strings.
    """
    # --- Known artist table ---
    known_parts = [
        '<table class="coefficient-table" id="known-predictions-table">',
        "<thead>",
        "<tr>",
        '    <th data-sort="string">Artist</th>',
        '    <th data-sort="string">Scenario</th>',
        '    <th data-sort="number">Predicted</th>',
        '    <th data-sort="string">90% CI</th>',
        '    <th data-sort="number">Last Score</th>',
        '    <th data-sort="number">Change</th>',
        '    <th data-sort="number">N Albums</th>',
        "</tr>",
        "</thead>",
        "<tbody>",
    ]

    # Map scenario codes to display names
    scenario_labels = {
        "same": "Same Features",
        "population_mean": "Population Mean",
        "artist_mean": "Artist Mean",
    }

    for _, row in known_predictions.iterrows():
        scenario = str(row.get("scenario", ""))
        scenario_display = scenario_labels.get(scenario, scenario)
        artist = html.escape(str(row.get("artist", "")))
        pred_q50 = row.get("pred_q50", float("nan"))
        pred_q05 = row.get("pred_q05", float("nan"))
        pred_q95 = row.get("pred_q95", float("nan"))
        last_score = row.get("last_score", float("nan"))
        n_albums = row.get("n_training_albums", "")

        change = (
            pred_q50 - last_score if pd.notna(pred_q50) and pd.notna(last_score) else float("nan")
        )
        change_str = f"{change:+.1f}" if pd.notna(change) else ""
        change_val = f"{change:.4f}" if pd.notna(change) else "0"

        known_parts.append(f'<tr data-scenario="{html.escape(scenario)}">')
        known_parts.append(f"    <td>{artist}</td>")
        known_parts.append(f"    <td>{html.escape(scenario_display)}</td>")
        known_parts.append(
            f'    <td data-value="{pred_q50:.4f}">{pred_q50:.1f}</td>'
            if pd.notna(pred_q50)
            else '    <td data-value="0">--</td>'
        )
        ci_str = (
            f"{pred_q05:.1f} -- {pred_q95:.1f}"
            if pd.notna(pred_q05) and pd.notna(pred_q95)
            else "--"
        )
        known_parts.append(f"    <td>{ci_str}</td>")
        known_parts.append(
            f'    <td data-value="{last_score:.4f}">{last_score:.1f}</td>'
            if pd.notna(last_score)
            else '    <td data-value="0">--</td>'
        )
        known_parts.append(f'    <td data-value="{change_val}">{change_str}</td>')
        known_parts.append(
            f'    <td data-value="{n_albums}">{n_albums}</td>'
            if pd.notna(n_albums)
            else '    <td data-value="0">--</td>'
        )
        known_parts.append("</tr>")

    known_parts.extend(["</tbody>", "</table>"])

    # --- New artist table ---
    new_parts = [
        '<table class="coefficient-table" id="new-predictions-table">',
        "<thead>",
        "<tr>",
        '    <th data-sort="string">Scenario</th>',
        '    <th data-sort="number">Predicted</th>',
        '    <th data-sort="number">Std Dev</th>',
        '    <th data-sort="string">90% CI</th>',
        "</tr>",
        "</thead>",
        "<tbody>",
    ]

    new_scenario_labels = {
        "population": "Population",
        "debut_defaults": "Debut Defaults",
    }

    for _, row in new_predictions.iterrows():
        scenario = str(row.get("scenario", ""))
        scenario_display = new_scenario_labels.get(scenario, scenario)
        pred_q50 = row.get("pred_q50", float("nan"))
        pred_std = row.get("pred_std", float("nan"))
        pred_q05 = row.get("pred_q05", float("nan"))
        pred_q95 = row.get("pred_q95", float("nan"))

        new_parts.append("<tr>")
        new_parts.append(f"    <td>{html.escape(scenario_display)}</td>")
        new_parts.append(
            f'    <td data-value="{pred_q50:.4f}">{pred_q50:.1f}</td>'
            if pd.notna(pred_q50)
            else '    <td data-value="0">--</td>'
        )
        new_parts.append(
            f'    <td data-value="{pred_std:.4f}">{pred_std:.2f}</td>'
            if pd.notna(pred_std)
            else '    <td data-value="0">--</td>'
        )
        ci_str = (
            f"{pred_q05:.1f} -- {pred_q95:.1f}"
            if pd.notna(pred_q05) and pd.notna(pred_q95)
            else "--"
        )
        new_parts.append(f"    <td>{ci_str}</td>")
        new_parts.append("</tr>")

    new_parts.extend(["</tbody>", "</table>"])

    return {
        "known_table": "\n".join(known_parts),
        "new_table": "\n".join(new_parts),
    }
