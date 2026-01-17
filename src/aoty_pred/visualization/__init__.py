"""Interactive visualization module for AOTY prediction results.

This module provides Plotly-based interactive charts and custom themes
for exploring model results. It includes:

- Custom Plotly templates (light and dark themes)
- Chart creation functions for diagnostics and predictions
- Static export pipeline (SVG, PNG via Kaleido)
- Dashboard assembly for multi-chart views
- Colorblind-safe color palette

Usage:
    >>> from aoty_pred.visualization import COLORBLIND_COLORS, create_trace_plot
    >>> import numpy as np
    >>> samples = np.random.randn(4, 1000)  # 4 chains, 1000 draws
    >>> fig = create_trace_plot(samples, "mu")
    >>> fig.show()
"""

from __future__ import annotations

# Import and register themes on package import
from aoty_pred.visualization.theme import COLORBLIND_COLORS, register_themes

# Import chart creation functions
from aoty_pred.visualization.charts import (
    create_forest_plot,
    create_posterior_plot,
    create_predictions_plot,
    create_reliability_plot,
    create_trace_plot,
)

# Import export functions
from aoty_pred.visualization.export import (
    ensure_kaleido_chrome,
    export_all_figures,
    export_dashboard_html,
    export_figure,
)

# Import dashboard functions
from aoty_pred.visualization.dashboard import (
    DashboardData,
    create_artist_view,
    create_dashboard_figures,
    get_artist_list,
)

# Import server functions
from aoty_pred.visualization.server import (
    app,
    run_server,
)

# Ensure themes are registered
register_themes()

__all__ = [
    "COLORBLIND_COLORS",
    "DashboardData",
    "app",
    "create_artist_view",
    "create_dashboard_figures",
    "create_forest_plot",
    "create_posterior_plot",
    "create_predictions_plot",
    "create_reliability_plot",
    "create_trace_plot",
    "ensure_kaleido_chrome",
    "export_all_figures",
    "export_dashboard_html",
    "export_figure",
    "get_artist_list",
    "register_themes",
    "run_server",
]
