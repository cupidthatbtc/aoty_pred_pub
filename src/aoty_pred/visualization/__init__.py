"""Interactive visualization module for AOTY prediction results.

This module provides Plotly-based interactive charts and custom themes
for exploring model results. It includes:

- Custom Plotly templates (light and dark themes)
- Chart creation functions for diagnostics and predictions
- Colorblind-safe color palette

Usage:
    >>> from aoty_pred.visualization import COLORBLIND_COLORS
    >>> from aoty_pred.visualization.charts import create_trace_plot
    >>> import numpy as np
    >>> samples = np.random.randn(4, 1000)  # 4 chains, 1000 draws
    >>> fig = create_trace_plot(samples, "mu")
    >>> fig.show()
"""

from __future__ import annotations

# Import and register themes on package import
from aoty_pred.visualization.theme import COLORBLIND_COLORS, register_themes

# Ensure themes are registered
register_themes()

__all__ = [
    "COLORBLIND_COLORS",
    "register_themes",
]
