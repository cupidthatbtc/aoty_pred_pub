"""FastAPI server for AOTY Model Dashboard.

This module provides a local web server for exploring model results interactively.
It serves the Plotly-based dashboard and provides endpoints for data access and
chart export.

Usage:
    >>> from aoty_pred.visualization.server import run_server
    >>> run_server(port=8050, open_browser=True)

CLI usage:
    aoty-pipeline visualize --port 8050
"""

from __future__ import annotations

import logging
import threading
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from aoty_pred.visualization.charts import (
    create_forest_plot,
    create_predictions_plot,
    create_reliability_plot,
    create_trace_plot,
)
from aoty_pred.visualization.dashboard import (
    DashboardData,
    create_artist_view,
    create_coefficients_table,
    create_dashboard_figures,
    get_artist_list,
)
from aoty_pred.visualization.export import ensure_kaleido_chrome

__all__ = [
    "app",
    "load_dashboard_data",
    "run_server",
]

logger = logging.getLogger(__name__)


def _find_project_root() -> Path:
    """Find project root by looking for pyproject.toml or .git upward from cwd."""
    for parent in [Path.cwd()] + list(Path.cwd().parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return Path.cwd()


# Module directory for templates
MODULE_DIR = Path(__file__).parent
TEMPLATES_DIR = MODULE_DIR / "templates"
STATIC_DIR = TEMPLATES_DIR / "static"

# FastAPI application
app = FastAPI(
    title="AOTY Model Dashboard",
    description="Interactive visualization of album score prediction model results.",
    version="0.1.0",
)

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Global state for cached data
_dashboard_data: DashboardData | None = None


def load_dashboard_data(run_dir: Path | None = None) -> DashboardData:
    """Load dashboard data from the most recent pipeline run or specified directory.

    Parameters
    ----------
    run_dir : Path | None, default None
        Path to a specific pipeline run directory (e.g., reports/2026-01-19_143052).
        If None, looks for the most recent run in reports/.

    Returns
    -------
    DashboardData
        Dataclass containing loaded data for dashboard views.
        Fields are None if corresponding data is not found.

    Notes
    -----
    Looks for:
    - InferenceData: models/*.nc or .json
    - Predictions: evaluation results
    - Coefficients: reports/tables/*.csv
    - Artist data: data/processed/*.parquet
    """
    global _dashboard_data

    # Start with empty data
    data = DashboardData()

    # Determine run directory
    if run_dir is None:
        # Look for most recent run in reports/
        reports_dir = Path("reports")
        if reports_dir.exists():
            run_dirs = sorted(
                [d for d in reports_dir.iterdir() if d.is_dir() and d.name[0].isdigit()],
                key=lambda d: d.name,
                reverse=True,
            )
            if run_dirs:
                run_dir = run_dirs[0]
                logger.info("Using most recent run: %s", run_dir)

    # Try to load inference data
    try:
        import arviz as az

        model_files = []
        if run_dir is not None:
            # Look for NetCDF files in run directory
            model_files = sorted(
                run_dir.glob("models/*.nc"), key=lambda f: f.stat().st_mtime, reverse=True
            )
            if not model_files:
                model_files = sorted(
                    run_dir.glob("*.nc"), key=lambda f: f.stat().st_mtime, reverse=True
                )

        # Fallback: check models/ directory relative to project root
        if not model_files:
            project_root = _find_project_root()
            models_dir = project_root / "models"
            if models_dir.exists():
                model_files = sorted(
                    models_dir.glob("*.nc"), key=lambda f: f.stat().st_mtime, reverse=True
                )
                if model_files:
                    logger.info("Using fallback models directory: %s", models_dir)

        if model_files:
            data.idata = az.from_netcdf(model_files[0])
            logger.info("Loaded inference data from %s", model_files[0])
    except (FileNotFoundError, ValueError, OSError, ImportError, TypeError) as e:
        logger.warning("Could not load inference data: %s", e)

    if run_dir is not None:
        # Try to load predictions from evaluation
        try:
            pred_files = list(run_dir.glob("evaluation/*.json"))
            if pred_files:
                import json

                with open(pred_files[0]) as f:
                    eval_data = json.load(f)

                if "predictions" in eval_data:
                    pred = eval_data["predictions"]
                    data.predictions = {
                        "y_true": np.array(pred.get("y_true", [])),
                        "y_pred_mean": np.array(pred.get("y_pred_mean", [])),
                        "y_pred_lower": np.array(pred.get("y_pred_lower", [])),
                        "y_pred_upper": np.array(pred.get("y_pred_upper", [])),
                    }
                    logger.info("Loaded predictions from %s", pred_files[0])
        except Exception as e:
            logger.debug("Could not load predictions: %s", e)

        # Try to load coefficient tables
        try:
            table_files = list(run_dir.glob("tables/*coefficient*.csv"))
            if not table_files:
                table_files = list(run_dir.glob("tables/*summary*.csv"))
            if table_files:
                data.coefficients = pd.read_csv(table_files[0])
                logger.info("Loaded coefficients from %s", table_files[0])
        except Exception as e:
            logger.debug("Could not load coefficients: %s", e)

        # Try to load calibration data
        try:
            cal_files = list(run_dir.glob("evaluation/*calibration*.json"))
            if cal_files:
                import json

                with open(cal_files[0]) as f:
                    cal_data = json.load(f)

                if all(k in cal_data for k in ["predicted_probs", "observed_freq", "counts"]):
                    data.reliability = {
                        "predicted_probs": np.array(cal_data["predicted_probs"]),
                        "observed_freq": np.array(cal_data["observed_freq"]),
                        "counts": np.array(cal_data["counts"]),
                    }
                    logger.info("Loaded calibration data from %s", cal_files[0])
        except Exception as e:
            logger.debug("Could not load calibration data: %s", e)

    # Try to load artist data from processed directory
    try:
        processed_dir = Path("data/processed")
        if processed_dir.exists():
            # Prefer user score data
            artist_files = list(processed_dir.glob("*user_score*.parquet"))
            if not artist_files:
                artist_files = list(processed_dir.glob("cleaned*.parquet"))
            if artist_files:
                df = pd.read_parquet(artist_files[0])
                # Ensure we have an artist column
                artist_col = None
                for col in ["artist", "Artist", "ARTIST"]:
                    if col in df.columns:
                        artist_col = col
                        break
                if artist_col:
                    if artist_col != "artist":
                        df = df.rename(columns={artist_col: "artist"})
                    data.artist_data = df
                    logger.info("Loaded artist data from %s", artist_files[0])
    except Exception as e:
        logger.debug("Could not load artist data: %s", e)

    # Load eval metrics from outputs/evaluation/metrics.json
    try:
        project_root = _find_project_root()
        metrics_path = project_root / "outputs" / "evaluation" / "metrics.json"
        if metrics_path.exists():
            import json

            with open(metrics_path) as f:
                data.eval_metrics = json.load(f)
            logger.info("Loaded eval metrics from %s", metrics_path)
    except Exception as e:
        logger.debug("Could not load eval metrics: %s", e)

    # Load known artist predictions from outputs/predictions/
    try:
        project_root = _find_project_root()
        known_path = project_root / "outputs" / "predictions" / "next_album_known_artists.csv"
        if known_path.exists():
            data.known_predictions = pd.read_csv(known_path)
            logger.info("Loaded known artist predictions from %s", known_path)
    except Exception as e:
        logger.debug("Could not load known artist predictions: %s", e)

    # Load new artist predictions from outputs/predictions/
    try:
        project_root = _find_project_root()
        new_path = project_root / "outputs" / "predictions" / "next_album_new_artist.csv"
        if new_path.exists():
            data.new_predictions = pd.read_csv(new_path)
            logger.info("Loaded new artist predictions from %s", new_path)
    except Exception as e:
        logger.debug("Could not load new artist predictions: %s", e)

    _dashboard_data = data
    return data


def _get_theme_from_request(request: Request) -> str:
    """Get theme from request cookie or default to light."""
    theme = request.cookies.get("theme", "light")
    return "aoty_dark" if theme == "dark" else "aoty_light"


@app.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    view: str = Query("overview", description="Dashboard view to display"),
) -> HTMLResponse:
    """Serve main dashboard page.

    Parameters
    ----------
    request : Request
        FastAPI request object.
    view : str, default "overview"
        Active view: overview, trace, predictions, coefficients, reliability.

    Returns
    -------
    HTMLResponse
        Rendered dashboard HTML.
    """
    global _dashboard_data

    if _dashboard_data is None:
        _dashboard_data = load_dashboard_data()

    theme = _get_theme_from_request(request)
    figures = create_dashboard_figures(_dashboard_data, theme=theme)

    # Get artist list for search autocomplete
    artists: list[str] = []
    if _dashboard_data.artist_data is not None:
        artists = get_artist_list(_dashboard_data.artist_data)

    # Generate coefficient table if available
    coefficients_table = None
    if _dashboard_data.coefficients is not None:
        coefficients_table = create_coefficients_table(_dashboard_data.coefficients)

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "title": "AOTY Model Dashboard",
            "active_view": view,
            "figures": figures,
            "artists": artists,
            "coefficients_table": coefficients_table,
            "eval_metrics": _dashboard_data.eval_metrics,
        },
    )


@app.get("/diagnostics", response_class=HTMLResponse)
async def diagnostics_view(request: Request) -> HTMLResponse:
    """Redirect to dashboard with trace view."""
    return await dashboard(request, view="trace")


@app.get("/predictions", response_class=HTMLResponse)
async def predictions_view(request: Request) -> HTMLResponse:
    """Redirect to dashboard with predictions view."""
    return await dashboard(request, view="predictions")


@app.get("/coefficients", response_class=HTMLResponse)
async def coefficients_view(request: Request) -> HTMLResponse:
    """Redirect to dashboard with coefficients view."""
    return await dashboard(request, view="coefficients")


@app.get("/calibration", response_class=HTMLResponse)
async def calibration_view(request: Request) -> HTMLResponse:
    """Redirect to dashboard with calibration view."""
    return await dashboard(request, view="reliability")


@app.get("/next-albums", response_class=HTMLResponse)
async def next_albums_view(request: Request) -> HTMLResponse:
    """Redirect to dashboard with next-albums view."""
    return await dashboard(request, view="next_albums")


@app.get("/artist", response_class=HTMLResponse)
async def artist_view(
    request: Request,
    name: str = Query(..., description="Artist name to display"),
) -> HTMLResponse:
    """Serve artist-specific view.

    Parameters
    ----------
    request : Request
        FastAPI request object.
    name : str
        Artist name to display.

    Returns
    -------
    HTMLResponse
        Rendered dashboard with artist view.
    """
    global _dashboard_data

    if _dashboard_data is None:
        _dashboard_data = load_dashboard_data()

    theme = _get_theme_from_request(request)
    figures = create_dashboard_figures(_dashboard_data, theme=theme)

    # Get artist list for search autocomplete
    artists: list[str] = []
    artist_chart = None
    if _dashboard_data.artist_data is not None:
        artists = get_artist_list(_dashboard_data.artist_data)
        artist_chart = create_artist_view(name, _dashboard_data.artist_data, theme=theme)

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "title": f"Artist: {name}",
            "active_view": "artist",
            "artist_name": name,
            "artist_chart": artist_chart,
            "figures": figures,
            "artists": artists,
        },
    )


@app.get("/api/artists", response_class=JSONResponse)
async def search_artists(
    q: str = Query("", description="Search query for artist names"),
) -> JSONResponse:
    """Search artists by name for autocomplete.

    Parameters
    ----------
    q : str, default ""
        Search query string (case-insensitive prefix match).

    Returns
    -------
    JSONResponse
        JSON array of matching artist names.
    """
    global _dashboard_data

    if _dashboard_data is None:
        _dashboard_data = load_dashboard_data()

    if _dashboard_data.artist_data is None:
        return JSONResponse(content=[])

    artists = get_artist_list(_dashboard_data.artist_data)

    if not q:
        return JSONResponse(content=artists[:50])  # Limit results

    # Filter by prefix (case-insensitive)
    q_lower = q.lower()
    matches = [a for a in artists if a.lower().startswith(q_lower)]

    return JSONResponse(content=matches[:50])


@app.get("/export/{chart_id}")
async def export_chart(
    chart_id: str,
    format: str = Query("png", description="Export format: svg, png, pdf"),
    width: int = Query(800, description="Figure width in pixels"),
    height: int = Query(600, description="Figure height in pixels"),
    scale: float = Query(2.0, description="Scale factor for raster output"),
) -> Response:
    """Export a single chart to static format.

    Parameters
    ----------
    chart_id : str
        Chart identifier: trace, posterior, predictions, coefficients, reliability.
    format : str, default "png"
        Export format: svg, png, pdf.
    width : int, default 800
        Figure width in pixels.
    height : int, default 600
        Figure height in pixels.
    scale : float, default 2.0
        Scale factor for raster formats.

    Returns
    -------
    Response
        Binary response with appropriate content type.
    """
    global _dashboard_data

    if _dashboard_data is None:
        _dashboard_data = load_dashboard_data()

    # Generate the requested figure
    fig: go.Figure | None = None

    if chart_id == "predictions" and _dashboard_data.predictions is not None:
        pred = _dashboard_data.predictions
        fig = create_predictions_plot(
            pred["y_true"],
            pred["y_pred_mean"],
            pred["y_pred_lower"],
            pred["y_pred_upper"],
        )
    elif chart_id == "coefficients" and _dashboard_data.coefficients is not None:
        fig = create_forest_plot(_dashboard_data.coefficients)
    elif chart_id == "reliability" and _dashboard_data.reliability is not None:
        rel = _dashboard_data.reliability
        fig = create_reliability_plot(
            rel["predicted_probs"],
            rel["observed_freq"],
            rel["counts"],
        )
    elif chart_id == "trace" and _dashboard_data.idata is not None:
        try:
            posterior = _dashboard_data.idata.posterior
            var_names = list(posterior.data_vars)
            if var_names:
                samples = posterior[var_names[0]].values
                if samples.ndim > 2:
                    samples = samples.reshape(samples.shape[0], -1)[:, 0:100]
                elif samples.ndim == 1:
                    samples = samples.reshape(1, -1)
                fig = create_trace_plot(samples, var_names[0])
        except Exception as e:
            logger.debug("Skipping trace chart due to unexpected idata format: %s", e)

    if fig is None:
        return Response(
            content=f"Chart '{chart_id}' not available.",
            status_code=404,
            media_type="text/plain",
        )

    # Ensure Kaleido is ready for export
    if format.lower() in ("png", "jpeg", "webp"):
        ensure_kaleido_chrome()

    # Export to bytes
    format_lower = format.lower()
    content_types = {
        "svg": "image/svg+xml",
        "png": "image/png",
        "jpeg": "image/jpeg",
        "pdf": "application/pdf",
        "webp": "image/webp",
    }

    if format_lower not in content_types:
        return Response(
            content=f"Unsupported format: {format}",
            status_code=400,
            media_type="text/plain",
        )

    try:
        effective_scale = scale if format_lower in ("png", "jpeg", "webp") else 1
        img_bytes = fig.to_image(
            format=format_lower,
            width=width,
            height=height,
            scale=effective_scale,
        )

        return Response(
            content=img_bytes,
            media_type=content_types[format_lower],
            headers={"Content-Disposition": f'attachment; filename="{chart_id}.{format_lower}"'},
        )
    except Exception as e:
        logger.error("Export failed for %s: %s", chart_id, e)
        return Response(
            content=f"Export failed: {e}",
            status_code=500,
            media_type="text/plain",
        )


@app.get("/health", response_class=JSONResponse)
async def health_check() -> JSONResponse:
    """Health check endpoint.

    Returns
    -------
    JSONResponse
        JSON object with status "ok".
    """
    return JSONResponse(content={"status": "ok"})


def open_browser_delayed(url: str, delay: float = 1.5) -> None:
    """Open browser after a delay.

    Parameters
    ----------
    url : str
        URL to open.
    delay : float, default 1.5
        Delay in seconds before opening browser.
    """

    def _open() -> None:
        webbrowser.open(url)

    timer = threading.Timer(delay, _open)
    timer.daemon = True
    timer.start()


def run_server(
    port: int = 8050,
    host: str = "127.0.0.1",
    open_browser: bool = True,
    run_dir: Path | None = None,
) -> None:
    """Start the dashboard server.

    Parameters
    ----------
    port : int, default 8050
        Server port number.
    host : str, default "127.0.0.1"
        Server host address.
    open_browser : bool, default True
        Whether to auto-open browser to dashboard URL.
    run_dir : Path | None, default None
        Path to pipeline run directory for loading data.
    """
    import uvicorn

    # Pre-load dashboard data
    load_dashboard_data(run_dir)

    url = f"http://{host}:{port}"
    logger.info("Starting AOTY Model Dashboard at %s", url)

    if open_browser:
        open_browser_delayed(url)

    uvicorn.run(app, host=host, port=port, log_level="info")
