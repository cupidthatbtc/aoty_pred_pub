"""Data flow diagram generation for pipeline visualization.

This module provides a builder class for creating publication-quality SVG diagrams
that illustrate the data transformation pipeline. Diagrams show boxes (nodes)
representing pipeline stages connected by arrows indicating data flow.

Design: Technical manual style with sharp rectangular boxes, thin lines, and
muted colors. Only key stages use subtle color accents; most boxes are grayscale.

Three theme variants are supported:
- light: White background with dark text
- dark: Dark background with light text
- transparent: No background for embedding

Usage:
    >>> from aoty_pred.visualization.diagrams import DataFlowDiagram, DiagramNode
    >>> diagram = DataFlowDiagram(detail_level="high", theme="light")
    >>> node = DiagramNode(
    ...     id="raw", x=0.1, y=0.8, width=0.15, height=0.05,
    ...     label="Raw Data", stage_type="data_input"
    ... )
    >>> diagram.add_node(node)
    >>> diagram.add_arrow((0.25, 0.82), (0.35, 0.82))
    >>> paths = diagram.save(Path("output"), formats=("svg",))
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Patch

if TYPE_CHECKING:
    pass

__all__ = [
    "DataFlowDiagram",
    "DetailLevel",
    "DiagramNode",
    "DiagramTheme",
    "STAGE_COLORS",
    "create_high_level_diagram",
    "create_intermediate_diagram",
    "create_detailed_diagram",
    "generate_all_diagrams",
]

# Type aliases for detail level and theme
DetailLevel = Literal["high", "intermediate", "detailed"]
DiagramTheme = Literal["light", "dark", "transparent"]

# Muted color palette for technical manual style
# Most boxes use grayscale; only key stages get subtle color accents
STAGE_COLORS: dict[str, str] = {
    "data_input": "#E8E8E8",      # Light gray - input
    "sanitization": "#D8D8D8",    # Medium-light gray - cleaning
    "splitting": "#C8C8C8",       # Medium gray - splits
    "features": "#B8D4E8",        # Muted blue - features (subtle accent)
    "model": "#E8D8B8",           # Muted tan/yellow - model (subtle accent)
    "validation": "#D8E8D8",      # Muted green - validation
    "output": "#E8D8D8",          # Muted pink/rose - output (subtle accent)
}


@dataclass
class DiagramNode:
    """A node (box) in the data flow diagram.

    Attributes
    ----------
    id : str
        Unique identifier for the node.
    x : float
        X position (0-1 normalized coordinates).
    y : float
        Y position (0-1 normalized coordinates).
    width : float
        Box width (0-1 normalized).
    height : float
        Box height (0-1 normalized).
    label : str
        Main label text displayed in the box.
    stage_type : str
        Key into STAGE_COLORS for box fill color.
    sublabel : str | None
        Optional smaller text displayed below main label.
    """

    id: str
    x: float
    y: float
    width: float
    height: float
    label: str
    stage_type: str
    sublabel: str | None = None


class DataFlowDiagram:
    """Builder for publication-quality data flow diagram SVGs.

    This class manages diagram construction, theming, and export. Nodes and
    arrows are added incrementally, then exported to multiple formats.

    Style: Technical manual aesthetic with sharp rectangular boxes, thin lines,
    and muted colors. Grid-aligned layouts with generous whitespace.

    Parameters
    ----------
    detail_level : DetailLevel
        Abstraction level ("high", "intermediate", "detailed").
    theme : DiagramTheme
        Visual theme ("light", "dark", "transparent").
    fig_width : float, default 12.0
        Figure width in inches.
    fig_height : float, default 8.0
        Figure height in inches.
    title : str | None, default None
        Optional diagram title.

    Examples
    --------
    >>> diagram = DataFlowDiagram("high", "light", title="Pipeline Overview")
    >>> diagram.add_node(DiagramNode(...))
    >>> diagram.add_arrow((0.2, 0.5), (0.4, 0.5))
    >>> diagram.save(Path("output"), formats=("svg", "png"))
    """

    def __init__(
        self,
        detail_level: DetailLevel,
        theme: DiagramTheme,
        fig_width: float = 12.0,
        fig_height: float = 8.0,
        title: str | None = None,
    ) -> None:
        """Initialize diagram with theme and size settings."""
        self.detail_level = detail_level
        self.theme = theme
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.title = title

        # Track nodes and arrows for reference
        self.nodes: list[DiagramNode] = []
        self._arrows: list[FancyArrowPatch] = []

        # Get theme-specific colors
        self.bg_color, self.text_color, self.border_color = self._get_theme_colors()

        # Create figure and axes
        self.fig, self.ax = plt.subplots(figsize=(fig_width, fig_height))
        self._setup_canvas()

    def _get_theme_colors(self) -> tuple[str, str, str]:
        """Return (background, text, border) colors for current theme.

        Returns
        -------
        tuple[str, str, str]
            Background color, text color, border color.
        """
        if self.theme == "light":
            return ("white", "#333333", "#555555")
        elif self.theme == "dark":
            return ("#1E1E1E", "#E0E0E0", "#888888")
        else:  # transparent
            return ("none", "#333333", "#555555")

    def _setup_canvas(self) -> None:
        """Configure axes for diagram drawing."""
        # Set coordinate system (0, 1) for normalized positioning
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        # Apply theme background
        if self.theme == "transparent":
            self.fig.patch.set_alpha(0)
            self.ax.patch.set_alpha(0)
        else:
            self.fig.patch.set_facecolor(self.bg_color)
            self.ax.set_facecolor(self.bg_color)

        # Add title if provided
        if self.title:
            self.ax.set_title(
                self.title,
                fontsize=12,
                fontweight="bold",
                color=self.text_color,
                pad=8,
                fontfamily="sans-serif",
            )

    def add_node(self, node: DiagramNode) -> None:
        """Add a node (box) to the diagram.

        Creates a sharp rectangular box with centered label text. If the node
        has a sublabel, it is displayed below the main label in smaller font.

        Parameters
        ----------
        node : DiagramNode
            Node specification with position, size, label, and stage type.
        """
        # Get fill color from stage type
        fill_color = STAGE_COLORS.get(node.stage_type, "#E0E0E0")

        # Create sharp rectangular box (no rounding)
        rect = Rectangle(
            (node.x, node.y),
            node.width,
            node.height,
            facecolor=fill_color,
            edgecolor=self.border_color,
            linewidth=0.8,
        )
        self.ax.add_patch(rect)

        # Calculate text vertical offset
        if node.sublabel:
            label_y_offset = 0.008  # Shift main label up if sublabel exists
        else:
            label_y_offset = 0

        # Add main label centered in box
        self.ax.text(
            node.x + node.width / 2,
            node.y + node.height / 2 + label_y_offset,
            node.label,
            ha="center",
            va="center",
            fontsize=8,
            color=self.text_color,
            fontweight="medium",
            fontfamily="sans-serif",
        )

        # Add sublabel if present
        if node.sublabel:
            self.ax.text(
                node.x + node.width / 2,
                node.y + node.height / 2 - 0.018,
                node.sublabel,
                ha="center",
                va="top",
                fontsize=6,
                color=self.text_color,
                fontfamily="sans-serif",
                alpha=0.7,
            )

        # Track node for reference
        self.nodes.append(node)

    def add_arrow(
        self,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        curved: bool = False,
        dashed: bool = False,
        label: str | None = None,
    ) -> None:
        """Add an arrow connecting two points.

        Parameters
        ----------
        start_xy : tuple[float, float]
            Starting point (x, y) in normalized coordinates.
        end_xy : tuple[float, float]
            Ending point (x, y) in normalized coordinates.
        curved : bool, default False
            If True, draw a curved arrow (useful for feedback loops).
        dashed : bool, default False
            If True, draw a dashed line instead of solid.
        label : str | None, default None
            Optional label to display at arrow midpoint.
        """
        # Connection style: curved or straight
        connection_style = "arc3,rad=0.15" if curved else "arc3,rad=0"

        # Line style: dashed or solid
        line_style = "--" if dashed else "-"

        # Create arrow patch with thin line
        arrow = FancyArrowPatch(
            start_xy,
            end_xy,
            arrowstyle="-|>",
            mutation_scale=8,
            color=self.border_color,
            connectionstyle=connection_style,
            linestyle=line_style,
            linewidth=0.8,
        )
        self.ax.add_patch(arrow)
        self._arrows.append(arrow)

        # Add label at midpoint if provided
        if label:
            mid_x = (start_xy[0] + end_xy[0]) / 2
            mid_y = (start_xy[1] + end_xy[1]) / 2
            self.ax.text(
                mid_x,
                mid_y + 0.015,  # Slight offset above arrow
                label,
                ha="center",
                va="bottom",
                fontsize=6,
                color=self.text_color,
                fontfamily="sans-serif",
                alpha=0.8,
            )

    def add_legend(self, location: str = "lower right") -> None:
        """Add color legend explaining stage types.

        Creates a legend with colored patches for each stage type
        defined in STAGE_COLORS.

        Parameters
        ----------
        location : str, default "lower right"
            Legend location (matplotlib loc string).
        """
        # Create legend handles for each stage (subset for clarity)
        handles = []
        labels = []

        # Only show key stages in legend to avoid clutter
        key_stages = ["data_input", "features", "model", "validation", "output"]

        for stage_name in key_stages:
            if stage_name in STAGE_COLORS:
                color = STAGE_COLORS[stage_name]
                patch = Patch(
                    facecolor=color,
                    edgecolor=self.border_color,
                    linewidth=0.5,
                    label=stage_name.replace("_", " ").title(),
                )
                handles.append(patch)
                labels.append(stage_name.replace("_", " ").title())

        # Add legend to axes
        legend = self.ax.legend(
            handles=handles,
            loc=location,
            frameon=True,
            fancybox=False,  # Sharp corners
            shadow=False,
            fontsize=7,
            framealpha=0.9 if self.theme != "transparent" else 0.7,
            edgecolor=self.border_color,
        )
        legend.get_frame().set_linewidth(0.5)

    def save(
        self,
        output_path: Path,
        formats: tuple[str, ...] = ("svg", "png", "pdf"),
    ) -> list[Path]:
        """Save diagram to multiple formats.

        Parameters
        ----------
        output_path : Path
            Base output path (without extension). Files will be created
            with appropriate extensions appended.
        formats : tuple[str, ...], default ("svg", "png", "pdf")
            Output formats to generate.

        Returns
        -------
        list[Path]
            List of paths to created files.
        """
        output_path = Path(output_path)

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        created_paths: list[Path] = []

        for fmt in formats:
            filepath = output_path.parent / f"{output_path.name}.{fmt}"

            # Common savefig parameters
            save_kwargs = {
                "format": fmt,
                "bbox_inches": "tight",
                "pad_inches": 0.1,
            }

            # Format-specific parameters
            if fmt == "png":
                save_kwargs["dpi"] = 300

            # Transparency for transparent theme
            if self.theme == "transparent":
                save_kwargs["transparent"] = True

            self.fig.savefig(filepath, **save_kwargs)
            created_paths.append(filepath)

        # Close figure to free memory
        plt.close(self.fig)

        return created_paths

    def close(self) -> None:
        """Close figure to free memory (if not already closed)."""
        try:
            plt.close(self.fig)
        except Exception:
            pass  # Figure may already be closed


# ============================================================================
# Layout Creation Functions
# ============================================================================


def create_high_level_diagram(theme: DiagramTheme = "light") -> DataFlowDiagram:
    """Create high-level data flow diagram (6 boxes) for README/overview.

    Shows the main pipeline stages at the highest abstraction level:
    Raw Data -> Cleaning -> Splitting -> Features -> Model -> Output

    Technical manual style: Sharp boxes, thin lines, muted colors, generous spacing.

    Parameters
    ----------
    theme : DiagramTheme, default "light"
        Visual theme ("light", "dark", "transparent").

    Returns
    -------
    DataFlowDiagram
        Configured diagram ready for export.

    Example
    -------
    >>> diagram = create_high_level_diagram("light")
    >>> diagram.save(Path("docs/figures/data_flow_high_light"))
    """
    # Determine title based on theme
    title = "AOTY Prediction Pipeline" if theme != "transparent" else None

    diagram = DataFlowDiagram(
        detail_level="high",
        theme=theme,
        fig_width=8,
        fig_height=10,
        title=title,
    )

    # Layout: Single column, top-to-bottom flow
    # Generous vertical spacing for readability
    box_width = 0.30
    box_height = 0.06
    center_x = 0.35  # Left of center to make room for legend

    # Y positions with generous spacing
    y_positions = [0.88, 0.74, 0.60, 0.46, 0.32, 0.18]

    # Row 0: Raw Data
    diagram.add_node(DiagramNode(
        id="raw",
        x=center_x,
        y=y_positions[0],
        width=box_width,
        height=box_height,
        label="Raw Data",
        stage_type="data_input",
        sublabel="CSV input",
    ))

    # Row 1: Data Cleaning
    diagram.add_node(DiagramNode(
        id="clean",
        x=center_x,
        y=y_positions[1],
        width=box_width,
        height=box_height,
        label="Data Cleaning",
        stage_type="sanitization",
        sublabel="schema validation",
    ))

    # Row 2: Train/Val/Test Split
    diagram.add_node(DiagramNode(
        id="split",
        x=center_x,
        y=y_positions[2],
        width=box_width,
        height=box_height,
        label="Train / Val / Test Split",
        stage_type="splitting",
        sublabel="temporal ordering",
    ))

    # Row 3: Feature Engineering
    diagram.add_node(DiagramNode(
        id="features",
        x=center_x,
        y=y_positions[3],
        width=box_width,
        height=box_height,
        label="Feature Engineering",
        stage_type="features",
        sublabel="6 feature blocks",
    ))

    # Row 4: Bayesian Model
    diagram.add_node(DiagramNode(
        id="model",
        x=center_x,
        y=y_positions[4],
        width=box_width,
        height=box_height,
        label="Bayesian Model + MCMC",
        stage_type="model",
        sublabel="hierarchical, 4 chains",
    ))

    # Row 5: Predictions
    diagram.add_node(DiagramNode(
        id="predictions",
        x=center_x,
        y=y_positions[5],
        width=box_width,
        height=box_height,
        label="Predictions",
        stage_type="output",
        sublabel="with uncertainty",
    ))

    # Arrows: Connect consecutive rows (straight down)
    arrow_x = center_x + box_width / 2

    for i in range(len(y_positions) - 1):
        diagram.add_arrow(
            (arrow_x, y_positions[i]),
            (arrow_x, y_positions[i + 1] + box_height),
        )

    # Add legend
    diagram.add_legend("lower right")

    return diagram


def create_intermediate_diagram(theme: DiagramTheme = "light") -> DataFlowDiagram:
    """Create intermediate data flow diagram (12 boxes) for paper figures.

    Shows pipeline stages with feature groupings and split strategies:
    - Data input with schema validation and cleaning
    - Two split strategies (temporal, artist-disjoint)
    - Feature blocks grouped by type
    - Model specification with MCMC
    - Validation stages

    Technical manual style: Sharp boxes, thin lines, muted colors.

    Parameters
    ----------
    theme : DiagramTheme, default "light"
        Visual theme ("light", "dark", "transparent").

    Returns
    -------
    DataFlowDiagram
        Configured diagram ready for export.

    Example
    -------
    >>> diagram = create_intermediate_diagram("dark")
    >>> diagram.save(Path("docs/figures/data_flow_intermediate_dark"))
    """
    title = "AOTY Prediction Pipeline" if theme != "transparent" else None

    diagram = DataFlowDiagram(
        detail_level="intermediate",
        theme=theme,
        fig_width=10,
        fig_height=12,
        title=title,
    )

    # Box dimensions - compact but readable
    box_h = 0.05
    box_w = 0.22
    small_w = 0.18

    # Y positions with good spacing (8 rows)
    y_pos = [0.90, 0.78, 0.66, 0.54, 0.42, 0.30, 0.18, 0.06]

    # Row 0: Raw CSV (centered)
    diagram.add_node(DiagramNode(
        id="raw_csv",
        x=0.39,
        y=y_pos[0],
        width=box_w,
        height=box_h,
        label="Raw CSV",
        stage_type="data_input",
        sublabel="all_albums_full.csv",
    ))

    # Row 1: Schema Validation | Cleaning Rules (two columns)
    diagram.add_node(DiagramNode(
        id="schema",
        x=0.15,
        y=y_pos[1],
        width=small_w,
        height=box_h,
        label="Schema Validation",
        stage_type="sanitization",
        sublabel="Pandera",
    ))

    diagram.add_node(DiagramNode(
        id="cleaning",
        x=0.55,
        y=y_pos[1],
        width=small_w,
        height=box_h,
        label="Cleaning Rules",
        stage_type="sanitization",
        sublabel="exclusions",
    ))

    # Row 2: Temporal Split | Artist-Disjoint Split (two columns)
    diagram.add_node(DiagramNode(
        id="temporal_split",
        x=0.15,
        y=y_pos[2],
        width=small_w,
        height=box_h,
        label="Temporal Split",
        stage_type="splitting",
        sublabel="within-artist",
    ))

    diagram.add_node(DiagramNode(
        id="artist_split",
        x=0.55,
        y=y_pos[2],
        width=small_w,
        height=box_h,
        label="Artist-Disjoint Split",
        stage_type="splitting",
        sublabel="holdout artists",
    ))

    # Row 3: Feature blocks (three columns)
    diagram.add_node(DiagramNode(
        id="temp_feat",
        x=0.05,
        y=y_pos[3],
        width=small_w,
        height=box_h,
        label="Temporal Features",
        stage_type="features",
        sublabel="5 features",
    ))

    diagram.add_node(DiagramNode(
        id="artist_feat",
        x=0.35,
        y=y_pos[3],
        width=small_w,
        height=box_h,
        label="Artist Features",
        stage_type="features",
        sublabel="LOO stats",
    ))

    diagram.add_node(DiagramNode(
        id="cat_feat",
        x=0.65,
        y=y_pos[3],
        width=small_w,
        height=box_h,
        label="Categorical Features",
        stage_type="features",
        sublabel="genre, type",
    ))

    # Row 4: Model Specification (centered)
    diagram.add_node(DiagramNode(
        id="model_spec",
        x=0.35,
        y=y_pos[4],
        width=box_w,
        height=box_h,
        label="Hierarchical Model",
        stage_type="model",
        sublabel="artist effects, priors",
    ))

    # Row 5: MCMC Fitting | GPU (two columns)
    diagram.add_node(DiagramNode(
        id="mcmc_fit",
        x=0.15,
        y=y_pos[5],
        width=small_w,
        height=box_h,
        label="MCMC Sampling",
        stage_type="model",
        sublabel="4 chains",
    ))

    diagram.add_node(DiagramNode(
        id="gpu",
        x=0.55,
        y=y_pos[5],
        width=small_w,
        height=box_h,
        label="GPU Acceleration",
        stage_type="model",
        sublabel="JAX / NumPyro",
    ))

    # Row 6: Validation (three columns)
    diagram.add_node(DiagramNode(
        id="convergence",
        x=0.05,
        y=y_pos[6],
        width=small_w,
        height=box_h,
        label="Convergence",
        stage_type="validation",
        sublabel="R-hat, ESS",
    ))

    diagram.add_node(DiagramNode(
        id="loo_cv",
        x=0.35,
        y=y_pos[6],
        width=small_w,
        height=box_h,
        label="LOO-CV",
        stage_type="validation",
        sublabel="PSIS",
    ))

    diagram.add_node(DiagramNode(
        id="calibration",
        x=0.65,
        y=y_pos[6],
        width=small_w,
        height=box_h,
        label="Calibration",
        stage_type="validation",
        sublabel="coverage",
    ))

    # Row 7: Predictions (centered)
    diagram.add_node(DiagramNode(
        id="predictions",
        x=0.30,
        y=y_pos[7],
        width=0.30,
        height=box_h,
        label="Predictions + Uncertainty",
        stage_type="output",
        sublabel="credible intervals",
    ))

    # ---- Arrows ----

    # Raw -> Schema and Cleaning
    diagram.add_arrow((0.50, y_pos[0]), (0.24, y_pos[1] + box_h))
    diagram.add_arrow((0.50, y_pos[0]), (0.64, y_pos[1] + box_h))

    # Schema & Cleaning -> Splits
    diagram.add_arrow((0.24, y_pos[1]), (0.24, y_pos[2] + box_h))
    diagram.add_arrow((0.64, y_pos[1]), (0.64, y_pos[2] + box_h))

    # Splits -> Features
    diagram.add_arrow((0.24, y_pos[2]), (0.14, y_pos[3] + box_h))
    diagram.add_arrow((0.24, y_pos[2]), (0.44, y_pos[3] + box_h))
    diagram.add_arrow((0.64, y_pos[2]), (0.74, y_pos[3] + box_h))

    # Features -> Model
    diagram.add_arrow((0.14, y_pos[3]), (0.40, y_pos[4] + box_h))
    diagram.add_arrow((0.44, y_pos[3]), (0.46, y_pos[4] + box_h))
    diagram.add_arrow((0.74, y_pos[3]), (0.52, y_pos[4] + box_h))

    # Model -> MCMC & GPU
    diagram.add_arrow((0.40, y_pos[4]), (0.24, y_pos[5] + box_h))
    diagram.add_arrow((0.52, y_pos[4]), (0.64, y_pos[5] + box_h))

    # MCMC & GPU -> Validation
    diagram.add_arrow((0.24, y_pos[5]), (0.14, y_pos[6] + box_h))
    diagram.add_arrow((0.24, y_pos[5]), (0.44, y_pos[6] + box_h))
    diagram.add_arrow((0.64, y_pos[5]), (0.74, y_pos[6] + box_h))

    # Validation -> Predictions
    diagram.add_arrow((0.14, y_pos[6]), (0.38, y_pos[7] + box_h))
    diagram.add_arrow((0.44, y_pos[6]), (0.45, y_pos[7] + box_h))
    diagram.add_arrow((0.74, y_pos[6]), (0.52, y_pos[7] + box_h))

    # Feedback loop: Convergence -> Model (dashed, curved)
    diagram.add_arrow(
        (0.05, y_pos[6] + box_h / 2),
        (0.35, y_pos[4] + box_h / 2),
        curved=True,
        dashed=True,
        label="refine",
    )

    # Add legend
    diagram.add_legend("lower right")

    return diagram


def create_detailed_diagram(theme: DiagramTheme = "light") -> DataFlowDiagram:
    """Create detailed data flow diagram (24 elements) for comprehensive docs.

    Shows the complete pipeline with all 6 feature blocks explicitly:
    - Data input with full cleaning chain
    - Split strategies with manifests
    - All 6 feature blocks: Temporal, AlbumType, ArtistHistory, ArtistReputation, Genre, Collaboration
    - Model architecture details
    - Full validation suite with feedback loops

    Technical manual style: Sharp boxes, thin lines, muted colors, grid layout.

    Parameters
    ----------
    theme : DiagramTheme, default "light"
        Visual theme ("light", "dark", "transparent").

    Returns
    -------
    DataFlowDiagram
        Configured diagram ready for export.

    Example
    -------
    >>> diagram = create_detailed_diagram("transparent")
    >>> diagram.save(Path("docs/figures/data_flow_detailed_transparent"))
    """
    title = "AOTY Prediction Pipeline" if theme != "transparent" else None

    diagram = DataFlowDiagram(
        detail_level="detailed",
        theme=theme,
        fig_width=14,
        fig_height=16,
        title=title,
    )

    # Box dimensions for detailed view - smaller to fit more
    box_h = 0.04
    std_w = 0.12
    small_w = 0.10

    # Y positions with good spacing (10 rows)
    y_pos = [0.92, 0.84, 0.74, 0.64, 0.54, 0.44, 0.34, 0.24, 0.14, 0.04]

    # ============ ROW 0: DATA INPUT ============
    diagram.add_node(DiagramNode(
        id="raw_csv",
        x=0.44,
        y=y_pos[0],
        width=std_w,
        height=box_h,
        label="Raw CSV",
        stage_type="data_input",
        sublabel="all_albums_full.csv",
    ))

    # ============ ROW 1: CLEANING (3 columns) ============
    diagram.add_node(DiagramNode(
        id="schema_val",
        x=0.10,
        y=y_pos[1],
        width=small_w,
        height=box_h,
        label="Schema Validation",
        stage_type="sanitization",
        sublabel="Pandera",
    ))

    diagram.add_node(DiagramNode(
        id="clean_rules",
        x=0.38,
        y=y_pos[1],
        width=std_w,
        height=box_h,
        label="Cleaning Rules",
        stage_type="sanitization",
        sublabel="nulls, dates",
    ))

    diagram.add_node(DiagramNode(
        id="min_ratings",
        x=0.68,
        y=y_pos[1],
        width=std_w,
        height=box_h,
        label="Min Ratings Filter",
        stage_type="sanitization",
        sublabel="threshold=10",
    ))

    # ============ ROW 2: AUDIT + SPLITS (3 columns) ============
    diagram.add_node(DiagramNode(
        id="audit_trail",
        x=0.10,
        y=y_pos[2],
        width=small_w,
        height=box_h,
        label="Audit Trail",
        stage_type="sanitization",
        sublabel="JSONL",
    ))

    diagram.add_node(DiagramNode(
        id="within_artist",
        x=0.38,
        y=y_pos[2],
        width=std_w,
        height=box_h,
        label="Within-Artist Split",
        stage_type="splitting",
        sublabel="temporal",
    ))

    diagram.add_node(DiagramNode(
        id="artist_disjoint",
        x=0.68,
        y=y_pos[2],
        width=std_w,
        height=box_h,
        label="Artist-Disjoint Split",
        stage_type="splitting",
        sublabel="holdout",
    ))

    # ============ ROW 3: TRAIN/VAL/TEST + MANIFESTS ============
    diagram.add_node(DiagramNode(
        id="train_set",
        x=0.10,
        y=y_pos[3],
        width=small_w,
        height=box_h,
        label="Train Set",
        stage_type="splitting",
        sublabel="N_train",
    ))

    diagram.add_node(DiagramNode(
        id="val_set",
        x=0.38,
        y=y_pos[3],
        width=small_w,
        height=box_h,
        label="Validation Set",
        stage_type="splitting",
        sublabel="N_val",
    ))

    diagram.add_node(DiagramNode(
        id="test_set",
        x=0.58,
        y=y_pos[3],
        width=small_w,
        height=box_h,
        label="Test Set",
        stage_type="splitting",
        sublabel="N_test",
    ))

    diagram.add_node(DiagramNode(
        id="manifests",
        x=0.78,
        y=y_pos[3],
        width=small_w,
        height=box_h,
        label="Manifests",
        stage_type="splitting",
        sublabel="hash verify",
    ))

    # ============ ROW 4: FEATURE BLOCKS (6 columns) ============
    feature_y = y_pos[4]
    feature_x_positions = [0.02, 0.17, 0.32, 0.47, 0.62, 0.77]

    diagram.add_node(DiagramNode(
        id="temporal_block",
        x=feature_x_positions[0],
        y=feature_y,
        width=std_w,
        height=box_h,
        label="TemporalBlock",
        stage_type="features",
        sublabel="5 temporal",
    ))

    diagram.add_node(DiagramNode(
        id="album_type_block",
        x=feature_x_positions[1],
        y=feature_y,
        width=std_w,
        height=box_h,
        label="AlbumTypeBlock",
        stage_type="features",
        sublabel="one-hot",
    ))

    diagram.add_node(DiagramNode(
        id="artist_history_block",
        x=feature_x_positions[2],
        y=feature_y,
        width=std_w,
        height=box_h,
        label="ArtistHistoryBlock",
        stage_type="features",
        sublabel="LOO stats",
    ))

    diagram.add_node(DiagramNode(
        id="artist_rep_block",
        x=feature_x_positions[3],
        y=feature_y,
        width=std_w,
        height=box_h,
        label="ArtistReputationBlock",
        stage_type="features",
        sublabel="prior mean",
    ))

    diagram.add_node(DiagramNode(
        id="genre_block",
        x=feature_x_positions[4],
        y=feature_y,
        width=std_w,
        height=box_h,
        label="GenreBlock",
        stage_type="features",
        sublabel="PCA",
    ))

    diagram.add_node(DiagramNode(
        id="collab_block",
        x=feature_x_positions[5],
        y=feature_y,
        width=std_w,
        height=box_h,
        label="CollaborationBlock",
        stage_type="features",
        sublabel="ordinal",
    ))

    # ============ ROW 5: FEATURE PIPELINE ============
    diagram.add_node(DiagramNode(
        id="feature_pipeline",
        x=0.35,
        y=y_pos[5],
        width=0.20,
        height=box_h,
        label="FeaturePipeline",
        stage_type="features",
        sublabel="fit train, transform all",
    ))

    # ============ ROW 6: MODEL COMPONENTS (5 columns) ============
    model_y = y_pos[6]
    model_x_positions = [0.05, 0.22, 0.39, 0.56, 0.73]

    diagram.add_node(DiagramNode(
        id="prior_config",
        x=model_x_positions[0],
        y=model_y,
        width=std_w,
        height=box_h,
        label="PriorConfig",
        stage_type="model",
        sublabel="9 hyperparams",
    ))

    diagram.add_node(DiagramNode(
        id="hierarchical",
        x=model_x_positions[1],
        y=model_y,
        width=std_w,
        height=box_h,
        label="Hierarchical",
        stage_type="model",
        sublabel="artist effects",
    ))

    diagram.add_node(DiagramNode(
        id="non_centered",
        x=model_x_positions[2],
        y=model_y,
        width=std_w,
        height=box_h,
        label="Non-centered",
        stage_type="model",
        sublabel="LocScaleReparam",
    ))

    diagram.add_node(DiagramNode(
        id="time_varying",
        x=model_x_positions[3],
        y=model_y,
        width=std_w,
        height=box_h,
        label="Time-varying",
        stage_type="model",
        sublabel="random walk",
    ))

    diagram.add_node(DiagramNode(
        id="ar1",
        x=model_x_positions[4],
        y=model_y,
        width=std_w,
        height=box_h,
        label="AR(1)",
        stage_type="model",
        sublabel="stationary",
    ))

    # ============ ROW 7: MCMC ============
    diagram.add_node(DiagramNode(
        id="mcmc_sampling",
        x=0.35,
        y=y_pos[7],
        width=0.20,
        height=box_h,
        label="MCMC Sampling",
        stage_type="model",
        sublabel="4 chains, GPU",
    ))

    # ============ ROW 8: VALIDATION (6 columns) ============
    val_y = y_pos[8]
    val_x_positions = [0.02, 0.17, 0.32, 0.47, 0.62, 0.77]

    diagram.add_node(DiagramNode(
        id="rhat",
        x=val_x_positions[0],
        y=val_y,
        width=std_w,
        height=box_h,
        label="R-hat",
        stage_type="validation",
        sublabel="< 1.01",
    ))

    diagram.add_node(DiagramNode(
        id="ess",
        x=val_x_positions[1],
        y=val_y,
        width=std_w,
        height=box_h,
        label="ESS",
        stage_type="validation",
        sublabel="> 400",
    ))

    diagram.add_node(DiagramNode(
        id="divergences",
        x=val_x_positions[2],
        y=val_y,
        width=std_w,
        height=box_h,
        label="Divergences",
        stage_type="validation",
        sublabel="= 0",
    ))

    diagram.add_node(DiagramNode(
        id="loo_psis",
        x=val_x_positions[3],
        y=val_y,
        width=std_w,
        height=box_h,
        label="LOO-CV",
        stage_type="validation",
        sublabel="PSIS",
    ))

    diagram.add_node(DiagramNode(
        id="pareto_k",
        x=val_x_positions[4],
        y=val_y,
        width=std_w,
        height=box_h,
        label="Pareto-k",
        stage_type="validation",
        sublabel="< 0.7",
    ))

    diagram.add_node(DiagramNode(
        id="calibration",
        x=val_x_positions[5],
        y=val_y,
        width=std_w,
        height=box_h,
        label="Calibration",
        stage_type="validation",
        sublabel="coverage",
    ))

    # ============ ROW 9: OUTPUTS (3 columns) ============
    diagram.add_node(DiagramNode(
        id="predictions",
        x=0.10,
        y=y_pos[9],
        width=std_w,
        height=box_h,
        label="Predictions",
        stage_type="output",
        sublabel="posterior mean",
    ))

    diagram.add_node(DiagramNode(
        id="uncertainty",
        x=0.38,
        y=y_pos[9],
        width=std_w,
        height=box_h,
        label="Uncertainty",
        stage_type="output",
        sublabel="95% CI",
    ))

    diagram.add_node(DiagramNode(
        id="pub_artifacts",
        x=0.66,
        y=y_pos[9],
        width=std_w,
        height=box_h,
        label="Publication",
        stage_type="output",
        sublabel="figures, tables",
    ))

    # ============ ARROWS ============

    # Row 0 -> Row 1: Raw -> Cleaning stages
    diagram.add_arrow((0.50, y_pos[0]), (0.15, y_pos[1] + box_h))
    diagram.add_arrow((0.50, y_pos[0]), (0.44, y_pos[1] + box_h))
    diagram.add_arrow((0.50, y_pos[0]), (0.74, y_pos[1] + box_h))

    # Row 1 -> Row 2: Cleaning -> Audit/Splits
    diagram.add_arrow((0.15, y_pos[1]), (0.15, y_pos[2] + box_h))
    diagram.add_arrow((0.44, y_pos[1]), (0.44, y_pos[2] + box_h))
    diagram.add_arrow((0.74, y_pos[1]), (0.74, y_pos[2] + box_h))

    # Row 2 -> Row 3: Splits -> Train/Val/Test
    diagram.add_arrow((0.44, y_pos[2]), (0.15, y_pos[3] + box_h))
    diagram.add_arrow((0.44, y_pos[2]), (0.43, y_pos[3] + box_h))
    diagram.add_arrow((0.74, y_pos[2]), (0.63, y_pos[3] + box_h))
    diagram.add_arrow((0.74, y_pos[2]), (0.83, y_pos[3] + box_h))

    # Row 3 -> Row 4: Data sets -> Feature blocks
    # Distribute arrows across all 6 feature blocks
    diagram.add_arrow((0.15, y_pos[3]), (0.08, y_pos[4] + box_h))
    diagram.add_arrow((0.15, y_pos[3]), (0.23, y_pos[4] + box_h))
    diagram.add_arrow((0.43, y_pos[3]), (0.38, y_pos[4] + box_h))
    diagram.add_arrow((0.43, y_pos[3]), (0.53, y_pos[4] + box_h))
    diagram.add_arrow((0.63, y_pos[3]), (0.68, y_pos[4] + box_h))
    diagram.add_arrow((0.63, y_pos[3]), (0.83, y_pos[4] + box_h))

    # Row 4 -> Row 5: Feature blocks -> FeaturePipeline
    for fx in feature_x_positions:
        diagram.add_arrow((fx + std_w / 2, y_pos[4]), (0.45, y_pos[5] + box_h))

    # Row 5 -> Row 6: FeaturePipeline -> Model components
    for mx in model_x_positions:
        diagram.add_arrow((0.45, y_pos[5]), (mx + std_w / 2, y_pos[6] + box_h))

    # Row 6 -> Row 7: Model components -> MCMC
    for mx in model_x_positions:
        diagram.add_arrow((mx + std_w / 2, y_pos[6]), (0.45, y_pos[7] + box_h))

    # Row 7 -> Row 8: MCMC -> Validation
    for vx in val_x_positions:
        diagram.add_arrow((0.45, y_pos[7]), (vx + std_w / 2, y_pos[8] + box_h))

    # Row 8 -> Row 9: Validation -> Outputs
    diagram.add_arrow((0.08, y_pos[8]), (0.16, y_pos[9] + box_h))
    diagram.add_arrow((0.53, y_pos[8]), (0.44, y_pos[9] + box_h))
    diagram.add_arrow((0.83, y_pos[8]), (0.72, y_pos[9] + box_h))

    # Feedback loops (dashed, curved)
    # Calibration -> Model (adjust loop)
    diagram.add_arrow(
        (0.83, y_pos[8] + box_h / 2),
        (0.28, y_pos[6] + box_h / 2),
        curved=True,
        dashed=True,
        label="tune",
    )

    # Add legend
    diagram.add_legend("lower right")

    return diagram


def generate_all_diagrams(output_dir: Path) -> dict[str, list[Path]]:
    """Generate all data flow diagram variants.

    Creates 9 diagrams: 3 detail levels x 3 themes.
    Each diagram is saved in SVG, PNG, and PDF formats.

    Parameters
    ----------
    output_dir : Path
        Directory for output files (created if needed).

    Returns
    -------
    dict[str, list[Path]]
        Dict mapping diagram name to list of created file paths.

    Example
    -------
    >>> results = generate_all_diagrams(Path("docs/figures"))
    >>> for name, paths in results.items():
    ...     print(f"{name}: {len(paths)} files")
    data_flow_high_light: 3 files
    data_flow_high_dark: 3 files
    ...
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[Path]] = {}

    creators = {
        "high": create_high_level_diagram,
        "intermediate": create_intermediate_diagram,
        "detailed": create_detailed_diagram,
    }
    themes: list[DiagramTheme] = ["light", "dark", "transparent"]

    for level, creator_fn in creators.items():
        for theme in themes:
            name = f"data_flow_{level}_{theme}"
            diagram = creator_fn(theme)
            paths = diagram.save(output_dir / name)
            results[name] = paths

    return results
