"""Data flow diagram generation for pipeline visualization.

This module provides a builder class for creating publication-quality SVG diagrams
that illustrate the data transformation pipeline. Diagrams show boxes (nodes)
representing pipeline stages connected by arrows indicating data flow.

All diagrams use the Wong (2011) colorblind-safe palette from Nature Methods:
https://www.nature.com/articles/nmeth.1618

Three theme variants are supported:
- light: White background with dark text
- dark: Dark background with light text
- transparent: No background (outline-based) for embedding

Usage:
    >>> from aoty_pred.visualization.diagrams import DataFlowDiagram, DiagramNode
    >>> diagram = DataFlowDiagram(detail_level="high", theme="light")
    >>> node = DiagramNode(
    ...     id="raw", x=0.1, y=0.8, width=0.15, height=0.08,
    ...     label="Raw Data", stage_type="data_input"
    ... )
    >>> diagram.add_node(node)
    >>> diagram.add_arrow((0.25, 0.84), (0.35, 0.84))
    >>> paths = diagram.save(Path("output"), formats=("svg",))
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from aoty_pred.visualization.theme import COLORBLIND_COLORS

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

# Stage colors using Wong (2011) colorblind-safe palette
# Maps pipeline stages to colors for consistent visual differentiation
STAGE_COLORS: dict[str, str] = {
    "data_input": COLORBLIND_COLORS[0],  # Blue
    "sanitization": COLORBLIND_COLORS[1],  # Orange
    "splitting": COLORBLIND_COLORS[2],  # Green
    "features": COLORBLIND_COLORS[3],  # Pink
    "model": COLORBLIND_COLORS[4],  # Yellow
    "validation": COLORBLIND_COLORS[5],  # Light blue
    "output": COLORBLIND_COLORS[6],  # Red-orange
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
            return ("white", "#333333", "#666666")
        elif self.theme == "dark":
            return ("#1E1E1E", "#E0E0E0", "#888888")
        else:  # transparent
            return ("none", "#333333", "#333333")

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
                fontsize=14,
                fontweight="bold",
                color=self.text_color,
                pad=10,
            )

    def add_node(self, node: DiagramNode) -> None:
        """Add a node (box) to the diagram.

        Creates a FancyBboxPatch with rounded corners and adds centered
        label text. If the node has a sublabel, it is displayed below
        the main label in smaller italic font.

        Parameters
        ----------
        node : DiagramNode
            Node specification with position, size, label, and stage type.
        """
        # Get fill color from stage type
        fill_color = STAGE_COLORS.get(node.stage_type, COLORBLIND_COLORS[0])

        # Create rounded box
        bbox = FancyBboxPatch(
            (node.x, node.y),
            node.width,
            node.height,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=fill_color,
            edgecolor=self.border_color,
            linewidth=1.5,
        )
        self.ax.add_patch(bbox)

        # Calculate text vertical offset
        if node.sublabel:
            label_y_offset = 0.01  # Shift main label up if sublabel exists
        else:
            label_y_offset = 0

        # Add main label centered in box
        self.ax.text(
            node.x + node.width / 2,
            node.y + node.height / 2 + label_y_offset,
            node.label,
            ha="center",
            va="center",
            fontsize=9,
            color=self.text_color,
            fontweight="bold",
        )

        # Add sublabel if present
        if node.sublabel:
            self.ax.text(
                node.x + node.width / 2,
                node.y + node.height / 2 - 0.03,
                node.sublabel,
                ha="center",
                va="top",
                fontsize=7,
                color=self.text_color,
                fontstyle="italic",
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
        connection_style = "arc3,rad=0.2" if curved else "arc3,rad=0"

        # Line style: dashed or solid
        line_style = "--" if dashed else "-"

        # Create arrow patch
        arrow = FancyArrowPatch(
            start_xy,
            end_xy,
            arrowstyle="-|>",
            mutation_scale=12,
            color=self.border_color,
            connectionstyle=connection_style,
            linestyle=line_style,
            linewidth=1.5,
        )
        self.ax.add_patch(arrow)
        self._arrows.append(arrow)

        # Add label at midpoint if provided
        if label:
            mid_x = (start_xy[0] + end_xy[0]) / 2
            mid_y = (start_xy[1] + end_xy[1]) / 2
            self.ax.text(
                mid_x,
                mid_y + 0.02,  # Slight offset above arrow
                label,
                ha="center",
                va="bottom",
                fontsize=7,
                color=self.text_color,
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
        # Create legend handles for each stage
        handles = []
        labels = []

        for stage_name, color in STAGE_COLORS.items():
            patch = Patch(
                facecolor=color,
                edgecolor=self.border_color,
                label=stage_name.replace("_", " ").title(),
            )
            handles.append(patch)
            labels.append(stage_name.replace("_", " ").title())

        # Add legend to axes
        self.ax.legend(
            handles=handles,
            loc=location,
            frameon=True,
            fancybox=True,
            shadow=False,
            fontsize=8,
            framealpha=0.9 if self.theme != "transparent" else 0.7,
        )

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
    """Create high-level data flow diagram (5-7 boxes) for README/overview.

    Shows the main pipeline stages at the highest abstraction level:
    Raw Data -> Cleaning -> Splitting -> Features -> Model -> Evaluation -> Predictions

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
        fig_width=10,
        fig_height=8,
        title=title,
    )

    # Y positions (top to bottom, normalized 0-1)
    y_positions = [0.85, 0.72, 0.59, 0.46, 0.33, 0.20, 0.07]

    # Standard box dimensions
    box_width = 0.20
    box_height = 0.08
    center_x = 0.40  # Centered in figure

    # Row 0: Raw Data
    diagram.add_node(DiagramNode(
        id="raw",
        x=center_x,
        y=y_positions[0],
        width=box_width,
        height=box_height,
        label="Raw Data",
        stage_type="data_input",
        sublabel="(N albums)",
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
        sublabel="(schema + rules)",
    ))

    # Row 2: Train/Val/Test Split
    diagram.add_node(DiagramNode(
        id="split",
        x=center_x,
        y=y_positions[2],
        width=box_width,
        height=box_height,
        label="Train/Val/Test Split",
        stage_type="splitting",
        sublabel="(temporal)",
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
        sublabel="(fit on train)",
    ))

    # Row 4: Side-by-side - Bayesian Model | MCMC Sampling
    model_x = 0.25
    mcmc_x = 0.55
    small_width = 0.18

    diagram.add_node(DiagramNode(
        id="model",
        x=model_x,
        y=y_positions[4],
        width=small_width,
        height=box_height,
        label="Bayesian Model",
        stage_type="model",
        sublabel="(hierarchical)",
    ))

    diagram.add_node(DiagramNode(
        id="mcmc",
        x=mcmc_x,
        y=y_positions[4],
        width=small_width,
        height=box_height,
        label="MCMC Sampling",
        stage_type="model",
        sublabel="(4 chains)",
    ))

    # Row 5: Evaluation
    diagram.add_node(DiagramNode(
        id="eval",
        x=center_x,
        y=y_positions[5],
        width=box_width,
        height=box_height,
        label="Evaluation",
        stage_type="validation",
        sublabel="(LOO-CV, calibration)",
    ))

    # Row 6: Predictions
    diagram.add_node(DiagramNode(
        id="predictions",
        x=center_x,
        y=y_positions[6],
        width=box_width,
        height=box_height,
        label="Predictions",
        stage_type="output",
        sublabel="(+ uncertainty)",
    ))

    # Arrows: Connect consecutive rows
    # Raw -> Clean
    diagram.add_arrow(
        (center_x + box_width / 2, y_positions[0]),
        (center_x + box_width / 2, y_positions[1] + box_height),
    )

    # Clean -> Split
    diagram.add_arrow(
        (center_x + box_width / 2, y_positions[1]),
        (center_x + box_width / 2, y_positions[2] + box_height),
    )

    # Split -> Features
    diagram.add_arrow(
        (center_x + box_width / 2, y_positions[2]),
        (center_x + box_width / 2, y_positions[3] + box_height),
    )

    # Features -> Both model boxes (split arrow)
    feat_bottom = y_positions[3]
    model_top = y_positions[4] + box_height

    # Features -> Bayesian Model
    diagram.add_arrow(
        (center_x + box_width / 2 - 0.05, feat_bottom),
        (model_x + small_width / 2, model_top),
    )

    # Features -> MCMC
    diagram.add_arrow(
        (center_x + box_width / 2 + 0.05, feat_bottom),
        (mcmc_x + small_width / 2, model_top),
    )

    # Model -> Evaluation
    diagram.add_arrow(
        (model_x + small_width / 2, y_positions[4]),
        (center_x + box_width / 2 - 0.05, y_positions[5] + box_height),
    )

    # MCMC -> Evaluation
    diagram.add_arrow(
        (mcmc_x + small_width / 2, y_positions[4]),
        (center_x + box_width / 2 + 0.05, y_positions[5] + box_height),
    )

    # Evaluation -> Predictions
    diagram.add_arrow(
        (center_x + box_width / 2, y_positions[5]),
        (center_x + box_width / 2, y_positions[6] + box_height),
    )

    # Add legend
    diagram.add_legend("lower right")

    return diagram


def create_intermediate_diagram(theme: DiagramTheme = "light") -> DataFlowDiagram:
    """Create intermediate data flow diagram (10-15 boxes) for paper figures.

    Shows pipeline stages with feature groupings and split strategies:
    - Data input with schema validation and cleaning
    - Two split strategies (temporal, artist-disjoint)
    - Feature blocks grouped by type
    - Model specification with MCMC
    - Validation with feedback loops

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
        fig_width=12,
        fig_height=10,
        title=title,
    )

    # Y positions for 10 rows
    y_pos = [0.90, 0.80, 0.70, 0.60, 0.51, 0.41, 0.31, 0.22, 0.12, 0.03]

    # Box dimensions
    box_h = 0.065
    box_w = 0.18
    small_w = 0.14
    tiny_w = 0.10

    # Row 0: Raw CSV
    diagram.add_node(DiagramNode(
        id="raw_csv",
        x=0.41,
        y=y_pos[0],
        width=box_w,
        height=box_h,
        label="Raw CSV",
        stage_type="data_input",
        sublabel="(N albums)",
    ))

    # Row 1: Schema Validation | Cleaning Rules
    diagram.add_node(DiagramNode(
        id="schema",
        x=0.22,
        y=y_pos[1],
        width=small_w,
        height=box_h,
        label="Schema Validation",
        stage_type="sanitization",
        sublabel="(Pandera)",
    ))

    diagram.add_node(DiagramNode(
        id="cleaning",
        x=0.55,
        y=y_pos[1],
        width=small_w,
        height=box_h,
        label="Cleaning Rules",
        stage_type="sanitization",
        sublabel="(exclusions)",
    ))

    # Row 2: Audit Trail
    diagram.add_node(DiagramNode(
        id="audit",
        x=0.41,
        y=y_pos[2],
        width=box_w,
        height=box_h,
        label="Audit Trail",
        stage_type="sanitization",
        sublabel="(JSONL)",
    ))

    # Row 3: Temporal Split | Artist-Disjoint Split
    diagram.add_node(DiagramNode(
        id="temporal_split",
        x=0.22,
        y=y_pos[3],
        width=small_w,
        height=box_h,
        label="Temporal Split",
        stage_type="splitting",
        sublabel="(within-artist)",
    ))

    diagram.add_node(DiagramNode(
        id="artist_split",
        x=0.55,
        y=y_pos[3],
        width=small_w,
        height=box_h,
        label="Artist-Disjoint Split",
        stage_type="splitting",
        sublabel="(holdout)",
    ))

    # Row 4: Train | Val | Test labels
    diagram.add_node(DiagramNode(
        id="train",
        x=0.15,
        y=y_pos[4],
        width=tiny_w,
        height=box_h,
        label="Train",
        stage_type="splitting",
    ))

    diagram.add_node(DiagramNode(
        id="val",
        x=0.38,
        y=y_pos[4],
        width=tiny_w,
        height=box_h,
        label="Validation",
        stage_type="splitting",
    ))

    diagram.add_node(DiagramNode(
        id="test",
        x=0.62,
        y=y_pos[4],
        width=tiny_w,
        height=box_h,
        label="Test",
        stage_type="splitting",
    ))

    # Row 5: Temporal Features | Artist Features | Categorical Features
    diagram.add_node(DiagramNode(
        id="temp_feat",
        x=0.12,
        y=y_pos[5],
        width=small_w,
        height=box_h,
        label="Temporal Features",
        stage_type="features",
        sublabel="(5 features)",
    ))

    diagram.add_node(DiagramNode(
        id="artist_feat",
        x=0.38,
        y=y_pos[5],
        width=small_w,
        height=box_h,
        label="Artist Features",
        stage_type="features",
        sublabel="(LOO stats)",
    ))

    diagram.add_node(DiagramNode(
        id="cat_feat",
        x=0.65,
        y=y_pos[5],
        width=small_w,
        height=box_h,
        label="Categorical Features",
        stage_type="features",
        sublabel="(genre, type)",
    ))

    # Row 6: Model Specification
    diagram.add_node(DiagramNode(
        id="model_spec",
        x=0.37,
        y=y_pos[6],
        width=0.20,
        height=box_h,
        label="Model Specification",
        stage_type="model",
        sublabel="(hierarchical priors)",
    ))

    # Row 7: MCMC Fitting | GPU Acceleration
    diagram.add_node(DiagramNode(
        id="mcmc_fit",
        x=0.22,
        y=y_pos[7],
        width=small_w,
        height=box_h,
        label="MCMC Fitting",
        stage_type="model",
        sublabel="(4 chains)",
    ))

    diagram.add_node(DiagramNode(
        id="gpu",
        x=0.55,
        y=y_pos[7],
        width=small_w,
        height=box_h,
        label="GPU Acceleration",
        stage_type="model",
        sublabel="(JAX/NumPyro)",
    ))

    # Row 8: Convergence | LOO-CV | Calibration
    diagram.add_node(DiagramNode(
        id="convergence",
        x=0.12,
        y=y_pos[8],
        width=tiny_w,
        height=box_h,
        label="Convergence",
        stage_type="validation",
        sublabel="(R-hat, ESS)",
    ))

    diagram.add_node(DiagramNode(
        id="loo_cv",
        x=0.38,
        y=y_pos[8],
        width=tiny_w,
        height=box_h,
        label="LOO-CV",
        stage_type="validation",
        sublabel="(PSIS)",
    ))

    diagram.add_node(DiagramNode(
        id="calibration",
        x=0.65,
        y=y_pos[8],
        width=tiny_w,
        height=box_h,
        label="Calibration",
        stage_type="validation",
        sublabel="(coverage)",
    ))

    # Row 9: Predictions + Uncertainty
    diagram.add_node(DiagramNode(
        id="predictions",
        x=0.32,
        y=y_pos[9],
        width=0.26,
        height=box_h,
        label="Predictions + Uncertainty",
        stage_type="output",
        sublabel="(credible intervals)",
    ))

    # ---- Arrows ----

    # Raw -> Schema and Cleaning
    diagram.add_arrow((0.50, y_pos[0]), (0.29, y_pos[1] + box_h))
    diagram.add_arrow((0.50, y_pos[0]), (0.62, y_pos[1] + box_h))

    # Schema & Cleaning -> Audit
    diagram.add_arrow((0.29, y_pos[1]), (0.45, y_pos[2] + box_h))
    diagram.add_arrow((0.62, y_pos[1]), (0.55, y_pos[2] + box_h))

    # Audit -> Both splits
    diagram.add_arrow((0.45, y_pos[2]), (0.29, y_pos[3] + box_h))
    diagram.add_arrow((0.55, y_pos[2]), (0.62, y_pos[3] + box_h))

    # Splits -> Train/Val/Test
    diagram.add_arrow((0.29, y_pos[3]), (0.20, y_pos[4] + box_h))
    diagram.add_arrow((0.29, y_pos[3]), (0.43, y_pos[4] + box_h))
    diagram.add_arrow((0.62, y_pos[3]), (0.67, y_pos[4] + box_h))

    # Train/Val/Test -> Feature blocks
    diagram.add_arrow((0.20, y_pos[4]), (0.19, y_pos[5] + box_h))
    diagram.add_arrow((0.43, y_pos[4]), (0.45, y_pos[5] + box_h))
    diagram.add_arrow((0.67, y_pos[4]), (0.72, y_pos[5] + box_h))

    # Feature blocks -> Model Specification
    diagram.add_arrow((0.19, y_pos[5]), (0.40, y_pos[6] + box_h))
    diagram.add_arrow((0.45, y_pos[5]), (0.47, y_pos[6] + box_h))
    diagram.add_arrow((0.72, y_pos[5]), (0.54, y_pos[6] + box_h))

    # Model Spec -> MCMC & GPU
    diagram.add_arrow((0.40, y_pos[6]), (0.29, y_pos[7] + box_h))
    diagram.add_arrow((0.54, y_pos[6]), (0.62, y_pos[7] + box_h))

    # MCMC & GPU -> Validation
    diagram.add_arrow((0.29, y_pos[7]), (0.17, y_pos[8] + box_h))
    diagram.add_arrow((0.29, y_pos[7]), (0.43, y_pos[8] + box_h))
    diagram.add_arrow((0.62, y_pos[7]), (0.70, y_pos[8] + box_h))

    # Validation -> Predictions
    diagram.add_arrow((0.17, y_pos[8]), (0.38, y_pos[9] + box_h))
    diagram.add_arrow((0.43, y_pos[8]), (0.45, y_pos[9] + box_h))
    diagram.add_arrow((0.70, y_pos[8]), (0.52, y_pos[9] + box_h))

    # Feedback loop: Validation -> Model Spec (dashed, curved)
    diagram.add_arrow(
        (0.17, y_pos[8] + box_h / 2),
        (0.37, y_pos[6] + box_h / 2),
        curved=True,
        dashed=True,
        label="refine",
    )

    # Add legend
    diagram.add_legend("lower right")

    return diagram


def create_detailed_diagram(theme: DiagramTheme = "light") -> DataFlowDiagram:
    """Create detailed data flow diagram (20+ elements) for comprehensive docs.

    Shows the complete pipeline with all 6 feature blocks explicitly:
    - Data input with full cleaning chain
    - Split strategies with manifests
    - All 6 feature blocks: Temporal, AlbumType, ArtistHistory, ArtistReputation, Genre, Collaboration
    - Model architecture details
    - Full validation suite with feedback loops

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
    title = "AOTY Prediction Pipeline (Detailed)" if theme != "transparent" else None

    diagram = DataFlowDiagram(
        detail_level="detailed",
        theme=theme,
        fig_width=14,
        fig_height=11,
        title=title,
    )

    # Y positions for detailed layout (more rows)
    # Data section: 0.92, 0.84, 0.77
    # Split section: 0.68, 0.60
    # Feature section: 0.50, 0.41
    # Model section: 0.31, 0.23
    # Validation section: 0.13
    # Output section: 0.03

    # Box dimensions for detailed view
    box_h = 0.055
    std_w = 0.13
    small_w = 0.11
    tiny_w = 0.09

    # ============ DATA INPUT SECTION ============
    # Row: Raw CSV
    diagram.add_node(DiagramNode(
        id="raw_csv",
        x=0.44,
        y=0.92,
        width=std_w,
        height=box_h,
        label="Raw CSV",
        stage_type="data_input",
        sublabel="(all_albums_full.csv)",
    ))

    # Row: Schema Validation | Cleaning Rules | Min Ratings
    diagram.add_node(DiagramNode(
        id="schema_val",
        x=0.15,
        y=0.84,
        width=small_w,
        height=box_h,
        label="Schema Validation",
        stage_type="sanitization",
        sublabel="(Pandera)",
    ))

    diagram.add_node(DiagramNode(
        id="clean_rules",
        x=0.40,
        y=0.84,
        width=small_w,
        height=box_h,
        label="Cleaning Rules",
        stage_type="sanitization",
        sublabel="(nulls, dates)",
    ))

    diagram.add_node(DiagramNode(
        id="min_ratings",
        x=0.65,
        y=0.84,
        width=small_w,
        height=box_h,
        label="Min Ratings Filter",
        stage_type="sanitization",
        sublabel="(threshold=10)",
    ))

    # Row: Audit Trail
    diagram.add_node(DiagramNode(
        id="audit_trail",
        x=0.44,
        y=0.77,
        width=std_w,
        height=box_h,
        label="Audit Trail",
        stage_type="sanitization",
        sublabel="(data/audit/*.jsonl)",
    ))

    # ============ SPLITTING SECTION ============
    # Row: Split strategies
    diagram.add_node(DiagramNode(
        id="within_artist",
        x=0.18,
        y=0.68,
        width=std_w,
        height=box_h,
        label="Within-Artist Split",
        stage_type="splitting",
        sublabel="(temporal ordering)",
    ))

    diagram.add_node(DiagramNode(
        id="artist_disjoint",
        x=0.52,
        y=0.68,
        width=std_w,
        height=box_h,
        label="Artist-Disjoint Split",
        stage_type="splitting",
        sublabel="(new artist holdout)",
    ))

    diagram.add_node(DiagramNode(
        id="manifests",
        x=0.78,
        y=0.68,
        width=small_w,
        height=box_h,
        label="Split Manifests",
        stage_type="splitting",
        sublabel="(hash verification)",
    ))

    # Row: Train/Val/Test sets
    diagram.add_node(DiagramNode(
        id="train_set",
        x=0.15,
        y=0.60,
        width=tiny_w,
        height=box_h,
        label="Train",
        stage_type="splitting",
        sublabel="(N_train)",
    ))

    diagram.add_node(DiagramNode(
        id="val_set",
        x=0.35,
        y=0.60,
        width=tiny_w,
        height=box_h,
        label="Validation",
        stage_type="splitting",
        sublabel="(N_val)",
    ))

    diagram.add_node(DiagramNode(
        id="test_set",
        x=0.55,
        y=0.60,
        width=tiny_w,
        height=box_h,
        label="Test",
        stage_type="splitting",
        sublabel="(N_test)",
    ))

    # ============ FEATURE SECTION ============
    # Row: All 6 feature blocks
    diagram.add_node(DiagramNode(
        id="temporal_block",
        x=0.05,
        y=0.50,
        width=small_w,
        height=box_h,
        label="TemporalBlock",
        stage_type="features",
        sublabel="(5 temporal)",
    ))

    diagram.add_node(DiagramNode(
        id="album_type_block",
        x=0.19,
        y=0.50,
        width=small_w,
        height=box_h,
        label="AlbumTypeBlock",
        stage_type="features",
        sublabel="(one-hot)",
    ))

    diagram.add_node(DiagramNode(
        id="artist_history_block",
        x=0.33,
        y=0.50,
        width=small_w,
        height=box_h,
        label="ArtistHistoryBlock",
        stage_type="features",
        sublabel="(LOO stats)",
    ))

    diagram.add_node(DiagramNode(
        id="artist_rep_block",
        x=0.47,
        y=0.50,
        width=small_w,
        height=box_h,
        label="ArtistReputationBlock",
        stage_type="features",
        sublabel="(prior mean)",
    ))

    diagram.add_node(DiagramNode(
        id="genre_block",
        x=0.61,
        y=0.50,
        width=small_w,
        height=box_h,
        label="GenreBlock",
        stage_type="features",
        sublabel="(PCA)",
    ))

    diagram.add_node(DiagramNode(
        id="collab_block",
        x=0.75,
        y=0.50,
        width=small_w,
        height=box_h,
        label="CollaborationBlock",
        stage_type="features",
        sublabel="(ordinal)",
    ))

    # Row: FeaturePipeline
    diagram.add_node(DiagramNode(
        id="feature_pipeline",
        x=0.35,
        y=0.41,
        width=0.20,
        height=box_h,
        label="FeaturePipeline",
        stage_type="features",
        sublabel="(fit on train, transform all)",
    ))

    # ============ MODEL SECTION ============
    # Row: Model components
    diagram.add_node(DiagramNode(
        id="prior_config",
        x=0.08,
        y=0.31,
        width=small_w,
        height=box_h,
        label="PriorConfig",
        stage_type="model",
        sublabel="(9 hyperparams)",
    ))

    diagram.add_node(DiagramNode(
        id="hierarchical",
        x=0.24,
        y=0.31,
        width=small_w,
        height=box_h,
        label="Hierarchical Model",
        stage_type="model",
        sublabel="(artist effects)",
    ))

    diagram.add_node(DiagramNode(
        id="non_centered",
        x=0.40,
        y=0.31,
        width=small_w,
        height=box_h,
        label="Non-centered Param",
        stage_type="model",
        sublabel="(LocScaleReparam)",
    ))

    diagram.add_node(DiagramNode(
        id="time_varying",
        x=0.56,
        y=0.31,
        width=small_w,
        height=box_h,
        label="Time-varying Effects",
        stage_type="model",
        sublabel="(random walk)",
    ))

    diagram.add_node(DiagramNode(
        id="ar1",
        x=0.72,
        y=0.31,
        width=small_w,
        height=box_h,
        label="AR(1) Structure",
        stage_type="model",
        sublabel="(stationary)",
    ))

    # Row: MCMC
    diagram.add_node(DiagramNode(
        id="mcmc_sampling",
        x=0.35,
        y=0.23,
        width=0.20,
        height=box_h,
        label="MCMC Sampling",
        stage_type="model",
        sublabel="(4 chains, GPU)",
    ))

    # ============ VALIDATION SECTION ============
    # Row: Convergence checks
    diagram.add_node(DiagramNode(
        id="rhat",
        x=0.05,
        y=0.13,
        width=tiny_w,
        height=box_h,
        label="R-hat",
        stage_type="validation",
        sublabel="(< 1.01)",
    ))

    diagram.add_node(DiagramNode(
        id="ess",
        x=0.17,
        y=0.13,
        width=tiny_w,
        height=box_h,
        label="ESS",
        stage_type="validation",
        sublabel="(> 400)",
    ))

    diagram.add_node(DiagramNode(
        id="divergences",
        x=0.29,
        y=0.13,
        width=tiny_w,
        height=box_h,
        label="Divergences",
        stage_type="validation",
        sublabel="(= 0)",
    ))

    diagram.add_node(DiagramNode(
        id="loo_psis",
        x=0.41,
        y=0.13,
        width=tiny_w,
        height=box_h,
        label="LOO-CV",
        stage_type="validation",
        sublabel="(PSIS)",
    ))

    diagram.add_node(DiagramNode(
        id="pareto_k",
        x=0.53,
        y=0.13,
        width=tiny_w,
        height=box_h,
        label="Pareto-k",
        stage_type="validation",
        sublabel="(< 0.7)",
    ))

    diagram.add_node(DiagramNode(
        id="calib",
        x=0.65,
        y=0.13,
        width=tiny_w,
        height=box_h,
        label="Calibration",
        stage_type="validation",
        sublabel="(coverage)",
    ))

    diagram.add_node(DiagramNode(
        id="sensitivity",
        x=0.77,
        y=0.13,
        width=tiny_w,
        height=box_h,
        label="Sensitivity",
        stage_type="validation",
        sublabel="(priors)",
    ))

    # ============ OUTPUT SECTION ============
    # Row: Outputs
    diagram.add_node(DiagramNode(
        id="predictions",
        x=0.15,
        y=0.03,
        width=small_w,
        height=box_h,
        label="Predictions",
        stage_type="output",
        sublabel="(posterior mean)",
    ))

    diagram.add_node(DiagramNode(
        id="uncertainty",
        x=0.35,
        y=0.03,
        width=small_w,
        height=box_h,
        label="Uncertainty Bands",
        stage_type="output",
        sublabel="(95% CI)",
    ))

    diagram.add_node(DiagramNode(
        id="pub_artifacts",
        x=0.55,
        y=0.03,
        width=std_w,
        height=box_h,
        label="Publication Artifacts",
        stage_type="output",
        sublabel="(figures, tables)",
    ))

    # ============ ARROWS ============

    # Data section
    diagram.add_arrow((0.50, 0.92), (0.21, 0.84 + box_h))  # Raw -> Schema
    diagram.add_arrow((0.50, 0.92), (0.46, 0.84 + box_h))  # Raw -> Cleaning
    diagram.add_arrow((0.50, 0.92), (0.71, 0.84 + box_h))  # Raw -> Min Ratings

    diagram.add_arrow((0.21, 0.84), (0.46, 0.77 + box_h))  # Schema -> Audit
    diagram.add_arrow((0.46, 0.84), (0.50, 0.77 + box_h))  # Cleaning -> Audit
    diagram.add_arrow((0.71, 0.84), (0.54, 0.77 + box_h))  # Min Ratings -> Audit

    # Audit -> Splits
    diagram.add_arrow((0.46, 0.77), (0.25, 0.68 + box_h))  # -> Within-artist
    diagram.add_arrow((0.54, 0.77), (0.58, 0.68 + box_h))  # -> Artist-disjoint
    diagram.add_arrow((0.54, 0.77), (0.84, 0.68 + box_h))  # -> Manifests

    # Splits -> Train/Val/Test
    diagram.add_arrow((0.25, 0.68), (0.20, 0.60 + box_h))  # Within-artist -> Train
    diagram.add_arrow((0.25, 0.68), (0.40, 0.60 + box_h))  # Within-artist -> Val
    diagram.add_arrow((0.58, 0.68), (0.60, 0.60 + box_h))  # Artist-disjoint -> Test

    # Train/Val/Test -> Feature blocks
    diagram.add_arrow((0.20, 0.60), (0.11, 0.50 + box_h))  # -> Temporal
    diagram.add_arrow((0.20, 0.60), (0.25, 0.50 + box_h))  # -> AlbumType
    diagram.add_arrow((0.40, 0.60), (0.39, 0.50 + box_h))  # -> ArtistHistory
    diagram.add_arrow((0.40, 0.60), (0.53, 0.50 + box_h))  # -> ArtistRep
    diagram.add_arrow((0.60, 0.60), (0.67, 0.50 + box_h))  # -> Genre
    diagram.add_arrow((0.60, 0.60), (0.81, 0.50 + box_h))  # -> Collab

    # Feature blocks -> FeaturePipeline
    diagram.add_arrow((0.11, 0.50), (0.38, 0.41 + box_h))
    diagram.add_arrow((0.25, 0.50), (0.42, 0.41 + box_h))
    diagram.add_arrow((0.39, 0.50), (0.45, 0.41 + box_h))
    diagram.add_arrow((0.53, 0.50), (0.48, 0.41 + box_h))
    diagram.add_arrow((0.67, 0.50), (0.52, 0.41 + box_h))
    diagram.add_arrow((0.81, 0.50), (0.55, 0.41 + box_h))

    # FeaturePipeline -> Model components
    diagram.add_arrow((0.38, 0.41), (0.14, 0.31 + box_h))  # -> PriorConfig
    diagram.add_arrow((0.42, 0.41), (0.30, 0.31 + box_h))  # -> Hierarchical
    diagram.add_arrow((0.45, 0.41), (0.46, 0.31 + box_h))  # -> Non-centered
    diagram.add_arrow((0.48, 0.41), (0.62, 0.31 + box_h))  # -> Time-varying
    diagram.add_arrow((0.52, 0.41), (0.78, 0.31 + box_h))  # -> AR1

    # Model components -> MCMC
    diagram.add_arrow((0.14, 0.31), (0.38, 0.23 + box_h))
    diagram.add_arrow((0.30, 0.31), (0.42, 0.23 + box_h))
    diagram.add_arrow((0.46, 0.31), (0.45, 0.23 + box_h))
    diagram.add_arrow((0.62, 0.31), (0.48, 0.23 + box_h))
    diagram.add_arrow((0.78, 0.31), (0.52, 0.23 + box_h))

    # MCMC -> Validation
    diagram.add_arrow((0.38, 0.23), (0.10, 0.13 + box_h))  # -> R-hat
    diagram.add_arrow((0.40, 0.23), (0.22, 0.13 + box_h))  # -> ESS
    diagram.add_arrow((0.42, 0.23), (0.34, 0.13 + box_h))  # -> Divergences
    diagram.add_arrow((0.45, 0.23), (0.46, 0.13 + box_h))  # -> LOO-CV
    diagram.add_arrow((0.48, 0.23), (0.58, 0.13 + box_h))  # -> Pareto-k
    diagram.add_arrow((0.50, 0.23), (0.70, 0.13 + box_h))  # -> Calibration
    diagram.add_arrow((0.52, 0.23), (0.82, 0.13 + box_h))  # -> Sensitivity

    # Validation -> Outputs
    diagram.add_arrow((0.10, 0.13), (0.21, 0.03 + box_h))  # -> Predictions
    diagram.add_arrow((0.22, 0.13), (0.41, 0.03 + box_h))  # -> Uncertainty
    diagram.add_arrow((0.70, 0.13), (0.61, 0.03 + box_h))  # -> Pub Artifacts

    # Feedback loops (dashed, curved)
    # Sensitivity -> PriorConfig (retrain loop)
    diagram.add_arrow(
        (0.82, 0.13 + box_h / 2),
        (0.14, 0.31 + box_h / 2),
        curved=True,
        dashed=True,
        label="tune priors",
    )

    # Calibration -> Model (adjust loop)
    diagram.add_arrow(
        (0.70, 0.13 + box_h / 2),
        (0.30, 0.31 + box_h / 2),
        curved=True,
        dashed=True,
        label="adjust",
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
