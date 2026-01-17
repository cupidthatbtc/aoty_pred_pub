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
