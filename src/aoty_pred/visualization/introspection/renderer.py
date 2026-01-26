"""Diagram renderer for producing Graphviz output from DiagramData.

This module provides DiagramRenderer which transforms aggregated DiagramData
into publication-quality Graphviz diagrams with academic styling, clustered
subgraphs, and theme support.

The renderer follows the styling patterns established in diagrams.py:
- Academic fonts (Times-Roman)
- Clean polyline splines
- Clustered subgraphs with minimal borders
- Theme-aware colors (light/dark/transparent)

Example:
    >>> from aoty_pred.visualization.introspection import (
    ...     StageIntrospector, DiagramAggregator, DiagramRenderer
    ... )
    >>> result = StageIntrospector().introspect()
    >>> data = DiagramAggregator().aggregate([result])
    >>> renderer = DiagramRenderer()
    >>> graph = renderer.render(data)
    >>> graph.render("pipeline", format="svg")
"""

from __future__ import annotations

import graphviz

from aoty_pred.visualization.diagrams import DiagramTheme
from aoty_pred.visualization.introspection.aggregator import DiagramData
from aoty_pred.visualization.introspection.base import EdgeSpec, NodeSpec
from aoty_pred.visualization.introspection.diagram_builder import (
    SECTION_HIERARCHY,
)

__all__ = [
    "DiagramRenderer",
    "EXTENDED_THEME_COLORS",
    "FEEDBACK_STYLES",
    "FEEDBACK_LOOPS",
    "LEGEND_SHAPES",
    "LEGEND_EDGE_TYPES",
]

# Extended theme colors with 10-section pastel fills and 9 edge type colors.
# Each section from SECTION_HIERARCHY gets a distinct fill color.
# Edge types include: data_flow, secondary, fit_only, retry, tune, refit, ablate
EXTENDED_THEME_COLORS: dict[str, dict[str, str]] = {
    "light": {
        # Base colors (from existing THEME_COLORS)
        "bgcolor": "#FFFFFF",
        "fontcolor": "#000000",
        "color": "#333333",
        "fillcolor": "#FFFFFF",
        # Section cluster fills - 10 sections (pastel for print-safe B&W)
        "CONFIG_fill": "#F5F5F5",  # Gray - neutral
        "DATA_RAW_fill": "#E8EEF4",  # Blue-gray - external
        "DATA_CLEAN_fill": "#FFF9E6",  # Yellow - validation
        "DATA_SPLIT_fill": "#E6F4F4",  # Cyan - division
        "FEATURES_fill": "#E8F5E8",  # Green - engineering
        "PRIORS_fill": "#F4E8F4",  # Lavender - statistical
        "MODEL_fill": "#E6EEF8",  # Blue - core
        "CONVERGENCE_fill": "#FFF0E0",  # Peach - diagnostics
        "EVALUATION_fill": "#F8E8F0",  # Pink - assessment
        "OUTPUT_fill": "#E0F0E0",  # Mint - results
        # Edge colors for 9 edge types
        "edge_data_flow": "#333333",
        "edge_secondary": "#666666",
        "edge_fit_only": "#2E7D32",
        "edge_retry": "#E67E22",
        "edge_tune": "#9B59B6",
        "edge_refit": "#3498DB",
        "edge_ablate": "#E74C3C",
        "edge_cross_ref": "#888888",
        "edge_annotation": "#999999",
        # Legacy edge colors for backward compatibility
        "edge_primary": "#333333",
    },
    "dark": {
        "bgcolor": "#1A1A1A",
        "fontcolor": "#E8E8E8",
        "color": "#808080",
        "fillcolor": "#2A2A2A",
        # Section fills - darkened (15-20% lightness, <30% saturation)
        "CONFIG_fill": "#2A2A2A",
        "DATA_RAW_fill": "#252A30",
        "DATA_CLEAN_fill": "#2A2820",
        "DATA_SPLIT_fill": "#202A2A",
        "FEATURES_fill": "#202A20",
        "PRIORS_fill": "#28202A",
        "MODEL_fill": "#202530",
        "CONVERGENCE_fill": "#2A2520",
        "EVALUATION_fill": "#2A2028",
        "OUTPUT_fill": "#202A20",
        # Edge colors - brighter for dark background
        "edge_data_flow": "#B0B0B0",
        "edge_secondary": "#808080",
        "edge_fit_only": "#4CAF50",
        "edge_retry": "#F39C12",
        "edge_tune": "#BB8FCE",
        "edge_refit": "#5DADE2",
        "edge_ablate": "#EC7063",
        "edge_cross_ref": "#A0A0A0",
        "edge_annotation": "#909090",
        # Legacy edge colors for backward compatibility
        "edge_primary": "#B0B0B0",
    },
    "transparent": {
        "bgcolor": "transparent",
        "fontcolor": "#000000",
        "color": "#333333",
        "fillcolor": "#FFFFFF",
        # Lighter fills for overlay use (50% lighter than light theme)
        "CONFIG_fill": "#FAFAFA",
        "DATA_RAW_fill": "#F0F4F8",
        "DATA_CLEAN_fill": "#FFFCF0",
        "DATA_SPLIT_fill": "#F0FAFA",
        "FEATURES_fill": "#F0FAF0",
        "PRIORS_fill": "#F8F0F8",
        "MODEL_fill": "#F0F4FC",
        "CONVERGENCE_fill": "#FFF8F0",
        "EVALUATION_fill": "#FCF0F8",
        "OUTPUT_fill": "#F0F8F0",
        # Same edge colors as light
        "edge_data_flow": "#333333",
        "edge_secondary": "#666666",
        "edge_fit_only": "#2E7D32",
        "edge_retry": "#E67E22",
        "edge_tune": "#9B59B6",
        "edge_refit": "#3498DB",
        "edge_ablate": "#E74C3C",
        "edge_cross_ref": "#888888",
        "edge_annotation": "#999999",
        # Legacy edge colors for backward compatibility
        "edge_primary": "#333333",
    },
}

# Category to shape mapping for nodes
CATEGORY_SHAPES: dict[str, str] = {
    "process": "box",
    "data": "ellipse",
    "decision": "diamond",
    "config": "note",
    "storage": "folder",
}

# Feedback loop styles - color palette for 4 feedback loop types
FEEDBACK_STYLES: dict[str, dict[str, str]] = {
    "retry": {"color": "#E67E22", "style": "dashed", "label": "retry"},
    "tune": {"color": "#9B59B6", "style": "dotted", "label": "tune"},
    "refit": {"color": "#3498DB", "style": "dashed", "label": "refit"},
    "ablate": {"color": "#E74C3C", "style": "dashed", "label": "ablate"},
}

# Feedback loop edge definitions - 4 loops from evaluation back to earlier stages
FEEDBACK_LOOPS: list[dict[str, str]] = [
    {"source": "stage:evaluate", "target": "stage:train", "type": "retry"},
    {"source": "stage:evaluate", "target": "prior:sigma_artist_scale", "type": "tune"},
    {"source": "stage:evaluate", "target": "stage:train", "type": "refit"},
    {"source": "stage:evaluate", "target": "stage:features", "type": "ablate"},
]

# Legend shape meanings for user reference
LEGEND_SHAPES: dict[str, str] = {
    "box": "Process",
    "ellipse": "Data",
    "diamond": "Decision",
    "note": "Config",
    "folder": "Storage",
}

# Legend edge type meanings: (display label, style, color)
LEGEND_EDGE_TYPES: dict[str, tuple[str, str, str]] = {
    "flow": ("Data flow", "solid", "#333333"),
    "retry": ("Retry", "dashed", "#E67E22"),
    "tune": ("Tune", "dotted", "#9B59B6"),
    "refit": ("Refit", "dashed", "#3498DB"),
    "ablate": ("Ablate", "dashed", "#E74C3C"),
}

# Category to fill color key mapping for nodes
CATEGORY_FILLS: dict[str, str] = {
    "process": "fillcolor",  # Default fill
    "data": "storage_fill",
    "decision": "decision_fill",
    "config": "note_fill",
}

# Edge category to style mapping
EDGE_CATEGORY_STYLES: dict[str, dict[str, str]] = {
    "flow": {"style": "solid"},
    "dependency": {"style": "solid"},
    "feedback": {"style": "dashed"},
}


def _sanitize_node_id(node_id: str) -> str:
    """Sanitize node ID for Graphviz compatibility.

    Replaces characters that Graphviz interprets as special (like colons
    for port separators) with safe alternatives.

    Args:
        node_id: Original node ID from introspection.

    Returns:
        Sanitized ID safe for Graphviz DOT format.
    """
    # Replace colons with underscores - Graphviz interprets : as port separator
    return node_id.replace(":", "_")


class DiagramRenderer:
    """Renderer for producing Graphviz diagrams from DiagramData.

    Transforms aggregated diagram data into publication-quality Graphviz
    output with academic styling. Supports theme variants (light/dark/transparent)
    and automatically handles clustered subgraphs.

    Styling follows the academic conventions from diagrams.py:
    - Times-Roman serif fonts for professional appearance
    - Polyline splines for clean angled connectors
    - Thin lines (1.0 penwidth) for academic look
    - Clustered subgraphs with dashed borders

    Attributes:
        theme: Visual theme for diagram colors.
        colors: Color palette for the selected theme.

    Example:
        >>> renderer = DiagramRenderer(theme="light")
        >>> data = DiagramData(nodes={}, edges=[], clusters={})
        >>> graph = renderer.render(data)
        >>> svg = graph.pipe(format="svg").decode("utf-8")
    """

    def __init__(self, theme: DiagramTheme = "light") -> None:
        """Initialize renderer with theme.

        Args:
            theme: Visual theme for colors. Options:
                - "light": White background, dark text (print-friendly)
                - "dark": Dark background, light text (presentations)
                - "transparent": No background (embedding)
        """
        self.theme = theme
        self.colors = EXTENDED_THEME_COLORS[theme]

    def render(self, data: DiagramData) -> graphviz.Digraph:
        """Render DiagramData to Graphviz Digraph.

        Creates a directed graph with:
        - Academic styling (serif fonts, clean lines)
        - 2-level nested clustered subgraphs for hierarchical grouping
        - Category-based node shapes and colors
        - Edge styles based on category
        - Feedback loop edges with constraint=false
        - Embedded legend showing node shapes and edge types

        Args:
            data: Aggregated diagram data from DiagramAggregator.

        Returns:
            Configured graphviz.Digraph ready for rendering to SVG/PNG/PDF.

        Example:
            >>> data = DiagramData(
            ...     nodes={"a": NodeSpec(id="a", label="A", category="process")},
            ...     edges=[],
            ...     clusters={},
            ... )
            >>> renderer = DiagramRenderer()
            >>> graph = renderer.render(data)
            >>> svg = graph.pipe(format="svg")
        """
        graph = graphviz.Digraph(
            name="IntrospectedPipeline",
            format="svg",
            engine="dot",
        )

        # Academic styling: serif fonts, clean layout
        graph.attr(
            rankdir="TB",
            fontname="Times-Roman",
            fontsize="14",
            label="Pipeline Diagram",
            labelloc="t",
            labeljust="c",
            pad="0.5",
            nodesep="0.6",
            ranksep="0.8",
            splines="polyline",
            compound="true",
            ordering="out",
            remincross="true",
        )

        # Set bgcolor only for non-transparent themes
        if self.theme != "transparent":
            graph.attr(bgcolor=self.colors["bgcolor"])

        # Default node style
        graph.attr(
            "node",
            fontname="Times-Roman",
            fontsize="10",
            shape="box",
            style="filled,rounded",
            fillcolor=self.colors["fillcolor"],
            color=self.colors["color"],
            fontcolor=self.colors["fontcolor"],
            penwidth="1.0",
            margin="0.15,0.10",
        )

        # Default edge style
        graph.attr(
            "edge",
            fontname="Times-Roman",
            fontsize="8",
            color=self.colors["edge_primary"],
            fontcolor=self.colors["fontcolor"],
            penwidth="1.0",
            arrowsize="0.7",
        )

        # Track shapes and edge types used for legend
        shapes_used: set[str] = set()
        edge_types_used: set[str] = set()

        # Collect shapes used by nodes
        for node in data.nodes.values():
            shape = CATEGORY_SHAPES.get(node.category, "box")
            shapes_used.add(shape)

        # Collect edge types used
        for edge in data.edges:
            edge_types_used.add(edge.category)

        # Render nested clusters using section hierarchy
        clustered_nodes = self._render_nested_clusters(graph, data)

        # Render unclustered nodes in main graph
        for node_id in sorted(data.nodes.keys()):
            if node_id not in clustered_nodes:
                self._add_node(graph, data.nodes[node_id])

        # Render edges
        for edge in data.edges:
            self._add_edge(graph, edge)

        # Add feedback loops with constraint=false
        self._add_feedback_loops(graph)
        for loop in FEEDBACK_LOOPS:
            edge_types_used.add(loop["type"])

        # Add legend cluster
        self._add_legend(graph, shapes_used, edge_types_used)

        return graph

    def _add_node(
        self,
        graph: graphviz.Digraph,
        node: NodeSpec,
    ) -> None:
        """Add a node to the graph with category-based styling.

        Args:
            graph: Graph or subgraph to add node to.
            node: NodeSpec to render.
        """
        if not isinstance(node, NodeSpec):
            return

        # Determine shape from category
        shape = CATEGORY_SHAPES.get(node.category, "box")

        # Determine fill color from category
        fill_key = CATEGORY_FILLS.get(node.category, "fillcolor")
        fillcolor = self.colors.get(fill_key, self.colors["fillcolor"])

        # Convert label newlines to Graphviz format
        label = node.label.replace("\n", "\\n")

        # Replace colons with underscores to prevent Graphviz port interpretation
        # The graphviz library auto-quotes but edges don't get the same treatment
        node_id = _sanitize_node_id(node.id)

        graph.node(
            node_id,
            label=label,
            shape=shape,
            fillcolor=fillcolor,
        )

    def _add_edge(
        self,
        graph: graphviz.Digraph,
        edge: EdgeSpec,
    ) -> None:
        """Add an edge to the graph with category-based styling.

        Args:
            graph: Graph to add edge to.
            edge: EdgeSpec to render.
        """
        if not isinstance(edge, EdgeSpec):
            return

        # Get style from category
        style_attrs = EDGE_CATEGORY_STYLES.get(edge.category, {"style": "solid"})

        # Build edge attributes
        attrs: dict[str, str] = {
            "style": style_attrs.get("style", edge.style),
        }

        if edge.label:
            attrs["label"] = edge.label

        # Replace colons with underscores to prevent Graphviz port interpretation
        source = _sanitize_node_id(edge.source)
        target = _sanitize_node_id(edge.target)

        graph.edge(source, target, **attrs)

    def _render_nested_clusters(
        self,
        graph: graphviz.Digraph,
        data: DiagramData,
    ) -> set[str]:
        """Render 2-level nested cluster hierarchy.

        Creates major section clusters containing sub-clusters.
        Uses SECTION_HIERARCHY to determine nesting structure.
        Uses section-specific fill colors from EXTENDED_THEME_COLORS.

        Args:
            graph: Graphviz graph to add clusters to.
            data: Diagram data containing nodes and clusters.

        Returns:
            Set of node IDs that were rendered in clusters.
        """
        clustered_nodes: set[str] = set()

        # Build mapping from cluster name to its node IDs
        cluster_to_nodes: dict[str, list[str]] = {}
        for cluster_name, node_ids in data.clusters.items():
            cluster_to_nodes[cluster_name] = list(node_ids)

        for major_section in SECTION_HIERARCHY:
            sub_clusters = SECTION_HIERARCHY[major_section]
            # Use section-specific fill from extended colors
            section_fill_key = f"{major_section}_fill"
            section_color = self.colors.get(section_fill_key, self.colors["fillcolor"])

            # Find all clusters belonging to this major section
            # They start with the section name (e.g., PRIORS_artist_pooling for PRIORS)
            section_clusters: list[str] = []
            for cluster_name in cluster_to_nodes:
                # Exact match for flat sections (e.g., CONFIG)
                if cluster_name == major_section:
                    section_clusters.append(cluster_name)
                # Prefix match for nested sections (e.g., PRIORS_artist_pooling)
                elif cluster_name.startswith(f"{major_section}_"):
                    section_clusters.append(cluster_name)

            if not section_clusters:
                continue

            with graph.subgraph(name=f"cluster_{major_section}") as major:
                major.attr(
                    label=major_section.replace("_", " ").title(),
                    style="rounded,filled",
                    fillcolor=section_color,
                    color="#888888",
                    penwidth="1.5",
                    fontname="Times-Roman",
                    fontsize="11",
                    labeljust="l",
                    labelloc="t",
                )

                if sub_clusters:
                    # Create nested sub-clusters for sections with hierarchy
                    for sub_name in sub_clusters:
                        full_cluster_name = f"{major_section}_{sub_name}"
                        if full_cluster_name not in cluster_to_nodes:
                            continue

                        node_ids = cluster_to_nodes[full_cluster_name]
                        if not node_ids:
                            continue

                        # Lighter fill for sub-clusters
                        sub_fill = self._lighten_color(section_color)

                        with major.subgraph(name=f"cluster_{full_cluster_name}") as sub:
                            sub.attr(
                                label=sub_name.replace("_", " ").title(),
                                style="rounded,filled",
                                fillcolor=sub_fill,
                                color="#AAAAAA",
                                penwidth="1.0",
                                fontname="Times-Roman",
                                fontsize="9",
                                labeljust="r",
                                labelloc="b",
                            )

                            for node_id in sorted(node_ids):
                                if node_id in data.nodes:
                                    self._add_node(sub, data.nodes[node_id])
                                    clustered_nodes.add(node_id)
                else:
                    # Flat section - add nodes directly to major cluster
                    for cluster_name in section_clusters:
                        for node_id in sorted(cluster_to_nodes[cluster_name]):
                            if node_id in data.nodes:
                                self._add_node(major, data.nodes[node_id])
                                clustered_nodes.add(node_id)

        return clustered_nodes

    def _lighten_color(self, hex_color: str) -> str:
        """Lighten a hex color by mixing with white.

        Args:
            hex_color: Hex color string (e.g., "#F4E8F4").

        Returns:
            Lightened hex color string.
        """
        # Parse hex color
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        # Mix with white (factor of 0.5)
        factor = 0.5
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)

        return f"#{r:02X}{g:02X}{b:02X}"

    def _add_feedback_loops(self, graph: graphviz.Digraph) -> None:
        """Add feedback loop edges with constraint=false.

        Adds 4 feedback loops from evaluation back to earlier stages:
        - retry: evaluation -> train (retry on failure)
        - tune: evaluation -> prior (tune hyperparameters)
        - refit: evaluation -> train (refit with new data)
        - ablate: evaluation -> features (ablation studies)

        Args:
            graph: Graphviz graph to add edges to.
        """
        for loop in FEEDBACK_LOOPS:
            style = FEEDBACK_STYLES[loop["type"]]
            source = _sanitize_node_id(loop["source"])
            target = _sanitize_node_id(loop["target"])
            graph.edge(
                source,
                target,
                label=style["label"],
                style=style["style"],
                color=style["color"],
                constraint="false",  # Critical: don't affect layout
                fontsize="8",
                fontcolor=style["color"],
            )

    def _add_legend(
        self,
        graph: graphviz.Digraph,
        shapes_used: set[str],
        edge_types_used: set[str],
    ) -> None:
        """Add legend cluster showing node shapes and edge types.

        Creates a cluster subgraph containing:
        - Sample nodes for each shape used
        - Sample edges for each edge type used

        Args:
            graph: Graphviz graph to add legend to.
            shapes_used: Set of shapes used in the diagram.
            edge_types_used: Set of edge categories used in the diagram.
        """
        with graph.subgraph(name="cluster_legend") as legend:
            legend.attr(
                label="Legend",
                labelloc="t",
                labeljust="r",
                style="rounded,filled",
                fillcolor="#FAFAFA",
                color="#CCCCCC",
                fontname="Times-Roman",
                fontsize="10",
                penwidth="1.0",
            )

            # Add shape samples (only shapes actually used)
            shape_idx = 0
            for shape, meaning in LEGEND_SHAPES.items():
                if shape in shapes_used:
                    legend.node(
                        f"legend_shape_{shape_idx}",
                        label=meaning,
                        shape=shape,
                        fontsize="8",
                        width="0.6",
                        height="0.25",
                        style="filled",
                        fillcolor="#FFFFFF",
                    )
                    shape_idx += 1

            # Add edge type samples
            for etype, (meaning, style, color) in LEGEND_EDGE_TYPES.items():
                if etype in edge_types_used or etype == "flow":
                    legend.node(
                        f"legend_{etype}_a",
                        label="",
                        shape="point",
                        width="0.1",
                        height="0.1",
                    )
                    legend.node(
                        f"legend_{etype}_b",
                        label=meaning,
                        shape="plaintext",
                        fontsize="8",
                    )
                    legend.edge(
                        f"legend_{etype}_a",
                        f"legend_{etype}_b",
                        style=style,
                        color=color,
                    )
