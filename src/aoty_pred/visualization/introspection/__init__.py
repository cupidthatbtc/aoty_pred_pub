"""Introspection infrastructure for ultra-detailed pipeline diagrams.

This package provides the v4.0 introspection architecture for generating
publication-quality pipeline diagrams from live code and data introspection.

The architecture follows a three-stage pipeline:
    1. Introspection: Extract metadata from code/data sources
    2. Aggregation: Merge multiple results into unified diagram data
    3. Rendering: Produce Graphviz output with academic styling

Core Components:
    - NodeSpec, EdgeSpec: Data structures for diagram elements
    - IntrospectionResult: Container for introspection output
    - Introspector: Protocol for pluggable introspectors
    - StageIntrospector: Proof-of-concept introspector for pipeline stages
    - DiagramAggregator: Merges multiple IntrospectionResults
    - DiagramData: Unified data structure for rendering
    - DiagramRenderer: Produces clustered Graphviz output

Example:
    >>> from aoty_pred.visualization.introspection import (
    ...     StageIntrospector, DiagramAggregator, DiagramRenderer
    ... )
    >>> # Introspect
    >>> si = StageIntrospector()
    >>> result = si.introspect()
    >>> # Aggregate
    >>> agg = DiagramAggregator()
    >>> data = agg.aggregate([result])
    >>> # Render
    >>> renderer = DiagramRenderer()
    >>> graph = renderer.render(data)
    >>> graph.render("pipeline", format="svg")
"""

from __future__ import annotations

from aoty_pred.visualization.introspection.aggregator import (
    DATA_FLOW_EDGES,
    DiagramAggregator,
    DiagramData,
)
from aoty_pred.visualization.introspection.base import (
    EdgeSpec,
    IntrospectionResult,
    Introspector,
    NodeSpec,
)
from aoty_pred.visualization.introspection.cleaning_introspector import (
    CleaningIntrospector,
)
from aoty_pred.visualization.introspection.config_introspector import (
    ConfigIntrospector,
)
from aoty_pred.visualization.introspection.csv_introspector import CSVIntrospector
from aoty_pred.visualization.introspection.diagram_builder import (
    SECTION_COLORS,
    SECTION_HIERARCHY,
    StructuredDiagramBuilder,
)
from aoty_pred.visualization.introspection.feature_introspector import (
    FeatureBlockIntrospector,
)
from aoty_pred.visualization.introspection.mcmc_introspector import MCMCIntrospector
from aoty_pred.visualization.introspection.model_introspector import ModelIntrospector
from aoty_pred.visualization.introspection.prior_introspector import PriorIntrospector
from aoty_pred.visualization.introspection.renderer import (
    EXTENDED_THEME_COLORS,
    FEEDBACK_LOOPS,
    FEEDBACK_STYLES,
    LEGEND_EDGE_TYPES,
    LEGEND_SHAPES,
    DiagramRenderer,
)
from aoty_pred.visualization.introspection.schema_introspector import SchemaIntrospector
from aoty_pred.visualization.introspection.split_introspector import SplitIntrospector
from aoty_pred.visualization.introspection.stage_introspector import StageIntrospector

__all__ = [
    # Base protocols and data structures
    "NodeSpec",
    "EdgeSpec",
    "IntrospectionResult",
    "Introspector",
    # Aggregation
    "DATA_FLOW_EDGES",
    "DiagramAggregator",
    "DiagramData",
    # Rendering
    "DiagramRenderer",
    "EXTENDED_THEME_COLORS",
    "FEEDBACK_LOOPS",
    "FEEDBACK_STYLES",
    "LEGEND_EDGE_TYPES",
    "LEGEND_SHAPES",
    # Diagram builder
    "StructuredDiagramBuilder",
    "SECTION_HIERARCHY",
    "SECTION_COLORS",
    # Introspectors
    "CleaningIntrospector",
    "ConfigIntrospector",
    "CSVIntrospector",
    "FeatureBlockIntrospector",
    "MCMCIntrospector",
    "ModelIntrospector",
    "PriorIntrospector",
    "SchemaIntrospector",
    "SplitIntrospector",
    "StageIntrospector",
]
