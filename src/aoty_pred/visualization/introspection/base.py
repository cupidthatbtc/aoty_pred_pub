"""Base protocols and data structures for pipeline introspection.

This module defines the core data structures and protocols for the v4.0
introspection architecture. All introspectors produce IntrospectionResult
objects that can be aggregated and rendered into publication-quality diagrams.

The protocol design uses dataclasses for structured data (per CONVENTIONS.md)
and Protocol for duck typing with type hints.

Data Flow:
    Introspector.introspect() -> IntrospectionResult
    DiagramAggregator.aggregate() -> DiagramData
    DiagramRenderer.render() -> graphviz.Digraph

Example:
    >>> from aoty_pred.visualization.introspection import StageIntrospector
    >>> introspector = StageIntrospector()
    >>> result = introspector.introspect()
    >>> print(f"Found {len(result.nodes)} nodes and {len(result.edges)} edges")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = [
    "NodeSpec",
    "EdgeSpec",
    "IntrospectionResult",
    "Introspector",
]


@dataclass
class NodeSpec:
    """Specification for a diagram node.

    Represents a single node in the introspected pipeline graph. Nodes can
    represent pipeline stages, data files, configuration options, or any
    other discrete element in the system.

    Attributes:
        id: Unique identifier for the node. Use namespaced format for clarity
            (e.g., "stage:data", "file:train.parquet", "config:seed").
        label: Display label for the node. Supports multi-line text using
            newline characters (\\n) for Graphviz rendering.
        category: Node category for styling (shape/color). Standard categories:
            - "process": Computational steps (box shape)
            - "data": Data files/artifacts (ellipse shape)
            - "decision": Decision points (diamond shape)
            - "config": Configuration nodes
        cluster: Optional cluster grouping identifier. Nodes with the same
            cluster value will be grouped in a Graphviz subgraph.
        metadata: Extra data for tooltips, debugging, or extended rendering.
            Can include original paths, descriptions, timestamps, etc.

    Example:
        >>> node = NodeSpec(
        ...     id="stage:train",
        ...     label="Train Model\\nMCMC Sampling",
        ...     category="process",
        ...     cluster="pipeline",
        ...     metadata={"description": "Fit Bayesian model"},
        ... )
    """

    id: str
    label: str
    category: str
    cluster: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeSpec:
    """Specification for a diagram edge (connection between nodes).

    Represents a directed edge connecting two nodes in the introspected graph.
    Edges can represent data flow, dependencies, feedback loops, or any
    other relationship between nodes.

    Attributes:
        source: Source node ID (must exist in nodes).
        target: Target node ID (must exist in nodes).
        label: Optional edge label for annotations (e.g., row counts,
            data types, or relationship descriptions).
        style: Edge line style for Graphviz:
            - "solid": Normal dependency or flow
            - "dashed": Optional or feedback relationship
            - "dotted": Weak or inferred relationship
        category: Edge category for color/styling:
            - "flow": Primary data flow (default)
            - "dependency": Stage dependencies
            - "feedback": Feedback/retry loops

    Example:
        >>> edge = EdgeSpec(
        ...     source="stage:features",
        ...     target="stage:train",
        ...     label="62K rows",
        ...     category="dependency",
        ... )
    """

    source: str
    target: str
    label: str = ""
    style: str = "solid"
    category: str = "flow"


@dataclass
class IntrospectionResult:
    """Result of introspecting a single source (stage, config, schema, etc.).

    Aggregates nodes, edges, and cluster information from one introspection
    source. Multiple IntrospectionResults can be merged by DiagramAggregator
    to produce a unified diagram.

    Attributes:
        source_type: Identifier for the introspection source (e.g., "stage",
            "csv", "schema", "config"). Used for sorting and debugging.
        nodes: List of nodes discovered during introspection.
        edges: List of edges (relationships) discovered during introspection.
        clusters: Mapping of cluster IDs to lists of node IDs that belong
            to each cluster. Used for Graphviz subgraph grouping.
        metadata: Extra information about the introspection run (timestamps,
            source file paths, version info, etc.).

    Example:
        >>> result = IntrospectionResult(
        ...     source_type="stage",
        ...     nodes=[NodeSpec(id="s:data", label="Data", category="process")],
        ...     edges=[EdgeSpec(source="s:data", target="s:splits")],
        ...     clusters={"pipeline": ["s:data", "s:splits"]},
        ... )
    """

    source_type: str
    nodes: list[NodeSpec]
    edges: list[EdgeSpec]
    clusters: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class Introspector(Protocol):
    """Protocol for introspection implementations.

    All introspectors must provide a source_type property and an introspect()
    method that returns an IntrospectionResult. This enables duck typing
    with type hints for pluggable introspector architectures.

    The Protocol pattern allows future introspectors (CSVIntrospector,
    SchemaIntrospector, ConfigIntrospector, etc.) to be added without
    modifying the aggregator or renderer.

    Example:
        >>> class MyIntrospector:
        ...     @property
        ...     def source_type(self) -> str:
        ...         return "my_source"
        ...
        ...     def introspect(self) -> IntrospectionResult:
        ...         return IntrospectionResult(
        ...             source_type=self.source_type,
        ...             nodes=[],
        ...             edges=[],
        ...         )
    """

    @property
    def source_type(self) -> str:
        """Return the identifier for this introspection source."""
        ...

    def introspect(self) -> IntrospectionResult:
        """Perform introspection and return results."""
        ...
