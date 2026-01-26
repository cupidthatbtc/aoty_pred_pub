"""Diagram aggregator for merging multiple introspection results.

This module provides DiagramAggregator which merges multiple IntrospectionResults
into a unified DiagramData structure suitable for rendering. The aggregator
handles node deduplication, edge validation, and cluster merging.

The aggregation strategy prioritizes:
1. Determinism: Results are sorted by source_type before processing
2. Later wins: Node collisions resolve in favor of later results
3. Edge validation: Edges referencing non-existent nodes are filtered
4. Cluster deduplication: Cluster node lists are sorted and deduplicated

Example:
    >>> from aoty_pred.visualization.introspection import (
    ...     StageIntrospector, DiagramAggregator
    ... )
    >>> si = StageIntrospector()
    >>> result = si.introspect()
    >>> agg = DiagramAggregator()
    >>> data = agg.aggregate([result])
    >>> print(f"Aggregated {len(data.nodes)} nodes")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aoty_pred.visualization.introspection.base import (
    EdgeSpec,
    IntrospectionResult,
    NodeSpec,
)

__all__ = ["DiagramAggregator", "DiagramData", "DATA_FLOW_EDGES"]


# Data flow edges that connect pipeline stages.
# Format: (source_pattern, target_pattern, label_template)
# source_pattern/target_pattern match node IDs (exact or prefix)
# label_template uses {row_count} placeholder for dynamic annotation
DATA_FLOW_EDGES: list[tuple[str, str, str]] = [
    # Raw data to first cleaning step
    ("csv:all_albums_full.csv", "cleaning:rename_columns", "{row_count} rows"),
    # Last cleaning step to splits
    ("cleaning:filter_for_critic_score_model", "stage:splits", ""),
    # Splits to features
    ("stage:splits", "stage:features", ""),
    # Features to model
    ("stage:features", "stage:train", "{row_count} samples"),
    # Model to evaluation
    ("stage:train", "stage:evaluate", ""),
]


@dataclass
class DiagramData:
    """Unified diagram data structure for rendering.

    Represents the complete, aggregated diagram ready for rendering.
    All nodes are deduplicated by ID, edges are validated, and clusters
    are merged with sorted node lists.

    Attributes:
        nodes: Dictionary mapping node ID to NodeSpec. Provides O(1) lookup
            for edge validation and prevents duplicate nodes.
        edges: List of all validated edges. Only includes edges where both
            source and target nodes exist in the nodes dict.
        clusters: Dictionary mapping cluster ID to sorted list of node IDs
            that belong to each cluster.
        metadata: Aggregation metadata including source types, counts, etc.

    Example:
        >>> data = DiagramData(
        ...     nodes={"a": NodeSpec(id="a", label="A", category="process")},
        ...     edges=[EdgeSpec(source="a", target="b")],
        ...     clusters={"group": ["a"]},
        ... )
    """

    nodes: dict[str, NodeSpec]
    edges: list[EdgeSpec]
    clusters: dict[str, list[str]]
    metadata: dict[str, Any] = field(default_factory=dict)


class DiagramAggregator:
    """Aggregator for merging multiple IntrospectionResults.

    Combines nodes, edges, and clusters from multiple introspection sources
    into a single DiagramData structure. Handles conflicts, validates
    relationships, and ensures deterministic output.

    Merge Semantics:
        - Nodes: Later results overwrite earlier nodes with same ID
        - Edges: All edges collected, invalid edges filtered
        - Clusters: Node lists extended and deduplicated
        - Data flow: Inter-cluster edges generated with row count annotations

    Example:
        >>> agg = DiagramAggregator()
        >>> result1 = IntrospectionResult(source_type="stage", nodes=[], edges=[])
        >>> result2 = IntrospectionResult(source_type="config", nodes=[], edges=[])
        >>> data = agg.aggregate([result1, result2])
    """

    def _find_node_by_pattern(
        self,
        nodes: dict[str, NodeSpec],
        pattern: str,
    ) -> str | None:
        """Find a node ID matching the given pattern.

        Supports exact match first, then prefix match.

        Args:
            nodes: Dict of node_id -> NodeSpec.
            pattern: Pattern to match (exact or prefix).

        Returns:
            Matching node ID or None.
        """
        # Exact match first
        if pattern in nodes:
            return pattern

        # Prefix match
        for node_id in nodes:
            if node_id.startswith(pattern.rstrip("*")):
                return node_id

        return None

    def _generate_data_flow_edges(
        self,
        nodes: dict[str, NodeSpec],
    ) -> list[EdgeSpec]:
        """Generate data flow edges with row count annotations.

        Matches DATA_FLOW_EDGES patterns against actual nodes and
        extracts row counts from node metadata for edge labels.

        Args:
            nodes: Dict of node_id -> NodeSpec with metadata.

        Returns:
            List of EdgeSpec for data flow connections.
        """
        edges: list[EdgeSpec] = []

        for source_pattern, target_pattern, label_template in DATA_FLOW_EDGES:
            # Find matching source node
            source_id = self._find_node_by_pattern(nodes, source_pattern)
            target_id = self._find_node_by_pattern(nodes, target_pattern)

            if source_id and target_id:
                # Extract row count from source node metadata
                source_node = nodes[source_id]
                row_count = source_node.metadata.get("row_count")

                # Format label with row count if available
                if row_count and "{row_count}" in label_template:
                    label = label_template.format(row_count=f"{row_count:,}")
                elif label_template and "{row_count}" not in label_template:
                    label = label_template
                else:
                    label = ""

                edges.append(EdgeSpec(
                    source=source_id,
                    target=target_id,
                    label=label,
                    category="flow",
                ))

        return edges

    def aggregate(self, results: list[IntrospectionResult]) -> DiagramData:
        """Merge multiple IntrospectionResults into unified DiagramData.

        Processes results in sorted order by source_type for determinism.
        Merges nodes, collects edges, combines clusters, and validates
        all relationships.

        Args:
            results: List of IntrospectionResults to merge. Can be empty.

        Returns:
            DiagramData with merged nodes, validated edges, and combined clusters.

        Example:
            >>> from aoty_pred.visualization.introspection import StageIntrospector
            >>> si = StageIntrospector()
            >>> result = si.introspect()
            >>> agg = DiagramAggregator()
            >>> data = agg.aggregate([result])
            >>> assert len(data.nodes) == 6
        """
        # Sort results by source_type for deterministic processing
        sorted_results = sorted(results, key=lambda r: r.source_type)

        # Merge nodes (later wins on collision)
        nodes: dict[str, NodeSpec] = {}
        for result in sorted_results:
            for node in result.nodes:
                nodes[node.id] = node

        # Collect all edges
        all_edges: list[EdgeSpec] = []
        for result in sorted_results:
            all_edges.extend(result.edges)

        # Validate edges: filter to only edges where source and target exist
        valid_edges = [
            edge for edge in all_edges
            if edge.source in nodes and edge.target in nodes
        ]

        # Generate data flow edges after merging all nodes
        flow_edges = self._generate_data_flow_edges(nodes)
        valid_edges.extend(flow_edges)

        # Merge clusters with deduplication
        clusters: dict[str, list[str]] = {}
        for result in sorted_results:
            for cluster_id, node_ids in result.clusters.items():
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].extend(node_ids)

        # Deduplicate and sort cluster node lists
        for cluster_id in clusters:
            clusters[cluster_id] = sorted(set(clusters[cluster_id]))

        # Build metadata
        source_types = [r.source_type for r in sorted_results]
        metadata = {
            "source_types": source_types,
            "total_nodes": len(nodes),
            "total_edges": len(valid_edges),
            "filtered_edges": len(all_edges) - len(valid_edges),
            "cluster_count": len(clusters),
        }

        return DiagramData(
            nodes=nodes,
            edges=valid_edges,
            clusters=clusters,
            metadata=metadata,
        )
