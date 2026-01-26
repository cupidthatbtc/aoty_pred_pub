"""Introspector for feature engineering blocks.

This module provides FeatureBlockIntrospector, which extracts metadata from all
feature blocks in the registry and produces diagram nodes representing each block
with input column counts and output feature estimates.

The introspector instantiates each block from the registry to access its metadata
(name, required_columns, requires) without fitting or transforming data.

Example:
    >>> from aoty_pred.visualization.introspection import FeatureBlockIntrospector
    >>> fi = FeatureBlockIntrospector()
    >>> result = fi.introspect()
    >>> print(f"Found {len(result.nodes)} feature blocks")
"""

from __future__ import annotations

from aoty_pred.features.registry import build_default_registry
from aoty_pred.visualization.introspection.base import (
    IntrospectionResult,
    NodeSpec,
)

__all__ = ["FeatureBlockIntrospector", "FEATURE_OUTPUT_ESTIMATES"]


# Estimated output feature counts for each block.
# Based on default parameters and typical usage.
FEATURE_OUTPUT_ESTIMATES: dict[str, int] = {
    "core_numeric": 3,      # The 3 numeric columns (User_Score, Critic_Score, etc.)
    "temporal": 5,          # album_sequence, career_years, release_gap_days, release_year, date_risk_ordinal
    "artist_reputation": 9, # Alias for artist_history
    "artist_history": 9,    # user_prior_*, critic_prior_*, is_debut (9 features)
    "genre": 30,            # n_components default (after PCA)
    "genre_pca": 30,        # Alias for genre
    "descriptor_pca": 10,   # n_components default
    "album_type": 4,        # One-hot categories (Album, EP, Mixtape, Compilation)
    "collaboration": 3,     # is_collaboration, num_artists, collab_type_ordinal
}


class FeatureBlockIntrospector:
    """Introspector for feature engineering blocks from the registry.

    Extracts metadata from all registered feature blocks and produces diagram
    nodes representing each block with input column counts and output feature
    estimates. Also produces a pipeline summary node.

    Example:
        >>> fi = FeatureBlockIntrospector()
        >>> result = fi.introspect()
        >>> for node in result.nodes:
        ...     print(f"{node.id}: {node.metadata.get('block_name', 'summary')}")
    """

    @property
    def source_type(self) -> str:
        """Return the identifier for this introspection source."""
        return "feature"

    def introspect(self) -> IntrospectionResult:
        """Introspect feature blocks and return diagram elements.

        Creates one node per feature block with input column count and
        output feature estimate in the label. Also creates a pipeline
        summary node with total block count and total features.

        Returns:
            IntrospectionResult containing feature block nodes.
        """
        nodes: list[NodeSpec] = []
        cluster_nodes: list[str] = []
        total_features = 0

        # Get registry and iterate over all blocks
        registry = build_default_registry()

        for name in sorted(registry._builders.keys()):
            # Try to instantiate the block with empty params
            try:
                block = registry._builders[name]({})
                required_columns = getattr(block, "required_columns", [])
                requires = getattr(block, "requires", [])
            except Exception:
                # If instantiation fails, use defaults
                required_columns = []
                requires = []

            # Get output estimate from module constant
            output_estimate = FEATURE_OUTPUT_ESTIMATES.get(name, 5)
            total_features += output_estimate

            node_id = f"feature:{name}"

            # Multi-line label: name, input cols, output features
            label = f"{name}\nIn: {len(required_columns)} cols\nOut: ~{output_estimate} features"

            node = NodeSpec(
                id=node_id,
                label=label,
                category="process",
                cluster="feature_blocks",
                metadata={
                    "block_name": name,
                    "required_columns": list(required_columns),
                    "requires": list(requires),
                    "output_estimate": output_estimate,
                },
            )
            nodes.append(node)
            cluster_nodes.append(node_id)

        # Create pipeline summary node
        block_count = len(nodes)
        summary_id = "feature:pipeline_summary"
        summary_label = f"Feature Pipeline\n{block_count} blocks\n~{total_features} features"

        summary_node = NodeSpec(
            id=summary_id,
            label=summary_label,
            category="data",
            cluster="feature_blocks",
            metadata={
                "block_count": block_count,
                "total_features": total_features,
            },
        )
        nodes.append(summary_node)
        cluster_nodes.append(summary_id)

        return IntrospectionResult(
            source_type=self.source_type,
            nodes=nodes,
            edges=[],  # Blocks are parallel, no sequential edges needed
            clusters={"feature_blocks": cluster_nodes},
            metadata={
                "block_count": block_count,
                "total_features": total_features,
            },
        )
