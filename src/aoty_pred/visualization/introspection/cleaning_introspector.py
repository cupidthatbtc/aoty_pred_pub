"""Introspector for data cleaning pipeline rules.

This module provides CleaningIntrospector, which extracts cleaning rule metadata
from the data pipeline and produces diagram nodes representing each cleaning step.

The introspector generates nodes with rule names and descriptions, connected by
edges showing the sequential processing order. Per RESEARCH.md, actual row counts
require running the pipeline with AuditLogger; for MVP we show rule definitions.

Example:
    >>> from aoty_pred.visualization.introspection import CleaningIntrospector
    >>> ci = CleaningIntrospector()
    >>> result = ci.introspect()
    >>> print(f"Found {len(result.nodes)} cleaning rules")
"""

from __future__ import annotations

from aoty_pred.visualization.introspection.base import (
    EdgeSpec,
    IntrospectionResult,
    NodeSpec,
)

__all__ = ["CleaningIntrospector", "CLEANING_RULES"]


# Default cleaning rules with names and descriptions.
# These represent the canonical cleaning pipeline order.
CLEANING_RULES: list[tuple[str, str]] = [
    ("rename_columns", "Rename raw columns to canonical format"),
    ("parse_release_dates", "Parse dates with 3-tier risk classification"),
    ("extract_collaboration_features", "Extract artist collaboration info"),
    ("extract_primary_genre", "Extract first genre from list"),
    ("flag_unknown_artist", "Flag [unknown artist] entries"),
    ("filter_for_user_score_model", "Filter to albums with valid user scores"),
    ("filter_for_critic_score_model", "Filter to albums with valid critic scores"),
]


class CleaningIntrospector:
    """Introspector for data cleaning pipeline rules.

    Extracts cleaning rule metadata and produces diagram nodes representing
    each step in the cleaning pipeline. Nodes are connected in sequence to
    show the processing order.

    Attributes:
        rules: List of (rule_name, description) tuples representing the
            cleaning pipeline steps.

    Example:
        >>> ci = CleaningIntrospector()
        >>> result = ci.introspect()
        >>> for node in result.nodes:
        ...     print(f"{node.id}: {node.metadata['rule_name']}")
    """

    def __init__(self, rules: list[tuple[str, str]] | None = None) -> None:
        """Initialize CleaningIntrospector.

        Args:
            rules: Optional list of (rule_name, description) tuples. Defaults
                to CLEANING_RULES module constant.
        """
        self.rules = rules if rules is not None else CLEANING_RULES

    @property
    def source_type(self) -> str:
        """Return the identifier for this introspection source."""
        return "cleaning"

    def introspect(self) -> IntrospectionResult:
        """Introspect cleaning pipeline rules and return diagram elements.

        Creates one node per cleaning rule with the rule name and description
        as the label. Nodes are connected by edges in pipeline order.

        Returns:
            IntrospectionResult containing cleaning rule nodes and edges.
        """
        nodes: list[NodeSpec] = []
        edges: list[EdgeSpec] = []
        cluster_nodes: list[str] = []

        for order, (rule_name, description) in enumerate(self.rules):
            node_id = f"cleaning:{rule_name}"

            # Multi-line label: rule name on top, description below
            label = f"{rule_name}\n{description}"

            node = NodeSpec(
                id=node_id,
                label=label,
                category="process",
                cluster="cleaning_pipeline",
                metadata={
                    "rule_name": rule_name,
                    "description": description,
                    "order": order,
                },
            )
            nodes.append(node)
            cluster_nodes.append(node_id)

        # Create sequential edges connecting rules in order
        for i in range(len(nodes) - 1):
            edge = EdgeSpec(
                source=nodes[i].id,
                target=nodes[i + 1].id,
                category="flow",
                style="solid",
            )
            edges.append(edge)

        return IntrospectionResult(
            source_type=self.source_type,
            nodes=nodes,
            edges=edges,
            clusters={"cleaning_pipeline": cluster_nodes},
            metadata={
                "rule_count": len(self.rules),
            },
        )
