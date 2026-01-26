"""Introspector for MCMC configuration parameters.

This module provides MCMCIntrospector, which extracts MCMC configuration
metadata from MCMCConfig and produces diagram nodes showing parameter names,
types, values, and semantic groupings.

The introspector groups parameters into semantic clusters (sampling, adaptation)
for organized diagram rendering.

Example:
    >>> from aoty_pred.visualization.introspection import MCMCIntrospector
    >>> mi = MCMCIntrospector()
    >>> result = mi.introspect()
    >>> print(f"Found {len(result.nodes)} MCMC parameters")
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from aoty_pred.models.bayes.fit import MCMCConfig
from aoty_pred.visualization.introspection.base import (
    IntrospectionResult,
    NodeSpec,
)

__all__ = ["MCMCIntrospector", "MCMC_GROUPS"]


# Semantic grouping of MCMC parameters.
# Maps group name to list of field names for diagram clustering.
MCMC_GROUPS: dict[str, list[str]] = {
    "sampling": ["num_warmup", "num_samples", "num_chains", "chain_method", "seed"],
    "adaptation": ["max_tree_depth", "target_accept_prob"],
}


def _get_field_group(field_name: str) -> str:
    """Determine which semantic group a field belongs to.

    Args:
        field_name: Name of the MCMCConfig field.

    Returns:
        Group name (e.g., "sampling", "adaptation").
    """
    for group_name, field_names in MCMC_GROUPS.items():
        if field_name in field_names:
            return group_name
    return "other"


class MCMCIntrospector:
    """Introspector for MCMC configuration parameters.

    Extracts parameter metadata from MCMCConfig and produces diagram nodes
    showing each parameter with its type, value, and semantic grouping.
    Parameters are clustered for organized diagram rendering.

    Attributes:
        config: The MCMCConfig instance to introspect (defaults to default config).

    Example:
        >>> mi = MCMCIntrospector()
        >>> result = mi.introspect()
        >>> for node in result.nodes:
        ...     print(f"{node.id}: {node.metadata['type_name']}")
    """

    def __init__(self, config: MCMCConfig | None = None) -> None:
        """Initialize MCMCIntrospector.

        Args:
            config: Optional MCMCConfig to introspect. Defaults to a
                new MCMCConfig() with default values.
        """
        self._config = config

    @property
    def source_type(self) -> str:
        """Return the identifier for this introspection source."""
        return "mcmc"

    def _get_config(self) -> MCMCConfig:
        """Get the MCMCConfig to introspect.

        Returns:
            The target MCMCConfig (default: MCMCConfig()).
        """
        if self._config is not None:
            return self._config
        return MCMCConfig()

    def _extract_field_info(self, field_name: str, value: Any) -> dict[str, Any]:
        """Extract metadata from a single MCMC config field.

        Args:
            field_name: Field name from MCMCConfig.
            value: Current value of the field.

        Returns:
            Dict with keys: field_name, value, type_name, group.
        """
        type_name = type(value).__name__
        group = _get_field_group(field_name)

        return {
            "field_name": field_name,
            "value": value,
            "type_name": type_name,
            "group": group,
        }

    def introspect(self) -> IntrospectionResult:
        """Introspect MCMC configuration and return diagram elements.

        Creates one node per MCMCConfig field with the parameter name,
        type, and value in the label. Parameters are grouped into semantic
        clusters for organized diagram rendering.

        Returns:
            IntrospectionResult containing MCMC parameter nodes.
        """
        config = self._get_config()

        nodes: list[NodeSpec] = []
        clusters: dict[str, list[str]] = {}

        # Process fields in sorted order for determinism
        config_fields = sorted(fields(config), key=lambda f: f.name)

        for field in config_fields:
            value = getattr(config, field.name)
            info = self._extract_field_info(field.name, value)

            # Format label with type and value
            label = f"{field.name}\n{info['type_name']}\n{value}"

            node_id = f"mcmc:{field.name}"
            group = info["group"]
            cluster_name = f"mcmc_{group}"

            node = NodeSpec(
                id=node_id,
                label=label,
                category="config",
                cluster=cluster_name,
                metadata=info,
            )
            nodes.append(node)

            # Track cluster membership
            if cluster_name not in clusters:
                clusters[cluster_name] = []
            clusters[cluster_name].append(node_id)

        return IntrospectionResult(
            source_type=self.source_type,
            nodes=nodes,
            edges=[],  # MCMC parameters don't have edges
            clusters=clusters,
            metadata={
                "param_count": len(nodes),
                "groups": sorted(set(n.metadata["group"] for n in nodes)),
            },
        )
