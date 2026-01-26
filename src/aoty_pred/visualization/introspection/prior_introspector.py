"""Introspector for Bayesian prior hyperparameters.

This module provides PriorIntrospector, which extracts prior configuration
metadata from PriorConfig and produces diagram nodes showing parameter names,
distribution types, values, and semantic groupings.

The introspector groups parameters into semantic clusters (artist pooling,
career dynamics, fixed effects, etc.) for organized diagram rendering.

Example:
    >>> from aoty_pred.visualization.introspection import PriorIntrospector
    >>> pi = PriorIntrospector()
    >>> result = pi.introspect()
    >>> print(f"Found {len(result.nodes)} prior parameters")
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from aoty_pred.models.bayes.priors import PriorConfig, get_default_priors
from aoty_pred.visualization.introspection.base import (
    IntrospectionResult,
    NodeSpec,
)

__all__ = ["PriorIntrospector", "PRIOR_DISTRIBUTIONS", "PRIOR_GROUPS"]


# Distribution type for each PriorConfig field.
# Maps field name to the distribution used in the NumPyro model.
PRIOR_DISTRIBUTIONS: dict[str, str] = {
    "mu_artist_loc": "Normal",
    "mu_artist_scale": "Normal",
    "sigma_artist_scale": "HalfNormal",
    "sigma_rw_scale": "HalfNormal",
    "rho_loc": "TruncatedNormal",
    "rho_scale": "TruncatedNormal",
    "beta_loc": "Normal",
    "beta_scale": "Normal",
    "sigma_obs_scale": "HalfNormal",
    "n_exponent_alpha": "Beta",
    "n_exponent_beta": "Beta",
}


# Semantic grouping of prior parameters.
# Maps group name to list of field names for diagram clustering.
PRIOR_GROUPS: dict[str, list[str]] = {
    "artist_pooling": ["mu_artist_loc", "mu_artist_scale", "sigma_artist_scale"],
    "career_dynamics": ["sigma_rw_scale", "rho_loc", "rho_scale"],
    "fixed_effects": ["beta_loc", "beta_scale"],
    "observation_noise": ["sigma_obs_scale"],
    "heteroscedastic": ["n_exponent_alpha", "n_exponent_beta"],
}


def _get_field_group(field_name: str) -> str:
    """Determine which semantic group a field belongs to.

    Args:
        field_name: Name of the PriorConfig field.

    Returns:
        Group name (e.g., "artist_pooling", "career_dynamics").
    """
    for group_name, field_names in PRIOR_GROUPS.items():
        if field_name in field_names:
            return group_name
    return "other"


class PriorIntrospector:
    """Introspector for Bayesian prior hyperparameters.

    Extracts parameter metadata from PriorConfig and produces diagram nodes
    showing each parameter with its distribution type, value, and semantic
    grouping. Parameters are clustered for organized diagram rendering.

    Attributes:
        priors: The PriorConfig instance to introspect (defaults to default priors).

    Example:
        >>> pi = PriorIntrospector()
        >>> result = pi.introspect()
        >>> for node in result.nodes:
        ...     print(f"{node.id}: {node.metadata['distribution']}")
    """

    def __init__(self, priors: PriorConfig | None = None) -> None:
        """Initialize PriorIntrospector.

        Args:
            priors: Optional PriorConfig to introspect. Defaults to the
                result of get_default_priors().
        """
        self._priors = priors

    @property
    def source_type(self) -> str:
        """Return the identifier for this introspection source."""
        return "prior"

    def _get_priors(self) -> PriorConfig:
        """Get the PriorConfig to introspect.

        Returns:
            The target PriorConfig (default: get_default_priors()).
        """
        if self._priors is not None:
            return self._priors
        return get_default_priors()

    def _extract_field_info(self, field_name: str, value: Any) -> dict[str, Any]:
        """Extract metadata from a single prior field.

        Args:
            field_name: Field name from PriorConfig.
            value: Current value of the field.

        Returns:
            Dict with keys: field_name, value, distribution, group.
        """
        distribution = PRIOR_DISTRIBUTIONS.get(field_name, "Unknown")
        group = _get_field_group(field_name)

        return {
            "field_name": field_name,
            "value": value,
            "distribution": distribution,
            "group": group,
        }

    def introspect(self) -> IntrospectionResult:
        """Introspect prior parameters and return diagram elements.

        Creates one node per PriorConfig field with the parameter name,
        distribution type, and value in the label. Parameters are grouped
        into semantic clusters for organized diagram rendering.

        Returns:
            IntrospectionResult containing prior parameter nodes.
        """
        priors = self._get_priors()

        nodes: list[NodeSpec] = []
        clusters: dict[str, list[str]] = {}

        # Process fields in sorted order for determinism
        prior_fields = sorted(fields(priors), key=lambda f: f.name)

        for field in prior_fields:
            value = getattr(priors, field.name)
            info = self._extract_field_info(field.name, value)

            # Format label with distribution and value
            label = f"{field.name}\n{info['distribution']}\n{value}"

            node_id = f"prior:{field.name}"
            group = info["group"]
            cluster_name = f"prior_{group}"

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
            edges=[],  # Prior parameters don't have edges
            clusters=clusters,
            metadata={
                "prior_count": len(nodes),
                "groups": sorted(set(n.metadata["group"] for n in nodes)),
            },
        )
