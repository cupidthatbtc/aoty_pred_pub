"""Introspector for Bayesian model sample sites via NumPyro trace.

This module provides ModelIntrospector, which uses NumPyro's handlers.trace
to extract model structure including sample sites, their distributions,
plate structure, and cross-references to prior hyperparameters.

The introspector executes the model with dummy data to discover all sample
sites dynamically, avoiding brittle source code parsing.

Example:
    >>> from aoty_pred.visualization.introspection import ModelIntrospector
    >>> mi = ModelIntrospector()
    >>> result = mi.introspect()
    >>> print(f"Found {len(result.nodes)} sample sites")
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import random
from numpyro.handlers import seed, trace

from aoty_pred.models.bayes.model import user_score_model
from aoty_pred.visualization.introspection.base import (
    EdgeSpec,
    IntrospectionResult,
    NodeSpec,
)

__all__ = ["ModelIntrospector", "SITE_CATEGORIES", "PRIOR_TO_SITE_MAPPING"]


# Categorization of sample sites by their role in the model.
# Maps site name (without prefix) to category.
SITE_CATEGORIES: dict[str, str] = {
    "mu_artist": "hyperprior",
    "sigma_artist": "hyperprior",
    "sigma_rw": "hyperprior",
    "rho": "hyperprior",
    "sigma_obs": "hyperprior",
    "n_exponent": "hyperprior",
    "init_artist_effect": "random_effect",
    "rw_raw": "random_effect",
    "beta": "fixed_effect",
    "y": "likelihood",
}


# Mapping from prior config fields to model sample sites.
# Format: prior_field -> (site_name, param_role)
PRIOR_TO_SITE_MAPPING: dict[str, tuple[str, str]] = {
    "mu_artist_loc": ("mu_artist", "loc"),
    "mu_artist_scale": ("mu_artist", "scale"),
    "sigma_artist_scale": ("sigma_artist", "scale"),
    "sigma_rw_scale": ("sigma_rw", "scale"),
    "rho_loc": ("rho", "loc"),
    "rho_scale": ("rho", "scale"),
    "beta_loc": ("beta", "loc"),
    "beta_scale": ("beta", "scale"),
    "sigma_obs_scale": ("sigma_obs", "scale"),
    "n_exponent_alpha": ("n_exponent", "concentration1"),
    "n_exponent_beta": ("n_exponent", "concentration0"),
}


def _get_site_category(site_name: str, is_observed: bool) -> str:
    """Determine the category for a sample site.

    Args:
        site_name: Name of the sample site (without prefix).
        is_observed: Whether the site is observed (likelihood).

    Returns:
        Category name (hyperprior, random_effect, fixed_effect, likelihood).
    """
    if is_observed:
        return "likelihood"
    return SITE_CATEGORIES.get(site_name, "other")


class ModelIntrospector:
    """Introspector for Bayesian model sample sites.

    Uses NumPyro's handlers.trace to execute the model with dummy data and
    extract all sample sites, their distributions, and plate structure.
    Produces cross-reference edges linking prior parameters to their
    corresponding sample site distribution parameters.

    Attributes:
        prefix: Model prefix (e.g., "user_" or "critic_").

    Example:
        >>> mi = ModelIntrospector()
        >>> result = mi.introspect()
        >>> for node in result.nodes:
        ...     print(f"{node.id}: {node.metadata['distribution']}")
    """

    def __init__(self, prefix: str = "user_") -> None:
        """Initialize ModelIntrospector.

        Args:
            prefix: Model prefix used for sample site names. Defaults to "user_"
                for the user_score_model. Use "critic_" for critic_score_model.
        """
        self._prefix = prefix

    @property
    def source_type(self) -> str:
        """Return the identifier for this introspection source."""
        return "model"

    def _create_dummy_args(self) -> dict:
        """Create minimal dummy data for model tracing.

        Uses max_seq=2 to ensure rw_raw sample site is captured
        (only appears when max_seq > 1 for random walk trajectory).

        Returns:
            Dict of dummy arguments matching model signature.
        """
        n_obs, n_features, n_artists = 10, 3, 5
        return {
            "artist_idx": jnp.zeros(n_obs, dtype=jnp.int32),
            # Use varying album_seq to enable random walk
            "album_seq": jnp.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2], dtype=jnp.int32),
            "prev_score": jnp.zeros(n_obs),
            "X": jnp.zeros((n_obs, n_features)),
            "y": jnp.zeros(n_obs),
            "n_artists": n_artists,
            "max_seq": 2,  # Enables rw_raw sample site
        }

    def _get_clean_name(self, site_name: str) -> str:
        """Remove prefix from site name for display.

        Args:
            site_name: Full site name (e.g., "user_mu_artist").

        Returns:
            Clean name without prefix (e.g., "mu_artist").
        """
        if site_name.startswith(self._prefix):
            return site_name[len(self._prefix) :]
        return site_name

    def _format_shape(self, shape: tuple) -> str:
        """Format shape tuple for display.

        Args:
            shape: Shape tuple from site value.

        Returns:
            Formatted shape string (e.g., "(5,)" or "scalar").
        """
        if not shape:
            return "scalar"
        return str(shape)

    def introspect(self) -> IntrospectionResult:
        """Introspect model sample sites and return diagram elements.

        Traces the model with dummy data to discover all sample sites,
        creates nodes for each site, and generates cross-reference edges
        from prior parameters to their corresponding sample sites.

        Returns:
            IntrospectionResult containing model sample site nodes and
            cross-reference edges to prior parameters.
        """
        # Trace model execution
        dummy_args = self._create_dummy_args()
        seeded_model = seed(user_score_model, random.PRNGKey(0))
        exec_trace = trace(seeded_model).get_trace(**dummy_args)

        nodes: list[NodeSpec] = []
        edges: list[EdgeSpec] = []
        clusters: dict[str, list[str]] = {}
        plates: list[str] = []
        site_names_in_trace: set[str] = set()

        # Process trace items in sorted order for determinism
        for name in sorted(exec_trace.keys()):
            site = exec_trace[name]
            site_type = site["type"]

            if site_type == "plate":
                plates.append(name)
                continue

            # Skip reparameterized sites (have _decentered suffix)
            # The original name appears as a deterministic site
            if "_decentered" in name:
                continue

            # Handle both sample sites and deterministic sites from reparameterization
            # (e.g., init_artist_effect becomes deterministic after LocScaleReparam)
            if site_type == "sample":
                is_observed = site["is_observed"]
                dist_obj = site["fn"]
                dist_name = dist_obj.__class__.__name__
            elif site_type == "deterministic":
                # Deterministic sites from reparameterization
                is_observed = False
                dist_name = "Reparameterized"
            else:
                continue

            clean_name = self._get_clean_name(name)
            site_names_in_trace.add(clean_name)

            # Get shape from value if available
            shape_str = "unknown"
            if site.get("value") is not None:
                shape_str = self._format_shape(site["value"].shape)

            # Categorize site
            category = _get_site_category(clean_name, is_observed)
            cluster_name = f"model_{category}"

            # Create node
            node_id = f"model:{name}"
            label = f"{clean_name}\n{dist_name}\n{shape_str}"

            node = NodeSpec(
                id=node_id,
                label=label,
                category="process" if is_observed else "config",
                cluster=cluster_name,
                metadata={
                    "site_name": name,
                    "clean_name": clean_name,
                    "distribution": dist_name,
                    "shape": shape_str,
                    "is_observed": is_observed,
                    "category": category,
                },
            )
            nodes.append(node)

            # Track cluster membership
            if cluster_name not in clusters:
                clusters[cluster_name] = []
            clusters[cluster_name].append(node_id)

        # Generate cross-reference edges from prior params to sample sites
        for prior_field, (site_name, param_role) in PRIOR_TO_SITE_MAPPING.items():
            # Check if this site exists in the trace
            if site_name in site_names_in_trace:
                full_site_name = f"{self._prefix}{site_name}"
                edge = EdgeSpec(
                    source=f"prior:{prior_field}",
                    target=f"model:{full_site_name}",
                    label=param_role,
                    style="dashed",
                    category="dependency",
                )
                edges.append(edge)

        return IntrospectionResult(
            source_type=self.source_type,
            nodes=nodes,
            edges=edges,
            clusters=clusters,
            metadata={
                "site_count": len(nodes),
                "plate_count": len(plates),
                "plates": plates,
                "prefix": self._prefix,
            },
        )
