"""Orchestrator for building structured pipeline diagrams from introspected data.

This module provides StructuredDiagramBuilder, which collects results from all
10 introspectors and organizes them into a 15-section hierarchical cluster
structure for the ultra-detailed pipeline diagram.

The builder remaps introspector-specific cluster names into a standardized
hierarchical format (e.g., PRIORS_artist_pooling, MODEL_sampling) that
enables consistent visual grouping in the final diagram.

Example:
    >>> from aoty_pred.visualization.introspection import StructuredDiagramBuilder
    >>> builder = StructuredDiagramBuilder()
    >>> data = builder.build()
    >>> print(f"Built diagram with {len(data.nodes)} nodes")
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from aoty_pred.visualization.introspection.aggregator import (
    DiagramAggregator,
    DiagramData,
)
from aoty_pred.visualization.introspection.base import (
    IntrospectionResult,
    NodeSpec,
)
from aoty_pred.visualization.introspection.cleaning_introspector import (
    CleaningIntrospector,
)
from aoty_pred.visualization.introspection.config_introspector import (
    ConfigIntrospector,
)
from aoty_pred.visualization.introspection.csv_introspector import CSVIntrospector
from aoty_pred.visualization.introspection.feature_introspector import (
    FeatureBlockIntrospector,
)
from aoty_pred.visualization.introspection.mcmc_introspector import MCMCIntrospector
from aoty_pred.visualization.introspection.model_introspector import ModelIntrospector
from aoty_pred.visualization.introspection.prior_introspector import PriorIntrospector
from aoty_pred.visualization.introspection.schema_introspector import SchemaIntrospector
from aoty_pred.visualization.introspection.split_introspector import SplitIntrospector
from aoty_pred.visualization.introspection.stage_introspector import StageIntrospector

__all__ = [
    "StructuredDiagramBuilder",
    "SECTION_HIERARCHY",
    "SECTION_COLORS",
    "INTROSPECTOR_TO_SECTION",
]


# Cluster hierarchy: major sections with optional sub-clusters.
# 10 major sections organized by pipeline stage (left-to-right flow).
# Sub-clusters within each section for semantic grouping.
SECTION_HIERARCHY: dict[str, list[str]] = {
    "CONFIG": [],  # Flat - all config nodes together
    "DATA_RAW": ["raw_csv"],  # Input data sources
    "DATA_CLEAN": ["schema", "cleaning"],  # Validation + cleaning
    "DATA_SPLIT": ["strategy", "partitions"],  # Split logic + train/val/test
    "FEATURES": ["blocks"],  # Feature engineering blocks
    "PRIORS": [
        "artist_pooling",
        "career_dynamics",
        "fixed_effects",
        "observation_noise",
        "heteroscedastic",
    ],  # 5 prior groups
    "MODEL": [
        "hyperprior",
        "random_effect",
        "fixed_effect",
        "likelihood",
        "sampling",
        "adaptation",
    ],  # Model architecture + MCMC
    "CONVERGENCE": ["diagnostics"],  # R-hat, ESS, divergences
    "EVALUATION": ["metrics", "calibration"],  # LOO-CV, CRPS, coverage
    "OUTPUT": ["pipeline"],  # Results + model files
}


# 10-color palette for light theme fills.
# Each major section gets a distinct pastel color for visual separation.
SECTION_COLORS: dict[str, str] = {
    "CONFIG": "#F5F5F5",  # Gray - neutral setup
    "DATA_RAW": "#E8EEF4",  # Blue-gray - external data
    "DATA_CLEAN": "#FFF9E6",  # Yellow - validation checks
    "DATA_SPLIT": "#E6F4F4",  # Cyan - data division
    "FEATURES": "#E8F5E8",  # Green - engineering
    "PRIORS": "#F4E8F4",  # Purple-light - statistical setup
    "MODEL": "#E6EEF8",  # Blue - core computation
    "CONVERGENCE": "#FFF0E0",  # Orange - diagnostics
    "EVALUATION": "#F8E8F0",  # Magenta - assessment
    "OUTPUT": "#E0F0E0",  # Green-bright - results
}


# Maps introspector source_type to major section.
# Used to categorize nodes during cluster remapping.
INTROSPECTOR_TO_SECTION: dict[str, str] = {
    "config": "CONFIG",
    "csv": "DATA_RAW",
    "schema": "DATA_CLEAN",
    "cleaning": "DATA_CLEAN",
    "split": "DATA_SPLIT",
    "feature": "FEATURES",
    "prior": "PRIORS",
    "mcmc": "MODEL",
    "model": "MODEL",
    "stage": "OUTPUT",
}


# Maps existing cluster names from introspectors to hierarchical names.
# Format: original_cluster -> SECTION_subcluster
_CLUSTER_REMAP: dict[str, str] = {
    # CSV introspector
    "data_files": "DATA_RAW_raw_csv",
    # Schema introspector
    "schemas": "DATA_CLEAN_schema",
    # Cleaning introspector
    "cleaning_pipeline": "DATA_CLEAN_cleaning",
    # Split introspector
    "splits": "DATA_SPLIT_partitions",
    # Feature introspector
    "feature_blocks": "FEATURES_blocks",
    # Config introspector (all go to flat CONFIG section)
    "config_ablation": "CONFIG",
    "config_convergence": "CONFIG",
    "config_data_filter": "CONFIG",
    "config_execution": "CONFIG",
    "config_mcmc": "CONFIG",
    "config_noise": "CONFIG",
    "config_preflight": "CONFIG",
    # Prior introspector
    "prior_artist_pooling": "PRIORS_artist_pooling",
    "prior_career_dynamics": "PRIORS_career_dynamics",
    "prior_fixed_effects": "PRIORS_fixed_effects",
    "prior_observation_noise": "PRIORS_observation_noise",
    "prior_heteroscedastic": "PRIORS_heteroscedastic",
    # MCMC introspector
    "mcmc_sampling": "MODEL_sampling",
    "mcmc_adaptation": "MODEL_adaptation",
    # Model introspector
    "model_fixed_effect": "MODEL_fixed_effect",
    "model_random_effect": "MODEL_random_effect",
    "model_hyperprior": "MODEL_hyperprior",
    "model_likelihood": "MODEL_likelihood",
    # Stage introspector
    "pipeline": "OUTPUT_pipeline",
}


@dataclass
class ClusterInfo:
    """Information about a cluster for rendering.

    Attributes:
        name: Hierarchical cluster name (e.g., "PRIORS_artist_pooling").
        section: Major section this cluster belongs to (e.g., "PRIORS").
        sub_cluster: Sub-cluster within section (e.g., "artist_pooling"), or None if flat.
        node_ids: List of node IDs in this cluster.
    """

    name: str
    section: str
    sub_cluster: str | None
    node_ids: list[str]


class StructuredDiagramBuilder:
    """Orchestrator for building structured pipeline diagrams.

    Collects introspection results from all 10 introspectors and organizes
    them into a 15-section hierarchical cluster structure. The builder
    handles cluster remapping to ensure consistent naming across the diagram.

    The build process:
    1. Instantiate all 10 introspectors
    2. Call introspect() on each
    3. Remap cluster names to hierarchical format
    4. Aggregate all results via DiagramAggregator
    5. Return unified DiagramData

    Example:
        >>> builder = StructuredDiagramBuilder()
        >>> data = builder.build()
        >>> print(f"Built {len(data.nodes)} nodes in {len(data.clusters)} clusters")
    """

    def __init__(self) -> None:
        """Initialize StructuredDiagramBuilder."""
        self._aggregator = DiagramAggregator()

    def _get_all_introspectors(self) -> list[Any]:
        """Return list of all 10 introspectors for full pipeline coverage.

        Returns instances of:
        1. CSVIntrospector - raw data files
        2. SchemaIntrospector - data schema validation
        3. CleaningIntrospector - data cleaning rules
        4. SplitIntrospector - train/test split strategy
        5. FeatureBlockIntrospector - feature engineering blocks
        6. ConfigIntrospector - pipeline configuration
        7. PriorIntrospector - Bayesian prior definitions
        8. MCMCIntrospector - MCMC sampling config
        9. ModelIntrospector - model architecture
        10. StageIntrospector - pipeline stage definitions

        Returns:
            List of 10 introspector instances.
        """
        return [
            CSVIntrospector(),
            SchemaIntrospector(),
            CleaningIntrospector(),
            SplitIntrospector(),
            FeatureBlockIntrospector(),
            ConfigIntrospector(),
            PriorIntrospector(),
            MCMCIntrospector(),
            ModelIntrospector(),
            StageIntrospector(),
        ]

    def _remap_cluster_name(self, original: str) -> str:
        """Remap an introspector cluster name to hierarchical format.

        Args:
            original: Original cluster name from introspector.

        Returns:
            Hierarchical cluster name (e.g., "PRIORS_artist_pooling").
        """
        return _CLUSTER_REMAP.get(original, original)

    def _remap_node_cluster(self, node: NodeSpec) -> NodeSpec:
        """Remap a node's cluster to hierarchical format.

        Args:
            node: Original NodeSpec with introspector cluster name.

        Returns:
            New NodeSpec with remapped cluster name.
        """
        if node.cluster is None:
            return node

        new_cluster = self._remap_cluster_name(node.cluster)
        return replace(node, cluster=new_cluster)

    def _remap_clusters(self, result: IntrospectionResult) -> IntrospectionResult:
        """Remap all cluster names in an IntrospectionResult.

        Args:
            result: Original IntrospectionResult from an introspector.

        Returns:
            New IntrospectionResult with remapped cluster names.
        """
        # Remap node clusters
        remapped_nodes = [self._remap_node_cluster(node) for node in result.nodes]

        # Remap cluster dictionary
        remapped_clusters: dict[str, list[str]] = {}
        for old_name, node_ids in result.clusters.items():
            new_name = self._remap_cluster_name(old_name)
            if new_name not in remapped_clusters:
                remapped_clusters[new_name] = []
            remapped_clusters[new_name].extend(node_ids)

        return IntrospectionResult(
            source_type=result.source_type,
            nodes=remapped_nodes,
            edges=result.edges,
            clusters=remapped_clusters,
            metadata=result.metadata,
        )

    def build(self) -> DiagramData:
        """Build the complete structured diagram from all introspectors.

        Orchestrates the full introspection pipeline:
        1. Get all 10 introspectors
        2. Run introspection on each
        3. Remap cluster names to hierarchical format
        4. Aggregate into unified DiagramData

        Returns:
            DiagramData with all nodes, edges, and hierarchical clusters.

        Example:
            >>> builder = StructuredDiagramBuilder()
            >>> data = builder.build()
            >>> # Should produce 80+ nodes and 10+ clusters
            >>> assert len(data.nodes) >= 40
        """
        introspectors = self._get_all_introspectors()

        # Collect and remap results from all introspectors
        remapped_results: list[IntrospectionResult] = []
        for introspector in introspectors:
            result = introspector.introspect()
            remapped = self._remap_clusters(result)
            remapped_results.append(remapped)

        # Aggregate all results
        data = self._aggregator.aggregate(remapped_results)

        # Add section metadata
        data.metadata["section_hierarchy"] = SECTION_HIERARCHY
        data.metadata["section_colors"] = SECTION_COLORS

        return data

    def _parse_cluster_name(self, cluster_name: str) -> tuple[str, str | None]:
        """Parse a cluster name into section and sub-cluster.

        Handles sections with underscores (e.g., DATA_CLEAN) by matching
        against known section names in SECTION_HIERARCHY.

        Args:
            cluster_name: Hierarchical cluster name (e.g., "DATA_CLEAN_cleaning").

        Returns:
            Tuple of (section, sub_cluster) where sub_cluster may be None.
        """
        # Check for exact section match (flat sections like CONFIG)
        if cluster_name in SECTION_HIERARCHY:
            return cluster_name, None

        # Find matching section prefix
        for section in SECTION_HIERARCHY:
            prefix = section + "_"
            if cluster_name.startswith(prefix):
                sub_cluster = cluster_name[len(prefix) :]
                return section, sub_cluster

        # Fallback: simple split (shouldn't happen with proper remapping)
        parts = cluster_name.split("_", 1)
        section = parts[0]
        sub_cluster = parts[1] if len(parts) > 1 else None
        return section, sub_cluster

    def get_section_info(self, data: DiagramData) -> dict[str, ClusterInfo]:
        """Extract structured section information from diagram data.

        Parses cluster names to extract section/sub-cluster relationships
        for use in rendering with nested subgraphs.

        Args:
            data: DiagramData from build().

        Returns:
            Dict mapping cluster name to ClusterInfo with section details.

        Example:
            >>> builder = StructuredDiagramBuilder()
            >>> data = builder.build()
            >>> info = builder.get_section_info(data)
            >>> priors_info = info.get("PRIORS_artist_pooling")
            >>> assert priors_info.section == "PRIORS"
        """
        result: dict[str, ClusterInfo] = {}

        for cluster_name, node_ids in data.clusters.items():
            section, sub_cluster = self._parse_cluster_name(cluster_name)

            result[cluster_name] = ClusterInfo(
                name=cluster_name,
                section=section,
                sub_cluster=sub_cluster,
                node_ids=list(node_ids),
            )

        return result
