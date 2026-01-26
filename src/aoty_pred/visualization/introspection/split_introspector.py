"""Introspector for data split manifests.

This module provides SplitIntrospector, which extracts train/val/test split
statistics from manifest files and produces diagram nodes with partition info.

The introspector handles missing directories and manifests gracefully by
creating placeholder nodes instead of raising errors.

Example:
    >>> from aoty_pred.visualization.introspection import SplitIntrospector
    >>> si = SplitIntrospector()
    >>> result = si.introspect()
    >>> for node in result.nodes:
    ...     print(f"{node.id}: train={node.metadata.get('train_rows')}")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aoty_pred.visualization.introspection.base import (
    IntrospectionResult,
    NodeSpec,
)

__all__ = ["SplitIntrospector", "SPLIT_TYPES"]


# Split types to introspect
SPLIT_TYPES: list[str] = ["within_artist_temporal", "artist_disjoint"]


class SplitIntrospector:
    """Introspector for data split manifests.

    Extracts split statistics from JSON manifest files in the splits directory.
    Creates nodes showing train/val/test partition sizes and percentages.
    Handles missing directories and manifests gracefully with placeholders.

    Attributes:
        splits_dir: Path to the splits directory containing split type subdirs.

    Example:
        >>> si = SplitIntrospector()
        >>> result = si.introspect()
        >>> for node in result.nodes:
        ...     print(f"{node.id}: {node.metadata.get('train_rows')} train rows")
    """

    def __init__(self, splits_dir: Path | None = None) -> None:
        """Initialize SplitIntrospector.

        Args:
            splits_dir: Path to splits directory. Defaults to Path("data/splits").
        """
        self.splits_dir = splits_dir if splits_dir is not None else Path("data/splits")

    @property
    def source_type(self) -> str:
        """Return the identifier for this introspection source."""
        return "split"

    def _title_case(self, split_type: str) -> str:
        """Convert split_type to title case with spaces.

        Args:
            split_type: Underscore-separated split type (e.g., "within_artist_temporal")

        Returns:
            Title-cased string with spaces (e.g., "Within Artist Temporal")
        """
        return " ".join(word.capitalize() for word in split_type.split("_"))

    def _create_placeholder_node(self, split_type: str, message: str) -> NodeSpec:
        """Create placeholder node for missing split data.

        Args:
            split_type: The split type identifier.
            message: Placeholder message to display.

        Returns:
            NodeSpec with placeholder label.
        """
        title = self._title_case(split_type)
        return NodeSpec(
            id=f"split:{split_type}",
            label=f"{title}\n{message}",
            category="data",
            cluster="splits",
            metadata={
                "split_type": split_type,
                "placeholder": True,
                "message": message,
            },
        )

    def _load_manifest(self, split_dir: Path) -> dict[str, Any] | None:
        """Load the most recent manifest from a split directory.

        Args:
            split_dir: Path to the split type directory.

        Returns:
            Parsed manifest dict, or None if no manifest found.
        """
        manifests = sorted(split_dir.glob("split_*.json"), reverse=True)
        if not manifests:
            return None

        with manifests[0].open() as f:
            return json.load(f)

    def _create_split_node(self, split_type: str, manifest: dict[str, Any]) -> NodeSpec:
        """Create node from split manifest data.

        Args:
            split_type: The split type identifier.
            manifest: Parsed manifest dictionary.

        Returns:
            NodeSpec with split statistics.
        """
        splits = manifest.get("splits", {})
        params = manifest.get("parameters", {})

        train = splits.get("train", {})
        val = splits.get("validation", {})
        test = splits.get("test", {})

        train_rows = train.get("row_count", 0)
        val_rows = val.get("row_count", 0)
        test_rows = test.get("row_count", 0)

        train_artists = train.get("unique_artists", 0)
        val_artists = val.get("unique_artists", 0)
        test_artists = test.get("unique_artists", 0)

        total = train_rows + val_rows + test_rows

        # Calculate percentages (avoid division by zero)
        if total > 0:
            train_pct = (train_rows / total) * 100
            val_pct = (val_rows / total) * 100
            test_pct = (test_rows / total) * 100
        else:
            train_pct = val_pct = test_pct = 0.0

        random_state = params.get("random_state", "N/A")
        title = self._title_case(split_type)

        # Format label per CONTEXT.md
        label = (
            f"{title}\n"
            f"Train: {train_rows:,} ({train_pct:.1f}%)\n"
            f"Val: {val_rows:,} ({val_pct:.1f}%)\n"
            f"Test: {test_rows:,} ({test_pct:.1f}%)\n"
            f"Seed: {random_state}"
        )

        return NodeSpec(
            id=f"split:{split_type}",
            label=label,
            category="data",
            cluster="splits",
            metadata={
                "split_type": split_type,
                "train_rows": train_rows,
                "val_rows": val_rows,
                "test_rows": test_rows,
                "train_artists": train_artists,
                "val_artists": val_artists,
                "test_artists": test_artists,
                "parameters": params,
            },
        )

    def introspect(self) -> IntrospectionResult:
        """Introspect split manifests and return diagram elements.

        For each split type, loads the most recent manifest and creates a node
        showing partition statistics. Missing directories or manifests result
        in placeholder nodes.

        Returns:
            IntrospectionResult containing split nodes.
        """
        nodes: list[NodeSpec] = []
        cluster_nodes: list[str] = []

        for split_type in SPLIT_TYPES:
            split_dir = self.splits_dir / split_type

            # Check if directory exists
            if not split_dir.exists():
                node = self._create_placeholder_node(split_type, "Split not found")
                nodes.append(node)
                cluster_nodes.append(node.id)
                continue

            # Try to load manifest
            manifest = self._load_manifest(split_dir)
            if manifest is None:
                node = self._create_placeholder_node(split_type, "No manifest found")
                nodes.append(node)
                cluster_nodes.append(node.id)
                continue

            # Create node with split statistics
            node = self._create_split_node(split_type, manifest)
            nodes.append(node)
            cluster_nodes.append(node.id)

        return IntrospectionResult(
            source_type=self.source_type,
            nodes=nodes,
            edges=[],  # Splits don't have edges between them
            clusters={"splits": cluster_nodes},
            metadata={
                "splits_dir": str(self.splits_dir),
                "split_types": SPLIT_TYPES,
            },
        )
