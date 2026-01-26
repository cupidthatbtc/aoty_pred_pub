"""CSV introspector for data file metadata extraction.

This module provides introspection of CSV files in the project, extracting
row counts, column names, file sizes, and modification dates into NodeSpec
format for diagram generation.

The CSVIntrospector enables pipeline diagrams to show actual data file
metadata from live introspection rather than hardcoded values.

Example:
    >>> from aoty_pred.visualization.introspection import CSVIntrospector
    >>> ci = CSVIntrospector()
    >>> result = ci.introspect()
    >>> print(f"Found {len(result.nodes)} CSV files")
    Found 6 CSV files
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from aoty_pred.visualization.introspection.base import (
    IntrospectionResult,
    NodeSpec,
)

__all__ = ["CSVIntrospector"]

# Default CSV paths to introspect
DEFAULT_CSV_PATHS = [
    Path("data/raw/all_albums_full.csv"),
    Path("data/processed/cleaned_all.csv"),
    Path("data/processed/critic_score.csv"),
    Path("data/processed/user_score_minratings_5.csv"),
    Path("data/processed/user_score_minratings_10.csv"),
    Path("data/processed/user_score_minratings_25.csv"),
]


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable form.

    Args:
        size_bytes: File size in bytes.

    Returns:
        Human-readable size string (e.g., "1.2 KB", "34.5 MB").

    Example:
        >>> _format_size(1234)
        '1.2 KB'
        >>> _format_size(1234567)
        '1.2 MB'
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


class CSVIntrospector:
    """Introspector for CSV file metadata.

    Examines CSV files to extract row counts, column names, file sizes,
    and modification dates. Missing files produce placeholder nodes
    rather than errors.

    Each CSV file becomes a node with:
    - ID: "csv:{filename}" (namespaced for uniqueness)
    - Label: Multi-line with filename, row count, column count, size
    - Category: "data" (data file)
    - Cluster: "data_files"
    - Metadata: Full path, row count, columns, size, modified date

    Attributes:
        csv_paths: List of Path objects to introspect.
        source_type: Always "csv" for this introspector.

    Example:
        >>> ci = CSVIntrospector()
        >>> result = ci.introspect()
        >>> for node in result.nodes:
        ...     print(f"{node.id}: {node.metadata.get('row_count')} rows")
    """

    def __init__(self, csv_paths: list[Path] | None = None) -> None:
        """Initialize CSVIntrospector.

        Args:
            csv_paths: List of CSV file paths to introspect. If None,
                uses DEFAULT_CSV_PATHS.
        """
        self.csv_paths = csv_paths if csv_paths is not None else DEFAULT_CSV_PATHS.copy()

    @property
    def source_type(self) -> str:
        """Return the introspection source type identifier."""
        return "csv"

    def introspect(self) -> IntrospectionResult:
        """Introspect CSV files and return structured result.

        For each CSV path, checks if the file exists. Existing files
        get full metadata extracted. Missing files produce placeholder
        nodes with status="missing".

        Returns:
            IntrospectionResult containing nodes for each CSV file and
            cluster information grouping all files.

        Example:
            >>> ci = CSVIntrospector()
            >>> result = ci.introspect()
            >>> print(result.source_type)
            csv
        """
        nodes: list[NodeSpec] = []
        cluster_nodes: list[str] = []

        # Sort paths by name for deterministic output
        sorted_paths = sorted(self.csv_paths, key=lambda p: p.name)

        for csv_path in sorted_paths:
            node_id = f"csv:{csv_path.name}"

            if not csv_path.exists():
                # Create placeholder for missing file
                node = NodeSpec(
                    id=node_id,
                    label=f"File not found: {csv_path.name}",
                    category="data",
                    cluster="data_files",
                    metadata={
                        "path": str(csv_path),
                        "exists": False,
                        "status": "missing",
                    },
                )
            else:
                # Extract metadata from existing file
                stat = csv_path.stat()
                size_bytes = stat.st_size
                modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d")

                # Count rows efficiently (without loading entire file)
                with open(csv_path, encoding="utf-8-sig") as f:
                    row_count = sum(1 for _ in f) - 1  # Subtract header

                # Get column names from first row only
                df_head = pd.read_csv(csv_path, nrows=1, encoding="utf-8-sig")
                columns = list(df_head.columns)
                col_count = len(columns)

                # Format size for display
                size_human = _format_size(size_bytes)

                # Build multi-line label
                label = f"{csv_path.name}\n{row_count:,} rows | {col_count} cols\nSize: {size_human}"

                node = NodeSpec(
                    id=node_id,
                    label=label,
                    category="data",
                    cluster="data_files",
                    metadata={
                        "path": str(csv_path),
                        "row_count": row_count,
                        "column_count": col_count,
                        "columns": columns,
                        "size_bytes": size_bytes,
                        "modified": modified,
                        "exists": True,
                    },
                )

            nodes.append(node)
            cluster_nodes.append(node_id)

        return IntrospectionResult(
            source_type=self.source_type,
            nodes=nodes,
            edges=[],
            clusters={"data_files": sorted(cluster_nodes)},
            metadata={
                "file_count": len(nodes),
                "existing_count": sum(1 for n in nodes if n.metadata.get("exists")),
            },
        )
