"""Schema introspector for Pandera schema metadata extraction.

This module provides introspection of Pandera DataFrameSchema objects,
extracting column names, data types, nullable status, and validation
checks into NodeSpec format for diagram generation.

The SchemaIntrospector enables pipeline diagrams to show actual schema
definitions from live introspection rather than hardcoded values.

Example:
    >>> from aoty_pred.visualization.introspection import SchemaIntrospector
    >>> si = SchemaIntrospector()
    >>> result = si.introspect()
    >>> print(f"Found {len(result.nodes)} schemas")
    Found 1 schemas
"""

from __future__ import annotations

from typing import Any

import pandera.pandas as pa

from aoty_pred.data.validation import RawAlbumSchema
from aoty_pred.visualization.introspection.base import (
    IntrospectionResult,
    NodeSpec,
)

__all__ = ["SchemaIntrospector"]

# Default schemas to introspect
DEFAULT_SCHEMAS: dict[str, pa.DataFrameSchema] = {
    "RawAlbumSchema": RawAlbumSchema,
}

# Maximum columns to show in label before truncating
MAX_COLUMNS_IN_LABEL = 8


def _format_check(check: Any) -> str:
    """Format a Pandera check for display.

    Args:
        check: Pandera Check object.

    Returns:
        Human-readable check description (e.g., "in_range(min=0, max=100)").
    """
    name = check.name if check.name else "custom"
    if check.statistics:
        params = ", ".join(f"{k}={v}" for k, v in check.statistics.items())
        return f"{name}({params})"
    return name


class SchemaIntrospector:
    """Introspector for Pandera schema metadata.

    Examines Pandera DataFrameSchema objects to extract column definitions,
    data types, nullable status, and validation checks.

    Each schema becomes a node with:
    - ID: "schema:{schema_name}" (namespaced for uniqueness)
    - Label: Multi-line with schema name and column definitions
    - Category: "config" (configuration/schema definition)
    - Cluster: "schemas"
    - Metadata: Full column details including checks

    Attributes:
        schemas: Dict mapping schema names to DataFrameSchema objects.
        source_type: Always "schema" for this introspector.

    Example:
        >>> si = SchemaIntrospector()
        >>> result = si.introspect()
        >>> for node in result.nodes:
        ...     print(f"{node.id}: {node.metadata.get('column_count')} columns")
    """

    def __init__(
        self, schemas: dict[str, pa.DataFrameSchema] | None = None
    ) -> None:
        """Initialize SchemaIntrospector.

        Args:
            schemas: Dict mapping schema names to DataFrameSchema objects.
                If None, uses DEFAULT_SCHEMAS (RawAlbumSchema).
        """
        self.schemas = schemas if schemas is not None else DEFAULT_SCHEMAS.copy()

    @property
    def source_type(self) -> str:
        """Return the introspection source type identifier."""
        return "schema"

    def introspect(self) -> IntrospectionResult:
        """Introspect Pandera schemas and return structured result.

        For each schema, extracts column names, types, nullable status,
        and validation checks.

        Returns:
            IntrospectionResult containing nodes for each schema and
            cluster information grouping all schemas.

        Example:
            >>> si = SchemaIntrospector()
            >>> result = si.introspect()
            >>> print(result.source_type)
            schema
        """
        nodes: list[NodeSpec] = []
        cluster_nodes: list[str] = []

        # Sort schemas by name for deterministic output
        sorted_schemas = sorted(self.schemas.items())

        for schema_name, schema in sorted_schemas:
            node_id = f"schema:{schema_name}"

            # Extract column information
            columns_info: list[dict[str, Any]] = []
            label_lines: list[str] = [schema_name]

            # Get columns sorted by name for determinism
            sorted_columns = sorted(schema.columns.items())

            for col_name, col in sorted_columns:
                # Get dtype as string
                dtype_str = str(col.dtype) if col.dtype else "any"

                # Nullable indicator
                nullable = col.nullable
                nullable_marker = "?" if nullable else ""

                # Extract checks
                checks_info: list[dict[str, Any]] = []
                for check in col.checks:
                    check_info = {
                        "name": check.name if check.name else "custom",
                        "statistics": check.statistics if check.statistics else {},
                    }
                    checks_info.append(check_info)

                columns_info.append(
                    {
                        "name": col_name,
                        "dtype": dtype_str,
                        "nullable": nullable,
                        "checks": checks_info,
                    }
                )

                # Add to label (limited to MAX_COLUMNS_IN_LABEL)
                if len(label_lines) <= MAX_COLUMNS_IN_LABEL:
                    label_lines.append(f"  {col_name}: {dtype_str}{nullable_marker}")

            # Add truncation indicator
            if len(sorted_columns) > MAX_COLUMNS_IN_LABEL:
                remaining = len(sorted_columns) - MAX_COLUMNS_IN_LABEL
                label_lines.append(f"  ... +{remaining} more")

            # Build label
            label = "\n".join(label_lines)

            node = NodeSpec(
                id=node_id,
                label=label,
                category="config",
                cluster="schemas",
                metadata={
                    "schema_name": schema_name,
                    "column_count": len(sorted_columns),
                    "strict": schema.strict,
                    "coerce": schema.coerce,
                    "columns": columns_info,
                },
            )

            nodes.append(node)
            cluster_nodes.append(node_id)

        return IntrospectionResult(
            source_type=self.source_type,
            nodes=nodes,
            edges=[],
            clusters={"schemas": sorted(cluster_nodes)},
            metadata={
                "schema_count": len(nodes),
            },
        )
