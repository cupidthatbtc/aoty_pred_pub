"""Stage introspector for pipeline stage metadata extraction.

This module provides introspection of pipeline stages defined in
aoty_pred.pipelines.stages, extracting stage names, dependencies,
input/output paths, and descriptions into NodeSpec/EdgeSpec format.

The StageIntrospector serves as the proof-of-concept introspector for
the v4.0 introspection architecture, demonstrating the full pipeline
from introspection through aggregation to rendering.

Example:
    >>> from aoty_pred.visualization.introspection import StageIntrospector
    >>> si = StageIntrospector()
    >>> result = si.introspect()
    >>> print(f"Found {len(result.nodes)} pipeline stages")
    Found 6 pipeline stages
"""

from __future__ import annotations

from aoty_pred.pipelines.stages import build_pipeline_stages
from aoty_pred.visualization.introspection.base import (
    EdgeSpec,
    IntrospectionResult,
    NodeSpec,
)

__all__ = ["StageIntrospector"]


class StageIntrospector:
    """Introspector for pipeline stage metadata.

    Wraps build_pipeline_stages() to extract stage information and produce
    NodeSpec/EdgeSpec data suitable for diagram generation.

    Each pipeline stage becomes a node with:
    - ID: "stage:{name}" (namespaced for uniqueness)
    - Label: Multi-line with name, inputs, outputs
    - Category: "process" (computational step)
    - Cluster: "pipeline"
    - Metadata: Full paths, description, dependencies

    Dependencies between stages become edges with:
    - Source: "stage:{dependency_name}"
    - Target: "stage:{stage_name}"
    - Category: "dependency"

    Attributes:
        source_type: Always "stage" for this introspector.

    Example:
        >>> si = StageIntrospector()
        >>> result = si.introspect()
        >>> node_ids = [n.id for n in result.nodes]
        >>> assert "stage:data" in node_ids
        >>> assert "stage:train" in node_ids
    """

    @property
    def source_type(self) -> str:
        """Return the introspection source type identifier."""
        return "stage"

    def introspect(self) -> IntrospectionResult:
        """Introspect pipeline stages and return structured result.

        Extracts all stages from build_pipeline_stages(), creating nodes
        for each stage and edges for dependencies. Results are sorted
        by stage name for deterministic output.

        Returns:
            IntrospectionResult containing nodes for each stage, edges for
            dependencies, and cluster information grouping all stages.

        Example:
            >>> si = StageIntrospector()
            >>> result = si.introspect()
            >>> print(result.source_type)
            stage
            >>> len(result.nodes)
            6
        """
        stages = build_pipeline_stages()

        # Sort stages by name for deterministic output
        sorted_stages = sorted(stages, key=lambda s: s.name)

        nodes: list[NodeSpec] = []
        edges: list[EdgeSpec] = []
        cluster_nodes: list[str] = []

        for stage in sorted_stages:
            node_id = f"stage:{stage.name}"

            # Truncate paths to filenames for label readability
            inputs = [p.name for p in stage.input_paths[:3]]
            outputs = [p.name for p in stage.output_paths[:3]]

            # Add ellipsis if truncated
            if len(stage.input_paths) > 3:
                inputs.append("...")
            if len(stage.output_paths) > 3:
                outputs.append("...")

            # Build multi-line label
            inputs_str = ", ".join(inputs) if inputs else "(none)"
            outputs_str = ", ".join(outputs) if outputs else "(none)"
            label = f"{stage.name}\nin: {inputs_str}\nout: {outputs_str}"

            # Create node with full metadata
            node = NodeSpec(
                id=node_id,
                label=label,
                category="process",
                cluster="pipeline",
                metadata={
                    "description": stage.description,
                    "input_paths": [str(p) for p in stage.input_paths],
                    "output_paths": [str(p) for p in stage.output_paths],
                    "depends_on": list(stage.depends_on),
                },
            )
            nodes.append(node)
            cluster_nodes.append(node_id)

            # Create edges for dependencies
            for dep_name in sorted(stage.depends_on):
                edge = EdgeSpec(
                    source=f"stage:{dep_name}",
                    target=node_id,
                    category="dependency",
                )
                edges.append(edge)

        return IntrospectionResult(
            source_type=self.source_type,
            nodes=nodes,
            edges=edges,
            clusters={"pipeline": sorted(cluster_nodes)},
            metadata={
                "stage_count": len(nodes),
                "edge_count": len(edges),
            },
        )
