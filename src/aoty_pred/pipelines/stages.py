"""Pipeline stage definitions with hash-based skip detection.

This module defines the computational graph of pipeline stages with their
dependencies, input/output paths, and hash-based skip logic for incremental runs.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from aoty_pred.utils.hashing import sha256_file

if TYPE_CHECKING:
    from aoty_pred.pipelines.manifest import RunManifest


@dataclass
class PipelineStage:
    """A pipeline stage with input tracking for incremental runs.

    Each stage defines its inputs, outputs, and dependencies on other stages.
    The compute_input_hash method enables skip detection by comparing current
    input hashes against previously recorded values.

    Attributes:
        name: Unique identifier for the stage (e.g., "data", "splits", "train")
        description: Human-readable description of what the stage does
        run_fn: Function to execute the stage, or None for placeholder stages
        input_paths: List of file paths this stage reads from
        output_paths: List of file paths this stage creates
        depends_on: List of stage names that must run before this stage

    Example:
        >>> stage = PipelineStage(
        ...     name="data",
        ...     description="Prepare and clean raw data",
        ...     run_fn=None,
        ...     input_paths=[Path("data/raw/albums.csv")],
        ...     output_paths=[Path("data/processed/cleaned.parquet")],
        ...     depends_on=[],
        ... )
        >>> stage.compute_input_hash()  # Returns hash of input files
    """

    name: str
    description: str
    run_fn: Callable[..., None] | None
    input_paths: list[Path] = field(default_factory=list)
    output_paths: list[Path] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)

    def compute_input_hash(self) -> str:
        """Compute combined hash of all input files.

        Hashes all existing input files and combines them into a single
        hash for comparison during skip detection.

        Returns:
            Combined SHA256 hash of all input files, or empty string if
            no input files exist.

        Example:
            >>> stage.compute_input_hash()
            'abc123def456...'
        """
        hashes: list[str] = []

        for path in sorted(self.input_paths):
            if path.exists():
                hashes.append(sha256_file(path))

        if not hashes:
            return ""

        # Combine hashes deterministically
        combined = hashlib.sha256("".join(sorted(hashes)).encode()).hexdigest()
        return combined

    def should_skip(
        self,
        manifest: RunManifest | None,
        force: bool = False,
    ) -> bool:
        """Check if stage can be skipped (outputs exist, inputs unchanged).

        A stage can be skipped only if:
        1. force is False
        2. A previous manifest exists
        3. The stage was run in that manifest
        4. The current input hash matches the recorded hash
        5. All output files exist

        Args:
            manifest: Previous run manifest to compare against, or None.
            force: If True, never skip (always return False).

        Returns:
            True if stage can be safely skipped, False otherwise.

        Example:
            >>> if stage.should_skip(previous_manifest):
            ...     print(f"Skipping {stage.name} (inputs unchanged)")
        """
        if force:
            return False

        if manifest is None:
            return False

        # Check if this stage was run before
        prev_hash = manifest.stage_hashes.get(self.name)
        if prev_hash is None:
            return False

        # Check if inputs have changed
        current_hash = self.compute_input_hash()
        if current_hash != prev_hash:
            return False

        # Check all outputs exist
        if not all(p.exists() for p in self.output_paths):
            return False

        return True


def _topological_sort(
    stages: list[PipelineStage],
    stage_names: set[str] | None = None,
) -> list[PipelineStage]:
    """Sort stages by dependencies using Kahn's algorithm.

    Args:
        stages: List of stages to sort.
        stage_names: Optional set of stage names to include. If None, include all.

    Returns:
        Stages in dependency order (dependencies first).

    Raises:
        ValueError: If there is a cycle in dependencies or missing dependency.
    """
    # Build adjacency list and in-degree count
    name_to_stage = {s.name: s for s in stages}

    # Filter to requested stages if specified
    if stage_names is not None:
        stages = [s for s in stages if s.name in stage_names]

    # Validate all dependencies exist
    all_stage_names = set(name_to_stage.keys())
    for stage in stages:
        for dep in stage.depends_on:
            if dep not in all_stage_names:
                raise ValueError(f"Stage '{stage.name}' depends on unknown stage '{dep}'")

    # Build in-degree map
    in_degree: dict[str, int] = {s.name: 0 for s in stages}
    for stage in stages:
        for dep in stage.depends_on:
            if dep in in_degree:  # Only count deps within our stage set
                pass  # dep is a prerequisite, handled below
            # Count only if the dependency is in our set
        for other in stages:
            if stage.name in other.depends_on:
                in_degree[stage.name] += 0  # We're building it wrong

    # Actually build in-degree correctly
    in_degree = {s.name: 0 for s in stages}
    for stage in stages:
        for dep in stage.depends_on:
            if dep in {s.name for s in stages}:
                # dep must come before stage
                pass
    # For each stage, count how many of its dependencies are in our set
    stage_name_set = {s.name for s in stages}
    in_degree = {s.name: 0 for s in stages}
    for stage in stages:
        for dep in stage.depends_on:
            if dep in stage_name_set:
                in_degree[stage.name] += 1

    # Kahn's algorithm
    queue = deque([s for s in stages if in_degree[s.name] == 0])
    result: list[PipelineStage] = []

    while queue:
        current = queue.popleft()
        result.append(current)

        # Decrease in-degree for stages that depend on current
        for stage in stages:
            if current.name in stage.depends_on:
                in_degree[stage.name] -= 1
                if in_degree[stage.name] == 0:
                    queue.append(stage)

    if len(result) != len(stages):
        # Cycle detected
        remaining = [s.name for s in stages if s not in result]
        raise ValueError(f"Circular dependency detected among stages: {remaining}")

    return result


# Define all pipeline stages
# Note: run_fn is None initially - orchestrator will wire them up
PIPELINE_STAGES: list[PipelineStage] = [
    PipelineStage(
        name="data",
        description="Prepare and clean raw album data",
        run_fn=None,
        input_paths=[Path("data/raw/all_albums_full.csv")],
        output_paths=[
            Path("data/processed/cleaned_all.parquet"),
            Path("data/processed/user_score_minratings_5.parquet"),
            Path("data/processed/user_score_minratings_10.parquet"),
            Path("data/processed/user_score_minratings_25.parquet"),
            Path("data/processed/critic_score.parquet"),
        ],
        depends_on=[],
    ),
    PipelineStage(
        name="splits",
        description="Create train/validation/test splits",
        run_fn=None,
        input_paths=[Path("data/processed/user_score_minratings_10.parquet")],
        output_paths=[
            Path("data/splits/within_artist_temporal/train.parquet"),
            Path("data/splits/within_artist_temporal/validation.parquet"),
            Path("data/splits/within_artist_temporal/test.parquet"),
            Path("data/splits/artist_disjoint/train.parquet"),
            Path("data/splits/artist_disjoint/validation.parquet"),
            Path("data/splits/artist_disjoint/test.parquet"),
        ],
        depends_on=["data"],
    ),
    PipelineStage(
        name="features",
        description="Build feature matrices from split data",
        run_fn=None,
        input_paths=[
            Path("data/splits/within_artist_temporal/train.parquet"),
            Path("data/splits/within_artist_temporal/validation.parquet"),
            Path("data/splits/within_artist_temporal/test.parquet"),
        ],
        output_paths=[
            Path("data/features/train_features.parquet"),
            Path("data/features/validation_features.parquet"),
            Path("data/features/test_features.parquet"),
        ],
        depends_on=["splits"],
    ),
    PipelineStage(
        name="train",
        description="Fit Bayesian models on training data",
        run_fn=None,
        input_paths=[
            Path("data/features/train_features.parquet"),
            Path("data/features/validation_features.parquet"),
        ],
        output_paths=[
            Path("models/user_score_model/"),
            Path("models/critic_score_model/"),
        ],
        depends_on=["features"],
    ),
    PipelineStage(
        name="evaluate",
        description="Run model evaluation and diagnostics",
        run_fn=None,
        input_paths=[
            Path("models/user_score_model/"),
            Path("models/critic_score_model/"),
            Path("data/features/test_features.parquet"),
        ],
        output_paths=[
            Path("outputs/evaluation/metrics.json"),
            Path("outputs/evaluation/diagnostics.json"),
        ],
        depends_on=["train"],
    ),
    PipelineStage(
        name="report",
        description="Generate publication artifacts (figures, tables)",
        run_fn=None,
        input_paths=[
            Path("outputs/evaluation/metrics.json"),
            Path("outputs/evaluation/diagnostics.json"),
        ],
        output_paths=[
            Path("reports/figures/"),
            Path("reports/tables/"),
        ],
        depends_on=["evaluate"],
    ),
]


def get_execution_order(stages: list[str] | None = None) -> list[PipelineStage]:
    """Get stages in dependency-respecting execution order.

    Args:
        stages: List of stage names to include, or None for all stages.
            If provided, stages are returned in topological order respecting
            dependencies between the specified stages.

    Returns:
        List of PipelineStage objects in execution order.

    Raises:
        KeyError: If an unknown stage name is provided.
        ValueError: If there is a circular dependency.

    Example:
        >>> order = get_execution_order()
        >>> [s.name for s in order]
        ['data', 'splits', 'features', 'train', 'evaluate', 'report']

        >>> order = get_execution_order(["features", "splits"])
        >>> [s.name for s in order]
        ['splits', 'features']
    """
    if stages is None:
        return _topological_sort(PIPELINE_STAGES)

    # Validate stage names
    valid_names = {s.name for s in PIPELINE_STAGES}
    for name in stages:
        if name not in valid_names:
            raise KeyError(f"Unknown stage: '{name}'. Valid stages: {sorted(valid_names)}")

    # Filter and sort
    stage_set = set(stages)
    return _topological_sort(PIPELINE_STAGES, stage_set)


def get_stage(name: str) -> PipelineStage:
    """Look up a stage by name.

    Args:
        name: Stage name to look up.

    Returns:
        PipelineStage with the given name.

    Raises:
        KeyError: If no stage with that name exists.

    Example:
        >>> stage = get_stage("train")
        >>> stage.description
        'Fit Bayesian models on training data'
    """
    for stage in PIPELINE_STAGES:
        if stage.name == name:
            return stage

    valid_names = sorted(s.name for s in PIPELINE_STAGES)
    raise KeyError(f"Unknown stage: '{name}'. Valid stages: {valid_names}")
