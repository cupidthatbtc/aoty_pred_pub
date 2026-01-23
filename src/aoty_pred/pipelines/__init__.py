"""Pipeline modules for end-to-end workflow.

This package provides the pipeline orchestrator and supporting components
for reproducible ML workflow execution.

Main Entry Points:
    run_pipeline: Convenience function to execute pipeline
    PipelineConfig: Configuration for pipeline execution
    PipelineOrchestrator: Full orchestrator class for advanced use

Example:
    >>> from aoty_pred.pipelines import run_pipeline, PipelineConfig
    >>> config = PipelineConfig(seed=42, dry_run=True)
    >>> exit_code = run_pipeline(config)
"""

from aoty_pred.pipelines.orchestrator import (
    PipelineConfig,
    PipelineOrchestrator,
    run_pipeline,
)
from aoty_pred.pipelines.manifest import (
    EnvironmentInfo,
    GitStateModel,
    RunManifest,
    capture_environment,
    generate_run_id,
    load_run_manifest,
    save_run_manifest,
)
from aoty_pred.pipelines.stages import (
    PipelineStage,
    build_pipeline_stages,
    get_execution_order,
    get_stage,
)
from aoty_pred.pipelines.errors import (
    ConvergenceError,
    DataValidationError,
    PipelineError,
    StageError,
)

__all__ = [
    # Orchestrator
    "PipelineConfig",
    "PipelineOrchestrator",
    "run_pipeline",
    # Manifest
    "EnvironmentInfo",
    "GitStateModel",
    "RunManifest",
    "capture_environment",
    "generate_run_id",
    "load_run_manifest",
    "save_run_manifest",
    # Stages
    "PipelineStage",
    "build_pipeline_stages",
    "get_execution_order",
    "get_stage",
    # Errors
    "ConvergenceError",
    "DataValidationError",
    "PipelineError",
    "StageError",
]
