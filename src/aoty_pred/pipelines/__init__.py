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
    PIPELINE_STAGES,
    PipelineStage,
    get_execution_order,
    get_stage,
)
from aoty_pred.pipelines.errors import (
    ConvergenceError,
    DataValidationError,
    PipelineError,
    StageError,
)

# Legacy pipeline modules (for backwards compatibility)
from aoty_pred.pipelines import (
    build_features,
    create_splits,
    predict_next,
    prepare_dataset,
    publication,
    sensitivity,
    train_bayes,
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
    "PIPELINE_STAGES",
    "PipelineStage",
    "get_execution_order",
    "get_stage",
    # Errors
    "ConvergenceError",
    "DataValidationError",
    "PipelineError",
    "StageError",
    # Legacy modules
    "build_features",
    "create_splits",
    "predict_next",
    "prepare_dataset",
    "publication",
    "sensitivity",
    "train_bayes",
]
