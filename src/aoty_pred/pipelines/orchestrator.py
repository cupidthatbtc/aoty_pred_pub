"""Pipeline orchestrator for end-to-end execution with progress tracking.

This module provides the PipelineOrchestrator class that executes pipeline
stages in dependency order, with features for:
- Progress display using Rich
- Hash-based skip logic for incremental runs
- Environment verification via pixi.lock
- Error handling with fail-fast semantics
- Manifest tracking for reproducibility
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import time
from typing import TYPE_CHECKING

import structlog
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from aoty_pred.pipelines.errors import (
    ConvergenceError,
    EnvironmentError,
    PipelineError,
    StageSkipped,
)
from aoty_pred.pipelines.manifest import (
    GitStateModel,
    RunManifest,
    capture_environment,
    generate_run_id,
    load_run_manifest,
    save_run_manifest,
)
from aoty_pred.pipelines.stages import PipelineStage, StageContext, get_execution_order
from aoty_pred.utils.environment import ensure_environment_locked, verify_environment
from aoty_pred.utils.git_state import capture_git_state
from aoty_pred.utils.logging import is_interactive, setup_pipeline_logging
from aoty_pred.utils.random import set_seeds

if TYPE_CHECKING:
    pass

log = structlog.get_logger()


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution.

    Attributes:
        seed: Random seed for reproducibility (default 42).
        skip_existing: If True, skip stages with unchanged inputs (default False).
        stages: List of stage names to run, or None for all stages.
        dry_run: If True, log what would run without executing (default False).
        strict: If True, fail if pixi.lock missing (default False).
        verbose: If True, enable DEBUG logging (default False).
        resume: Run ID to resume, or None for fresh run.
        max_albums: Maximum albums per artist for model training (default 50).
            Albums beyond this limit use the same artist effect as the max position.

    Example:
        >>> config = PipelineConfig(seed=42, dry_run=True)
        >>> config.stages is None  # Run all stages
        True
    """

    seed: int = 42
    skip_existing: bool = False
    stages: list[str] | None = None
    dry_run: bool = False
    strict: bool = False
    verbose: bool = False
    resume: str | None = None
    max_albums: int = 50


class PipelineOrchestrator:
    """Orchestrates pipeline execution with progress tracking and error handling.

    The orchestrator manages the full pipeline lifecycle:
    1. Verify environment (pixi.lock check)
    2. Create run directory and manifest
    3. Execute stages in dependency order
    4. Track progress with Rich display
    5. Handle errors with fail-fast semantics
    6. Create outputs/latest symlink on success

    Attributes:
        config: Pipeline configuration options.
        output_base: Base directory for output runs (default "outputs").
        run_dir: Path to current run directory (set during run).
        manifest: Current run manifest (set during run).

    Example:
        >>> config = PipelineConfig(seed=42, dry_run=True)
        >>> orchestrator = PipelineOrchestrator(config)
        >>> exit_code = orchestrator.run()
    """

    def __init__(
        self,
        config: PipelineConfig,
        output_base: Path | str = Path("outputs"),
    ) -> None:
        """Initialize orchestrator with configuration.

        Args:
            config: Pipeline configuration.
            output_base: Base directory for outputs (default "outputs").
        """
        self.config = config
        self.output_base = Path(output_base)
        self.run_dir: Path | None = None
        self.manifest: RunManifest | None = None
        self._start_time: float = 0.0

    def run(self) -> int:
        """Execute the pipeline and return exit code.

        Runs all configured stages in dependency order with progress tracking.
        Creates run manifest, handles errors, and maintains output structure.

        Returns:
            Exit code: 0 on success, error's exit_code on failure.

        Raises:
            EnvironmentError: If strict=True and pixi.lock missing.
        """
        self._start_time = time()

        # 1. Verify environment
        try:
            self._verify_environment()
        except EnvironmentError as e:
            log.error("environment_verification_failed", error=str(e))
            return e.exit_code

        # 2. Set up run directory and manifest
        self._setup_run()

        # 3. Set up logging
        log_file = self.run_dir / "pipeline.log.json" if self.run_dir else None
        setup_pipeline_logging(verbose=self.config.verbose, log_file=log_file)

        # 4. Set random seeds
        set_seeds(self.config.seed)
        log.info(
            "pipeline_started",
            run_id=self.manifest.run_id if self.manifest else "unknown",
            seed=self.config.seed,
            dry_run=self.config.dry_run,
            stages=self.config.stages,
        )

        # 5. Get execution order
        try:
            stages = get_execution_order(self.config.stages)
        except KeyError as e:
            log.error("invalid_stage", error=str(e))
            return 1

        if not stages:
            log.warning("no_stages_to_execute")
            self._finalize_success()
            return 0

        # 6. Execute stages
        try:
            self._execute_stages(stages)
            self._finalize_success()
            return 0
        except PipelineError as e:
            self._handle_failure(e, e.stage)
            return e.exit_code
        except Exception as e:
            self._handle_failure(e, "unknown")
            return 1

    def _verify_environment(self) -> None:
        """Verify environment is locked for reproducibility.

        In strict mode, raises EnvironmentError if pixi.lock is not found.
        In non-strict mode, logs a warning but continues.
        """
        log.debug("verifying_environment", strict=self.config.strict)

        try:
            ensure_environment_locked(strict=self.config.strict)
        except Exception as e:
            # Re-raise as our EnvironmentError for consistent exit code
            raise EnvironmentError(str(e)) from e

        # Log environment status
        status = verify_environment()
        if status.is_reproducible:
            log.info(
                "environment_verified",
                pixi_lock_hash=status.pixi_lock_hash[:12] if status.pixi_lock_hash else None,
            )
        else:
            log.warning("environment_not_locked", warnings=status.warnings)

    def _setup_run(self) -> None:
        """Create run directory and initialize manifest."""
        # Handle resume vs fresh run
        if self.config.resume:
            self._setup_resume()
            return

        # Generate new run ID and directory
        run_id = generate_run_id()
        self.run_dir = self.output_base / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Capture git state and environment
        git_state = capture_git_state()
        environment = capture_environment()

        # Build command string for manifest
        command = self._build_command_string()

        # Create manifest
        self.manifest = RunManifest(
            run_id=run_id,
            created_at=datetime.now().isoformat(),
            command=command,
            flags={
                "seed": self.config.seed,
                "skip_existing": self.config.skip_existing,
                "stages": self.config.stages,
                "dry_run": self.config.dry_run,
                "strict": self.config.strict,
                "verbose": self.config.verbose,
                "resume": self.config.resume,
                "max_albums": self.config.max_albums,
            },
            seed=self.config.seed,
            git=GitStateModel.from_git_state(git_state),
            environment=environment,
            input_hashes={},
            stage_hashes={},
            stages_completed=[],
            stages_skipped=[],
            outputs={},
            success=False,
            error=None,
            duration_seconds=0.0,
        )

        # Save initial manifest
        save_run_manifest(self.manifest, self.run_dir)

    def _setup_resume(self) -> None:
        """Set up for resuming a previous run."""
        resume_id = self.config.resume

        # Try to find the run directory
        run_dir = self.output_base / resume_id
        failed_dir = self.output_base / "failed" / resume_id

        if run_dir.exists():
            self.run_dir = run_dir
        elif failed_dir.exists():
            # Move back from failed for retry
            self.run_dir = run_dir
            shutil.move(str(failed_dir), str(run_dir))
        else:
            raise PipelineError(
                f"Cannot find run to resume: {resume_id}",
                stage="setup",
            )

        # Load existing manifest
        manifest_path = self.run_dir / "manifest.json"
        if not manifest_path.exists():
            raise PipelineError(
                f"No manifest.json in run directory: {resume_id}",
                stage="setup",
            )

        self.manifest = load_run_manifest(manifest_path)
        log.info(
            "resuming_run",
            run_id=resume_id,
            completed_stages=self.manifest.stages_completed,
        )

    def _build_command_string(self) -> str:
        """Build command string representation for manifest."""
        parts = ["aoty-pipeline run"]

        if self.config.seed != 42:
            parts.append(f"--seed {self.config.seed}")
        if self.config.skip_existing:
            parts.append("--skip-existing")
        if self.config.stages:
            parts.append(f"--stages {','.join(self.config.stages)}")
        if self.config.dry_run:
            parts.append("--dry-run")
        if self.config.strict:
            parts.append("--strict")
        if self.config.verbose:
            parts.append("--verbose")
        if self.config.max_albums != 50:
            parts.append(f"--max-albums {self.config.max_albums}")

        return " ".join(parts)

    def _execute_stages(self, stages: list[PipelineStage]) -> None:
        """Execute stages with progress display.

        Args:
            stages: List of stages in execution order.
        """
        # Load previous manifest for skip detection
        previous_manifest: RunManifest | None = None
        if self.config.skip_existing and self.run_dir:
            latest_link = self.output_base / "latest"
            if latest_link.exists():
                try:
                    prev_manifest_path = latest_link / "manifest.json"
                    if prev_manifest_path.exists():
                        previous_manifest = load_run_manifest(prev_manifest_path)
                except Exception as e:
                    log.debug("could_not_load_previous_manifest", error=str(e), exc_info=True)

        # Set up progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            disable=not is_interactive(),
        ) as progress:
            task_id = progress.add_task("Pipeline", total=len(stages))

            for stage in stages:
                progress.update(task_id, description=f"[cyan]{stage.name}")

                # Check if this stage was already completed (for resume)
                if (
                    self.manifest
                    and stage.name in self.manifest.stages_completed
                ):
                    log.info(
                        "stage_already_completed",
                        stage=stage.name,
                    )
                    progress.advance(task_id)
                    continue

                # Check if stage should be skipped
                if self.config.skip_existing and not self.config.dry_run:
                    if stage.should_skip(previous_manifest, force=False):
                        log.info(
                            "stage_skipped",
                            stage=stage.name,
                            reason="inputs unchanged",
                        )
                        if self.manifest:
                            self.manifest.stages_skipped.append(stage.name)
                            save_run_manifest(self.manifest, self.run_dir)
                        progress.advance(task_id)
                        continue

                # Execute stage
                self._execute_stage(stage)
                progress.advance(task_id)

    def _create_stage_context(self) -> StageContext:
        """Create StageContext for stage execution.

        Returns:
            StageContext with current configuration.
        """
        return StageContext(
            run_dir=self.run_dir or Path("outputs"),
            seed=self.config.seed,
            strict=self.config.strict,
            verbose=self.config.verbose,
            manifest=self.manifest,
            max_albums=self.config.max_albums,
        )

    def _execute_stage(self, stage: PipelineStage) -> None:
        """Execute a single pipeline stage.

        Args:
            stage: Stage to execute.
        """
        log.info("stage_started", stage=stage.name, description=stage.description)

        if self.config.dry_run:
            log.info("stage_dry_run", stage=stage.name, would_run=stage.description)
            if self.manifest:
                self.manifest.stages_completed.append(stage.name)
                self.manifest.stage_hashes[stage.name] = stage.compute_input_hash()
                save_run_manifest(self.manifest, self.run_dir)
            return

        # Create stage context
        ctx = self._create_stage_context()

        # Execute the stage's run function
        if stage.run_fn is None:
            log.warning(
                "stage_no_run_fn",
                stage=stage.name,
                message="Stage has no run function defined",
            )
        else:
            try:
                stage.run_fn(ctx)
            except StageSkipped as e:
                log.info("stage_skipped", stage=stage.name, reason=e.message)
                if self.manifest:
                    self.manifest.stages_skipped.append(stage.name)
                    save_run_manifest(self.manifest, self.run_dir)
                return
            except ConvergenceError as e:
                # Handle convergence errors: fail in strict mode, warn otherwise
                if self.config.strict:
                    raise
                log.warning(
                    "convergence_warning",
                    stage=stage.name,
                    error=str(e),
                    message="Continuing despite convergence issues (strict=False)",
                )
            except PipelineError:
                raise
            except Exception as e:
                # Wrap unexpected errors
                raise PipelineError(str(e), stage=stage.name) from e

        # Update manifest
        if self.manifest:
            self.manifest.stages_completed.append(stage.name)
            self.manifest.stage_hashes[stage.name] = stage.compute_input_hash()
            save_run_manifest(self.manifest, self.run_dir)

        log.info("stage_completed", stage=stage.name)

    def _handle_failure(self, error: Exception, stage: str) -> None:
        """Handle pipeline failure with cleanup.

        Args:
            error: The exception that caused failure.
            stage: Name of the stage that failed.
        """
        log.error(
            "pipeline_failed",
            stage=stage,
            error=str(error),
            exc_info=True,
        )

        # Update manifest
        if self.manifest:
            self.manifest.success = False
            self.manifest.error = str(error)
            self.manifest.duration_seconds = time() - self._start_time
            if self.run_dir:
                save_run_manifest(self.manifest, self.run_dir)

        # Close logging handlers before moving directory (Windows file lock issue)
        self._close_log_handlers()

        # Move to failed directory
        if self.run_dir and self.run_dir.exists():
            failed_dir = self.output_base / "failed"
            failed_dir.mkdir(parents=True, exist_ok=True)
            failed_path = failed_dir / self.run_dir.name

            # Remove existing failed dir if present
            if failed_path.exists():
                shutil.rmtree(failed_path)

            try:
                shutil.move(str(self.run_dir), str(failed_path))
                log.info("run_moved_to_failed", path=str(failed_path))
            except PermissionError as e:
                # On Windows, file locks can persist; log but don't fail
                log.warning(
                    "failed_to_move_to_failed",
                    error=str(e),
                    run_dir=str(self.run_dir),
                )

    def _close_log_handlers(self) -> None:
        """Close file handlers to release locks (needed for Windows).

        On Windows, file handlers keep files locked which prevents moving
        directories containing log files. This closes all handlers on the
        root logger to release those locks.
        """
        root_logger = logging.getLogger()
        handlers_to_remove = []

        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                handlers_to_remove.append(handler)

        for handler in handlers_to_remove:
            root_logger.removeHandler(handler)

    def _finalize_success(self) -> None:
        """Finalize successful run with manifest update and symlink."""
        # Update manifest
        if self.manifest:
            self.manifest.success = True
            self.manifest.duration_seconds = time() - self._start_time
            if self.run_dir:
                save_run_manifest(self.manifest, self.run_dir)

        # Create latest symlink
        if self.run_dir:
            self._create_latest_link()

        log.info(
            "pipeline_completed",
            run_id=self.manifest.run_id if self.manifest else "unknown",
            duration=f"{self.manifest.duration_seconds:.2f}s" if self.manifest else "unknown",
            stages_completed=len(self.manifest.stages_completed) if self.manifest else 0,
            stages_skipped=len(self.manifest.stages_skipped) if self.manifest else 0,
        )

    def _create_latest_link(self) -> None:
        """Create outputs/latest symlink/junction to current run."""
        if not self.run_dir:
            return

        latest_link = self.output_base / "latest"

        # Remove existing link/junction
        if latest_link.exists() or latest_link.is_symlink():
            try:
                if sys.platform == "win32" and latest_link.is_dir():
                    # On Windows, junctions appear as directories
                    os.rmdir(latest_link)
                else:
                    latest_link.unlink()
            except Exception as e:
                log.warning("failed_to_remove_latest_link", error=str(e))
                return

        # Create new link
        try:
            if sys.platform == "win32":
                # Try symlink first (requires Developer Mode or admin)
                try:
                    os.symlink(self.run_dir, latest_link, target_is_directory=True)
                    log.debug("created_symlink", target=str(self.run_dir))
                except OSError:
                    # Fall back to directory junction (no special permissions)
                    # Validate paths don't contain shell metacharacters
                    link_str = str(latest_link)
                    target_str = str(self.run_dir)
                    if any(c in link_str + target_str for c in '&|;<>`$^%\r\n'):
                        log.warning("unsafe_path_characters", link=link_str, target=target_str)
                        return
                    result = subprocess.run(
                        ["cmd", "/c", "mklink", "/J", link_str, target_str],
                        capture_output=True,
                        check=True,
                    )
                    log.debug("created_junction", target=target_str)
            else:
                os.symlink(self.run_dir, latest_link, target_is_directory=True)
                log.debug("created_symlink", target=str(self.run_dir))
        except Exception as e:
            log.warning("failed_to_create_latest_link", error=str(e))


def run_pipeline(config: PipelineConfig, output_base: Path | str = Path("outputs")) -> int:
    """Convenience function to run pipeline with given configuration.

    Creates an orchestrator and runs the pipeline, returning the exit code.

    Args:
        config: Pipeline configuration.
        output_base: Base directory for outputs (default "outputs").

    Returns:
        Exit code: 0 on success, error's exit_code on failure.

    Example:
        >>> exit_code = run_pipeline(PipelineConfig(seed=42))
    """
    orchestrator = PipelineOrchestrator(config, output_base=output_base)
    return orchestrator.run()
