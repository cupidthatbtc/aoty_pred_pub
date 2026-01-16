"""Tests for pipeline orchestrator."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aoty_pred.pipelines.errors import (
    ConvergenceError,
    EnvironmentError,
    PipelineError,
    StageError,
    StageSkipped,
)
from aoty_pred.pipelines.orchestrator import (
    PipelineConfig,
    PipelineOrchestrator,
    run_pipeline,
)


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_default_values(self):
        """PipelineConfig has sensible defaults."""
        config = PipelineConfig()
        assert config.seed == 42
        assert config.skip_existing is False
        assert config.stages is None
        assert config.dry_run is False
        assert config.strict is False
        assert config.verbose is False
        assert config.resume is None

    def test_custom_values(self):
        """PipelineConfig accepts custom values."""
        config = PipelineConfig(
            seed=123,
            skip_existing=True,
            stages=["data", "splits"],
            dry_run=True,
            strict=True,
            verbose=True,
            resume="2026-01-19_143052",
        )
        assert config.seed == 123
        assert config.skip_existing is True
        assert config.stages == ["data", "splits"]
        assert config.dry_run is True
        assert config.strict is True
        assert config.verbose is True
        assert config.resume == "2026-01-19_143052"


class TestPipelineOrchestratorInit:
    """Tests for PipelineOrchestrator initialization."""

    def test_basic_init(self):
        """Orchestrator initializes with config."""
        config = PipelineConfig()
        orchestrator = PipelineOrchestrator(config)

        assert orchestrator.config == config
        assert orchestrator.output_base == Path("outputs")
        assert orchestrator.run_dir is None
        assert orchestrator.manifest is None

    def test_custom_output_base(self, tmp_path: Path):
        """Orchestrator accepts custom output base."""
        config = PipelineConfig()
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)

        assert orchestrator.output_base == tmp_path


class TestDryRunMode:
    """Tests for dry_run mode."""

    @patch("aoty_pred.pipelines.orchestrator.ensure_environment_locked")
    @patch("aoty_pred.pipelines.orchestrator.verify_environment")
    def test_dry_run_does_not_execute_stages(
        self,
        mock_verify: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
    ):
        """Dry run mode logs but doesn't execute stage functions."""
        # Set up mock environment verification
        mock_verify.return_value = MagicMock(
            is_reproducible=True,
            pixi_lock_hash="abc123def456",
            warnings=[],
        )

        # Create a mock run_fn that would fail if called for real execution
        was_called = {"value": False}

        def failing_run_fn():
            was_called["value"] = True
            raise RuntimeError("Should not be called in dry run")

        # Patch the stages to have a controlled run_fn
        with patch("aoty_pred.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_stage = MagicMock()
            mock_stage.name = "test_stage"
            mock_stage.description = "Test stage"
            mock_stage.run_fn = failing_run_fn
            mock_stage.compute_input_hash.return_value = "hash123"
            mock_order.return_value = [mock_stage]

            config = PipelineConfig(dry_run=True)
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            exit_code = orchestrator.run()

            assert exit_code == 0
            assert was_called["value"] is False  # run_fn was NOT called

    @patch("aoty_pred.pipelines.orchestrator.ensure_environment_locked")
    @patch("aoty_pred.pipelines.orchestrator.verify_environment")
    def test_dry_run_records_stage_in_manifest(
        self,
        mock_verify: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
    ):
        """Dry run still records stages in manifest."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True,
            pixi_lock_hash="abc123",
            warnings=[],
        )

        with patch("aoty_pred.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_stage = MagicMock()
            mock_stage.name = "data"
            mock_stage.description = "Prepare data"
            mock_stage.run_fn = None
            mock_stage.compute_input_hash.return_value = "hash123"
            mock_order.return_value = [mock_stage]

            config = PipelineConfig(dry_run=True)
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            orchestrator.run()

            assert orchestrator.manifest is not None
            assert "data" in orchestrator.manifest.stages_completed


class TestSkipExisting:
    """Tests for skip_existing mode."""

    @patch("aoty_pred.pipelines.orchestrator.ensure_environment_locked")
    @patch("aoty_pred.pipelines.orchestrator.verify_environment")
    def test_skip_existing_uses_manifest_hashes(
        self,
        mock_verify: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
    ):
        """Skip existing checks hash from previous manifest."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True,
            pixi_lock_hash="abc123",
            warnings=[],
        )

        with patch("aoty_pred.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_stage = MagicMock()
            mock_stage.name = "data"
            mock_stage.description = "Prepare data"
            mock_stage.run_fn = None
            # Make should_skip return True to simulate unchanged inputs
            mock_stage.should_skip.return_value = True
            mock_stage.compute_input_hash.return_value = "same_hash"
            mock_order.return_value = [mock_stage]

            config = PipelineConfig(skip_existing=True)
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            orchestrator.run()

            # Stage should have been checked for skip
            mock_stage.should_skip.assert_called()


class TestErrorHandling:
    """Tests for error handling."""

    @patch("aoty_pred.pipelines.orchestrator.ensure_environment_locked")
    @patch("aoty_pred.pipelines.orchestrator.verify_environment")
    def test_error_returns_correct_exit_code(
        self,
        mock_verify: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
    ):
        """Failed runs return correct error exit code."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True,
            pixi_lock_hash="abc123",
            warnings=[],
        )

        with patch("aoty_pred.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_stage = MagicMock()
            mock_stage.name = "data"
            mock_stage.description = "Prepare data"
            mock_stage.run_fn = MagicMock(side_effect=StageError("Test error", "data"))
            mock_stage.compute_input_hash.return_value = "hash123"
            mock_order.return_value = [mock_stage]

            config = PipelineConfig()
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            exit_code = orchestrator.run()

            # Should return error exit code
            assert exit_code == 4  # StageError exit code

    @patch("aoty_pred.pipelines.orchestrator.ensure_environment_locked")
    @patch("aoty_pred.pipelines.orchestrator.verify_environment")
    def test_error_updates_manifest_success_false(
        self,
        mock_verify: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
    ):
        """Failed run updates manifest success to False."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True,
            pixi_lock_hash="abc123",
            warnings=[],
        )

        with patch("aoty_pred.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_stage = MagicMock()
            mock_stage.name = "train"
            mock_stage.description = "Train model"
            mock_stage.run_fn = MagicMock(side_effect=ConvergenceError("R-hat exceeded", "train"))
            mock_stage.compute_input_hash.return_value = "hash123"
            mock_order.return_value = [mock_stage]

            config = PipelineConfig(strict=True)
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            exit_code = orchestrator.run()

            assert exit_code == 2  # ConvergenceError exit code

            # Manifest should record failure
            assert orchestrator.manifest is not None
            assert orchestrator.manifest.success is False
            assert "R-hat exceeded" in orchestrator.manifest.error

    @patch("aoty_pred.pipelines.orchestrator.ensure_environment_locked")
    @patch("aoty_pred.pipelines.orchestrator.verify_environment")
    def test_error_attempts_to_move_to_failed(
        self,
        mock_verify: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
    ):
        """Failed runs attempt to move to outputs/failed/."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True,
            pixi_lock_hash="abc123",
            warnings=[],
        )

        with patch("aoty_pred.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_stage = MagicMock()
            mock_stage.name = "data"
            mock_stage.description = "Prepare data"
            mock_stage.run_fn = MagicMock(side_effect=StageError("Test error", "data"))
            mock_stage.compute_input_hash.return_value = "hash123"
            mock_order.return_value = [mock_stage]

            with patch("aoty_pred.pipelines.orchestrator.shutil.move") as mock_move:
                config = PipelineConfig()
                orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
                orchestrator.run()

                # Should attempt to move
                mock_move.assert_called()


class TestManifestSaving:
    """Tests for manifest persistence."""

    @patch("aoty_pred.pipelines.orchestrator.ensure_environment_locked")
    @patch("aoty_pred.pipelines.orchestrator.verify_environment")
    def test_manifest_saved_after_each_stage(
        self,
        mock_verify: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
    ):
        """Manifest is saved incrementally after each stage."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True,
            pixi_lock_hash="abc123",
            warnings=[],
        )

        save_count = {"value": 0}

        def count_saves(manifest, run_dir):
            save_count["value"] += 1
            # Actually save the manifest
            from aoty_pred.pipelines.manifest import save_run_manifest as real_save
            return real_save(manifest, run_dir)

        with patch("aoty_pred.pipelines.orchestrator.get_execution_order") as mock_order:
            stage1 = MagicMock()
            stage1.name = "stage1"
            stage1.description = "Stage 1"
            stage1.run_fn = None
            stage1.compute_input_hash.return_value = "hash1"

            stage2 = MagicMock()
            stage2.name = "stage2"
            stage2.description = "Stage 2"
            stage2.run_fn = None
            stage2.compute_input_hash.return_value = "hash2"

            mock_order.return_value = [stage1, stage2]

            with patch(
                "aoty_pred.pipelines.orchestrator.save_run_manifest",
                side_effect=count_saves,
            ):
                config = PipelineConfig()
                orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
                orchestrator.run()

            # Initial save + 2 stages + final success = at least 4 saves
            assert save_count["value"] >= 3


class TestEnvironmentVerification:
    """Tests for environment verification."""

    def test_environment_verified_at_startup(self, tmp_path: Path):
        """Environment verification is called at pipeline start."""
        with patch("aoty_pred.pipelines.orchestrator.ensure_environment_locked") as mock_ensure:
            with patch("aoty_pred.pipelines.orchestrator.verify_environment") as mock_verify:
                mock_verify.return_value = MagicMock(
                    is_reproducible=True,
                    pixi_lock_hash="abc123",
                    warnings=[],
                )

                with patch("aoty_pred.pipelines.orchestrator.get_execution_order") as mock_order:
                    mock_order.return_value = []

                    config = PipelineConfig()
                    orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
                    orchestrator.run()

                    # ensure_environment_locked should be called
                    mock_ensure.assert_called_once()

    def test_strict_mode_fails_when_pixi_lock_missing(self, tmp_path: Path):
        """Strict mode fails if pixi.lock is not found."""
        with patch(
            "aoty_pred.pipelines.orchestrator.ensure_environment_locked",
            side_effect=EnvironmentError("pixi.lock not found"),
        ):
            config = PipelineConfig(strict=True)
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            exit_code = orchestrator.run()

            assert exit_code == 5  # EnvironmentError exit code


class TestRunPipeline:
    """Tests for run_pipeline convenience function."""

    @patch("aoty_pred.pipelines.orchestrator.ensure_environment_locked")
    @patch("aoty_pred.pipelines.orchestrator.verify_environment")
    def test_run_pipeline_returns_exit_code(
        self,
        mock_verify: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
    ):
        """run_pipeline returns orchestrator exit code."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True,
            pixi_lock_hash="abc123",
            warnings=[],
        )

        with patch("aoty_pred.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_order.return_value = []

            config = PipelineConfig()
            exit_code = run_pipeline(config, output_base=tmp_path)

            assert exit_code == 0


class TestStageSkipped:
    """Tests for StageSkipped control flow."""

    @patch("aoty_pred.pipelines.orchestrator.ensure_environment_locked")
    @patch("aoty_pred.pipelines.orchestrator.verify_environment")
    def test_stage_skipped_is_not_error(
        self,
        mock_verify: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
    ):
        """StageSkipped exception doesn't cause pipeline failure."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True,
            pixi_lock_hash="abc123",
            warnings=[],
        )

        with patch("aoty_pred.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_stage = MagicMock()
            mock_stage.name = "data"
            mock_stage.description = "Prepare data"
            mock_stage.run_fn = MagicMock(side_effect=StageSkipped("Inputs unchanged"))
            mock_stage.compute_input_hash.return_value = "hash123"
            mock_order.return_value = [mock_stage]

            config = PipelineConfig()
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            exit_code = orchestrator.run()

            # Should succeed (not fail on StageSkipped)
            assert exit_code == 0
            # Stage should be in skipped list
            assert "data" in orchestrator.manifest.stages_skipped


class TestLatestSymlink:
    """Tests for outputs/latest symlink/junction creation."""

    @patch("aoty_pred.pipelines.orchestrator.ensure_environment_locked")
    @patch("aoty_pred.pipelines.orchestrator.verify_environment")
    def test_latest_link_created_on_success(
        self,
        mock_verify: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
    ):
        """Successful run creates outputs/latest link."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True,
            pixi_lock_hash="abc123",
            warnings=[],
        )

        with patch("aoty_pred.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_order.return_value = []

            config = PipelineConfig()
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            exit_code = orchestrator.run()

            assert exit_code == 0

            latest_link = tmp_path / "latest"
            # On Windows, this might be a junction (appears as dir)
            # On Unix, this is a symlink
            assert latest_link.exists() or latest_link.is_symlink()

    @patch("aoty_pred.pipelines.orchestrator.ensure_environment_locked")
    @patch("aoty_pred.pipelines.orchestrator.verify_environment")
    def test_latest_link_not_created_on_failure(
        self,
        mock_verify: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
    ):
        """Failed run does not create/update outputs/latest link."""
        mock_verify.return_value = MagicMock(
            is_reproducible=True,
            pixi_lock_hash="abc123",
            warnings=[],
        )

        with patch("aoty_pred.pipelines.orchestrator.get_execution_order") as mock_order:
            mock_stage = MagicMock()
            mock_stage.name = "data"
            mock_stage.description = "Prepare data"
            mock_stage.run_fn = MagicMock(side_effect=StageError("Test error", "data"))
            mock_stage.compute_input_hash.return_value = "hash123"
            mock_order.return_value = [mock_stage]

            config = PipelineConfig()
            orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
            exit_code = orchestrator.run()

            assert exit_code == 4

            # latest link should NOT exist (or point to previous successful run, not this failed one)
            latest_link = tmp_path / "latest"
            # For a fresh run that fails, latest should not exist
            assert not latest_link.exists() and not latest_link.is_symlink()


class TestBuildCommandString:
    """Tests for command string building."""

    def test_default_command(self, tmp_path: Path):
        """Default config produces simple command."""
        config = PipelineConfig()
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        cmd = orchestrator._build_command_string()

        assert cmd == "aoty-pipeline run"

    def test_command_with_options(self, tmp_path: Path):
        """Options are included in command string."""
        config = PipelineConfig(
            seed=123,
            skip_existing=True,
            stages=["data", "splits"],
            dry_run=True,
            strict=True,
            verbose=True,
        )
        orchestrator = PipelineOrchestrator(config, output_base=tmp_path)
        cmd = orchestrator._build_command_string()

        assert "--seed 123" in cmd
        assert "--skip-existing" in cmd
        assert "--stages data,splits" in cmd
        assert "--dry-run" in cmd
        assert "--strict" in cmd
        assert "--verbose" in cmd
