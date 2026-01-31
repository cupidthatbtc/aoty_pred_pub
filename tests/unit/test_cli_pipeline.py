"""Tests for CLI pipeline commands.

Tests the CLI entry points for pipeline execution, including the main
'run' command and individual stage subcommands.
"""

import re

import pytest
from typer.testing import CliRunner

from aoty_pred.cli import __version__, app

runner = CliRunner()


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class TestCLIHelp:
    """Tests for CLI help output."""

    def test_main_help_shows_commands(self):
        """Main help shows run and stage commands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.stdout
        assert "stage" in result.stdout

    def test_run_help_shows_all_options(self):
        """Run command help shows all expected options."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0

        # Check all flags are documented (strip ANSI codes for CI)
        output = strip_ansi(result.stdout)
        assert "--seed" in output
        assert "--skip-existing" in output
        assert "--stages" in output
        assert "--dry-run" in output
        assert "--strict" in output
        assert "--verbose" in output
        assert "--resume" in output

    def test_stage_help_shows_subcommands(self):
        """Stage help shows all individual stages."""
        result = runner.invoke(app, ["stage", "--help"])
        assert result.exit_code == 0
        assert "data" in result.stdout
        assert "splits" in result.stdout
        assert "features" in result.stdout
        assert "train" in result.stdout
        assert "evaluate" in result.stdout
        assert "report" in result.stdout

    def test_stage_data_help_works(self):
        """Individual stage help works."""
        result = runner.invoke(app, ["stage", "data", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.stdout)
        assert "--seed" in output
        assert "--verbose" in output


class TestCLIVersion:
    """Tests for version flag."""

    def test_version_flag_shows_version(self):
        """--version flag shows version string."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.stdout
        assert "aoty-pred version" in result.stdout

    def test_version_short_flag(self):
        """Short -V flag also shows version."""
        result = runner.invoke(app, ["-V"])
        assert result.exit_code == 0
        assert __version__ in result.stdout


class TestRunCommand:
    """Tests for main run command."""

    def test_run_dry_run_succeeds(self):
        """Run with --dry-run completes without executing stages."""
        result = runner.invoke(app, ["run", "--dry-run"])
        # Exit code 0 means success (or early exit from dry-run)
        assert result.exit_code == 0

    def test_run_dry_run_verbose(self):
        """Run with --dry-run --verbose works."""
        result = runner.invoke(app, ["run", "--dry-run", "--verbose"])
        assert result.exit_code == 0

    def test_run_with_seed(self):
        """Run accepts custom seed."""
        result = runner.invoke(app, ["run", "--dry-run", "--seed", "123"])
        assert result.exit_code == 0

    def test_run_with_stages_filter(self):
        """Run accepts stages filter."""
        result = runner.invoke(app, ["run", "--dry-run", "--stages", "data,splits"])
        assert result.exit_code == 0


class TestStageCommands:
    """Tests for individual stage commands."""

    @pytest.mark.parametrize(
        "stage_name",
        ["data", "splits", "features", "train", "evaluate", "predict", "report"],
    )
    def test_stage_command_exists(self, stage_name):
        """Each stage command exists and shows help."""
        result = runner.invoke(app, ["stage", stage_name, "--help"])
        assert result.exit_code == 0

    def test_stage_train_has_strict_option(self):
        """Train stage has --strict option."""
        result = runner.invoke(app, ["stage", "train", "--help"])
        assert result.exit_code == 0
        assert "--strict" in strip_ansi(result.stdout)

    def test_stage_predict_help_has_options(self):
        """Predict stage help shows expected options."""
        result = runner.invoke(app, ["stage", "predict", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.stdout)
        assert "--seed" in output
        assert "--verbose" in output


class TestPackageExports:
    """Tests for package exports."""

    def test_run_pipeline_import(self):
        """run_pipeline can be imported from pipelines package."""
        from aoty_pred.pipelines import run_pipeline

        assert callable(run_pipeline)

    def test_pipeline_config_import(self):
        """PipelineConfig can be imported from pipelines package."""
        from aoty_pred.pipelines import PipelineConfig

        # Can instantiate with defaults
        config = PipelineConfig()
        assert config.seed == 42
        assert config.dry_run is False

    def test_pipeline_orchestrator_import(self):
        """PipelineOrchestrator can be imported."""
        from aoty_pred.pipelines import PipelineOrchestrator

        assert PipelineOrchestrator is not None

    def test_stages_import(self):
        """Stage-related exports work."""
        from aoty_pred.pipelines import (
            build_pipeline_stages,
            get_execution_order,
            get_stage,
        )

        assert len(build_pipeline_stages()) > 0
        assert callable(get_execution_order)
        assert callable(get_stage)

    def test_errors_import(self):
        """Error classes can be imported."""
        from aoty_pred.pipelines import (
            ConvergenceError,
            DataValidationError,
            PipelineError,
            StageError,
        )

        # Check error hierarchy
        assert issubclass(ConvergenceError, PipelineError)
        assert issubclass(DataValidationError, PipelineError)
        assert issubclass(StageError, PipelineError)

    def test_manifest_import(self):
        """Manifest-related exports work."""
        from aoty_pred.pipelines import (
            generate_run_id,
        )

        # generate_run_id should return string
        run_id = generate_run_id()
        assert isinstance(run_id, str)
        assert len(run_id) == 17  # YYYY-MM-DD_HHMMSS


class TestCLIExitCodes:
    """Tests for CLI exit code behavior."""

    def test_no_command_shows_help(self):
        """Running with no command shows help and exits cleanly."""
        result = runner.invoke(app, [])
        # With invoke_without_command=True, shows help
        assert result.exit_code == 0

    def test_invalid_stage_in_stages_fails(self):
        """Invalid stage name in --stages causes failure."""
        result = runner.invoke(app, ["run", "--dry-run", "--stages", "invalid_stage"])
        assert result.exit_code != 0
