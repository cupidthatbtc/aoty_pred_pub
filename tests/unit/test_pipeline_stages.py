"""Tests for pipeline stage definitions and execution order."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aoty_pred.pipelines.stages import (
    PipelineStage,
    build_pipeline_stages,
    get_execution_order,
    get_stage,
)


class TestPipelineStage:
    """Tests for PipelineStage dataclass."""

    def test_basic_creation(self):
        """PipelineStage can be created with required fields."""
        stage = PipelineStage(
            name="test",
            description="Test stage",
            run_fn=None,
        )
        assert stage.name == "test"
        assert stage.description == "Test stage"
        assert stage.run_fn is None
        assert stage.input_paths == []
        assert stage.output_paths == []
        assert stage.depends_on == []

    def test_with_paths(self, tmp_path: Path):
        """PipelineStage stores input and output paths."""
        stage = PipelineStage(
            name="data",
            description="Process data",
            run_fn=None,
            input_paths=[tmp_path / "input.csv"],
            output_paths=[tmp_path / "output.parquet"],
            depends_on=["prior_stage"],
        )
        assert len(stage.input_paths) == 1
        assert len(stage.output_paths) == 1
        assert stage.depends_on == ["prior_stage"]


class TestComputeInputHash:
    """Tests for PipelineStage.compute_input_hash."""

    def test_empty_when_no_inputs(self):
        """Returns empty string when no input paths defined."""
        stage = PipelineStage(
            name="test",
            description="Test",
            run_fn=None,
            input_paths=[],
        )
        assert stage.compute_input_hash() == ""

    def test_empty_when_inputs_not_exist(self, tmp_path: Path):
        """Returns empty string when input files don't exist."""
        stage = PipelineStage(
            name="test",
            description="Test",
            run_fn=None,
            input_paths=[tmp_path / "nonexistent.csv"],
        )
        assert stage.compute_input_hash() == ""

    def test_hash_when_input_exists(self, tmp_path: Path):
        """Returns SHA256 hash when input file exists."""
        input_file = tmp_path / "input.csv"
        input_file.write_text("col1,col2\n1,2\n3,4\n")

        stage = PipelineStage(
            name="test",
            description="Test",
            run_fn=None,
            input_paths=[input_file],
        )
        hash_value = stage.compute_input_hash()

        # SHA256 is 64 hex characters
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_hash_changes_with_content(self, tmp_path: Path):
        """Hash changes when file content changes."""
        input_file = tmp_path / "input.csv"
        input_file.write_text("content_v1")

        stage = PipelineStage(
            name="test",
            description="Test",
            run_fn=None,
            input_paths=[input_file],
        )
        hash_v1 = stage.compute_input_hash()

        input_file.write_text("content_v2")
        hash_v2 = stage.compute_input_hash()

        assert hash_v1 != hash_v2

    def test_hash_deterministic(self, tmp_path: Path):
        """Same content produces same hash."""
        input_file = tmp_path / "input.csv"
        input_file.write_text("consistent content")

        stage = PipelineStage(
            name="test",
            description="Test",
            run_fn=None,
            input_paths=[input_file],
        )

        assert stage.compute_input_hash() == stage.compute_input_hash()

    def test_hash_combines_multiple_files(self, tmp_path: Path):
        """Hash combines multiple input files."""
        file1 = tmp_path / "file1.csv"
        file2 = tmp_path / "file2.csv"
        file1.write_text("content1")
        file2.write_text("content2")

        stage = PipelineStage(
            name="test",
            description="Test",
            run_fn=None,
            input_paths=[file1, file2],
        )
        hash_both = stage.compute_input_hash()

        # Different from single file hash
        stage_single = PipelineStage(
            name="test2",
            description="Test",
            run_fn=None,
            input_paths=[file1],
        )
        hash_single = stage_single.compute_input_hash()

        assert hash_both != hash_single


class TestShouldSkip:
    """Tests for PipelineStage.should_skip."""

    def test_no_skip_when_force_true(self, tmp_path: Path):
        """Never skip when force=True."""
        stage = PipelineStage(name="test", description="Test", run_fn=None)
        # Even with a manifest, force=True means no skip
        mock_manifest = MagicMock()
        mock_manifest.stage_hashes = {"test": "somehash"}

        assert stage.should_skip(mock_manifest, force=True) is False

    def test_no_skip_when_manifest_none(self):
        """No skip when no previous manifest."""
        stage = PipelineStage(name="test", description="Test", run_fn=None)

        assert stage.should_skip(None) is False

    def test_no_skip_when_stage_not_in_manifest(self):
        """No skip when stage not recorded in manifest."""
        stage = PipelineStage(name="new_stage", description="Test", run_fn=None)
        mock_manifest = MagicMock()
        mock_manifest.stage_hashes = {"other_stage": "hash"}

        assert stage.should_skip(mock_manifest) is False

    def test_no_skip_when_hash_changed(self, tmp_path: Path):
        """No skip when input hash has changed."""
        input_file = tmp_path / "input.csv"
        input_file.write_text("new_content")

        stage = PipelineStage(
            name="test",
            description="Test",
            run_fn=None,
            input_paths=[input_file],
        )

        mock_manifest = MagicMock()
        mock_manifest.stage_hashes = {"test": "old_hash_value"}

        assert stage.should_skip(mock_manifest) is False

    def test_no_skip_when_outputs_missing(self, tmp_path: Path):
        """No skip when output files don't exist."""
        input_file = tmp_path / "input.csv"
        input_file.write_text("content")

        stage = PipelineStage(
            name="test",
            description="Test",
            run_fn=None,
            input_paths=[input_file],
            output_paths=[tmp_path / "missing_output.parquet"],
        )

        # Compute actual hash to match
        actual_hash = stage.compute_input_hash()
        mock_manifest = MagicMock()
        mock_manifest.stage_hashes = {"test": actual_hash}

        # Should not skip because output doesn't exist
        assert stage.should_skip(mock_manifest) is False

    def test_skip_when_all_conditions_met(self, tmp_path: Path):
        """Skip when hash matches and outputs exist."""
        input_file = tmp_path / "input.csv"
        input_file.write_text("content")
        output_file = tmp_path / "output.parquet"
        output_file.write_text("output")

        stage = PipelineStage(
            name="test",
            description="Test",
            run_fn=None,
            input_paths=[input_file],
            output_paths=[output_file],
        )

        # Compute actual hash
        actual_hash = stage.compute_input_hash()
        mock_manifest = MagicMock()
        mock_manifest.stage_hashes = {"test": actual_hash}

        assert stage.should_skip(mock_manifest) is True


class TestPipelineStages:
    """Tests for build_pipeline_stages registry."""

    def test_all_stages_defined(self):
        """All expected stages are defined."""
        stages = build_pipeline_stages()
        stage_names = {s.name for s in stages}
        expected = {"data", "splits", "features", "train", "evaluate", "report"}
        assert stage_names == expected

    def test_stages_have_descriptions(self):
        """All stages have non-empty descriptions."""
        for stage in build_pipeline_stages():
            assert stage.description, f"Stage {stage.name} missing description"

    def test_stages_have_valid_dependencies(self):
        """All stage dependencies reference existing stages."""
        stages = build_pipeline_stages()
        valid_names = {s.name for s in stages}
        for stage in stages:
            for dep in stage.depends_on:
                assert dep in valid_names, f"Stage {stage.name} has invalid dependency: {dep}"

    def test_data_stage_has_no_dependencies(self):
        """Data stage is the root with no dependencies."""
        data_stage = get_stage("data")
        assert data_stage.depends_on == []

    def test_report_stage_depends_on_evaluate(self):
        """Report stage depends on evaluate."""
        report_stage = get_stage("report")
        assert "evaluate" in report_stage.depends_on

    def test_splits_input_path_reflects_min_ratings(self):
        """Splits stage input_paths use the correct min_ratings parquet file."""
        from pathlib import Path

        stages_10 = build_pipeline_stages(min_ratings=10)
        stages_30 = build_pipeline_stages(min_ratings=30)

        splits_10 = next(s for s in stages_10 if s.name == "splits")
        splits_30 = next(s for s in stages_30 if s.name == "splits")

        assert splits_10.input_paths == [Path("data/processed/user_score_minratings_10.parquet")]
        assert splits_30.input_paths == [Path("data/processed/user_score_minratings_30.parquet")]


class TestGetExecutionOrder:
    """Tests for get_execution_order function."""

    def test_returns_all_stages_by_default(self):
        """Returns all stages when no filter provided."""
        order = get_execution_order()
        assert len(order) == len(build_pipeline_stages())

    def test_respects_dependencies(self):
        """Stages come after their dependencies."""
        order = get_execution_order()
        names = [s.name for s in order]

        # data before splits
        assert names.index("data") < names.index("splits")
        # splits before features
        assert names.index("splits") < names.index("features")
        # features before train
        assert names.index("features") < names.index("train")
        # train before evaluate
        assert names.index("train") < names.index("evaluate")
        # evaluate before report
        assert names.index("evaluate") < names.index("report")

    def test_filters_to_specified_stages(self):
        """Returns only specified stages when filter provided."""
        order = get_execution_order(["data", "splits"])
        names = [s.name for s in order]

        assert len(names) == 2
        assert set(names) == {"data", "splits"}

    def test_filtered_stages_in_order(self):
        """Filtered stages still respect dependency order."""
        order = get_execution_order(["splits", "data"])  # Reverse order in input
        names = [s.name for s in order]

        # Still returns in correct order
        assert names.index("data") < names.index("splits")

    def test_unknown_stage_raises_keyerror(self):
        """Raises KeyError for unknown stage name."""
        with pytest.raises(KeyError) as exc_info:
            get_execution_order(["nonexistent"])

        assert "nonexistent" in str(exc_info.value)

    def test_empty_list_returns_empty(self):
        """Empty stage list returns empty result."""
        order = get_execution_order([])
        assert order == []


class TestGetStage:
    """Tests for get_stage function."""

    def test_finds_existing_stage(self):
        """Returns stage when name exists."""
        stage = get_stage("data")
        assert stage.name == "data"
        assert isinstance(stage, PipelineStage)

    def test_unknown_stage_raises_keyerror(self):
        """Raises KeyError for unknown stage."""
        with pytest.raises(KeyError) as exc_info:
            get_stage("nonexistent")

        assert "nonexistent" in str(exc_info.value)
        # Error message should list valid stages
        assert "data" in str(exc_info.value)

    def test_all_defined_stages_findable(self):
        """All defined stages can be found by name."""
        for stage in build_pipeline_stages():
            found = get_stage(stage.name)
            assert found.name == stage.name
