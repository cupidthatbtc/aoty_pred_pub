"""Tests for pipeline run manifest schema and I/O."""

import json
from pathlib import Path

import pytest

from aoty_pred.pipelines.manifest import (
    EnvironmentInfo,
    GitStateModel,
    RunManifest,
    capture_environment,
    generate_run_id,
    load_run_manifest,
    save_run_manifest,
)
from aoty_pred.utils.git_state import GitState


class TestEnvironmentInfo:
    """Tests for EnvironmentInfo model."""

    def test_captures_python_version(self):
        """capture_environment returns real Python version."""
        env = capture_environment()
        # Python version should be like "3.11.5"
        assert env.python_version
        parts = env.python_version.split(".")
        assert len(parts) >= 2
        assert int(parts[0]) >= 3

    def test_captures_jax_version(self):
        """capture_environment returns JAX version or 'not installed'."""
        env = capture_environment()
        assert env.jax_version
        # Either a version string or "not installed"
        assert env.jax_version != "unknown" or env.jax_version == "not installed"

    def test_captures_platform(self):
        """capture_environment returns platform info."""
        env = capture_environment()
        assert env.platform
        # Should contain OS name
        assert any(os_name in env.platform for os_name in ["Windows", "Linux", "Darwin"])

    def test_pixi_lock_hash_is_string_or_none(self):
        """pixi_lock_hash is either a SHA256 string or None."""
        env = capture_environment()
        if env.pixi_lock_hash is not None:
            # SHA256 is 64 hex characters
            assert len(env.pixi_lock_hash) == 64
            assert all(c in "0123456789abcdef" for c in env.pixi_lock_hash)

    def test_numpyro_version_optional(self):
        """numpyro_version is optional (None if not installed)."""
        env = capture_environment()
        # Just verify it's either a string or None
        assert env.numpyro_version is None or isinstance(env.numpyro_version, str)

    def test_arviz_version_optional(self):
        """arviz_version is optional (None if not installed)."""
        env = capture_environment()
        assert env.arviz_version is None or isinstance(env.arviz_version, str)


class TestGitStateModel:
    """Tests for GitStateModel Pydantic wrapper."""

    def test_from_git_state(self):
        """GitStateModel can be created from GitState dataclass."""
        git_state = GitState(
            commit="abc123def456",
            branch="main",
            dirty=True,
            untracked_count=3,
        )
        model = GitStateModel.from_git_state(git_state)
        assert model.commit == "abc123def456"
        assert model.branch == "main"
        assert model.dirty is True
        assert model.untracked_count == 3

    def test_to_git_state(self):
        """GitStateModel can convert back to GitState dataclass."""
        model = GitStateModel(
            commit="abc123def456",
            branch="develop",
            dirty=False,
            untracked_count=0,
        )
        git_state = model.to_git_state()
        assert isinstance(git_state, GitState)
        assert git_state.commit == "abc123def456"
        assert git_state.branch == "develop"

    def test_json_serialization(self):
        """GitStateModel serializes to JSON correctly."""
        model = GitStateModel(
            commit="abc123",
            branch="main",
            dirty=True,
            untracked_count=5,
        )
        json_str = model.model_dump_json()
        data = json.loads(json_str)
        assert data["commit"] == "abc123"
        assert data["dirty"] is True


class TestGenerateRunId:
    """Tests for run ID generation."""

    def test_format(self):
        """Run ID has correct format: YYYY-MM-DD_HHMMSS."""
        run_id = generate_run_id()
        # Format: "2026-01-19_143052"
        assert len(run_id) == 17
        assert run_id[4] == "-"
        assert run_id[7] == "-"
        assert run_id[10] == "_"

    def test_parseable(self):
        """Run ID can be parsed back to datetime components."""
        run_id = generate_run_id()
        # Should parse without error
        date_part, time_part = run_id.split("_")
        year, month, day = date_part.split("-")
        assert 2020 <= int(year) <= 2100
        assert 1 <= int(month) <= 12
        assert 1 <= int(day) <= 31
        assert len(time_part) == 6
        hour = int(time_part[:2])
        minute = int(time_part[2:4])
        second = int(time_part[4:])
        assert 0 <= hour <= 23
        assert 0 <= minute <= 59
        assert 0 <= second <= 59


class TestRunManifest:
    """Tests for RunManifest schema."""

    @pytest.fixture
    def sample_manifest(self) -> RunManifest:
        """Create a sample RunManifest for testing."""
        return RunManifest(
            run_id="2026-01-19_143052",
            created_at="2026-01-19T14:30:52Z",
            command="aoty-pipeline run --seed 42",
            flags={"seed": 42, "skip_existing": False, "verbose": True},
            seed=42,
            git=GitStateModel(
                commit="abc123def456789012345678901234567890abcd",
                branch="main",
                dirty=False,
                untracked_count=0,
            ),
            environment=EnvironmentInfo(
                python_version="3.11.5",
                jax_version="0.4.26",
                numpyro_version="0.15.0",
                arviz_version="0.18.0",
                platform="Windows 11",
                pixi_lock_hash="abc123def456" * 5 + "abcd",
            ),
            input_hashes={
                "data/raw/albums.csv": "hash1",
                "data/processed/cleaned.parquet": "hash2",
            },
            stage_hashes={"data": "hash_data", "splits": "hash_splits"},
            stages_completed=["data", "splits"],
            stages_skipped=["features"],
            outputs={"model": "outputs/2026-01-19_143052/model.pkl"},
            success=True,
        )

    def test_serialization_roundtrip(self, sample_manifest: RunManifest):
        """RunManifest serializes and deserializes correctly."""
        json_str = sample_manifest.model_dump_json(indent=2)
        restored = RunManifest.model_validate_json(json_str)

        assert restored.run_id == sample_manifest.run_id
        assert restored.seed == sample_manifest.seed
        assert restored.flags == sample_manifest.flags
        assert restored.git.commit == sample_manifest.git.commit
        assert restored.environment.python_version == sample_manifest.environment.python_version
        assert restored.environment.pixi_lock_hash == sample_manifest.environment.pixi_lock_hash
        assert restored.stages_completed == sample_manifest.stages_completed
        assert restored.success == sample_manifest.success

    def test_preserves_pixi_lock_hash(self, sample_manifest: RunManifest):
        """Serialization preserves pixi_lock_hash field."""
        json_str = sample_manifest.model_dump_json()
        data = json.loads(json_str)
        assert "pixi_lock_hash" in data["environment"]
        assert data["environment"]["pixi_lock_hash"] == sample_manifest.environment.pixi_lock_hash

    def test_error_field_optional(self):
        """Error field is optional (None by default)."""
        manifest = RunManifest(
            run_id="test",
            created_at="2026-01-19T00:00:00Z",
            command="test",
            flags={},
            seed=42,
            git=GitStateModel(commit="x", branch="y", dirty=False, untracked_count=0),
            environment=EnvironmentInfo(
                python_version="3.11.0",
                jax_version="0.4.0",
                numpyro_version=None,
                arviz_version=None,
                platform="Test",
                pixi_lock_hash=None,
            ),
            input_hashes={},
            stage_hashes={},
            stages_completed=[],
            stages_skipped=[],
            outputs={},
            success=True,
        )
        assert manifest.error is None

    def test_duration_default_zero(self):
        """Duration field defaults to 0.0."""
        manifest = RunManifest(
            run_id="test",
            created_at="2026-01-19T00:00:00Z",
            command="test",
            flags={},
            seed=42,
            git=GitStateModel(commit="x", branch="y", dirty=False, untracked_count=0),
            environment=EnvironmentInfo(
                python_version="3.11.0",
                jax_version="0.4.0",
                numpyro_version=None,
                arviz_version=None,
                platform="Test",
                pixi_lock_hash=None,
            ),
            input_hashes={},
            stage_hashes={},
            stages_completed=[],
            stages_skipped=[],
            outputs={},
            success=True,
        )
        assert manifest.duration_seconds == 0.0


class TestSaveLoadManifest:
    """Tests for manifest file I/O."""

    @pytest.fixture
    def sample_manifest(self) -> RunManifest:
        """Create a sample RunManifest for testing."""
        return RunManifest(
            run_id="2026-01-19_143052",
            created_at="2026-01-19T14:30:52Z",
            command="aoty-pipeline run --seed 42",
            flags={"seed": 42},
            seed=42,
            git=GitStateModel(
                commit="abc123def456789012345678901234567890abcd",
                branch="main",
                dirty=False,
                untracked_count=0,
            ),
            environment=EnvironmentInfo(
                python_version="3.11.5",
                jax_version="0.4.26",
                numpyro_version="0.15.0",
                arviz_version="0.18.0",
                platform="Windows 11",
                pixi_lock_hash="a" * 64,
            ),
            input_hashes={"input.csv": "hash1"},
            stage_hashes={"data": "hash_data"},
            stages_completed=["data"],
            stages_skipped=[],
            outputs={"output": "path/to/output"},
            success=True,
            duration_seconds=123.45,
        )

    def test_save_creates_file(self, tmp_path: Path, sample_manifest: RunManifest):
        """save_run_manifest creates manifest.json file."""
        run_dir = tmp_path / "test_run"
        path = save_run_manifest(sample_manifest, run_dir)

        assert path.exists()
        assert path.name == "manifest.json"
        assert path.parent == run_dir

    def test_save_creates_directory(self, tmp_path: Path, sample_manifest: RunManifest):
        """save_run_manifest creates run directory if needed."""
        run_dir = tmp_path / "nested" / "run" / "dir"
        path = save_run_manifest(sample_manifest, run_dir)

        assert run_dir.exists()
        assert path.exists()

    def test_load_restores_manifest(self, tmp_path: Path, sample_manifest: RunManifest):
        """load_run_manifest restores saved manifest."""
        run_dir = tmp_path / "test_run"
        path = save_run_manifest(sample_manifest, run_dir)

        loaded = load_run_manifest(path)

        assert loaded.run_id == sample_manifest.run_id
        assert loaded.seed == sample_manifest.seed
        assert loaded.git.commit == sample_manifest.git.commit
        assert loaded.environment.pixi_lock_hash == sample_manifest.environment.pixi_lock_hash
        assert loaded.duration_seconds == sample_manifest.duration_seconds

    def test_save_load_roundtrip_preserves_all_fields(
        self, tmp_path: Path, sample_manifest: RunManifest
    ):
        """Save/load roundtrip preserves all fields including pixi_lock_hash."""
        run_dir = tmp_path / "roundtrip_test"
        path = save_run_manifest(sample_manifest, run_dir)
        loaded = load_run_manifest(path)

        # Compare as JSON to check all fields
        original_json = json.loads(sample_manifest.model_dump_json())
        loaded_json = json.loads(loaded.model_dump_json())

        assert original_json == loaded_json

    def test_load_nonexistent_raises(self, tmp_path: Path):
        """load_run_manifest raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_run_manifest(tmp_path / "nonexistent.json")

    def test_saved_json_is_pretty_printed(self, tmp_path: Path, sample_manifest: RunManifest):
        """Saved JSON is pretty-printed with indentation."""
        run_dir = tmp_path / "test_run"
        path = save_run_manifest(sample_manifest, run_dir)

        content = path.read_text(encoding="utf-8")
        # Pretty-printed JSON has newlines
        assert "\n" in content
        # And indentation
        assert "  " in content
