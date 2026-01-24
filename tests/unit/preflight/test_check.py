"""Tests for preflight check logic."""

from __future__ import annotations

from unittest import mock

from aoty_pred.gpu_memory import GpuMemoryInfo
from aoty_pred.pipelines.errors import GpuMemoryError
from aoty_pred.preflight import PreflightStatus, run_preflight_check

# Standard test params that produce a small memory estimate (~0.02 GB)
# Shared by multiple test classes to avoid duplication
TEST_PARAMS: dict[str, int] = {
    "n_observations": 1000,
    "n_features": 20,
    "n_artists": 50,
    "max_seq": 10,
    "num_chains": 4,
    "num_samples": 1000,
    "num_warmup": 1000,
}


class TestPreflightStatusDetermination:
    """Tests for preflight status determination logic."""

    @mock.patch("aoty_pred.preflight.check.query_gpu_memory")
    def test_preflight_pass_with_headroom(self, mock_query):
        """Estimate well under available memory -> PASS."""
        # 10 GB available, small estimate should easily pass
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=10 * 1024**3,  # 10 GB
            used_bytes=0,
            free_bytes=10 * 1024**3,  # 10 GB free
        )

        result = run_preflight_check(**TEST_PARAMS)

        assert result.status == PreflightStatus.PASS

    @mock.patch("aoty_pred.preflight.check.query_gpu_memory")
    def test_preflight_warning_low_headroom(self, mock_query):
        """Estimate close to available memory (low headroom) -> WARNING."""
        # Use a larger estimate by increasing params, or reducing available mem
        # Estimate ~0.02 GB, so 0.024 GB available = ~83% used = low headroom
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=int(0.024 * 1024**3),
            used_bytes=0,
            free_bytes=int(0.024 * 1024**3),
        )

        result = run_preflight_check(**TEST_PARAMS)

        assert result.status == PreflightStatus.WARNING

    @mock.patch("aoty_pred.preflight.check.query_gpu_memory")
    def test_preflight_fail_exceeds_available(self, mock_query):
        """Estimate exceeds available memory -> FAIL."""
        # Very little available memory
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=int(0.01 * 1024**3),  # 0.01 GB
            used_bytes=0,
            free_bytes=int(0.01 * 1024**3),  # 0.01 GB
        )

        result = run_preflight_check(**TEST_PARAMS)

        assert result.status == PreflightStatus.FAIL

    @mock.patch("aoty_pred.preflight.check.query_gpu_memory")
    def test_preflight_cannot_check_no_gpu(self, mock_query):
        """GPU query raises GpuMemoryError -> CANNOT_CHECK."""
        mock_query.side_effect = GpuMemoryError("No GPU detected")

        result = run_preflight_check(**TEST_PARAMS)

        assert result.status == PreflightStatus.CANNOT_CHECK


class TestPreflightExitCodes:
    """Tests for exit codes returned by PreflightResult."""

    @mock.patch("aoty_pred.preflight.check.query_gpu_memory")
    def test_exit_code_pass(self, mock_query):
        """PASS status -> exit_code 0."""
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=10 * 1024**3,
            used_bytes=0,
            free_bytes=10 * 1024**3,
        )

        result = run_preflight_check(**TEST_PARAMS)

        assert result.status == PreflightStatus.PASS
        assert result.exit_code == 0

    @mock.patch("aoty_pred.preflight.check.query_gpu_memory")
    def test_exit_code_fail(self, mock_query):
        """FAIL status -> exit_code 1."""
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=int(0.001 * 1024**3),  # Tiny
            used_bytes=0,
            free_bytes=int(0.001 * 1024**3),
        )

        result = run_preflight_check(**TEST_PARAMS)

        assert result.status == PreflightStatus.FAIL
        assert result.exit_code == 1

    @mock.patch("aoty_pred.preflight.check.query_gpu_memory")
    def test_exit_code_warning(self, mock_query):
        """WARNING status -> exit_code 2."""
        # Tight fit with low headroom
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=int(0.024 * 1024**3),
            used_bytes=0,
            free_bytes=int(0.024 * 1024**3),
        )

        result = run_preflight_check(**TEST_PARAMS)

        assert result.status == PreflightStatus.WARNING
        assert result.exit_code == 2

    @mock.patch("aoty_pred.preflight.check.query_gpu_memory")
    def test_exit_code_cannot_check(self, mock_query):
        """CANNOT_CHECK status -> exit_code 2."""
        mock_query.side_effect = GpuMemoryError("No GPU")

        result = run_preflight_check(**TEST_PARAMS)

        assert result.status == PreflightStatus.CANNOT_CHECK
        assert result.exit_code == 2


class TestPreflightSuggestions:
    """Tests for suggestions generated on FAIL/WARNING."""

    @mock.patch("aoty_pred.preflight.check.query_gpu_memory")
    def test_suggestions_on_fail(self, mock_query):
        """FAIL status includes suggestions."""
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=int(0.001 * 1024**3),
            used_bytes=0,
            free_bytes=int(0.001 * 1024**3),
        )

        result = run_preflight_check(
            n_observations=1000,
            n_features=20,
            n_artists=50,
            max_seq=10,
            num_chains=4,
            num_samples=1000,
            num_warmup=1000,
        )

        assert result.status == PreflightStatus.FAIL
        assert len(result.suggestions) > 0

    @mock.patch("aoty_pred.preflight.check.query_gpu_memory")
    def test_suggestions_reduce_chains(self, mock_query):
        """Suggestions include reducing chains when chains > 1."""
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=int(0.001 * 1024**3),
            used_bytes=0,
            free_bytes=int(0.001 * 1024**3),
        )

        result = run_preflight_check(
            n_observations=1000,
            n_features=20,
            n_artists=50,
            max_seq=10,
            num_chains=4,  # More than 1 chain
            num_samples=1000,
            num_warmup=1000,
        )

        assert result.status == PreflightStatus.FAIL
        chain_suggestion = [s for s in result.suggestions if "--num-chains" in s]
        assert len(chain_suggestion) > 0

    @mock.patch("aoty_pred.preflight.check.query_gpu_memory")
    def test_suggestions_reduce_samples(self, mock_query):
        """Suggestions include reducing samples when samples > 500."""
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=int(0.001 * 1024**3),
            used_bytes=0,
            free_bytes=int(0.001 * 1024**3),
        )

        result = run_preflight_check(
            n_observations=1000,
            n_features=20,
            n_artists=50,
            max_seq=10,
            num_chains=4,
            num_samples=1000,  # More than 500
            num_warmup=1000,
        )

        assert result.status == PreflightStatus.FAIL
        sample_suggestion = [s for s in result.suggestions if "--num-samples" in s]
        assert len(sample_suggestion) > 0

    @mock.patch("aoty_pred.preflight.check.query_gpu_memory")
    def test_no_suggestions_on_pass(self, mock_query):
        """PASS status has empty suggestions list."""
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=10 * 1024**3,
            used_bytes=0,
            free_bytes=10 * 1024**3,
        )

        result = run_preflight_check(
            n_observations=1000,
            n_features=20,
            n_artists=50,
            max_seq=10,
            num_chains=4,
            num_samples=1000,
            num_warmup=1000,
        )

        assert result.status == PreflightStatus.PASS
        assert result.suggestions == []


class TestPreflightMessages:
    """Tests for message content in PreflightResult."""

    @mock.patch("aoty_pred.preflight.check.query_gpu_memory")
    def test_message_contains_estimate(self, mock_query):
        """Message includes estimate GB value."""
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=10 * 1024**3,
            used_bytes=0,
            free_bytes=10 * 1024**3,
        )

        result = run_preflight_check(
            n_observations=1000,
            n_features=20,
            n_artists=50,
            max_seq=10,
            num_chains=4,
            num_samples=1000,
            num_warmup=1000,
        )

        # Message should contain "GB" (from estimate value)
        assert "GB" in result.message

    @mock.patch("aoty_pred.preflight.check.query_gpu_memory")
    def test_message_contains_available(self, mock_query):
        """Message includes available GB value."""
        mock_query.return_value = GpuMemoryInfo(
            device_name="Test GPU",
            total_bytes=10 * 1024**3,
            used_bytes=0,
            free_bytes=10 * 1024**3,  # 10 GB
        )

        result = run_preflight_check(
            n_observations=1000,
            n_features=20,
            n_artists=50,
            max_seq=10,
            num_chains=4,
            num_samples=1000,
            num_warmup=1000,
        )

        # Message should reference available memory
        assert "10.0 GB available" in result.message
