"""Tests for pipeline error classes."""

from __future__ import annotations

import pytest

from aoty_pred.pipelines.errors import GpuMemoryError, PipelineError


class TestGpuMemoryError:
    """Tests for GpuMemoryError exception."""

    def test_exit_code_is_6(self):
        """GpuMemoryError has exit code 6."""
        error = GpuMemoryError("GPU check failed")
        assert error.exit_code == 6

    def test_inherits_from_pipeline_error(self):
        """GpuMemoryError inherits from PipelineError."""
        error = GpuMemoryError("test")
        assert isinstance(error, PipelineError)

    def test_default_stage_is_gpu_check(self):
        """GpuMemoryError defaults to stage='gpu_check'."""
        error = GpuMemoryError("test message")
        assert error.stage == "gpu_check"
        assert "[gpu_check]" in str(error)

    def test_custom_stage(self):
        """GpuMemoryError accepts custom stage."""
        error = GpuMemoryError("test", stage="preflight")
        assert error.stage == "preflight"

    def test_message_preserved(self):
        """GpuMemoryError preserves message."""
        error = GpuMemoryError("Insufficient GPU memory")
        assert error.message == "Insufficient GPU memory"

    def test_can_be_raised_and_caught(self):
        """GpuMemoryError can be raised and caught."""
        with pytest.raises(GpuMemoryError) as exc_info:
            raise GpuMemoryError("No GPU detected")
        assert exc_info.value.exit_code == 6
