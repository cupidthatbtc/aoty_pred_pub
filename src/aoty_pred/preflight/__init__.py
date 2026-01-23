"""Preflight check module for GPU memory verification.

This module provides preflight checking capabilities to verify GPU memory
availability before starting long-running MCMC sampling jobs.

The preflight system can:
- Estimate memory requirements for a given model configuration
- Compare against available GPU memory
- Report pass/fail/warning status
- Suggest configuration adjustments

Example:
    >>> from aoty_pred.preflight import PreflightStatus, PreflightResult
    >>> # After running preflight check:
    >>> if result.status == PreflightStatus.FAIL:
    ...     raise SystemExit(result.exit_code)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aoty_pred.gpu_memory import MemoryEstimate


class PreflightStatus(Enum):
    """Status of preflight check.

    Values:
        PASS: Sufficient memory available, safe to proceed.
        FAIL: Insufficient memory, MCMC run will likely OOM.
        WARNING: Memory is tight, may work but risky.
        CANNOT_CHECK: Unable to determine memory availability
            (e.g., no GPU, NVML not available).
    """

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    CANNOT_CHECK = "cannot_check"


@dataclass(frozen=True)
class PreflightResult:
    """Result of preflight memory check.

    Provides detailed information about memory availability and
    estimated requirements, with actionable suggestions if needed.

    Attributes:
        status: Overall pass/fail/warning status.
        estimate: Memory estimate breakdown (None if cannot check).
        available_gb: Available (free) GPU memory in GB.
        total_gpu_gb: Total GPU memory in GB.
        headroom_percent: Remaining memory percentage after estimate.
            Negative if estimate exceeds available.
        message: Human-readable summary message.
        suggestions: List of configuration adjustment suggestions.
        device_name: GPU device name (None if cannot check).
    """

    status: PreflightStatus
    estimate: MemoryEstimate | None
    available_gb: float
    total_gpu_gb: float
    headroom_percent: float
    message: str
    suggestions: list[str]
    device_name: str | None = None

    @property
    def exit_code(self) -> int:
        """Exit code for CLI usage.

        Returns:
            0 for PASS (safe to proceed).
            1 for FAIL (do not proceed).
            2 for WARNING or CANNOT_CHECK (proceed with caution).
        """
        match self.status:
            case PreflightStatus.PASS:
                return 0
            case PreflightStatus.FAIL:
                return 1
            case PreflightStatus.WARNING | PreflightStatus.CANNOT_CHECK:
                return 2


__all__ = [
    "PreflightStatus",
    "PreflightResult",
]
