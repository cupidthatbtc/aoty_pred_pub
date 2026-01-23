"""GPU memory preflight check infrastructure.

This module provides:
- Platform detection (WSL2 vs native Linux vs other)
- GPU memory queries via NVML
- Dataclasses for structured results

Example:
    >>> from aoty_pred.gpu_memory import query_gpu_memory, detect_platform
    >>> platform = detect_platform()
    >>> if platform.supports_gpu:
    ...     info = query_gpu_memory()
    ...     print(info.format_display())
"""

from aoty_pred.gpu_memory.estimate import (
    MemoryEstimate,
    estimate_memory_gb,
)
from aoty_pred.gpu_memory.platform import (
    PlatformInfo,
    PlatformType,
    detect_platform,
)
from aoty_pred.gpu_memory.query import (
    GpuMemoryInfo,
    query_gpu_memory,
)

__all__ = [
    # Memory estimation
    "MemoryEstimate",
    "estimate_memory_gb",
    # Platform detection
    "PlatformType",
    "PlatformInfo",
    "detect_platform",
    # GPU memory query
    "GpuMemoryInfo",
    "query_gpu_memory",
]
