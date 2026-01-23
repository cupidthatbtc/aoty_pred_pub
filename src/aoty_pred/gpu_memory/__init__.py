"""GPU memory preflight check package.

Provides utilities for querying GPU memory availability and performing
preflight checks before expensive MCMC operations.
"""

from aoty_pred.gpu_memory.platform import PlatformInfo, PlatformType, detect_platform

__all__ = ["PlatformInfo", "PlatformType", "detect_platform"]
