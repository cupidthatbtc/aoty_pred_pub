"""JAX device memory statistics wrapper.

Provides typed access to JAX's jax.Device.memory_stats() API for accurate
peak GPU memory measurement during MCMC runs.

Example:
    >>> stats = get_jax_memory_stats()
    >>> print(f"Peak memory: {stats.peak_gb:.2f} GB")
    Peak memory: 4.25 GB
"""

from __future__ import annotations

from dataclasses import dataclass

import jax


@dataclass(frozen=True)
class JaxMemoryStats:
    """JAX device memory statistics.

    Wraps jax.Device.memory_stats() return values with type safety
    and convenient property accessors for GB values.

    Attributes:
        bytes_in_use: Current bytes being used on the device.
        peak_bytes_in_use: Maximum bytes used since device initialization.
        bytes_limit: Maximum available bytes on the device.
        bytes_reserved: Pre-allocated bytes from system.
    """

    bytes_in_use: int
    peak_bytes_in_use: int
    bytes_limit: int
    bytes_reserved: int

    @property
    def peak_gb(self) -> float:
        """Peak memory usage in GB (1024^3 bytes)."""
        return self.peak_bytes_in_use / (1024**3)

    @property
    def limit_gb(self) -> float:
        """Memory limit in GB (1024^3 bytes)."""
        return self.bytes_limit / (1024**3)

    @property
    def in_use_gb(self) -> float:
        """Current memory usage in GB (1024^3 bytes)."""
        return self.bytes_in_use / (1024**3)

    @property
    def reserved_gb(self) -> float:
        """Reserved memory in GB (1024^3 bytes)."""
        return self.bytes_reserved / (1024**3)


def get_jax_memory_stats(device_index: int = 0) -> JaxMemoryStats:
    """Get JAX memory statistics for specified GPU device.

    Queries jax.Device.memory_stats() for the specified GPU and returns
    a typed JaxMemoryStats dataclass with peak memory usage.

    Args:
        device_index: GPU device index (default 0 for first GPU).

    Returns:
        JaxMemoryStats with current and peak memory usage.

    Raises:
        RuntimeError: If no GPU devices available or device_index out of range.

    Example:
        >>> stats = get_jax_memory_stats()
        >>> print(f"Peak: {stats.peak_gb:.2f} GB of {stats.limit_gb:.2f} GB")
        Peak: 4.25 GB of 24.00 GB
    """
    try:
        devices = jax.devices("gpu")
    except RuntimeError as e:
        raise RuntimeError(f"No GPU devices available for JAX: {e}") from e

    if not devices:
        raise RuntimeError("No GPU devices available for JAX")

    if device_index >= len(devices):
        raise RuntimeError(
            f"GPU index {device_index} out of range. "
            f"Available GPUs: 0-{len(devices) - 1}"
        )

    device = devices[device_index]
    stats = device.memory_stats()

    # Handle case where memory_stats() returns None (e.g., on some platforms)
    if stats is None:
        raise RuntimeError(
            f"Device {device_index} does not support memory_stats(). "
            "This may occur on non-CUDA backends."
        )

    return JaxMemoryStats(
        bytes_in_use=stats.get("bytes_in_use", 0),
        peak_bytes_in_use=stats.get("peak_bytes_in_use", 0),
        bytes_limit=stats.get("bytes_limit", 0),
        bytes_reserved=stats.get("bytes_reserved", 0),
    )
