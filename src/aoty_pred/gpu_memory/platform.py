"""Platform detection for GPU memory preflight checks.

Detects WSL2 vs native Linux vs other platforms to enable platform-specific
error messaging and behavior adjustments.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class PlatformType(Enum):
    """Detected platform type."""

    NATIVE_LINUX = "native_linux"
    WSL2 = "wsl2"
    WSL1 = "wsl1"
    MACOS = "macos"
    WINDOWS = "windows"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PlatformInfo:
    """Platform detection result.

    Attributes:
        platform_type: Detected platform category.
        kernel_version: Linux kernel version if available.
        wsl_distro: WSL distribution name if running under WSL.
    """

    platform_type: PlatformType
    kernel_version: str | None = None
    wsl_distro: str | None = None

    @property
    def is_wsl(self) -> bool:
        """Check if running under any WSL version."""
        return self.platform_type in (PlatformType.WSL1, PlatformType.WSL2)

    @property
    def supports_gpu(self) -> bool:
        """Check if platform can support NVIDIA GPU.

        WSL2 and native Linux support GPU passthrough.
        WSL1 does not support GPU passthrough.
        macOS uses Metal, not NVIDIA CUDA.
        """
        return self.platform_type in (
            PlatformType.NATIVE_LINUX,
            PlatformType.WSL2,
            PlatformType.WINDOWS,
        )


def detect_platform() -> PlatformInfo:
    """Detect the current platform.

    Uses /proc/version and environment variables to distinguish:
    - Native Linux
    - WSL2 (supports GPU passthrough)
    - WSL1 (no GPU support)
    - macOS
    - Windows

    Returns:
        PlatformInfo with platform type and relevant metadata.
    """
    system = platform.system()

    if system == "Darwin":
        return PlatformInfo(platform_type=PlatformType.MACOS)

    if system == "Windows":
        return PlatformInfo(platform_type=PlatformType.WINDOWS)

    if system != "Linux":
        return PlatformInfo(platform_type=PlatformType.UNKNOWN)

    # Linux - check for WSL
    kernel_version = None
    proc_version = ""
    try:
        proc_version = Path("/proc/version").read_text().lower()
        parts = proc_version.split()
        if len(parts) > 2:
            kernel_version = parts[2]
    except OSError:
        pass

    # WSL detection: "microsoft" or "wsl" in /proc/version
    is_wsl = "microsoft" in proc_version or "wsl" in proc_version

    if not is_wsl:
        return PlatformInfo(
            platform_type=PlatformType.NATIVE_LINUX,
            kernel_version=kernel_version,
        )

    # Distinguish WSL1 vs WSL2
    wsl_distro = os.environ.get("WSL_DISTRO_NAME")

    # WSL2 indicators:
    # 1. /run/WSL directory exists
    # 2. "wsl2" appears in /proc/version
    # 3. WSL_INTEROP environment variable is set
    # 4. Kernel version 5.x+ (WSL2 uses real Linux kernel)
    # WSL2 uses kernel 5.x+; check major version >= 5
    kernel_major = None
    if kernel_version is not None:
        try:
            kernel_major = int(kernel_version.split(".")[0])
        except (ValueError, IndexError):
            pass

    is_wsl2 = (
        Path("/run/WSL").exists()
        or "wsl2" in proc_version
        or os.environ.get("WSL_INTEROP") is not None
        or (kernel_major is not None and kernel_major >= 5)
    )

    return PlatformInfo(
        platform_type=PlatformType.WSL2 if is_wsl2 else PlatformType.WSL1,
        kernel_version=kernel_version,
        wsl_distro=wsl_distro,
    )
