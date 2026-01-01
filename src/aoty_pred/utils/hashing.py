"""Hashing utilities for reproducibility verification."""

import hashlib
from pathlib import Path


def sha256_file(path: Path | str, block_size: int = 65536) -> str:
    """
    Compute SHA256 hash of file in memory-efficient chunks.

    Args:
        path: Path to the file to hash
        block_size: Size of chunks to read (default 64KB for memory efficiency)

    Returns:
        Hexadecimal SHA256 digest (64 characters)

    Example:
        >>> h = sha256_file("data/raw/all_albums_full.csv")
        >>> len(h)
        64
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            sha256.update(block)
    return sha256.hexdigest()
