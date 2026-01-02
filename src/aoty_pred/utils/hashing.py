"""Hashing utilities for reproducibility verification."""

import hashlib
from pathlib import Path

import pandas as pd


def hash_dataframe(df: pd.DataFrame) -> str:
    """
    Generate deterministic SHA256 hash of DataFrame contents.

    Uses pandas internal hashing (handles NaN, dtypes consistently)
    then combines into single digest.

    Args:
        df: DataFrame to hash

    Returns:
        Hexadecimal SHA256 digest (64 characters)

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, 2, 3]})
        >>> h = hash_dataframe(df)
        >>> len(h)
        64
    """
    # Get per-row hashes as uint64 series
    row_hashes = pd.util.hash_pandas_object(df, index=True)
    # Combine into bytes and hash
    hash_bytes = row_hashes.values.tobytes()
    return hashlib.sha256(hash_bytes).hexdigest()


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
