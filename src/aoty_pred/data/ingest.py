"""Raw data ingestion with validation and metadata."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from aoty_pred.data.validation import validate_raw_dataframe
from aoty_pred.io.readers import read_csv
from aoty_pred.utils.hashing import sha256_file


@dataclass
class LoadMetadata:
    """Metadata about the loaded dataset."""

    file_path: str
    file_hash: str
    load_timestamp: str
    row_count: int
    column_count: int


def load_raw_albums(
    path: str | Path = "data/raw/all_albums_full.csv",
    validate: bool = False,
) -> tuple[pd.DataFrame, LoadMetadata]:
    """
    Load raw AOTY album data with validation and metadata.

    Args:
        path: Path to raw CSV file
        validate: Whether to validate against schema (default False).
            Raw data may have quality issues (e.g., 5 rows with null Album)
            that are caught by validation. Set to True to enforce strict schema.

    Returns:
        Tuple of (DataFrame, LoadMetadata)
        - DataFrame has 'original_row_id' column preserving raw CSV row numbers
        - LoadMetadata contains file hash and load info for reproducibility

    Raises:
        FileNotFoundError: If path does not exist
        pa.errors.SchemaErrors: If validation fails
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    # Compute hash before loading for reproducibility
    file_hash = sha256_file(path)

    # Load with BOM-handling encoding
    df = read_csv(path, encoding="utf-8-sig")

    # Preserve original row IDs for audit trail (before any filtering)
    df["original_row_id"] = df.index

    # Validate if requested
    if validate:
        df = validate_raw_dataframe(df)

    # Build metadata
    metadata = LoadMetadata(
        file_path=str(path.resolve()),
        file_hash=file_hash,
        load_timestamp=datetime.now().isoformat(),
        row_count=len(df),
        column_count=len(df.columns),
    )

    return df, metadata


def load_raw_dataset(path: str, encoding: str = "utf-8-sig") -> pd.DataFrame:
    """
    Legacy function for backward compatibility.

    Use load_raw_albums() for new code - it returns metadata.
    """
    df, _ = load_raw_albums(path, validate=False)
    return df
