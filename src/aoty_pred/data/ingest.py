"""Raw data ingestion with validation and metadata."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from aoty_pred.data.validation import validate_raw_dataframe
from aoty_pred.io.readers import read_csv
from aoty_pred.utils.hashing import sha256_file

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataDimensions:
    """Data dimensions for memory estimation.

    Attributes:
        n_observations: Number of rows after filtering.
        n_artists: Number of unique artists.
        source: Description of data source (e.g., "from data: filename.csv").
    """

    n_observations: int
    n_artists: int
    source: str

    @classmethod
    def from_defaults(cls) -> DataDimensions:
        """Create with conservative defaults when data unavailable."""
        return cls(
            n_observations=1000,
            n_artists=100,
            source="defaults (data unavailable)",
        )


def extract_data_dimensions(
    csv_path: Path | str = "data/raw/all_albums_full.csv",
    min_ratings: int = 10,
) -> DataDimensions:
    """Extract observation and artist counts from raw CSV.

    Loads only required columns for performance (~0.2s vs ~30s full load).

    Args:
        csv_path: Path to raw CSV file.
        min_ratings: Minimum user ratings filter (matches CLI default).

    Returns:
        DataDimensions with counts and source indicator.
        Returns conservative defaults if file not found or on error.
    """
    path = Path(csv_path)

    if not path.exists():
        return DataDimensions.from_defaults()

    try:
        # Only load columns needed for counting
        df = pd.read_csv(
            path,
            usecols=["Artist", "User Ratings"],
            encoding="utf-8-sig",  # Handle BOM
        )

        # Apply same filter as training pipeline
        df = df[df["User Ratings"] >= min_ratings]

        return DataDimensions(
            n_observations=len(df),
            n_artists=df["Artist"].nunique(),
            source=f"from data: {path.name}",
        )
    except Exception as e:
        logger.warning(
            "Failed to extract dimensions from %s (min_ratings=%d): %s. "
            "Falling back to defaults.",
            path.name,
            min_ratings,
            e,
        )
        return DataDimensions.from_defaults()


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
