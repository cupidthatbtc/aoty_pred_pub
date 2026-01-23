"""Schema validation for raw and processed data."""

import pandas as pd
import pandera.pandas as pa


# Expected column names from raw CSV (all 18 columns)
REQUIRED_RAW_COLUMNS = [
    "Artist",
    "Album",
    "Year",
    "Release Date",
    "Genres",
    "Descriptors",
    "Critic Score",
    "User Score",
    "User Ratings",
    "Critic Reviews",
    "Tracks",
    "Runtime (min)",
    "Avg Track Runtime (min)",
    "Album Type",
    "Album URL",
    "All Artists",
]


# Define the schema for raw data validation
# Note: Use float for numeric columns because pandas represents int+NaN as float
RawAlbumSchema = pa.DataFrameSchema(
    {
        "Artist": pa.Column(str, nullable=False),
        "Album": pa.Column(str, nullable=False),
        "Year": pa.Column(float, pa.Check.in_range(1900, 2030), nullable=True),
        "Release Date": pa.Column(str, nullable=True),
        "Genres": pa.Column(str, nullable=True),
        "Descriptors": pa.Column(str, nullable=True),
        "Critic Score": pa.Column(float, pa.Check.in_range(0, 100), nullable=True),
        "User Score": pa.Column(float, pa.Check.in_range(0, 100), nullable=True),
        "User Ratings": pa.Column(float, pa.Check.ge(0), nullable=True),
        "Critic Reviews": pa.Column(float, pa.Check.ge(0), nullable=True),
        "Tracks": pa.Column(float, pa.Check.ge(0), nullable=True),
        "Runtime (min)": pa.Column(float, pa.Check.ge(0), nullable=True),
        "Avg Track Runtime (min)": pa.Column(float, pa.Check.ge(0), nullable=True),
        "Album Type": pa.Column(str, nullable=True),
        "Album URL": pa.Column(str, nullable=True),
        "All Artists": pa.Column(str, nullable=True),
    },
    strict=False,  # Allow extra columns (original_row_id added later)
    coerce=True,  # Coerce types where possible
)


def validate_raw_dataframe(df: pd.DataFrame, lazy: bool = True) -> pd.DataFrame:
    """
    Validate raw DataFrame against RawAlbumSchema.

    Args:
        df: DataFrame loaded from raw CSV
        lazy: If True, collect all errors before raising. If False, fail fast.

    Returns:
        Validated DataFrame (with types coerced)

    Raises:
        pa.errors.SchemaErrors: If validation fails (contains failure_cases attribute)
    """
    try:
        validated = RawAlbumSchema.validate(df, lazy=lazy)
        return validated
    except pa.errors.SchemaErrors as e:
        # Re-raise with helpful context
        raise pa.errors.SchemaErrors(
            schema_errors=e.schema_errors, data=e.data
        ) from None


def validate_raw_schema(df: pd.DataFrame) -> None:
    """
    Legacy function for backward compatibility.

    Raises ValueError if required columns are missing.
    """
    missing = [col for col in REQUIRED_RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw columns: {missing}")
