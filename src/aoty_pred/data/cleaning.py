"""Data cleaning and filtering pipeline."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from aoty_pred.data.lineage import AuditLogger

# Column name mapping from raw to canonical
RAW_TO_CANONICAL = {
    "Release Date": "Release_Date",
    "Critic Score": "Critic_Score",
    "User Score": "User_Score",
    "Avg Track Score": "Avg_Track_Score",
    "User Ratings": "User_Ratings",
    "Critic Reviews": "Critic_Reviews",
    "Tracks": "Num_Tracks",
    "Runtime (min)": "Runtime_Min",
    "Avg Track Runtime (min)": "Avg_Runtime",
    "Album URL": "Album_URL",
    "All Artists": "All_Artists",
    "Album Type": "Album_Type",
}


@dataclass
class CleaningConfig:
    """Configuration for cleaning pipeline."""

    min_year: int = 1950
    max_year: int = 2025
    score_min: float = 0.0
    score_max: float = 100.0
    drop_descriptors: bool = True  # Per research: 4.2% coverage, severe selection bias


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns from raw format to canonical (underscore-separated)."""
    return df.rename(columns=RAW_TO_CANONICAL)


def parse_release_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse Release_Date column with three-tier risk classification.

    Creates columns:
    - Release_Date_Parsed: datetime
    - date_risk: 'low' | 'medium' | 'high'
    - date_imputation_type: 'none' | 'jan1' | 'artist_inferred'
    """
    df = df.copy()

    # Parse existing dates (format: "April 10, 2018")
    df["Release_Date_Parsed"] = pd.to_datetime(
        df["Release_Date"],
        format="%B %d, %Y",
        errors="coerce",
    )

    # Initialize risk columns
    df["date_risk"] = "low"
    df["date_imputation_type"] = "none"

    # Tier 2: Has Year but no Release_Date
    tier2_mask = df["Release_Date_Parsed"].isna() & df["Year"].notna()
    df.loc[tier2_mask, "date_risk"] = "medium"
    df.loc[tier2_mask, "date_imputation_type"] = "jan1"
    df.loc[tier2_mask, "Release_Date_Parsed"] = pd.to_datetime(
        df.loc[tier2_mask, "Year"].astype(int).astype(str) + "-01-01"
    )

    # Tier 3: Missing both (will be handled separately - requires artist inference)
    tier3_mask = df["Release_Date"].isna() & df["Year"].isna()
    df.loc[tier3_mask, "date_risk"] = "high"
    df.loc[tier3_mask, "date_imputation_type"] = "artist_inferred"

    # Flag edge cases
    df["flag_future_year"] = df["Year"] > 2025
    df["flag_sparse_era"] = df["Year"] < 1950

    return df


def extract_collaboration_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract collaboration features from All_Artists column.

    Creates columns:
    - num_artists: count of artists
    - is_collaboration: boolean
    - collab_type: 'solo' | 'duo' | 'small_group' | 'ensemble'
    """
    df = df.copy()

    def parse_artist_count(all_artists_str: str, primary: str) -> int:
        if pd.isna(all_artists_str):
            return 1
        if " | " in str(all_artists_str):
            return len(all_artists_str.split(" | "))
        return 1

    df["num_artists"] = df.apply(
        lambda row: parse_artist_count(row.get("All_Artists"), row["Artist"]),
        axis=1,
    )
    df["is_collaboration"] = df["num_artists"] > 1

    def classify_collab(n: int) -> str:
        if n == 1:
            return "solo"
        elif n == 2:
            return "duo"
        elif n <= 4:
            return "small_group"
        return "ensemble"

    df["collab_type"] = df["num_artists"].apply(classify_collab)

    return df


def extract_primary_genre(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract primary genre (first listed) from Genres column.

    Creates column:
    - primary_genre: first genre in comma-separated list
    """
    df = df.copy()
    df["primary_genre"] = df["Genres"].str.split(", ").str[0]
    return df


def flag_unknown_artist(df: pd.DataFrame) -> pd.DataFrame:
    """Flag [unknown artist] for special handling."""
    df = df.copy()
    df["is_unknown_artist"] = df["Artist"] == "[unknown artist]"
    return df


def apply_exclusion_filter(
    df: pd.DataFrame,
    condition: pd.Series,
    reason: str,
    logger: Optional[AuditLogger] = None,
    value_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply filter and log exclusions.

    Args:
        df: DataFrame to filter
        condition: Boolean Series - True = keep, False = exclude
        reason: Reason string for audit log
        logger: Optional AuditLogger to record exclusions
        value_col: Column name to include in exclusion value

    Returns:
        Filtered DataFrame (rows where condition is True)
    """
    excluded = df[~condition]
    kept = df[condition].copy()

    if logger is not None:
        logger.log_exclusions_bulk(excluded, reason=reason, value_col=value_col)
        logger.log_filter_stats(
            filter_name=reason,
            rows_before=len(df),
            rows_excluded=len(excluded),
            rows_after=len(kept),
        )

    return kept


def clean_albums(
    df: pd.DataFrame,
    config: Optional[CleaningConfig] = None,
    logger: Optional[AuditLogger] = None,
) -> pd.DataFrame:
    """
    Apply full cleaning pipeline to raw album data.

    Steps:
    1. Rename columns to canonical format
    2. Parse dates with risk classification
    3. Extract collaboration features
    4. Extract primary genre
    5. Flag unknown artist
    6. Drop Descriptors column (per research)

    Does NOT apply filtering - use apply_exclusion_filter separately.

    Args:
        df: Raw DataFrame (with original_row_id)
        config: Cleaning configuration
        logger: Optional audit logger

    Returns:
        Cleaned DataFrame with new columns
    """
    config = config or CleaningConfig()

    # Apply transformations
    df = rename_columns(df)
    df = parse_release_dates(df)
    df = extract_collaboration_features(df)
    df = extract_primary_genre(df)
    df = flag_unknown_artist(df)

    # Drop Descriptors per research (4.2% coverage, severe selection bias)
    if config.drop_descriptors and "Descriptors" in df.columns:
        df = df.drop(columns=["Descriptors"])
        if logger:
            logger.log.info(
                "column_dropped", column="Descriptors", reason="low_coverage_selection_bias"
            )

    return df


def filter_for_user_score_model(
    df: pd.DataFrame,
    min_ratings: int,
    logger: Optional[AuditLogger] = None,
) -> pd.DataFrame:
    """
    Filter dataset for user score modeling.

    Requires:
    - Valid User_Score (0-100)
    - User_Ratings >= min_ratings
    """
    # Filter: has valid user score
    df = apply_exclusion_filter(
        df,
        condition=df["User_Score"].notna(),
        reason="missing_user_score",
        logger=logger,
        value_col="User_Score",
    )

    # Filter: score in valid range
    df = apply_exclusion_filter(
        df,
        condition=(df["User_Score"] >= 0) & (df["User_Score"] <= 100),
        reason="invalid_user_score_range",
        logger=logger,
        value_col="User_Score",
    )

    # Filter: minimum ratings
    df = apply_exclusion_filter(
        df,
        condition=df["User_Ratings"] >= min_ratings,
        logger=logger,
        reason=f"below_min_ratings_{min_ratings}",
        value_col="User_Ratings",
    )

    return df


def filter_for_critic_score_model(
    df: pd.DataFrame,
    min_reviews: int = 1,
    logger: Optional[AuditLogger] = None,
) -> pd.DataFrame:
    """
    Filter dataset for critic score modeling.

    Requires:
    - Valid Critic_Score (0-100)
    - Critic_Reviews >= min_reviews
    """
    # Filter: has valid critic score
    df = apply_exclusion_filter(
        df,
        condition=df["Critic_Score"].notna(),
        reason="missing_critic_score",
        logger=logger,
        value_col="Critic_Score",
    )

    # Filter: score in valid range
    df = apply_exclusion_filter(
        df,
        condition=(df["Critic_Score"] >= 0) & (df["Critic_Score"] <= 100),
        reason="invalid_critic_score_range",
        logger=logger,
        value_col="Critic_Score",
    )

    # Filter: minimum reviews
    df = apply_exclusion_filter(
        df,
        condition=df["Critic_Reviews"] >= min_reviews,
        reason=f"below_min_reviews_{min_reviews}",
        logger=logger,
        value_col="Critic_Reviews",
    )

    return df
