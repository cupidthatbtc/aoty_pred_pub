"""Leak-safe splitting logic for artist album prediction."""

from typing import Tuple
import warnings

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def within_artist_temporal_split(
    df: pd.DataFrame,
    artist_col: str = "Artist",
    date_col: str = "Release_Date_Parsed",
    test_albums: int = 1,
    val_albums: int = 1,
    min_train_albums: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data holding out last N albums per artist for test/validation.

    This is the PRIMARY evaluation strategy: tests model's ability to predict
    an artist's next album given their history.

    Args:
        df: Cleaned album DataFrame with Artist and date columns
        artist_col: Column name for artist grouping
        date_col: Column name for temporal ordering (Release_Date_Parsed)
        test_albums: Number of most recent albums per artist for test set
        val_albums: Number of second-most-recent albums per artist for validation
        min_train_albums: Minimum albums required in training set per artist

    Returns:
        Tuple of (train_df, val_df, test_df)

    Note:
        Artists with fewer than (test_albums + val_albums + min_train_albums)
        albums are excluded from all splits.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "Artist": ["A"]*4 + ["B"]*3,
        ...     "Album": list(range(7)),
        ...     "Release_Date_Parsed": pd.date_range("2020", periods=7, freq="YS"),
        ... })
        >>> train, val, test = within_artist_temporal_split(df)
        >>> len(train), len(val), len(test)
        (3, 2, 2)
    """
    # Sort by artist and date to ensure temporal ordering
    df_sorted = df.sort_values([artist_col, date_col])

    # Count albums per artist
    album_counts = df_sorted.groupby(artist_col).size()
    min_required = test_albums + val_albums + min_train_albums
    valid_artists = album_counts[album_counts >= min_required].index
    df_valid = df_sorted[df_sorted[artist_col].isin(valid_artists)].copy()

    # Extract last N per artist for test
    test_df = df_valid.groupby(artist_col).tail(test_albums)
    remaining = df_valid.drop(test_df.index)

    # Extract second-to-last N per artist for validation
    val_df = remaining.groupby(artist_col).tail(val_albums)
    train_df = remaining.drop(val_df.index)

    return train_df, val_df, test_df


def artist_disjoint_split(
    df: pd.DataFrame,
    artist_col: str = "Artist",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data ensuring no artist appears in multiple splits.

    This is the SECONDARY evaluation strategy: tests model's ability to
    generalize to unseen artists (cold-start evaluation).

    Uses two-stage GroupShuffleSplit:
    1. Split test set (artist-disjoint)
    2. Split validation from remaining (artist-disjoint)

    Args:
        df: Cleaned album DataFrame
        artist_col: Column name for artist grouping
        test_size: Proportion of data for test set (by artist groups)
        val_size: Proportion of data for validation set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "Artist": [f"Artist_{i//3}" for i in range(60)],
        ...     "Album": list(range(60)),
        ...     "Score": [70]*60,
        ... })
        >>> train, val, test = artist_disjoint_split(df, random_state=42)
        >>> # No artist overlap between splits
        >>> train_a = set(train["Artist"])
        >>> test_a = set(test["Artist"])
        >>> len(train_a & test_a)
        0
    """
    groups = df[artist_col].values

    # Stage 1: Separate test set
    gss_test = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_val_idx, test_idx = next(gss_test.split(df, groups=groups))

    test_df = df.iloc[test_idx].copy()
    train_val_df = df.iloc[train_val_idx]

    # Stage 2: Separate validation from train
    val_proportion = val_size / (1 - test_size)
    gss_val = GroupShuffleSplit(
        n_splits=1,
        test_size=val_proportion,
        random_state=random_state + 1,  # Different seed for second split
    )
    train_val_groups = train_val_df[artist_col].values
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_groups))

    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()

    return train_df, val_df, test_df


def assert_no_artist_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    artist_col: str = "Artist",
) -> None:
    """
    Verify no artist appears in multiple splits (artist-disjoint property).

    Raises:
        ValueError: If any artist overlap is detected

    Note:
        This should ONLY be called for artist-disjoint splits.
        Within-artist temporal splits intentionally have artist overlap.
    """
    train_artists = set(train_df[artist_col])
    val_artists = set(val_df[artist_col])
    test_artists = set(test_df[artist_col])

    overlap_train_val = train_artists & val_artists
    overlap_train_test = train_artists & test_artists
    overlap_val_test = val_artists & test_artists

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError(
            f"Artist overlap detected: "
            f"train-val={len(overlap_train_val)}, "
            f"train-test={len(overlap_train_test)}, "
            f"val-test={len(overlap_val_test)}"
        )


def validate_temporal_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    artist_col: str = "Artist",
    date_col: str = "Release_Date_Parsed",
) -> None:
    """
    Verify temporal ordering is correct for within-artist splits.

    For each artist, checks that:
    - Test albums are chronologically after or equal to validation albums
    - Validation albums are chronologically after or equal to training albums

    Note:
        Same-date albums are allowed since the split function uses groupby.tail()
        which provides consistent ordering. Only strictly backwards ordering
        (training data after test data) is flagged as a violation.

    Raises:
        ValueError: If temporal ordering is violated (train after test)
    """
    # Get artists that appear in all splits (expected for temporal split)
    train_artists = set(train_df[artist_col])
    test_artists = set(test_df[artist_col])
    val_artists = set(val_df[artist_col])

    # Only validate artists present in both train and test
    common_artists = train_artists & test_artists

    for artist in common_artists:
        train_max = train_df[train_df[artist_col] == artist][date_col].max()
        test_min = test_df[test_df[artist_col] == artist][date_col].min()

        # Strict check: training data must not come AFTER test data
        # Same-date albums are OK (tail() provides consistent ordering)
        if train_max > test_min:
            raise ValueError(
                f"Temporal violation for {artist}: "
                f"train max date {train_max} > test min date {test_min}"
            )

        # Check validation if artist present
        if artist in val_artists:
            val_dates = val_df[val_df[artist_col] == artist][date_col]
            val_min = val_dates.min()
            val_max = val_dates.max()

            if train_max > val_min:
                raise ValueError(
                    f"Temporal violation for {artist}: "
                    f"train max {train_max} > val min {val_min}"
                )
            if val_max > test_min:
                raise ValueError(
                    f"Temporal violation for {artist}: "
                    f"val max {val_max} > test min {test_min}"
                )


def group_split(df, group_col: str, seed: int):
    """
    DEPRECATED: Use within_artist_temporal_split or artist_disjoint_split instead.

    This stub remains for backward compatibility only.
    """
    warnings.warn(
        "group_split is deprecated. Use within_artist_temporal_split or "
        "artist_disjoint_split instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return df, df, df
