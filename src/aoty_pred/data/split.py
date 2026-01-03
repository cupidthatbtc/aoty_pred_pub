"""Leak-safe splitting logic for artist album prediction."""

from typing import Tuple
import warnings

import pandas as pd


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
