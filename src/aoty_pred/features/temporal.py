"""Temporal feature block for album career context.

Computes temporal features that capture career trajectory context:
- album_sequence: Sequential album number for artist (1, 2, 3...)
- career_years: Years since artist's first album
- release_gap_days: Days since artist's previous album (0 for debuts)
- release_year: Calendar year for trend capture
- date_risk_ordinal: Risk level of date accuracy (low=0, medium=1, high=2)
"""

from __future__ import annotations

import pandas as pd

from .base import BaseFeatureBlock, FeatureContext, FeatureOutput


class TemporalBlock(BaseFeatureBlock):
    """Feature block computing temporal context features.

    This block is stateless - no statistics are learned during fit.
    The fit() method validates required columns and sets fitted state.

    Required columns: Artist, Release_Date_Parsed, Year, date_risk, Album

    Features computed:
        - album_sequence: 1-indexed album number within artist
        - career_years: Years since artist's first album
        - release_gap_days: Days since previous album (0 for debuts)
        - release_year: Calendar year of release
        - date_risk_ordinal: Ordinal encoding of date risk level

    Examples
    --------
    >>> block = TemporalBlock()
    >>> block.fit(train_df, ctx)
    >>> output = block.transform(test_df, ctx)
    >>> output.feature_names
    ['album_sequence', 'career_years', 'release_gap_days', 'release_year', 'date_risk_ordinal']
    """

    name = "temporal"
    requires: list[str] = []
    required_columns: list[str] = [
        "Artist",
        "Release_Date_Parsed",
        "Year",
        "date_risk",
        "Album",
    ]

    def fit(self, df, ctx: FeatureContext) -> "TemporalBlock":
        """Fit the temporal block on training data.

        Validates required columns exist. This block is stateless,
        so no statistics are learned from training data.

        Parameters
        ----------
        df : DataFrame
            Training data with required columns.
        ctx : FeatureContext
            Shared context (unused for this stateless block).

        Returns
        -------
        TemporalBlock
            Self, for method chaining.
        """
        self.validate_columns(df)
        self._fitted_ = True
        return self

    def transform(self, df, ctx: FeatureContext) -> FeatureOutput:
        """Transform data to compute temporal features.

        Parameters
        ----------
        df : DataFrame
            Data to transform (train, val, or test).
        ctx : FeatureContext
            Shared context (unused for this block).

        Returns
        -------
        FeatureOutput
            DataFrame with 5 temporal feature columns.

        Raises
        ------
        NotFittedError
            If fit() has not been called.
        """
        self._check_is_fitted()

        # Sort by Artist, Release_Date_Parsed, Album for deterministic ordering
        # Album as tiebreaker ensures same-date albums have consistent order
        df_sorted = df.sort_values(
            ["Artist", "Release_Date_Parsed", "Album"]
        ).copy()

        # Album sequence (1-indexed): cumcount + 1 within artist
        df_sorted["album_sequence"] = (
            df_sorted.groupby("Artist").cumcount() + 1
        )

        # Career length: years since artist's first album
        df_sorted["first_release"] = df_sorted.groupby("Artist")[
            "Release_Date_Parsed"
        ].transform("min")
        df_sorted["career_years"] = (
            df_sorted["Release_Date_Parsed"] - df_sorted["first_release"]
        ).dt.days / 365.25

        # Release gap: days since previous album (0 for debuts)
        df_sorted["prev_release"] = df_sorted.groupby("Artist")[
            "Release_Date_Parsed"
        ].shift(1)
        df_sorted["release_gap_days"] = (
            df_sorted["Release_Date_Parsed"] - df_sorted["prev_release"]
        ).dt.days
        df_sorted["release_gap_days"] = df_sorted["release_gap_days"].fillna(0)

        # Release year for trend capture
        df_sorted["release_year"] = df_sorted["Release_Date_Parsed"].dt.year

        # Date risk as ordinal (low=0, medium=1, high=2)
        risk_map = {"low": 0, "medium": 1, "high": 2}
        df_sorted["date_risk_ordinal"] = (
            df_sorted["date_risk"].map(risk_map).fillna(1)
        )

        # Re-align to original index before returning
        result = df_sorted.loc[df.index]

        feature_cols = [
            "album_sequence",
            "career_years",
            "release_gap_days",
            "release_year",
            "date_risk_ordinal",
        ]

        return FeatureOutput(
            data=result[feature_cols],
            feature_names=feature_cols,
            metadata={"block": self.name, "params": self.params},
        )
