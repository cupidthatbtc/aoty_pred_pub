"""Artist history feature block with leave-one-out computation.

Computes artist track record features using expanding windows with shift()
to prevent data leakage. Each album sees only prior albums from that artist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseFeatureBlock, FeatureContext, FeatureOutput


def _compute_trajectory_slope(scores: pd.Series) -> float:
    """Compute linear slope of prior scores.

    Parameters
    ----------
    scores : pd.Series
        Series of prior album scores (may contain NaN).

    Returns
    -------
    float
        Slope of linear fit, or NaN if fewer than 2 valid scores.
    """
    valid = scores.dropna()
    if len(valid) < 2:
        return np.nan
    x = np.arange(len(valid))
    slope, _ = np.polyfit(x, valid.values, 1)
    return slope


class ArtistHistoryBlock(BaseFeatureBlock):
    """Artist history features using leave-one-out expanding windows.

    Computes prior mean, std, count, and trajectory for each artist,
    excluding the current album (LOO pattern via shift()).

    Required columns: Artist, Release_Date_Parsed, User_Score, Critic_Score, Album

    Output features (9 total):
    - user_prior_mean, user_prior_std, user_prior_count, user_trajectory
    - critic_prior_mean, critic_prior_std, critic_prior_count, critic_trajectory
    - is_debut

    Debut albums have NaN prior statistics which are imputed with global
    means learned from training data during fit().

    Examples
    --------
    >>> block = ArtistHistoryBlock()
    >>> block.fit(train_df, ctx)
    >>> output = block.transform(test_df, ctx)
    >>> output.feature_names
    ['user_prior_mean', 'user_prior_std', ...]
    """

    name = "artist_history"
    requires: list[str] = []
    required_columns = ["Artist", "Release_Date_Parsed", "User_Score", "Critic_Score", "Album"]

    def fit(self, df: pd.DataFrame, ctx: FeatureContext) -> "ArtistHistoryBlock":
        """Learn global statistics from training data for debut imputation.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with required columns.
        ctx : FeatureContext
            Shared context with config and random state.

        Returns
        -------
        ArtistHistoryBlock
            Self, for method chaining.
        """
        self.validate_columns(df)

        # Compute global statistics for debut imputation
        self._global_user_mean_ = df["User_Score"].mean()
        self._global_user_std_ = df["User_Score"].std()
        self._global_critic_mean_ = df["Critic_Score"].mean()
        self._global_critic_std_ = df["Critic_Score"].std()

        self._fitted_ = True
        return self

    def transform(self, df: pd.DataFrame, ctx: FeatureContext) -> FeatureOutput:
        """Compute LOO artist history features.

        Parameters
        ----------
        df : pd.DataFrame
            Data to transform (can be train, val, or test).
        ctx : FeatureContext
            Shared context with config and random state.

        Returns
        -------
        FeatureOutput
            Transformed features with 9 columns.

        Raises
        ------
        NotFittedError
            If fit() has not been called.
        """
        self._check_is_fitted()
        self.validate_columns(df)

        # Sort by Artist, date, and Album (Album for deterministic tie-breaking)
        df_sorted = df.sort_values(["Artist", "Release_Date_Parsed", "Album"]).copy()

        # Compute LOO expanding statistics for each score type
        for score_col, prefix in [("User_Score", "user"), ("Critic_Score", "critic")]:
            grp = df_sorted.groupby("Artist", sort=False)[score_col]

            # shift(1) excludes current album, expanding() accumulates prior
            df_sorted[f"{prefix}_prior_mean"] = grp.transform(
                lambda x: x.shift(1).expanding().mean()
            )
            df_sorted[f"{prefix}_prior_std"] = grp.transform(
                lambda x: x.shift(1).expanding().std()
            )
            df_sorted[f"{prefix}_prior_count"] = grp.transform(
                lambda x: x.shift(1).expanding().count()
            )

            # Compute trajectory slope (requires 2+ prior albums)
            df_sorted[f"{prefix}_trajectory"] = grp.transform(
                lambda x: x.shift(1).expanding().apply(_compute_trajectory_slope, raw=False)
            )

        # Mark debuts BEFORE imputation (using user_prior_mean NaN to detect)
        # Note: count() returns 0 for debuts, but mean() returns NaN
        df_sorted["is_debut"] = df_sorted["user_prior_mean"].isna().astype(int)

        # Impute debut NaN values with global statistics from fit()
        df_sorted["user_prior_mean"] = df_sorted["user_prior_mean"].fillna(self._global_user_mean_)
        df_sorted["user_prior_std"] = df_sorted["user_prior_std"].fillna(self._global_user_std_)
        df_sorted["user_prior_count"] = df_sorted["user_prior_count"].fillna(0)
        df_sorted["user_trajectory"] = df_sorted["user_trajectory"].fillna(0)

        df_sorted["critic_prior_mean"] = df_sorted["critic_prior_mean"].fillna(self._global_critic_mean_)
        df_sorted["critic_prior_std"] = df_sorted["critic_prior_std"].fillna(self._global_critic_std_)
        df_sorted["critic_prior_count"] = df_sorted["critic_prior_count"].fillna(0)
        df_sorted["critic_trajectory"] = df_sorted["critic_trajectory"].fillna(0)

        # Re-align to original index order
        result = df_sorted.loc[df.index]

        feature_cols = [
            "user_prior_mean",
            "user_prior_std",
            "user_prior_count",
            "user_trajectory",
            "critic_prior_mean",
            "critic_prior_std",
            "critic_prior_count",
            "critic_trajectory",
            "is_debut",
        ]

        return FeatureOutput(
            data=result[feature_cols],
            feature_names=feature_cols,
            metadata={
                "block": self.name,
                "global_user_mean": self._global_user_mean_,
                "global_user_std": self._global_user_std_,
                "global_critic_mean": self._global_critic_mean_,
                "global_critic_std": self._global_critic_std_,
            },
        )


# Backwards compatibility alias
ArtistReputationBlock = ArtistHistoryBlock
