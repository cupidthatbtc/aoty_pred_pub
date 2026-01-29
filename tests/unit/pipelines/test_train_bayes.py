"""Unit tests for train_bayes pipeline data preparation functions.

Tests cover:
- load_training_data: DataFrame alignment and overlap handling
- prepare_model_data: Artist indexing, album sequences, n_reviews validation
- _apply_max_albums_cap: Most-recent album capping logic

These tests do NOT run MCMC - they focus on pure data preparation logic.
"""

import numpy as np
import pandas as pd
import pytest

from aoty_pred.pipelines.train_bayes import (
    _apply_max_albums_cap,
    load_training_data,
    prepare_model_data,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_features_df():
    """Create a sample features DataFrame."""
    return pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_2": [0.1, 0.2, 0.3, 0.4, 0.5],
            "n_reviews": [10, 20, 30, 40, 50],
        },
        index=pd.RangeIndex(5),
    )


@pytest.fixture
def sample_splits_df():
    """Create a sample splits DataFrame with required columns."""
    return pd.DataFrame(
        {
            "Artist": ["A", "A", "B", "B", "B"],
            "User_Score": [70.0, 75.0, 80.0, 85.0, 90.0],
            "Album": ["a1", "a2", "b1", "b2", "b3"],
        },
        index=pd.RangeIndex(5),
    )


@pytest.fixture
def sample_train_df():
    """Create a sample training DataFrame for prepare_model_data tests."""
    return pd.DataFrame(
        {
            "Artist": ["A", "A", "A", "B", "B", "C"],
            "User_Score": [70.0, 75.0, 80.0, 85.0, 90.0, 65.0],
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feature_2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "n_reviews": [10, 20, 30, 40, 50, 60],
        }
    )


# =============================================================================
# Tests for load_training_data
# =============================================================================


class TestLoadTrainingData:
    """Tests for load_training_data function."""

    def test_length_mismatch_raises_value_error(self, tmp_path, sample_splits_df):
        """Should raise ValueError when DataFrames have different lengths."""
        # Create features with different length
        features_df = pd.DataFrame(
            {
                "feature_1": [1.0, 2.0, 3.0],  # Only 3 rows vs 5 in splits
                "n_reviews": [10, 20, 30],
            }
        )

        # Save to parquet
        features_path = tmp_path / "features.parquet"
        splits_path = tmp_path / "splits.parquet"
        features_df.to_parquet(features_path)
        sample_splits_df.to_parquet(splits_path)

        with pytest.raises(ValueError, match="DataFrame length mismatch"):
            load_training_data(features_path, splits_path)

    def test_index_mismatch_raises_value_error(self, tmp_path, sample_splits_df):
        """Should raise ValueError when DataFrames have mismatched indices."""
        # Create features with different indices
        features_df = pd.DataFrame(
            {
                "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "n_reviews": [10, 20, 30, 40, 50],
            },
            index=[10, 11, 12, 13, 14],  # Different from splits 0-4
        )

        features_path = tmp_path / "features.parquet"
        splits_path = tmp_path / "splits.parquet"
        features_df.to_parquet(features_path)
        sample_splits_df.to_parquet(splits_path)

        with pytest.raises(ValueError, match="DataFrame index mismatch"):
            load_training_data(features_path, splits_path)

    def test_overlap_columns_dropped(self, tmp_path, sample_features_df):
        """Should drop overlapping columns from splits before join."""
        # Create splits with a column that overlaps features
        splits_df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B", "B", "B"],
                "User_Score": [70.0, 75.0, 80.0, 85.0, 90.0],
                "feature_1": [999.0, 999.0, 999.0, 999.0, 999.0],  # Overlap
                "n_reviews": [100, 200, 300, 400, 500],  # Overlap - different values
            },
            index=pd.RangeIndex(5),
        )

        features_path = tmp_path / "features.parquet"
        splits_path = tmp_path / "splits.parquet"
        sample_features_df.to_parquet(features_path)
        splits_df.to_parquet(splits_path)

        model_args, feature_cols, train_df = load_training_data(features_path, splits_path)

        # Feature values should be from features_df, not splits_df
        assert train_df["feature_1"].iloc[0] == 1.0, "feature_1 should come from features"
        assert train_df["n_reviews"].iloc[0] == 10, "n_reviews should come from features"

    def test_nan_features_filled(self, tmp_path, sample_splits_df):
        """Should fill NaN values in feature columns with 0."""
        features_df = pd.DataFrame(
            {
                "feature_1": [1.0, np.nan, 3.0, np.nan, 5.0],
                "n_reviews": [10, 20, 30, 40, 50],
            },
            index=pd.RangeIndex(5),
        )

        features_path = tmp_path / "features.parquet"
        splits_path = tmp_path / "splits.parquet"
        features_df.to_parquet(features_path)
        sample_splits_df.to_parquet(splits_path)

        model_args, feature_cols, train_df = load_training_data(features_path, splits_path)

        # NaN values should be filled with 0
        assert not train_df["feature_1"].isna().any(), "NaN values should be filled"
        assert train_df["feature_1"].iloc[1] == 0.0
        assert train_df["feature_1"].iloc[3] == 0.0


# =============================================================================
# Tests for prepare_model_data
# =============================================================================


class TestPrepareModelData:
    """Tests for prepare_model_data function."""

    def test_artist_index_mapping(self, sample_train_df):
        """Should create sequential artist indices."""
        model_args, valid_mask = prepare_model_data(
            sample_train_df, ["feature_1", "feature_2"], min_albums_filter=1
        )

        artist_idx = model_args["artist_idx"]

        # Should have unique indices for each artist
        unique_indices = set(artist_idx)
        assert len(unique_indices) == 3, "Should have 3 unique artist indices"
        # Indices should be 0, 1, 2
        assert unique_indices == {0, 1, 2}

    def test_album_seq_computation(self, sample_train_df):
        """Should compute 1-indexed album sequence within artist."""
        model_args, valid_mask = prepare_model_data(
            sample_train_df, ["feature_1", "feature_2"], min_albums_filter=1
        )

        album_seq = model_args["album_seq"]

        # Artist A has 3 albums -> seq 1, 2, 3
        # Artist B has 2 albums -> seq 1, 2
        # Artist C has 1 album -> seq 1
        # Order: A, A, A, B, B, C
        expected = np.array([1, 2, 3, 1, 2, 1])
        np.testing.assert_array_equal(album_seq, expected)

    def test_min_albums_filter_clamps_seq(self):
        """Artists below min_albums_filter should get album_seq=1."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "A", "B", "C"],  # A:3, B:1, C:1
                "User_Score": [70.0, 75.0, 80.0, 85.0, 90.0],
                "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "n_reviews": [10, 20, 30, 40, 50],
            }
        )

        model_args, valid_mask = prepare_model_data(df, ["feature_1"], min_albums_filter=2)

        album_seq = model_args["album_seq"]

        # A has 3 albums >= 2 -> seq 1, 2, 3
        # B has 1 album < 2 -> seq 1 (clamped)
        # C has 1 album < 2 -> seq 1 (clamped)
        expected = np.array([1, 2, 3, 1, 1])
        np.testing.assert_array_equal(album_seq, expected)

    def test_prev_score_computation(self, sample_train_df):
        """Should compute shifted prev_score within artist."""
        model_args, valid_mask = prepare_model_data(
            sample_train_df, ["feature_1", "feature_2"], min_albums_filter=1
        )

        prev_score = model_args["prev_score"]
        y = model_args["y"]

        # Global mean of User_Score
        global_mean = sample_train_df["User_Score"].mean()

        # First album of each artist should have global mean
        # A's first (idx 0): global_mean
        # A's second (idx 1): A's first score = 70.0
        # A's third (idx 2): A's second score = 75.0
        # B's first (idx 3): global_mean
        # B's second (idx 4): B's first score = 85.0
        # C's first (idx 5): global_mean
        assert np.isclose(prev_score[0], global_mean)
        assert np.isclose(prev_score[1], 70.0)
        assert np.isclose(prev_score[2], 75.0)
        assert np.isclose(prev_score[3], global_mean)
        assert np.isclose(prev_score[4], 85.0)
        assert np.isclose(prev_score[5], global_mean)

    def test_n_reviews_validation_filters_invalid(self):
        """Should filter rows with NaN or <= 0 n_reviews."""
        # Keep less than 50% invalid to avoid the "too many invalid" error
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "A", "B", "B", "C"],
                "User_Score": [70.0, 75.0, 80.0, 85.0, 90.0, 95.0],
                "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "n_reviews": [10, np.nan, 30, 40, 0, 60],  # 2 invalid out of 6 (33%)
            }
        )

        model_args, valid_mask = prepare_model_data(df, ["feature_1"], min_albums_filter=1)

        # Should keep only valid rows (indices 0, 2, 3, 5)
        assert len(model_args["y"]) == 4
        assert valid_mask.sum() == 4
        # Verify kept rows
        np.testing.assert_array_equal(valid_mask, [True, False, True, True, False, True])

    def test_n_reviews_too_many_invalid_raises(self):
        """Should raise ValueError when >50% n_reviews are invalid."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B", "B"],
                "User_Score": [70.0, 75.0, 80.0, 85.0],
                "feature_1": [1.0, 2.0, 3.0, 4.0],
                "n_reviews": [np.nan, np.nan, np.nan, 40],  # 75% invalid
            }
        )

        with pytest.raises(ValueError, match="Too many invalid n_reviews"):
            prepare_model_data(df, ["feature_1"], min_albums_filter=1)

    def test_model_args_has_required_keys(self, sample_train_df):
        """Should include all required keys in model_args."""
        model_args, valid_mask = prepare_model_data(
            sample_train_df, ["feature_1", "feature_2"], min_albums_filter=1
        )

        required_keys = [
            "artist_idx",
            "album_seq",
            "prev_score",
            "X",
            "y",
            "n_reviews",
            "n_artists",
            "artist_album_counts",
        ]

        for key in required_keys:
            assert key in model_args, f"Missing key: {key}"

    def test_feature_matrix_shape(self, sample_train_df):
        """Feature matrix should have correct shape."""
        model_args, valid_mask = prepare_model_data(
            sample_train_df, ["feature_1", "feature_2"], min_albums_filter=1
        )

        X = model_args["X"]
        assert X.shape == (6, 2), "X should be (n_obs, n_features)"
        assert X.dtype == np.float32

    def test_uses_user_ratings_fallback(self):
        """Should use User_Ratings if n_reviews not in features."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B"],
                "User_Score": [70.0, 75.0, 80.0],
                "User_Ratings": [10, 20, 30],  # Fallback column
                "feature_1": [1.0, 2.0, 3.0],
            }
        )

        model_args, valid_mask = prepare_model_data(df, ["feature_1"], min_albums_filter=1)

        # Should use User_Ratings as n_reviews
        np.testing.assert_array_equal(model_args["n_reviews"], [10, 20, 30])

    def test_missing_n_reviews_and_user_ratings_raises(self):
        """Should raise ValueError if neither n_reviews nor User_Ratings present."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A"],
                "User_Score": [70.0, 75.0],
                "feature_1": [1.0, 2.0],
            }
        )

        with pytest.raises(ValueError, match="n_reviews column not found"):
            prepare_model_data(df, ["feature_1"], min_albums_filter=1)


# =============================================================================
# Tests for _apply_max_albums_cap
# =============================================================================


class TestApplyMaxAlbumsCap:
    """Tests for _apply_max_albums_cap function."""

    def test_capping_keeps_most_recent(self):
        """Should keep most recent albums when capping."""
        # Artist 0 has 5 albums (seq 1-5), cap at 3 -> keep seq 3, 4, 5 (renumbered to 1, 2, 3)
        model_args = {
            "album_seq": np.array([1, 2, 3, 4, 5, 1, 2]),  # Artist 0: 5, Artist 1: 2
            "artist_idx": np.array([0, 0, 0, 0, 0, 1, 1]),
        }
        artist_album_counts = pd.Series([5, 2])

        result = _apply_max_albums_cap(
            model_args, max_albums_cap=3, artist_album_counts=artist_album_counts
        )

        # Artist 0: offset = 5-3 = 2
        # new_seq = max(1, original - 2) = [1-2=1, 2-2=1, 3-2=1, 4-2=2, 5-2=3]
        # Artist 1: offset = 2-3 = 0 (no change)
        expected_seq = np.array([1, 1, 1, 2, 3, 1, 2])
        np.testing.assert_array_equal(result["album_seq"], expected_seq)

    def test_max_seq_derived_correctly(self):
        """Should compute max_seq from capped album_seq."""
        model_args = {
            "album_seq": np.array([1, 2, 3, 4, 5, 1, 2, 3, 4]),
            "artist_idx": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1]),
        }
        artist_album_counts = pd.Series([5, 4])

        result = _apply_max_albums_cap(
            model_args, max_albums_cap=3, artist_album_counts=artist_album_counts
        )

        assert result["max_seq"] == 3, "max_seq should be max of capped sequences"

    def test_max_albums_one(self):
        """Edge case: max_albums=1 should give all seq=1."""
        model_args = {
            "album_seq": np.array([1, 2, 3, 1, 2]),
            "artist_idx": np.array([0, 0, 0, 1, 1]),
        }
        artist_album_counts = pd.Series([3, 2])

        result = _apply_max_albums_cap(
            model_args, max_albums_cap=1, artist_album_counts=artist_album_counts
        )

        expected = np.array([1, 1, 1, 1, 1])
        np.testing.assert_array_equal(result["album_seq"], expected)
        assert result["max_seq"] == 1

    def test_artists_with_exactly_max_albums(self):
        """Artists with exactly max_albums should not be modified."""
        model_args = {
            "album_seq": np.array([1, 2, 3, 1, 2, 3]),
            "artist_idx": np.array([0, 0, 0, 1, 1, 1]),
        }
        artist_album_counts = pd.Series([3, 3])

        result = _apply_max_albums_cap(
            model_args, max_albums_cap=3, artist_album_counts=artist_album_counts
        )

        # No change needed
        expected = np.array([1, 2, 3, 1, 2, 3])
        np.testing.assert_array_equal(result["album_seq"], expected)
        assert result["max_seq"] == 3

    def test_no_artists_above_cap(self):
        """Should handle case where no artists exceed cap."""
        model_args = {
            "album_seq": np.array([1, 2, 1, 2]),
            "artist_idx": np.array([0, 0, 1, 1]),
        }
        artist_album_counts = pd.Series([2, 2])

        result = _apply_max_albums_cap(
            model_args, max_albums_cap=5, artist_album_counts=artist_album_counts
        )

        # No change
        expected = np.array([1, 2, 1, 2])
        np.testing.assert_array_equal(result["album_seq"], expected)
        assert result["max_seq"] == 2

    def test_guards_against_non_positive_cap(self):
        """Should handle max_albums_cap <= 0 by clamping to 1."""
        model_args = {
            "album_seq": np.array([1, 2, 3]),
            "artist_idx": np.array([0, 0, 0]),
        }
        artist_album_counts = pd.Series([3])

        # Zero cap should be treated as 1
        result = _apply_max_albums_cap(
            model_args, max_albums_cap=0, artist_album_counts=artist_album_counts
        )
        assert result["max_seq"] == 1

        # Negative cap should be treated as 1
        result = _apply_max_albums_cap(
            model_args, max_albums_cap=-5, artist_album_counts=artist_album_counts
        )
        assert result["max_seq"] == 1


# =============================================================================
# Tests for n_ref computation
# =============================================================================


class TestNRefComputation:
    """Tests verifying n_ref (reference review count) computation from model data.

    n_ref = median(n_reviews) is computed by train_models() and added to model_args
    before calling fit_model(). These tests verify the formula and that
    prepare_model_data() returns the base keys correctly (n_ref is added later
    by train_models, not prepare_model_data).
    """

    def test_n_ref_equals_median_of_n_reviews(self):
        """n_ref should be the median of n_reviews values from model_args."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B", "B", "C"],
                "User_Score": [70.0, 75.0, 80.0, 85.0, 90.0],
                "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "n_reviews": [10, 20, 30, 40, 50],
            }
        )

        model_args, valid_mask = prepare_model_data(df, ["feature_1"], min_albums_filter=1)

        # n_ref formula: median of n_reviews
        expected_n_ref = float(np.median(model_args["n_reviews"]))
        assert expected_n_ref == 30.0, f"Expected median 30.0, got {expected_n_ref}"

    def test_model_args_keys_include_expected_heteroscedastic_keys(self):
        """prepare_model_data should return base keys but NOT n_ref (added by train_models)."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B", "B"],
                "User_Score": [70.0, 75.0, 80.0, 85.0],
                "feature_1": [1.0, 2.0, 3.0, 4.0],
                "n_reviews": [10, 20, 30, 40],
            }
        )

        model_args, valid_mask = prepare_model_data(df, ["feature_1"], min_albums_filter=1)

        # Base keys from prepare_model_data
        expected_base_keys = {
            "artist_idx",
            "album_seq",
            "prev_score",
            "X",
            "y",
            "n_reviews",
            "n_artists",
            "artist_album_counts",
        }
        assert expected_base_keys.issubset(
            set(model_args.keys())
        ), f"Missing keys: {expected_base_keys - set(model_args.keys())}"

        # n_ref is NOT added by prepare_model_data -- it's added by train_models()
        assert "n_ref" not in model_args, (
            "n_ref should NOT be in model_args from prepare_model_data; "
            "it is added by train_models() after max_albums capping"
        )
