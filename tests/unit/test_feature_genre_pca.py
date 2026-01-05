"""Tests for GenreBlock with multi-hot encoding and PCA.

These tests verify genre vocabulary learning, multi-hot encoding,
PCA dimensionality reduction, and proper fit/transform enforcement.
"""

import numpy as np
import pandas as pd
import pytest

from aoty_pred.features.base import FeatureContext, FeatureOutput
from aoty_pred.features.errors import NotFittedError
from aoty_pred.features.genre import GenreBlock, GenrePCABlock


@pytest.fixture
def ctx():
    """Create a test FeatureContext."""
    return FeatureContext(config={}, random_state=42)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with genres for testing."""
    return pd.DataFrame({
        "Genres": [
            "Rock, Alternative",
            "Rock, Indie Rock",
            "Electronic, Ambient",
            "Rock, Punk",
            "Alternative, Indie Rock",
        ],
        "Artist": ["A", "B", "C", "D", "E"],
    })


@pytest.fixture
def large_df():
    """Create a larger DataFrame for PCA testing."""
    # Create data with many genres to test PCA reduction
    genres_list = [
        "Rock, Alternative",
        "Rock, Indie Rock",
        "Electronic, Ambient",
        "Rock, Punk",
        "Alternative, Indie Rock",
        "Hip Hop, Rap",
        "Pop, Synth Pop",
        "Metal, Heavy Metal",
        "Jazz, Fusion",
        "Classical, Orchestral",
    ] * 10  # 100 samples
    return pd.DataFrame({
        "Genres": genres_list,
        "Artist": [f"Artist_{i}" for i in range(100)],
    })


class TestFitTransformEnforcement:
    """Tests for fit/transform pattern enforcement."""

    def test_transform_before_fit_raises_error(self, sample_df, ctx):
        """Transform without fit must raise NotFittedError."""
        block = GenreBlock()

        with pytest.raises(NotFittedError) as exc_info:
            block.transform(sample_df, ctx)

        assert "genre" in str(exc_info.value)
        assert "has not been fitted yet" in str(exc_info.value)

    def test_is_fitted_false_initially(self):
        """is_fitted should be False before fit() is called."""
        block = GenreBlock()
        assert block.is_fitted is False

    def test_is_fitted_true_after_fit(self, sample_df, ctx):
        """is_fitted should be True after fit() is called."""
        block = GenreBlock({"min_genre_count": 1})
        block.fit(sample_df, ctx)
        assert block.is_fitted is True


class TestVocabularyLearning:
    """Tests for genre vocabulary learning."""

    def test_fit_learns_vocabulary(self, sample_df, ctx):
        """fit() should learn genre vocabulary from training data."""
        block = GenreBlock({"min_genre_count": 1, "n_components": None})
        block.fit(sample_df, ctx)

        # Should have learned unique genres
        assert len(block._genre_vocab_) > 0
        assert "Rock" in block._genre_vocab_
        assert "Alternative" in block._genre_vocab_

    def test_rare_genres_excluded(self, ctx):
        """Genres below min_genre_count should be excluded."""
        df = pd.DataFrame({
            "Genres": [
                "Rock, Rare1",
                "Rock, Alternative",
                "Rock, Alternative",
                "Rock, Rare2",
                "Alternative, Rare3",
            ]
        })
        block = GenreBlock({"min_genre_count": 3, "n_components": None})
        block.fit(df, ctx)

        # Rock appears 4 times, Alternative 3 times - both included
        # Rare1, Rare2, Rare3 each appear once - excluded
        assert "Rock" in block._genre_vocab_
        assert "Alternative" in block._genre_vocab_
        assert "Rare1" not in block._genre_vocab_
        assert "Rare2" not in block._genre_vocab_
        assert "Rare3" not in block._genre_vocab_

    def test_vocabulary_is_sorted(self, sample_df, ctx):
        """Vocabulary should be sorted for deterministic ordering."""
        block = GenreBlock({"min_genre_count": 1, "n_components": None})
        block.fit(sample_df, ctx)

        vocab_list = list(block._genre_vocab_)
        assert vocab_list == sorted(vocab_list)


class TestMultiHotEncoding:
    """Tests for multi-hot encoding."""

    def test_multi_hot_encoding_correct(self, ctx):
        """Multi-hot encoding should set 1 for present genres."""
        df = pd.DataFrame({
            "Genres": [
                "A",
                "B",
                "A, B",
            ]
        })
        block = GenreBlock({"min_genre_count": 1, "n_components": None})
        block.fit(df, ctx)
        output = block.transform(df, ctx)

        # Vocabulary should be sorted: ["A", "B"]
        assert block._genre_vocab_ == ("A", "B")

        # Row 0: "A" -> [1, 0]
        assert output.data.iloc[0]["genre_A"] == 1.0
        assert output.data.iloc[0]["genre_B"] == 0.0

        # Row 1: "B" -> [0, 1]
        assert output.data.iloc[1]["genre_A"] == 0.0
        assert output.data.iloc[1]["genre_B"] == 1.0

        # Row 2: "A, B" -> [1, 1]
        assert output.data.iloc[2]["genre_A"] == 1.0
        assert output.data.iloc[2]["genre_B"] == 1.0

    def test_multi_genre_album_multiple_ones(self, ctx):
        """Albums with multiple genres should have multiple 1s."""
        df = pd.DataFrame({
            "Genres": [
                "Rock, Alternative, Indie",
            ]
        })
        block = GenreBlock({"min_genre_count": 1, "n_components": None})
        block.fit(df, ctx)
        output = block.transform(df, ctx)

        # Should have 3 genres, all set to 1
        row = output.data.iloc[0]
        assert row.sum() == 3.0
        assert row["genre_Rock"] == 1.0
        assert row["genre_Alternative"] == 1.0
        assert row["genre_Indie"] == 1.0

    def test_unknown_genre_ignored(self, ctx):
        """Genres not in vocabulary should produce zeros."""
        train_df = pd.DataFrame({
            "Genres": ["A", "A", "B", "B"]
        })
        test_df = pd.DataFrame({
            "Genres": ["A, Unknown", "C"]
        })

        block = GenreBlock({"min_genre_count": 1, "n_components": None})
        block.fit(train_df, ctx)
        output = block.transform(test_df, ctx)

        # Row 0: "A, Unknown" -> A=1, Unknown ignored
        assert output.data.iloc[0]["genre_A"] == 1.0
        assert "genre_Unknown" not in output.data.columns

        # Row 1: "C" -> all zeros (C not in vocab)
        assert output.data.iloc[1].sum() == 0.0

    def test_missing_genres_column_all_zeros(self, ctx):
        """Missing/null Genres should produce all zeros."""
        train_df = pd.DataFrame({
            "Genres": ["A", "B", "A", "B"]
        })
        test_df = pd.DataFrame({
            "Genres": [None, "", "A"]
        })

        block = GenreBlock({"min_genre_count": 1, "n_components": None})
        block.fit(train_df, ctx)
        output = block.transform(test_df, ctx)

        # Row 0: None -> all zeros
        assert output.data.iloc[0].sum() == 0.0

        # Row 1: "" -> all zeros
        assert output.data.iloc[1].sum() == 0.0

        # Row 2: "A" -> [1, 0]
        assert output.data.iloc[2]["genre_A"] == 1.0


class TestPCAReduction:
    """Tests for PCA dimensionality reduction."""

    def test_pca_reduces_dimensions(self, large_df, ctx):
        """PCA should reduce dimensions to n_components."""
        # Low min_genre_count to get many genres for PCA
        block = GenreBlock({"min_genre_count": 5, "n_components": 3})
        block.fit(large_df, ctx)
        output = block.transform(large_df, ctx)

        # Output should have n_components columns
        assert output.data.shape[1] == 3
        assert block._use_pca_ is True

    def test_pca_output_column_names(self, large_df, ctx):
        """PCA output should have genre_pca_N column names."""
        block = GenreBlock({"min_genre_count": 5, "n_components": 3})
        block.fit(large_df, ctx)
        output = block.transform(large_df, ctx)

        expected_names = ["genre_pca_0", "genre_pca_1", "genre_pca_2"]
        assert output.feature_names == expected_names
        assert list(output.data.columns) == expected_names

    def test_no_pca_when_disabled(self, sample_df, ctx):
        """n_components=None should skip PCA."""
        block = GenreBlock({"min_genre_count": 1, "n_components": None})
        block.fit(sample_df, ctx)
        output = block.transform(sample_df, ctx)

        # Should have genre_NAME columns, not genre_pca_N
        assert block._use_pca_ is False
        assert all(col.startswith("genre_") and not col.startswith("genre_pca_")
                   for col in output.data.columns)

    def test_explained_variance_in_metadata(self, large_df, ctx):
        """Metadata should contain explained variance when PCA used."""
        block = GenreBlock({"min_genre_count": 5, "n_components": 3})
        block.fit(large_df, ctx)
        output = block.transform(large_df, ctx)

        assert "explained_variance_ratio" in output.metadata
        assert "total_explained_variance" in output.metadata
        assert len(output.metadata["explained_variance_ratio"]) == 3
        assert 0 < output.metadata["total_explained_variance"] <= 1.0

    def test_pca_skipped_if_fewer_genres_than_components(self, ctx):
        """PCA should be skipped if vocab size < n_components."""
        df = pd.DataFrame({
            "Genres": ["A", "B", "A", "B"]  # Only 2 genres
        })
        block = GenreBlock({"min_genre_count": 1, "n_components": 10})
        block.fit(df, ctx)

        # n_components=10 > vocab_size=2, so PCA should be skipped
        assert block._use_pca_ is False


class TestOutputFormat:
    """Tests for output format and index preservation."""

    def test_output_preserves_original_index(self, ctx):
        """Output DataFrame should preserve the original index."""
        df = pd.DataFrame(
            {"Genres": ["Rock", "Pop", "Jazz"]},
            index=[100, 200, 300]
        )
        block = GenreBlock({"min_genre_count": 1, "n_components": None})
        block.fit(df, ctx)
        output = block.transform(df, ctx)

        assert list(output.data.index) == [100, 200, 300]

    def test_output_has_correct_structure(self, sample_df, ctx):
        """Output should be a FeatureOutput with proper structure."""
        block = GenreBlock({"min_genre_count": 1, "n_components": None})
        output = block.fit_transform(sample_df, ctx)

        assert isinstance(output, FeatureOutput)
        assert isinstance(output.data, pd.DataFrame)
        assert isinstance(output.feature_names, list)
        assert isinstance(output.metadata, dict)

    def test_metadata_contains_vocab_size(self, sample_df, ctx):
        """Metadata should contain n_genres_in_vocab."""
        block = GenreBlock({"min_genre_count": 1, "n_components": None})
        output = block.fit_transform(sample_df, ctx)

        assert "n_genres_in_vocab" in output.metadata
        assert output.metadata["n_genres_in_vocab"] == len(block._genre_vocab_)


class TestBackwardsCompatibility:
    """Tests for backwards compatibility."""

    def test_genre_pca_block_alias_exists(self):
        """GenrePCABlock alias should exist for backwards compatibility."""
        assert GenrePCABlock is GenreBlock

    def test_genre_pca_block_works_same_as_genre_block(self, sample_df, ctx):
        """GenrePCABlock should work identically to GenreBlock."""
        block = GenrePCABlock({"min_genre_count": 1, "n_components": None})
        output = block.fit_transform(sample_df, ctx)

        assert isinstance(output, FeatureOutput)
        assert output.data is not None
