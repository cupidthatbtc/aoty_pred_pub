"""Tests for CollaborationBlock feature encoding.

These tests verify collab_type vocabulary learning, passthrough of
is_collaboration and num_artists, and proper fit/transform enforcement.
"""

import pandas as pd
import pytest

from aoty_pred.features.base import FeatureContext, FeatureOutput
from aoty_pred.features.collaboration import CollaborationBlock
from aoty_pred.features.errors import NotFittedError


@pytest.fixture
def ctx():
    """Create a test FeatureContext."""
    return FeatureContext(config={}, random_state=42)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with collaboration columns."""
    return pd.DataFrame(
        {
            "is_collaboration": [False, True, False, True, False],
            "collab_type": ["solo", "duo", "solo", "small_group", "solo"],
            "num_artists": [1, 2, 1, 3, 1],
            "Artist": ["A", "B & C", "D", "E, F & G", "H"],
        }
    )


class TestFitTransformEnforcement:
    """Tests for fit/transform pattern enforcement."""

    def test_transform_before_fit_raises_error(self, sample_df, ctx):
        """Transform without fit must raise NotFittedError."""
        block = CollaborationBlock()

        with pytest.raises(NotFittedError) as exc_info:
            block.transform(sample_df, ctx)

        assert "collaboration" in str(exc_info.value)
        assert "has not been fitted yet" in str(exc_info.value)

    def test_is_fitted_false_initially(self):
        """is_fitted should be False before fit() is called."""
        block = CollaborationBlock()
        assert block.is_fitted is False

    def test_is_fitted_true_after_fit(self, sample_df, ctx):
        """is_fitted should be True after fit() is called."""
        block = CollaborationBlock()
        block.fit(sample_df, ctx)
        assert block.is_fitted is True


class TestVocabularyLearning:
    """Tests for collab_type vocabulary learning."""

    def test_fit_learns_collab_types(self, sample_df, ctx):
        """fit() should learn collab_type vocabulary from training data."""
        block = CollaborationBlock()
        block.fit(sample_df, ctx)

        vocab = block._collab_type_vocab_
        assert vocab is not None
        assert "solo" in vocab.categories
        assert "duo" in vocab.categories
        assert "small_group" in vocab.categories

    def test_collab_type_uses_default_ordering(self, sample_df, ctx):
        """Default collab_type ordering should be solo < duo < small_group < ensemble."""
        block = CollaborationBlock()
        block.fit(sample_df, ctx)

        vocab = block._collab_type_vocab_
        assert vocab is not None
        # Check ordering
        assert vocab.category_to_idx["solo"] == 0
        assert vocab.category_to_idx["duo"] == 1
        assert vocab.category_to_idx["small_group"] == 2


class TestFeaturePassthrough:
    """Tests for feature passthrough and encoding."""

    def test_is_collaboration_passthrough(self, sample_df, ctx):
        """is_collaboration should be passed through as 0/1."""
        block = CollaborationBlock()
        output = block.fit_transform(sample_df, ctx)

        expected = [0, 1, 0, 1, 0]
        assert list(output.data["is_collaboration"]) == expected

    def test_num_artists_passthrough(self, sample_df, ctx):
        """num_artists should be passed through directly."""
        block = CollaborationBlock()
        output = block.fit_transform(sample_df, ctx)

        expected = [1, 2, 1, 3, 1]
        assert list(output.data["num_artists"]) == expected

    def test_collab_type_encoding(self, sample_df, ctx):
        """collab_type should be encoded as ordinal values."""
        block = CollaborationBlock()
        output = block.fit_transform(sample_df, ctx)

        # solo=0, duo=1, small_group=2
        expected = [0, 1, 0, 2, 0]
        assert list(output.data["collab_type_ordinal"]) == expected

    def test_unknown_collab_type_uses_unknown_idx(self, ctx):
        """Unknown collab_type should use unknown_idx (-1)."""
        train_df = pd.DataFrame(
            {
                "is_collaboration": [False, True],
                "collab_type": ["solo", "duo"],
                "num_artists": [1, 2],
            }
        )
        test_df = pd.DataFrame(
            {
                "is_collaboration": [True],
                "collab_type": ["mega_ensemble"],  # Not in training
                "num_artists": [10],
            }
        )

        block = CollaborationBlock()
        block.fit(train_df, ctx)
        output = block.transform(test_df, ctx)

        # Unknown type should get -1
        assert output.data["collab_type_ordinal"].iloc[0] == -1


class TestOutputFormat:
    """Tests for output format and structure."""

    def test_output_has_correct_columns(self, sample_df, ctx):
        """Output should have all expected columns."""
        block = CollaborationBlock()
        output = block.fit_transform(sample_df, ctx)

        expected_columns = ["is_collaboration", "num_artists", "collab_type_ordinal"]
        assert list(output.data.columns) == expected_columns
        assert output.feature_names == expected_columns

    def test_output_preserves_original_index(self, ctx):
        """Output DataFrame should preserve the original index."""
        df = pd.DataFrame(
            {
                "is_collaboration": [False, True, False],
                "collab_type": ["solo", "duo", "solo"],
                "num_artists": [1, 2, 1],
            },
            index=[100, 200, 300],
        )

        block = CollaborationBlock()
        output = block.fit_transform(df, ctx)

        assert list(output.data.index) == [100, 200, 300]

    def test_output_has_correct_structure(self, sample_df, ctx):
        """Output should be a FeatureOutput with proper structure."""
        block = CollaborationBlock()
        output = block.fit_transform(sample_df, ctx)

        assert isinstance(output, FeatureOutput)
        assert isinstance(output.data, pd.DataFrame)
        assert isinstance(output.feature_names, list)
        assert isinstance(output.metadata, dict)

    def test_metadata_contains_categories(self, sample_df, ctx):
        """Metadata should contain collab_type categories."""
        block = CollaborationBlock()
        output = block.fit_transform(sample_df, ctx)

        assert "collab_type_categories" in output.metadata
        assert "solo" in output.metadata["collab_type_categories"]


class TestMissingColumns:
    """Tests for handling missing required columns."""

    def test_missing_column_raises_error(self, ctx):
        """Missing required column should raise ValueError."""
        df = pd.DataFrame(
            {
                "is_collaboration": [False],
                # Missing collab_type and num_artists
            }
        )

        block = CollaborationBlock()

        with pytest.raises(ValueError) as exc_info:
            block.fit(df, ctx)

        assert "missing required columns" in str(exc_info.value).lower()
