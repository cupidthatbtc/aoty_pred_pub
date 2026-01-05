"""Tests for TemporalBlock feature computation.

Tests verify correct temporal feature calculation including:
- Album sequence numbering within artists
- Career length in years
- Release gap handling (debuts = 0)
- Date risk ordinal encoding
- Index preservation and deterministic ordering
"""

import pandas as pd
import pytest

from aoty_pred.features.base import FeatureContext, FeatureOutput
from aoty_pred.features.errors import NotFittedError
from aoty_pred.features.temporal import TemporalBlock


@pytest.fixture
def ctx():
    """Create a test FeatureContext."""
    return FeatureContext(config={}, random_state=42)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with required columns."""
    return pd.DataFrame(
        {
            "Artist": ["Artist A", "Artist A", "Artist A"],
            "Album": ["Album 1", "Album 2", "Album 3"],
            "Release_Date_Parsed": pd.to_datetime(
                ["2020-01-01", "2021-06-15", "2022-12-31"]
            ),
            "Year": [2020, 2021, 2022],
            "date_risk": ["low", "medium", "high"],
        }
    )


class TestFitTransformEnforcement:
    """Tests for fit/transform state enforcement."""

    def test_transform_before_fit_raises_error(self, sample_df, ctx):
        """Transform without fit must raise NotFittedError."""
        block = TemporalBlock()

        with pytest.raises(NotFittedError) as exc_info:
            block.transform(sample_df, ctx)

        assert "temporal" in str(exc_info.value)
        assert "has not been fitted yet" in str(exc_info.value)

    def test_is_fitted_false_initially(self):
        """is_fitted should be False before fit() is called."""
        block = TemporalBlock()
        assert block.is_fitted is False

    def test_fit_sets_fitted_true(self, sample_df, ctx):
        """is_fitted should be True after fit() is called."""
        block = TemporalBlock()
        block.fit(sample_df, ctx)
        assert block.is_fitted is True

    def test_fit_returns_self(self, sample_df, ctx):
        """fit() should return self for method chaining."""
        block = TemporalBlock()
        result = block.fit(sample_df, ctx)
        assert result is block


class TestAlbumSequence:
    """Tests for album_sequence feature computation."""

    def test_album_sequence_single_artist(self, ctx):
        """Album sequence should be 1, 2, 3... for single artist."""
        df = pd.DataFrame(
            {
                "Artist": ["Artist A", "Artist A", "Artist A"],
                "Album": ["A", "B", "C"],
                "Release_Date_Parsed": pd.to_datetime(
                    ["2020-01-01", "2021-01-01", "2022-01-01"]
                ),
                "Year": [2020, 2021, 2022],
                "date_risk": ["low", "low", "low"],
            }
        )

        block = TemporalBlock()
        output = block.fit_transform(df, ctx)

        # Should be [1, 2, 3] when aligned to original index
        assert output.data["album_sequence"].tolist() == [1, 2, 3]

    def test_album_sequence_multiple_artists(self, ctx):
        """Album sequence should restart for each artist."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "B", "B"],
                "Album": ["A1", "A2", "B1", "B2"],
                "Release_Date_Parsed": pd.to_datetime(
                    ["2020-01-01", "2021-01-01", "2020-06-01", "2021-06-01"]
                ),
                "Year": [2020, 2021, 2020, 2021],
                "date_risk": ["low", "low", "low", "low"],
            }
        )

        block = TemporalBlock()
        output = block.fit_transform(df, ctx)

        # Sequence restarts per artist: A1=1, A2=2, B1=1, B2=2
        assert output.data["album_sequence"].tolist() == [1, 2, 1, 2]


class TestCareerYears:
    """Tests for career_years feature computation."""

    def test_career_years_first_album_is_zero(self, ctx):
        """First album should have career_years = 0."""
        df = pd.DataFrame(
            {
                "Artist": ["Artist A"],
                "Album": ["Debut"],
                "Release_Date_Parsed": pd.to_datetime(["2020-01-01"]),
                "Year": [2020],
                "date_risk": ["low"],
            }
        )

        block = TemporalBlock()
        output = block.fit_transform(df, ctx)

        assert output.data["career_years"].iloc[0] == 0.0

    def test_career_years_calculation(self, ctx):
        """Career years should be correctly computed from first album."""
        df = pd.DataFrame(
            {
                "Artist": ["Artist A", "Artist A"],
                "Album": ["First", "Second"],
                "Release_Date_Parsed": pd.to_datetime(
                    ["2020-01-01", "2022-01-01"]
                ),
                "Year": [2020, 2022],
                "date_risk": ["low", "low"],
            }
        )

        block = TemporalBlock()
        output = block.fit_transform(df, ctx)

        # First album: 0 years
        assert output.data["career_years"].iloc[0] == 0.0
        # Second album: ~2 years (731 days / 365.25)
        expected = 731 / 365.25  # 2 years (including leap day)
        assert abs(output.data["career_years"].iloc[1] - expected) < 0.01


class TestReleaseGapDays:
    """Tests for release_gap_days feature computation."""

    def test_release_gap_days_debut_is_zero(self, ctx):
        """Debut album should have release_gap_days = 0."""
        df = pd.DataFrame(
            {
                "Artist": ["Artist A"],
                "Album": ["Debut"],
                "Release_Date_Parsed": pd.to_datetime(["2020-01-01"]),
                "Year": [2020],
                "date_risk": ["low"],
            }
        )

        block = TemporalBlock()
        output = block.fit_transform(df, ctx)

        assert output.data["release_gap_days"].iloc[0] == 0.0

    def test_release_gap_days_subsequent_albums(self, ctx):
        """Subsequent albums should have correct gap in days."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "A"],
                "Album": ["A1", "A2", "A3"],
                "Release_Date_Parsed": pd.to_datetime(
                    ["2020-01-01", "2020-01-11", "2020-01-31"]
                ),
                "Year": [2020, 2020, 2020],
                "date_risk": ["low", "low", "low"],
            }
        )

        block = TemporalBlock()
        output = block.fit_transform(df, ctx)

        # First: 0 (debut), Second: 10 days, Third: 20 days
        assert output.data["release_gap_days"].tolist() == [0.0, 10.0, 20.0]


class TestDateRiskOrdinal:
    """Tests for date_risk_ordinal feature computation."""

    def test_date_risk_ordinal_mapping(self, ctx):
        """Date risk should map to ordinal: low=0, medium=1, high=2."""
        df = pd.DataFrame(
            {
                "Artist": ["A", "A", "A"],
                "Album": ["1", "2", "3"],
                "Release_Date_Parsed": pd.to_datetime(
                    ["2020-01-01", "2020-02-01", "2020-03-01"]
                ),
                "Year": [2020, 2020, 2020],
                "date_risk": ["low", "medium", "high"],
            }
        )

        block = TemporalBlock()
        output = block.fit_transform(df, ctx)

        assert output.data["date_risk_ordinal"].tolist() == [0, 1, 2]

    def test_date_risk_ordinal_unknown_defaults_to_medium(self, ctx):
        """Unknown date_risk values should default to medium (1)."""
        df = pd.DataFrame(
            {
                "Artist": ["A"],
                "Album": ["1"],
                "Release_Date_Parsed": pd.to_datetime(["2020-01-01"]),
                "Year": [2020],
                "date_risk": ["unknown_value"],
            }
        )

        block = TemporalBlock()
        output = block.fit_transform(df, ctx)

        assert output.data["date_risk_ordinal"].iloc[0] == 1

    def test_date_risk_ordinal_null_defaults_to_medium(self, ctx):
        """Null date_risk values should default to medium (1)."""
        df = pd.DataFrame(
            {
                "Artist": ["A"],
                "Album": ["1"],
                "Release_Date_Parsed": pd.to_datetime(["2020-01-01"]),
                "Year": [2020],
                "date_risk": [None],
            }
        )

        block = TemporalBlock()
        output = block.fit_transform(df, ctx)

        assert output.data["date_risk_ordinal"].iloc[0] == 1


class TestIndexPreservation:
    """Tests for original index preservation."""

    def test_output_preserves_original_index(self, ctx):
        """Output should have same index as input DataFrame."""
        # Create DataFrame with non-sequential index
        df = pd.DataFrame(
            {
                "Artist": ["A", "A"],
                "Album": ["1", "2"],
                "Release_Date_Parsed": pd.to_datetime(
                    ["2020-01-01", "2021-01-01"]
                ),
                "Year": [2020, 2021],
                "date_risk": ["low", "low"],
            },
            index=[100, 200],
        )

        block = TemporalBlock()
        output = block.fit_transform(df, ctx)

        assert output.data.index.tolist() == [100, 200]


class TestDeterministicOrdering:
    """Tests for deterministic ordering of same-date albums."""

    def test_deterministic_ordering_same_date(self, ctx):
        """Albums on same date should be ordered by Album name."""
        # Two albums with same artist and same date
        df = pd.DataFrame(
            {
                "Artist": ["A", "A"],
                "Album": ["Zebra", "Alpha"],  # Alphabetically: Alpha < Zebra
                "Release_Date_Parsed": pd.to_datetime(
                    ["2020-01-01", "2020-01-01"]
                ),
                "Year": [2020, 2020],
                "date_risk": ["low", "low"],
            }
        )

        block = TemporalBlock()
        output = block.fit_transform(df, ctx)

        # After sorting by Album: Alpha=1, Zebra=2
        # Original order: Zebra (idx 0), Alpha (idx 1)
        # So sequence at idx 0 should be 2 (Zebra is 2nd alphabetically)
        # and sequence at idx 1 should be 1 (Alpha is 1st alphabetically)
        assert output.data.loc[0, "album_sequence"] == 2  # Zebra
        assert output.data.loc[1, "album_sequence"] == 1  # Alpha


class TestFeatureOutput:
    """Tests for FeatureOutput structure."""

    def test_output_is_feature_output(self, sample_df, ctx):
        """Output should be FeatureOutput instance."""
        block = TemporalBlock()
        output = block.fit_transform(sample_df, ctx)
        assert isinstance(output, FeatureOutput)

    def test_output_has_correct_feature_names(self, sample_df, ctx):
        """Output should have all 5 feature columns."""
        block = TemporalBlock()
        output = block.fit_transform(sample_df, ctx)

        expected_features = [
            "album_sequence",
            "career_years",
            "release_gap_days",
            "release_year",
            "date_risk_ordinal",
        ]
        assert output.feature_names == expected_features
        assert list(output.data.columns) == expected_features

    def test_output_has_metadata(self, sample_df, ctx):
        """Output should include block metadata."""
        block = TemporalBlock()
        output = block.fit_transform(sample_df, ctx)

        assert "block" in output.metadata
        assert output.metadata["block"] == "temporal"


class TestMissingColumns:
    """Tests for missing column validation."""

    def test_fit_raises_on_missing_columns(self, ctx):
        """fit() should raise ValueError if required columns missing."""
        df = pd.DataFrame(
            {
                "Artist": ["A"],
                "Album": ["1"],
                # Missing: Release_Date_Parsed, Year, date_risk
            }
        )

        block = TemporalBlock()
        with pytest.raises(ValueError) as exc_info:
            block.fit(df, ctx)

        assert "missing required columns" in str(exc_info.value).lower()
