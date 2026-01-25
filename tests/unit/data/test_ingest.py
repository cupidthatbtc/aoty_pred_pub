"""Unit tests for data dimension extraction."""

import dataclasses

import pytest

from aoty_pred.data.ingest import DataDimensions, extract_data_dimensions


class TestDataDimensions:
    """Tests for DataDimensions dataclass."""

    def test_from_defaults_returns_conservative_estimates(self):
        """from_defaults() returns n_observations=1000, n_artists=100."""
        dims = DataDimensions.from_defaults()

        assert dims.n_observations == 1000
        assert dims.n_artists == 100

    def test_from_defaults_source_indicates_fallback(self):
        """from_defaults() source contains 'defaults'."""
        dims = DataDimensions.from_defaults()

        assert "defaults" in dims.source

    def test_dataclass_frozen(self):
        """DataDimensions is frozen and cannot be modified."""
        dims = DataDimensions.from_defaults()

        with pytest.raises(dataclasses.FrozenInstanceError):
            dims.n_observations = 2000  # type: ignore[misc]


class TestExtractDataDimensions:
    """Tests for extract_data_dimensions function."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create sample CSV with known values for testing."""
        csv_path = tmp_path / "test_albums.csv"
        csv_path.write_text(
            "Artist,User Ratings,Album\n"
            "Artist A,15,Album 1\n"
            "Artist A,20,Album 2\n"
            "Artist B,5,Album 3\n"  # Below min_ratings=10, should be filtered
            "Artist B,12,Album 4\n"
            "Artist C,25,Album 5\n"
        )
        return csv_path

    def test_extract_from_nonexistent_file_returns_defaults(self, tmp_path):
        """Path doesn't exist -> returns from_defaults()."""
        nonexistent = tmp_path / "does_not_exist.csv"

        dims = extract_data_dimensions(nonexistent)

        assert dims.n_observations == 1000
        assert dims.n_artists == 100
        assert "defaults" in dims.source

    def test_extract_applies_min_ratings_filter(self, sample_csv):
        """Rows below min_ratings threshold are excluded from count."""
        dims = extract_data_dimensions(sample_csv, min_ratings=10)

        # Row 3 (Artist B, 5 ratings) is filtered out
        # Remaining: 4 observations
        assert dims.n_observations == 4

    def test_extract_counts_unique_artists(self, sample_csv):
        """n_artists is nunique() of Artist column after filtering."""
        dims = extract_data_dimensions(sample_csv, min_ratings=10)

        # Artists A, B, C all have at least one row >= min_ratings
        assert dims.n_artists == 3

    def test_extract_source_contains_filename(self, sample_csv):
        """Source includes the CSV filename."""
        dims = extract_data_dimensions(sample_csv, min_ratings=10)

        assert "test_albums.csv" in dims.source

    def test_extract_handles_bom_encoding(self, tmp_path):
        """CSV with BOM is handled correctly."""
        csv_path = tmp_path / "bom_test.csv"
        # Write with UTF-8 BOM
        csv_path.write_bytes(
            b"\xef\xbb\xbf"  # UTF-8 BOM
            b"Artist,User Ratings,Album\n"
            b"Artist A,15,Album 1\n"
        )

        dims = extract_data_dimensions(csv_path, min_ratings=10)

        assert dims.n_observations == 1
        assert dims.n_artists == 1

    def test_extract_returns_defaults_on_malformed_csv(self, tmp_path):
        """Malformed CSV returns defaults gracefully."""
        csv_path = tmp_path / "malformed.csv"
        csv_path.write_text("This is not,a valid,CSV\nwith wrong,columns")

        dims = extract_data_dimensions(csv_path, min_ratings=10)

        # Should fall back to defaults on error
        assert dims.n_observations == 1000
        assert "defaults" in dims.source
