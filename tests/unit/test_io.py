"""Unit tests for io module (readers, writers, paths).

Tests cover:
- read_csv with basic content, BOM handling, kwargs
- read_csv error handling for missing files
- write_csv with basic content, no index
- project_root path functions
"""

import pandas as pd
import pytest
from pathlib import Path

from aoty_pred.io.readers import read_csv
from aoty_pred.io.writers import write_csv
from aoty_pred.io.paths import project_root


# =============================================================================
# Test Class: TestReaders
# =============================================================================


class TestReaders:
    """Tests for io.readers module."""

    def test_read_csv_basic(self, tmp_path: Path):
        """read_csv should read CSV with known content correctly."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("a,b,c\n1,2,3\n4,5,6\n")

        df = read_csv(csv_path)

        assert list(df.columns) == ["a", "b", "c"]
        assert len(df) == 2
        assert df["a"].tolist() == [1, 4]
        assert df["b"].tolist() == [2, 5]
        assert df["c"].tolist() == [3, 6]

    def test_read_csv_with_bom(self, tmp_path: Path):
        """read_csv should handle UTF-8 BOM correctly."""
        csv_path = tmp_path / "bom.csv"
        # Write CSV with UTF-8 BOM
        bom_content = "\ufeffa,b,c\n1,2,3\n"
        csv_path.write_text(bom_content, encoding="utf-8")

        # Default encoding utf-8-sig should handle BOM
        df = read_csv(csv_path)

        # First column should NOT have BOM prefix
        assert df.columns[0] == "a"
        assert "\ufeff" not in df.columns[0]
        assert list(df.columns) == ["a", "b", "c"]

    def test_read_csv_with_kwargs(self, tmp_path: Path):
        """read_csv should pass kwargs to pd.read_csv."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("a,b,c,d\n1,2,3,4\n5,6,7,8\n")

        df = read_csv(csv_path, usecols=["a", "c"])

        assert list(df.columns) == ["a", "c"]
        assert len(df) == 2
        assert df["a"].tolist() == [1, 5]
        assert df["c"].tolist() == [3, 7]

    def test_read_csv_file_not_found(self, tmp_path: Path):
        """read_csv should raise FileNotFoundError for non-existent file."""
        nonexistent = tmp_path / "does_not_exist.csv"

        with pytest.raises(FileNotFoundError):
            read_csv(nonexistent)

    def test_read_csv_with_different_encoding(self, tmp_path: Path):
        """read_csv should respect encoding parameter."""
        csv_path = tmp_path / "utf8.csv"
        content = "name,value\ntest,42\n"
        csv_path.write_text(content, encoding="utf-8")

        df = read_csv(csv_path, encoding="utf-8")

        assert list(df.columns) == ["name", "value"]
        assert df["name"].iloc[0] == "test"

    def test_read_csv_with_nrows(self, tmp_path: Path):
        """read_csv should respect nrows kwarg."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("a,b\n1,2\n3,4\n5,6\n7,8\n")

        df = read_csv(csv_path, nrows=2)

        assert len(df) == 2
        assert df["a"].tolist() == [1, 3]


# =============================================================================
# Test Class: TestWriters
# =============================================================================


class TestWriters:
    """Tests for io.writers module."""

    def test_write_csv_basic(self, tmp_path: Path):
        """write_csv should write DataFrame to CSV correctly."""
        csv_path = tmp_path / "output.csv"
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        write_csv(df, str(csv_path))

        # Verify file exists and content is correct
        assert csv_path.exists()
        result = pd.read_csv(csv_path)
        assert list(result.columns) == ["x", "y"]
        assert result["x"].tolist() == [1, 2, 3]
        assert result["y"].tolist() == [4, 5, 6]

    def test_write_csv_no_index(self, tmp_path: Path):
        """write_csv should not write index as column."""
        csv_path = tmp_path / "output.csv"
        df = pd.DataFrame(
            {"x": [1, 2], "y": [3, 4]},
            index=["row1", "row2"]
        )

        write_csv(df, str(csv_path))

        # Read back and verify no index column
        result = pd.read_csv(csv_path)
        assert list(result.columns) == ["x", "y"]
        assert len(result.columns) == 2
        assert "Unnamed: 0" not in result.columns

    def test_write_csv_roundtrip(self, tmp_path: Path):
        """write_csv then read_csv should roundtrip correctly."""
        csv_path = tmp_path / "roundtrip.csv"
        original = pd.DataFrame({
            "name": ["alice", "bob", "charlie"],
            "score": [85.5, 92.0, 78.5],
        })

        write_csv(original, str(csv_path))
        recovered = read_csv(csv_path)

        pd.testing.assert_frame_equal(original, recovered)

    def test_write_csv_overwrites_existing(self, tmp_path: Path):
        """write_csv should overwrite existing file."""
        csv_path = tmp_path / "output.csv"

        # Write first version
        df1 = pd.DataFrame({"a": [1, 2]})
        write_csv(df1, str(csv_path))

        # Write second version
        df2 = pd.DataFrame({"b": [3, 4, 5]})
        write_csv(df2, str(csv_path))

        # Verify second version
        result = pd.read_csv(csv_path)
        assert list(result.columns) == ["b"]
        assert len(result) == 3


# =============================================================================
# Test Class: TestPaths
# =============================================================================


class TestPaths:
    """Tests for io.paths module."""

    def test_project_root_is_directory(self):
        """project_root should return a directory path."""
        root = project_root()

        assert isinstance(root, Path)
        assert root.is_dir()

    def test_project_root_contains_expected_files(self):
        """project_root should point to project with expected structure."""
        root = project_root()

        # Should contain pyproject.toml or src/ directory
        has_pyproject = (root / "pyproject.toml").exists()
        has_src = (root / "src").is_dir()

        assert has_pyproject or has_src, (
            f"project_root ({root}) should contain pyproject.toml or src/"
        )

    def test_project_root_consistent(self):
        """project_root should return same path on repeated calls."""
        root1 = project_root()
        root2 = project_root()

        assert root1 == root2

    def test_project_root_is_absolute(self):
        """project_root should return absolute path."""
        root = project_root()

        assert root.is_absolute()

    def test_project_root_contains_aoty_pred_source(self):
        """project_root should contain the aoty_pred source package."""
        root = project_root()

        aoty_pred_path = root / "src" / "aoty_pred"
        assert aoty_pred_path.is_dir(), (
            f"Expected {aoty_pred_path} to exist"
        )
