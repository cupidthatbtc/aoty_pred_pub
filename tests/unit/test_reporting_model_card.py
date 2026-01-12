"""Unit tests for model card generation."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aoty_pred.reporting.model_card import (
    ModelCardData,
    create_default_model_card_data,
    generate_model_card,
    update_model_card_with_results,
    write_model_card,
)


@pytest.fixture
def sample_model_card_data():
    """Create sample ModelCardData for testing."""
    return ModelCardData(
        model_name="Test Model",
        model_version="0.1.0",
        model_type="Test Type",
        authors=["Test Author"],
        created_date="2026-01-19",
        last_updated="2026-01-19",
        dataset_name="Test Dataset",
        dataset_size=1000,
        dataset_description="Test description",
        data_preprocessing="Test preprocessing",
        architecture_summary="Test architecture",
        priors_description="Test priors",
        hyperparameters={"param1": 1.0, "param2": "value"},
        convergence_summary="R-hat < 1.01",
        calibration_summary="95% coverage: 94%",
        predictive_summary="MAE: 5.0",
        loo_elpd=-1234.5,
        limitations=["Limitation 1", "Limitation 2"],
        ethical_considerations=["Consideration 1", "Consideration 2"],
        intended_use="Test use case",
        out_of_scope_use="Not for production",
        load_example="model = load()",
        predict_example="pred = predict()",
        interpret_example="print(pred)",
    )


class TestModelCardData:
    """Tests for ModelCardData dataclass."""

    def test_dataclass_fields_present(self, sample_model_card_data):
        """All required fields should be present."""
        data = sample_model_card_data

        # Model identity
        assert hasattr(data, "model_name")
        assert hasattr(data, "model_version")
        assert hasattr(data, "model_type")

        # Authors and dates
        assert hasattr(data, "authors")
        assert hasattr(data, "created_date")
        assert hasattr(data, "last_updated")

        # Training data
        assert hasattr(data, "dataset_name")
        assert hasattr(data, "dataset_size")
        assert hasattr(data, "dataset_description")
        assert hasattr(data, "data_preprocessing")

        # Model details
        assert hasattr(data, "architecture_summary")
        assert hasattr(data, "priors_description")
        assert hasattr(data, "hyperparameters")

        # Evaluation metrics
        assert hasattr(data, "convergence_summary")
        assert hasattr(data, "calibration_summary")
        assert hasattr(data, "predictive_summary")
        assert hasattr(data, "loo_elpd")

        # Limitations and ethics
        assert hasattr(data, "limitations")
        assert hasattr(data, "ethical_considerations")
        assert hasattr(data, "intended_use")
        assert hasattr(data, "out_of_scope_use")

        # Code examples
        assert hasattr(data, "load_example")
        assert hasattr(data, "predict_example")
        assert hasattr(data, "interpret_example")

    def test_default_values(self):
        """Optional fields should have sensible defaults."""
        data = ModelCardData(
            model_name="Test",
            model_version="0.1.0",
            model_type="Test",
            authors=["Author"],
            created_date="2026-01-19",
            last_updated="2026-01-19",
            dataset_name="Test",
            dataset_size=100,
            dataset_description="Test",
            data_preprocessing="Test",
            architecture_summary="Test",
            priors_description="Test",
        )

        # Check defaults
        assert data.hyperparameters == {}
        assert data.convergence_summary == "Not yet evaluated"
        assert data.calibration_summary == "Not yet evaluated"
        assert data.predictive_summary == "Not yet evaluated"
        assert data.loo_elpd is None
        assert data.limitations == []
        assert data.ethical_considerations == []
        assert data.intended_use == ""
        assert data.out_of_scope_use == ""
        assert data.load_example == ""
        assert data.predict_example == ""
        assert data.interpret_example == ""

    def test_create_default_returns_valid_data(self):
        """create_default_model_card_data should return valid ModelCardData."""
        data = create_default_model_card_data()

        assert isinstance(data, ModelCardData)
        assert data.model_name == "AOTY Artist Score Prediction Model"
        assert data.model_type == "Bayesian Hierarchical Regression with Time-Varying Effects"
        assert len(data.authors) > 0
        assert len(data.limitations) > 0
        assert len(data.ethical_considerations) > 0

    def test_create_default_has_code_examples(self):
        """Default model card should have code examples."""
        data = create_default_model_card_data()

        assert len(data.load_example) > 0
        assert len(data.predict_example) > 0
        assert len(data.interpret_example) > 0
        assert "load_model" in data.load_example
        assert "predict" in data.predict_example.lower()


class TestGenerateModelCard:
    """Tests for generate_model_card function."""

    def test_markdown_format(self, sample_model_card_data):
        """Should generate valid markdown structure."""
        card = generate_model_card(sample_model_card_data, format="markdown")

        assert isinstance(card, str)
        assert "# Model Card:" in card
        assert "## Model Details" in card
        assert "## Intended Use" in card

    def test_latex_format(self, sample_model_card_data):
        """Should generate valid LaTeX structure."""
        card = generate_model_card(sample_model_card_data, format="latex")

        assert isinstance(card, str)
        assert "\\documentclass{article}" in card
        assert "\\begin{document}" in card
        assert "\\end{document}" in card
        assert "\\section*{Model Card:" in card

    def test_invalid_format_raises(self, sample_model_card_data):
        """Should raise ValueError for invalid format."""
        with pytest.raises(ValueError, match="format must be"):
            generate_model_card(sample_model_card_data, format="html")

    def test_contains_model_name(self, sample_model_card_data):
        """Model name should appear in output."""
        card = generate_model_card(sample_model_card_data, format="markdown")

        assert sample_model_card_data.model_name in card

    def test_contains_all_sections(self, sample_model_card_data):
        """All major sections should be present in markdown."""
        card = generate_model_card(sample_model_card_data, format="markdown")

        expected_sections = [
            "## Model Details",
            "## Intended Use",
            "## Training Data",
            "## Model Architecture",
            "### Prior Distributions",
            "### Hyperparameters",
            "## Evaluation Results",
            "### Convergence Diagnostics",
            "### Calibration",
            "### Predictive Performance",
            "## Limitations",
            "## Ethical Considerations",
            "## How to Use",
            "### Loading the Model",
            "### Making Predictions",
            "### Interpreting Results",
        ]

        for section in expected_sections:
            assert section in card, f"Missing section: {section}"

    def test_code_blocks_formatted(self, sample_model_card_data):
        """Code examples should have proper fencing."""
        card = generate_model_card(sample_model_card_data, format="markdown")

        # Should have python code blocks
        assert "```python" in card
        assert "```" in card

        # Code examples should be included
        assert sample_model_card_data.load_example in card
        assert sample_model_card_data.predict_example in card
        assert sample_model_card_data.interpret_example in card

    def test_hyperparameters_table(self, sample_model_card_data):
        """Hyperparameters should be formatted as table in markdown."""
        card = generate_model_card(sample_model_card_data, format="markdown")

        assert "| Parameter | Value |" in card
        assert "|-----------|-------|" in card
        assert "param1" in card
        assert "1.0" in card

    def test_latex_code_listing(self, sample_model_card_data):
        """LaTeX output should use lstlisting for code."""
        card = generate_model_card(sample_model_card_data, format="latex")

        assert "\\begin{lstlisting}" in card
        assert "\\end{lstlisting}" in card

    def test_latex_escapes_special_chars(self):
        """LaTeX generator should escape special characters."""
        data = ModelCardData(
            model_name="Test_Model & Co.",
            model_version="0.1.0",
            model_type="Test Type with $special% chars",
            authors=["Author #1"],
            created_date="2026-01-19",
            last_updated="2026-01-19",
            dataset_name="Test",
            dataset_size=100,
            dataset_description="Test with 100% coverage",
            data_preprocessing="Test",
            architecture_summary="Test",
            priors_description="Test",
        )

        card = generate_model_card(data, format="latex")

        # Special characters should be escaped
        assert "\\_" in card  # underscore
        assert "\\&" in card  # ampersand
        assert "\\$" in card  # dollar
        assert "\\%" in card  # percent
        assert "\\#" in card  # hash


class TestWriteModelCard:
    """Tests for write_model_card function."""

    def test_creates_markdown_file(self, sample_model_card_data):
        """Should create .md file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model_card"
            paths = write_model_card(sample_model_card_data, output_path, formats=("md",))

            assert len(paths) == 1
            assert paths[0].suffix == ".md"
            assert paths[0].exists()

    def test_creates_latex_file(self, sample_model_card_data):
        """Should create .tex file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model_card"
            paths = write_model_card(sample_model_card_data, output_path, formats=("tex",))

            assert len(paths) == 1
            assert paths[0].suffix == ".tex"
            assert paths[0].exists()

    def test_creates_both_formats(self, sample_model_card_data):
        """Should create both .md and .tex files by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model_card"
            paths = write_model_card(sample_model_card_data, output_path)

            assert len(paths) == 2
            suffixes = {p.suffix for p in paths}
            assert ".md" in suffixes
            assert ".tex" in suffixes

    def test_returns_paths(self, sample_model_card_data):
        """Should return list of created paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model_card"
            paths = write_model_card(sample_model_card_data, output_path)

            assert isinstance(paths, list)
            assert all(isinstance(p, Path) for p in paths)
            assert all(p.exists() for p in paths)

    def test_creates_parent_directories(self, sample_model_card_data):
        """Should create parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "deep" / "nested" / "path" / "model_card"
            paths = write_model_card(sample_model_card_data, output_path)

            assert all(p.exists() for p in paths)

    def test_unsupported_format_raises(self, sample_model_card_data):
        """Should raise ValueError for unsupported format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model_card"
            with pytest.raises(ValueError, match="Unsupported format"):
                write_model_card(sample_model_card_data, output_path, formats=("html",))


class TestUpdateWithResults:
    """Tests for update_model_card_with_results function."""

    def test_fills_convergence(self, sample_model_card_data):
        """Should populate convergence summary from diagnostics."""
        # Create mock convergence diagnostics
        convergence = MagicMock()
        convergence.passed = True
        convergence.rhat_max = 1.005
        convergence.ess_bulk_min = 2000
        convergence.ess_tail_min = 1800
        convergence.divergences = 0
        convergence.failing_params = []

        updated = update_model_card_with_results(
            sample_model_card_data, convergence=convergence
        )

        assert "PASSED" in updated.convergence_summary
        assert "1.0050" in updated.convergence_summary
        assert "2,000" in updated.convergence_summary

    def test_fills_convergence_failed(self, sample_model_card_data):
        """Should show FAILED status when convergence fails."""
        convergence = MagicMock()
        convergence.passed = False
        convergence.rhat_max = 1.02
        convergence.ess_bulk_min = 500
        convergence.ess_tail_min = 400
        convergence.divergences = 10
        convergence.failing_params = ["param1", "param2"]

        updated = update_model_card_with_results(
            sample_model_card_data, convergence=convergence
        )

        assert "FAILED" in updated.convergence_summary
        assert "param1" in updated.convergence_summary

    def test_fills_calibration(self, sample_model_card_data):
        """Should populate calibration summary from coverage results."""
        # Create mock coverage results
        coverage_50 = MagicMock()
        coverage_50.empirical = 0.48
        coverage_95 = MagicMock()
        coverage_95.empirical = 0.94

        coverage_results = {0.50: coverage_50, 0.95: coverage_95}

        updated = update_model_card_with_results(
            sample_model_card_data, coverage_results=coverage_results
        )

        assert "50% CI:" in updated.calibration_summary
        assert "95% CI:" in updated.calibration_summary
        assert "48.0%" in updated.calibration_summary
        assert "94.0%" in updated.calibration_summary

    def test_fills_elpd(self, sample_model_card_data):
        """Should fill LOO ELPD value from LOOResult."""
        loo_result = MagicMock()
        loo_result.elpd_loo = -5432.1
        loo_result.se_elpd = 123.4

        updated = update_model_card_with_results(
            sample_model_card_data, loo_result=loo_result
        )

        assert updated.loo_elpd == -5432.1
        assert "-5432.1" in updated.predictive_summary
        assert "123.4" in updated.predictive_summary

    def test_fills_point_metrics(self, sample_model_card_data):
        """Should fill point metrics summary."""
        point_metrics = MagicMock()
        point_metrics.mae = 4.56
        point_metrics.rmse = 6.78
        point_metrics.r2 = 0.789

        updated = update_model_card_with_results(
            sample_model_card_data, point_metrics=point_metrics
        )

        assert "MAE: 4.56" in updated.predictive_summary
        assert "RMSE: 6.78" in updated.predictive_summary
        assert "R-squared: 0.789" in updated.predictive_summary

    def test_preserves_other_fields(self, sample_model_card_data):
        """Should preserve fields not being updated."""
        convergence = MagicMock()
        convergence.passed = True
        convergence.rhat_max = 1.005
        convergence.ess_bulk_min = 2000
        convergence.ess_tail_min = 1800
        convergence.divergences = 0
        convergence.failing_params = []

        updated = update_model_card_with_results(
            sample_model_card_data, convergence=convergence
        )

        # Should preserve non-updated fields
        assert updated.model_name == sample_model_card_data.model_name
        assert updated.limitations == sample_model_card_data.limitations
        assert updated.load_example == sample_model_card_data.load_example

    def test_updates_last_updated_date(self, sample_model_card_data):
        """Should update last_updated to today's date."""
        from datetime import date

        convergence = MagicMock()
        convergence.passed = True
        convergence.rhat_max = 1.005
        convergence.ess_bulk_min = 2000
        convergence.ess_tail_min = 1800
        convergence.divergences = 0
        convergence.failing_params = []

        updated = update_model_card_with_results(
            sample_model_card_data, convergence=convergence
        )

        assert updated.last_updated == date.today().isoformat()

    def test_no_updates_returns_copy(self, sample_model_card_data):
        """Calling with no updates should return data with only last_updated changed."""
        updated = update_model_card_with_results(sample_model_card_data)

        # Should be a different object
        assert updated is not sample_model_card_data
        # But with same content (except last_updated)
        assert updated.model_name == sample_model_card_data.model_name
        assert updated.convergence_summary == sample_model_card_data.convergence_summary
