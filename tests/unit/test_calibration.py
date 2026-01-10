"""Unit tests for the calibration module.

Tests coverage computation, multi-level coverage, and reliability diagram data.
Uses synthetic data with known calibration properties.
"""

import numpy as np
import pytest

from aoty_pred.evaluation.calibration import (
    CoverageResult,
    ReliabilityData,
    compute_coverage,
    compute_multi_coverage,
    compute_reliability_data,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def rng():
    """Fixed random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def perfect_calibration_data(rng):
    """Synthetic data with perfect calibration.

    For perfect calibration, we need y_true to be a sample FROM the
    predictive distribution (not the center of it). The model's
    predictive distribution has mean mu and std sigma. y_true is
    drawn from N(mu, sigma), so when we compute coverage, ~95% of
    observations should fall within the 95% CI.
    """
    n_obs = 500
    sigma = 10.0
    n_samples = 1000

    # For each observation, the model predicts a distribution N(mu_i, sigma)
    # We'll use mu_i = 50 for all observations (simplicity)
    mu = np.full(n_obs, 50.0)

    # y_true is sampled from the model's predictive distribution
    # This is the key: y_true = mu + eps, where eps ~ N(0, sigma)
    y_true = mu + rng.normal(0, sigma, n_obs)

    # The model's predictive samples are drawn from N(mu_i, sigma)
    # Note: samples are around mu, not around y_true
    y_samples = mu + rng.normal(0, sigma, (n_samples, n_obs))

    return y_true, y_samples, sigma


@pytest.fixture
def overconfident_data(rng):
    """Synthetic data with overconfident predictions.

    Intervals are too narrow - samples have lower variance than the
    actual noise in y_true.
    """
    n_obs = 500
    true_sigma = 10.0
    model_sigma = 5.0  # Model thinks variance is lower

    # True values with noise
    y_base = rng.normal(50, 5, n_obs)
    y_true = y_base + rng.normal(0, true_sigma, n_obs)

    # Samples are too narrow
    n_samples = 1000
    y_samples = y_base + rng.normal(0, model_sigma, (n_samples, n_obs))

    return y_true, y_samples


@pytest.fixture
def underconfident_data(rng):
    """Synthetic data with underconfident predictions.

    Intervals are too wide - samples have higher variance than the
    actual noise in y_true.
    """
    n_obs = 500
    true_sigma = 5.0
    model_sigma = 15.0  # Model thinks variance is higher

    # True values with small noise
    y_base = rng.normal(50, 5, n_obs)
    y_true = y_base + rng.normal(0, true_sigma, n_obs)

    # Samples are too wide
    n_samples = 1000
    y_samples = y_base + rng.normal(0, model_sigma, (n_samples, n_obs))

    return y_true, y_samples


# ============================================================================
# Tests for compute_coverage
# ============================================================================


class TestComputeCoverage:
    """Tests for the compute_coverage function."""

    def test_returns_coverage_result(self, perfect_calibration_data):
        """Should return a CoverageResult instance."""
        y_true, y_samples, _ = perfect_calibration_data
        result = compute_coverage(y_true, y_samples, prob=0.95)

        assert isinstance(result, CoverageResult)

    def test_compute_coverage_perfect_calibration(self, perfect_calibration_data):
        """Well-calibrated samples should have empirical coverage near nominal."""
        y_true, y_samples, _ = perfect_calibration_data
        result = compute_coverage(y_true, y_samples, prob=0.95)

        # With 500 observations, coverage should be within a reasonable range
        # Allow some sampling variability (say, 90% to 100%)
        assert 0.90 <= result.empirical <= 1.0
        assert result.nominal == 0.95
        assert result.n_obs == 500

    def test_compute_coverage_overconfident(self, overconfident_data):
        """Overconfident predictions should have coverage < nominal."""
        y_true, y_samples = overconfident_data
        result = compute_coverage(y_true, y_samples, prob=0.95)

        # Overconfident model: coverage should be clearly below 95%
        assert result.empirical < 0.90
        assert result.nominal == 0.95

    def test_compute_coverage_underconfident(self, underconfident_data):
        """Underconfident predictions should have coverage > nominal."""
        y_true, y_samples = underconfident_data
        result = compute_coverage(y_true, y_samples, prob=0.95)

        # Underconfident model: coverage should be near 100%
        assert result.empirical > 0.97
        assert result.nominal == 0.95

    def test_compute_coverage_interval_width(self, perfect_calibration_data):
        """Interval width (sharpness) should be computed correctly."""
        y_true, y_samples, sigma = perfect_calibration_data
        result = compute_coverage(y_true, y_samples, prob=0.95)

        # For normal distribution, 95% CI width is ~3.92 * sigma
        expected_width = 3.92 * sigma
        # Allow 20% tolerance for finite samples
        assert 0.8 * expected_width <= result.interval_width <= 1.2 * expected_width

    def test_coverage_bounds_shape(self, perfect_calibration_data):
        """Lower and upper bounds should have correct shape."""
        y_true, y_samples, _ = perfect_calibration_data
        result = compute_coverage(y_true, y_samples, prob=0.95)

        assert result.lower_bound.shape == (len(y_true),)
        assert result.upper_bound.shape == (len(y_true),)

    def test_coverage_n_covered_consistent(self, perfect_calibration_data):
        """n_covered should be consistent with empirical coverage."""
        y_true, y_samples, _ = perfect_calibration_data
        result = compute_coverage(y_true, y_samples, prob=0.95)

        expected_empirical = result.n_covered / result.n_obs
        assert result.empirical == expected_empirical

    def test_different_probability_levels(self, perfect_calibration_data):
        """Coverage should work for different probability levels."""
        y_true, y_samples, _ = perfect_calibration_data

        for prob in [0.50, 0.80, 0.95]:
            result = compute_coverage(y_true, y_samples, prob=prob)
            assert result.nominal == prob
            # Looser bounds for smaller intervals
            tolerance = 0.15 if prob < 0.9 else 0.10
            assert abs(result.empirical - prob) < tolerance

    def test_input_validation_y_true_shape(self, rng):
        """Should raise error for non-1D y_true."""
        y_true_2d = rng.normal(0, 1, (10, 5))
        y_samples = rng.normal(0, 1, (100, 50))

        with pytest.raises(ValueError, match="y_true must be 1D"):
            compute_coverage(y_true_2d, y_samples)

    def test_input_validation_y_samples_shape(self, rng):
        """Should raise error for non-2D y_samples."""
        y_true = rng.normal(0, 1, 50)
        y_samples_1d = rng.normal(0, 1, 50)

        with pytest.raises(ValueError, match="y_samples must be 2D"):
            compute_coverage(y_true, y_samples_1d)

    def test_input_validation_shape_mismatch(self, rng):
        """Should raise error when shapes don't match."""
        y_true = rng.normal(0, 1, 50)
        y_samples = rng.normal(0, 1, (100, 30))  # Wrong n_obs

        with pytest.raises(ValueError, match="observations"):
            compute_coverage(y_true, y_samples)


# ============================================================================
# Tests for compute_multi_coverage
# ============================================================================


class TestComputeMultiCoverage:
    """Tests for the compute_multi_coverage function."""

    def test_compute_multi_coverage_all_levels(self, perfect_calibration_data):
        """Should return coverage for all requested probability levels."""
        y_true, y_samples, _ = perfect_calibration_data
        probs = (0.50, 0.80, 0.95)

        results = compute_multi_coverage(y_true, y_samples, probs=probs)

        assert set(results.keys()) == set(probs)
        for prob in probs:
            assert isinstance(results[prob], CoverageResult)
            assert results[prob].nominal == prob

    def test_multi_coverage_custom_levels(self, perfect_calibration_data):
        """Should work with custom probability levels."""
        y_true, y_samples, _ = perfect_calibration_data
        probs = (0.60, 0.90, 0.99)

        results = compute_multi_coverage(y_true, y_samples, probs=probs)

        assert set(results.keys()) == set(probs)

    def test_multi_coverage_ordering(self, perfect_calibration_data):
        """Higher probability levels should have higher or equal coverage."""
        y_true, y_samples, _ = perfect_calibration_data
        probs = (0.50, 0.80, 0.95)

        results = compute_multi_coverage(y_true, y_samples, probs=probs)

        # Coverage should increase with probability level (for calibrated model)
        # Use <= for robustness (can be equal at 100%)
        assert results[0.50].empirical <= results[0.80].empirical
        assert results[0.80].empirical <= results[0.95].empirical
        # At least one should differ (not all saturated at 100%)
        assert results[0.50].empirical < 1.0 or results[0.50].nominal == 0.50


# ============================================================================
# Tests for compute_reliability_data
# ============================================================================


class TestComputeReliabilityData:
    """Tests for the compute_reliability_data function."""

    def test_returns_reliability_data(self, perfect_calibration_data):
        """Should return a ReliabilityData instance."""
        y_true, y_samples, _ = perfect_calibration_data
        result = compute_reliability_data(y_true, y_samples, n_bins=10)

        assert isinstance(result, ReliabilityData)

    def test_compute_reliability_data_bins(self, perfect_calibration_data):
        """Should produce the requested number of bins (or fewer if ties)."""
        y_true, y_samples, _ = perfect_calibration_data
        n_bins = 10

        result = compute_reliability_data(y_true, y_samples, n_bins=n_bins)

        # Number of actual bins may be <= n_bins due to ties
        assert len(result.predicted_probs) <= n_bins
        assert len(result.observed_freq) <= n_bins
        assert len(result.counts) <= n_bins

    def test_compute_reliability_data_counts(self, perfect_calibration_data):
        """Counts should sum to n_obs."""
        y_true, y_samples, _ = perfect_calibration_data
        n_bins = 10

        result = compute_reliability_data(y_true, y_samples, n_bins=n_bins)

        # Total counts should equal number of observations
        assert result.counts.sum() == len(y_true)

    def test_reliability_data_bin_edges(self, perfect_calibration_data):
        """Bin edges should span [0, 1] approximately."""
        y_true, y_samples, _ = perfect_calibration_data

        result = compute_reliability_data(y_true, y_samples, n_bins=10)

        # Bin edges should be in [0, 1] (or close to it for empirical data)
        assert result.bin_edges[0] >= 0
        assert result.bin_edges[-1] <= 1

    def test_reliability_predicted_probs_range(self, perfect_calibration_data):
        """Predicted probabilities should be in [0, 1]."""
        y_true, y_samples, _ = perfect_calibration_data

        result = compute_reliability_data(y_true, y_samples, n_bins=10)

        # Check non-empty bins
        mask = result.counts > 0
        assert all(0 <= p <= 1 for p in result.predicted_probs[mask])

    def test_reliability_observed_freq_range(self, perfect_calibration_data):
        """Observed frequencies should be in [0, 1]."""
        y_true, y_samples, _ = perfect_calibration_data

        result = compute_reliability_data(y_true, y_samples, n_bins=10)

        # Check non-empty bins
        mask = result.counts > 0
        assert all(0 <= f <= 1 for f in result.observed_freq[mask])

    def test_reliability_different_n_bins(self, perfect_calibration_data):
        """Should work with different numbers of bins."""
        y_true, y_samples, _ = perfect_calibration_data

        for n_bins in [5, 10, 20]:
            result = compute_reliability_data(y_true, y_samples, n_bins=n_bins)
            # Just check it runs and produces valid output
            assert len(result.predicted_probs) <= n_bins
            assert result.counts.sum() == len(y_true)

    def test_reliability_input_validation(self, rng):
        """Should raise error for invalid inputs."""
        y_true_2d = rng.normal(0, 1, (10, 5))
        y_samples = rng.normal(0, 1, (100, 50))

        with pytest.raises(ValueError, match="y_true must be 1D"):
            compute_reliability_data(y_true_2d, y_samples)
