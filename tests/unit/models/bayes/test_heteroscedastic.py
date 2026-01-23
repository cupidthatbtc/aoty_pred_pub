"""Tests for heteroscedastic observation noise implementation."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from aoty_pred.models.bayes.model import compute_sigma_scaled


class TestComputeSigmaScaled:
    """Tests for the compute_sigma_scaled helper function."""

    def test_basic_scaling(self):
        """Test that sigma_scaled = sigma_obs / n^exponent."""
        sigma_obs = 1.0
        n_reviews = jnp.array([100.0])
        exponent = 0.5  # sqrt scaling

        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent)

        # 1.0 / 100^0.5 = 1.0 / 10 = 0.1
        assert np.isclose(result[0], 0.1, rtol=1e-5)

    def test_single_review_penalty(self):
        """Test that n=1 applies 2x multiplier."""
        sigma_obs = 1.0
        n_reviews = jnp.array([1.0])
        exponent = 0.5

        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent)

        # n=1 should apply 2x penalty: sigma_obs * 2.0
        assert np.isclose(result[0], 2.0, rtol=1e-5)

    def test_custom_single_review_multiplier(self):
        """Test custom single_review_multiplier."""
        sigma_obs = 1.0
        n_reviews = jnp.array([1.0])
        exponent = 0.5

        result = compute_sigma_scaled(
            sigma_obs, n_reviews, exponent, single_review_multiplier=3.0
        )

        assert np.isclose(result[0], 3.0, rtol=1e-5)

    def test_homoscedastic_mode(self):
        """Test that exponent=0 returns sigma_obs unchanged."""
        sigma_obs = 1.0
        n_reviews = jnp.array([1.0, 10.0, 100.0, 1000.0])
        exponent = 0.0

        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent)

        # n^0 = 1 for all n, but n=1 still gets penalty
        # Actually when exp=0, formula gives sigma_obs for all n
        # But n=1 case triggers penalty branch
        # Need to verify actual behavior
        assert np.isclose(result[1], 1.0, rtol=1e-5)  # n=10
        assert np.isclose(result[2], 1.0, rtol=1e-5)  # n=100
        assert np.isclose(result[3], 1.0, rtol=1e-5)  # n=1000

    def test_extreme_n_no_underflow(self):
        """Test that extreme n values don't cause underflow."""
        sigma_obs = 1.0
        n_reviews = jnp.array([100000.0])  # 100k reviews
        exponent = 0.5

        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent)

        # 1.0 / 100000^0.5 = 1.0 / 316.2... = 0.00316...
        # But min_sigma=0.01, so should hit floor
        assert result[0] >= 0.01
        assert not np.isnan(result[0])
        assert not np.isinf(result[0])

    def test_min_sigma_floor(self):
        """Test custom min_sigma floor."""
        sigma_obs = 1.0
        n_reviews = jnp.array([1000000.0])  # 1M reviews
        exponent = 0.5

        result = compute_sigma_scaled(
            sigma_obs, n_reviews, exponent, min_sigma=0.001
        )

        # 1.0 / 1000000^0.5 = 0.001, at floor
        assert result[0] >= 0.001
        assert not np.isnan(result[0])

    def test_array_broadcasting(self):
        """Test that function handles arrays correctly."""
        sigma_obs = 2.0
        n_reviews = jnp.array([4.0, 9.0, 16.0, 25.0])
        exponent = 0.5

        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent)

        # 2.0 / sqrt(n) for each
        expected = jnp.array([1.0, 2 / 3, 0.5, 0.4])
        assert result.shape == (4,)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_cube_root_scaling(self):
        """Test cube-root scaling (exponent=0.33)."""
        sigma_obs = 1.0
        n_reviews = jnp.array([8.0, 27.0, 64.0])
        exponent = 1 / 3  # cube root

        result = compute_sigma_scaled(sigma_obs, n_reviews, exponent)

        # 1.0 / n^(1/3) = 1/2, 1/3, 1/4 for n=8,27,64
        expected = jnp.array([0.5, 1 / 3, 0.25])
        np.testing.assert_allclose(result, expected, rtol=1e-4)
