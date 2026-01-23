"""Unit tests for compute_sigma_scaled function."""

import jax.numpy as jnp
import pytest

from aoty_pred.models.bayes.model import compute_sigma_scaled


class TestComputeSigmaScaledValidation:
    """Tests for input validation in compute_sigma_scaled."""

    def test_sigma_obs_positive_works(self):
        """Test that positive sigma_obs values work correctly."""
        n_reviews = jnp.array([100.0])

        # Should not raise
        result = compute_sigma_scaled(1.0, n_reviews, exponent=0.5)

        # sigma = 1.0 / sqrt(100) = 0.1
        assert jnp.isclose(result[0], 0.1, atol=1e-6)
