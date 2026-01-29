"""Integration tests for sigma_ref training pipeline.

Tests verify:
1. n_ref and n_ref_method are stored in InferenceData constant_data
2. sigma_ref posterior stats are extractable and positive
3. sigma_obs deterministic is present when sigma_ref mode is active
4. Divergence rate is computable as float in [0, 1]
5. Quick validation: 200-sample MCMC with synthetic data passes divergence check

These tests use fit_model() directly (bypassing train_models() which needs file I/O)
with synthetic data to test the pipeline integration.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from aoty_pred.models.bayes.fit import MCMCConfig, fit_model
from aoty_pred.models.bayes.model import user_score_model
from aoty_pred.models.bayes.priors import PriorConfig

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _make_sample_data():
    """Create sample data dict for model tests (matches test_sigma_ref.py pattern)."""
    key = random.PRNGKey(42)
    n_obs = 50
    n_artists = 10
    n_features = 3

    return {
        "artist_idx": jnp.array([i % n_artists for i in range(n_obs)]),
        "album_seq": jnp.array([(i // n_artists) + 1 for i in range(n_obs)]),
        "prev_score": jnp.zeros(n_obs),
        "X": random.normal(key, (n_obs, n_features)),
        "n_artists": n_artists,
        "max_seq": (n_obs // n_artists),
        "n_reviews": jnp.array([10 + i * 5 for i in range(n_obs)], dtype=jnp.float32),
    }


def _make_model_args_for_fit(n_ref=50.0, learn_n_exponent=True):
    """Create full model_args dict suitable for fit_model().

    Adds observed y, priors, and heteroscedastic config to base sample data.
    """
    data = _make_sample_data()
    data["y"] = np.asarray(random.normal(random.PRNGKey(1), (50,)) * 5 + 70)
    data["n_ref"] = n_ref
    data["n_exponent"] = 0.0
    data["learn_n_exponent"] = learn_n_exponent
    data["n_exponent_prior"] = "logit-normal"
    data["priors"] = PriorConfig()
    return data


# ===========================================================================
# Class 1: n_ref in InferenceData constant_data
# ===========================================================================


@pytest.mark.slow
class TestNRefInInferenceData:
    """Verify n_ref and n_ref_method are stored in InferenceData constant_data."""

    @pytest.fixture(scope="class")
    def fit_result_with_n_ref(self):
        """Run a short MCMC with n_ref=50.0 (shared across class tests)."""
        model_args = _make_model_args_for_fit(n_ref=50.0, learn_n_exponent=True)
        config = MCMCConfig(num_warmup=20, num_samples=20, num_chains=1)
        return fit_model(user_score_model, model_args, config=config, progress_bar=False)

    def test_n_ref_stored_in_constant_data(self, fit_result_with_n_ref):
        """constant_data should contain n_ref=50.0 when sigma_ref mode is active."""
        idata = fit_result_with_n_ref.idata
        assert "n_ref" in idata.constant_data, "n_ref missing from constant_data"
        assert float(idata.constant_data["n_ref"]) == 50.0

    def test_n_ref_method_stored_in_constant_data(self, fit_result_with_n_ref):
        """constant_data should contain n_ref_method='median'."""
        idata = fit_result_with_n_ref.idata
        assert "n_ref_method" in idata.constant_data, "n_ref_method missing from constant_data"
        assert str(idata.constant_data["n_ref_method"].values) == "median"

    def test_n_ref_not_in_constant_data_when_none(self):
        """constant_data should NOT contain n_ref when n_ref=None (homoscedastic)."""
        model_args = _make_model_args_for_fit(n_ref=None, learn_n_exponent=False)
        model_args["n_exponent"] = 0.0
        config = MCMCConfig(num_warmup=20, num_samples=20, num_chains=1)
        result = fit_model(user_score_model, model_args, config=config, progress_bar=False)
        assert "n_ref" not in result.idata.constant_data


# ===========================================================================
# Class 2: Training summary fields
# ===========================================================================


@pytest.mark.slow
class TestSigmaRefTrainingSummaryFields:
    """Verify sigma_ref stats, sigma_obs deterministic, and divergence_rate.

    These tests run short MCMC fits and verify the posterior contains the
    expected sites and stats needed for training_summary.json construction.
    """

    @pytest.fixture(scope="class")
    def fit_result_sigma_ref(self):
        """Run a short MCMC with sigma_ref mode active."""
        model_args = _make_model_args_for_fit(n_ref=50.0, learn_n_exponent=True)
        config = MCMCConfig(num_warmup=20, num_samples=20, num_chains=1)
        return fit_model(user_score_model, model_args, config=config, progress_bar=False)

    def test_sigma_ref_stats_in_summary(self, fit_result_sigma_ref):
        """sigma_ref posterior should have positive samples extractable for summary stats."""
        idata = fit_result_sigma_ref.idata
        sigma_ref_samples = idata.posterior["user_sigma_ref"].values.flatten()

        # Should have samples
        assert sigma_ref_samples.shape[0] > 0, "No sigma_ref samples found"

        # Mean should be positive (HalfNormal prior)
        mean_val = float(np.mean(sigma_ref_samples))
        assert mean_val > 0, f"sigma_ref mean should be positive, got {mean_val}"

    def test_sigma_obs_deterministic_in_posterior(self, fit_result_sigma_ref):
        """sigma_obs should be a deterministic site equal to sigma_ref * n_ref^n_exp."""
        idata = fit_result_sigma_ref.idata

        # sigma_obs must exist in posterior (deterministic sites included by fit_model)
        assert "user_sigma_obs" in idata.posterior, "user_sigma_obs missing from posterior"

        # Verify algebraic identity: sigma_obs = sigma_ref * n_ref^n_exp
        sigma_ref = idata.posterior["user_sigma_ref"].values.flatten()
        sigma_obs = idata.posterior["user_sigma_obs"].values.flatten()
        n_exp = idata.posterior["user_n_exponent"].values.flatten()
        n_ref = 50.0

        expected = sigma_ref * np.power(n_ref, n_exp)
        np.testing.assert_allclose(
            sigma_obs,
            expected,
            rtol=1e-5,
            err_msg="sigma_obs != sigma_ref * n_ref^n_exp in posterior",
        )

    def test_divergence_rate_computation(self, fit_result_sigma_ref):
        """Divergence rate should be a float in [0, 1]."""
        config = MCMCConfig(num_warmup=20, num_samples=20, num_chains=1)
        divergence_rate = float(
            fit_result_sigma_ref.divergences / (config.num_samples * config.num_chains)
        )

        assert isinstance(divergence_rate, float)
        assert 0.0 <= divergence_rate <= 1.0, f"Divergence rate {divergence_rate} not in [0, 1]"


# ===========================================================================
# Class 3: Quick validation (automated Phase 46 success check)
# ===========================================================================


@pytest.mark.slow
class TestQuickValidation:
    """Quick validation MCMC run confirming sigma_ref mode works end-to-end.

    This is the automated check for Phase 46 success criterion #1:
    "A quick validation MCMC run with learn_n_exponent confirms sigma_ref mode
    works end-to-end."

    Uses SYNTHETIC data (not real data). The divergence rate threshold of 50%
    is generous for a quick run with small synthetic data. The real validation
    on actual training data is a manual CLI step.

    Manual validation CLI command (uses real data):
        python -m aoty_pred.pipelines.orchestrator \\
            --learn-n-exponent --samples 200 --warmup 200 --chains 1 \\
            --allow-divergences
        # Check models/training_summary.json for divergence_rate < 0.50
    """

    def test_quick_validation_200_samples_sigma_ref(self):
        """200-sample MCMC with sigma_ref mode should have divergence_rate < 0.50."""
        model_args = _make_model_args_for_fit(n_ref=None, learn_n_exponent=True)
        # Compute n_ref as median of n_reviews (same as train_models)
        n_ref = float(np.median(np.asarray(model_args["n_reviews"])))
        model_args["n_ref"] = n_ref

        num_warmup = 200
        num_samples = 200
        num_chains = 1

        config = MCMCConfig(
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
        )
        result = fit_model(user_score_model, model_args, config=config, progress_bar=False)

        divergence_rate = result.divergences / (num_samples * num_chains)
        print(
            f"\nQuick validation: divergences={result.divergences}, "
            f"rate={divergence_rate:.4f}, runtime={result.runtime_seconds:.1f}s"
        )

        assert divergence_rate < 0.50, (
            f"Divergence rate {divergence_rate:.4f} >= 0.50 threshold. "
            f"Got {result.divergences} divergences in {num_samples * num_chains} samples."
        )
