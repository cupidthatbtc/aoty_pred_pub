"""Tests for next-album prediction pipeline."""

from __future__ import annotations

import pytest

from aoty_pred.pipelines.predict_next import (
    SCENARIOS_KNOWN,
    SCENARIOS_NEW,
    _extract_posterior_samples,
    predict_next_albums,
)


def test_module_imports():
    """Module-level imports succeed without errors."""
    import aoty_pred.pipelines.predict_next  # noqa: F401


def test_predict_next_albums_signature():
    """predict_next_albums has correct signature accepting StageContext."""
    import inspect

    sig = inspect.signature(predict_next_albums)
    params = list(sig.parameters.keys())
    assert "ctx" in params


def test_scenario_constants():
    """Scenario name constants are defined correctly."""
    assert SCENARIOS_KNOWN == ["same", "population_mean", "artist_mean"]
    assert SCENARIOS_NEW == ["population", "debut_defaults"]


def test_extract_posterior_samples_function_exists():
    """_extract_posterior_samples helper exists and is callable."""
    assert callable(_extract_posterior_samples)


class TestExtractPosteriorSamples:
    """Tests for _extract_posterior_samples with mock InferenceData."""

    def test_extracts_and_flattens(self):
        """Samples are flattened from (chains, draws, ...) to (n_samples, ...)."""
        import numpy as np

        # Build a minimal mock that mimics idata.posterior
        class MockDataArray:
            def __init__(self, values):
                self._values = values

            @property
            def values(self):
                return self._values

        class MockPosterior:
            def __init__(self, data_vars_dict):
                self._data_vars = data_vars_dict

            @property
            def data_vars(self):
                return self._data_vars

            def __getitem__(self, key):
                return MockDataArray(self._data_vars[key])

        class MockIData:
            def __init__(self, posterior_dict):
                # Store raw numpy arrays keyed by name
                self.posterior = MockPosterior(posterior_dict)

        # 2 chains, 3 draws, scalar parameter
        mock_data = {
            "user_mu_artist": np.random.randn(2, 3),
            "user_beta": np.random.randn(2, 3, 5),
        }
        idata = MockIData(mock_data)

        result = _extract_posterior_samples(idata)

        # Scalar: flattened to (6,)
        assert result["user_mu_artist"].shape == (6,)
        # Vector: flattened to (6, 5)
        assert result["user_beta"].shape == (6, 5)
