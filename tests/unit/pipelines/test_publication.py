"""Tests for publication artifact helpers."""

import pytest

from aoty_pred.pipelines.publication import _get_coefficient_var_names


class MockPosterior:
    def __init__(self, var_names):
        self._var_names = set(var_names)

    def __contains__(self, key):
        return key in self._var_names


class MockIData:
    def __init__(self, var_names):
        self.posterior = MockPosterior(var_names)


class TestGetCoefficientVarNames:
    def test_basic_vars_always_present(self):
        idata = MockIData(["user_beta", "user_mu_artist", "user_sigma_artist", "user_sigma_obs"])
        result = _get_coefficient_var_names(idata)
        assert "user_beta" in result
        assert "user_mu_artist" in result
        assert "user_sigma_artist" in result
        assert "user_sigma_obs" in result

    def test_includes_sigma_ref_when_present(self):
        idata = MockIData(
            [
                "user_beta",
                "user_mu_artist",
                "user_sigma_artist",
                "user_sigma_obs",
                "user_sigma_ref",
            ]
        )
        result = _get_coefficient_var_names(idata)
        assert "user_sigma_ref" in result

    def test_excludes_sigma_ref_when_absent(self):
        idata = MockIData(["user_beta", "user_mu_artist", "user_sigma_artist", "user_sigma_obs"])
        result = _get_coefficient_var_names(idata)
        assert "user_sigma_ref" not in result

    def test_includes_n_exponent_when_present(self):
        idata = MockIData(
            [
                "user_beta",
                "user_mu_artist",
                "user_sigma_artist",
                "user_sigma_obs",
                "user_n_exponent",
            ]
        )
        result = _get_coefficient_var_names(idata)
        assert "user_n_exponent" in result

    def test_excludes_n_exponent_when_absent(self):
        idata = MockIData(["user_beta", "user_mu_artist", "user_sigma_artist", "user_sigma_obs"])
        result = _get_coefficient_var_names(idata)
        assert "user_n_exponent" not in result

    def test_custom_prefix(self):
        idata = MockIData(
            [
                "critic_beta",
                "critic_mu_artist",
                "critic_sigma_artist",
                "critic_sigma_obs",
            ]
        )
        result = _get_coefficient_var_names(idata, prefix="critic_")
        assert "critic_beta" in result
        assert "critic_sigma_obs" in result

    def test_ordering_sigma_ref_before_sigma_obs(self):
        idata = MockIData(
            [
                "user_beta",
                "user_mu_artist",
                "user_sigma_artist",
                "user_sigma_obs",
                "user_sigma_ref",
            ]
        )
        result = _get_coefficient_var_names(idata)
        ref_idx = result.index("user_sigma_ref")
        obs_idx = result.index("user_sigma_obs")
        assert ref_idx < obs_idx
