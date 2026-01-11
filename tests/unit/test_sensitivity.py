"""Unit tests for sensitivity analysis module.

Tests data structures, prior configurations, and aggregation functions.
Integration tests requiring actual model fitting are marked for skip.
"""

import numpy as np
import pandas as pd
import pytest

from aoty_pred.pipelines.sensitivity import (
    SensitivityResult,
    PRIOR_CONFIGS,
    aggregate_sensitivity_results,
    create_coefficient_comparison_df,
    extract_coefficient_summary,
)
from aoty_pred.evaluation.cv import LOOResult
from aoty_pred.evaluation.metrics import CRPSResult
from aoty_pred.models.bayes.diagnostics import ConvergenceDiagnostics


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_convergence():
    """Create a mock ConvergenceDiagnostics with passing values."""
    return ConvergenceDiagnostics(
        rhat_max=1.002,
        ess_bulk_min=2500,
        ess_tail_min=2200,
        divergences=0,
        passed=True,
        failing_params=[],
        summary_df=pd.DataFrame({
            "mean": [0.5, 0.3],
            "sd": [0.1, 0.05],
            "r_hat": [1.001, 1.002],
            "ess_bulk": [2500, 2700],
            "ess_tail": [2200, 2400],
        }, index=["param1", "param2"]),
    )


@pytest.fixture
def mock_loo_result():
    """Create a mock LOOResult with synthetic ELPD."""
    # Create a minimal mock that has the required attributes
    class MockELPDData:
        elpd_loo = -1234.5
        se = 45.2
        p_loo = 120.3
        pareto_k = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.8])
        warning = None

    return LOOResult(
        loo=MockELPDData(),
        elpd_loo=-1234.5,
        se_elpd=45.2,
        p_loo=120.3,
        n_high_pareto_k=1,  # One value > 0.7
        high_pareto_k_indices=np.array([5]),
        warning=None,
    )


@pytest.fixture
def mock_crps_result():
    """Create a mock CRPSResult."""
    return CRPSResult(
        mean_crps=5.8,
        crps_values=np.array([4.5, 5.2, 6.1, 5.8, 6.4]),
        n_obs=5,
    )


@pytest.fixture
def mock_coefficients():
    """Create a mock coefficient summary DataFrame."""
    return pd.DataFrame({
        "mean": [0.15, 0.08, 12.5],
        "sd": [0.03, 0.02, 1.2],
        "hdi_3%": [0.09, 0.04, 10.1],
        "hdi_97%": [0.21, 0.12, 14.9],
    }, index=["user_rho", "user_beta[0]", "sigma_obs"])


@pytest.fixture
def mock_sensitivity_results(mock_convergence, mock_loo_result, mock_coefficients):
    """Create a set of mock SensitivityResults for testing aggregation."""
    results = {}

    # Default config result
    results["default"] = SensitivityResult(
        name="default",
        config={"priors": {"mu_artist_scale": 1.0}},
        idata=None,
        convergence=mock_convergence,
        loo=mock_loo_result,
        crps=None,
        coefficients=mock_coefficients,
    )

    # Diffuse config result - slightly different ELPD
    diffuse_loo = LOOResult(
        loo=type("MockELPD", (), {
            "elpd_loo": -1256.3,
            "se": 48.1,
            "p_loo": 135.2,
            "pareto_k": np.array([0.3, 0.4, 0.5]),
            "warning": None,
        })(),
        elpd_loo=-1256.3,
        se_elpd=48.1,
        p_loo=135.2,
        n_high_pareto_k=0,
        high_pareto_k_indices=np.array([]),
        warning=None,
    )

    diffuse_coef = mock_coefficients.copy()
    diffuse_coef["mean"] = [0.142, 0.072, 13.1]
    diffuse_coef["hdi_3%"] = [0.078, 0.032, 10.7]
    diffuse_coef["hdi_97%"] = [0.206, 0.112, 15.5]

    results["diffuse"] = SensitivityResult(
        name="diffuse",
        config={"priors": {"mu_artist_scale": 5.0}},
        idata=None,
        convergence=mock_convergence,
        loo=diffuse_loo,
        crps=None,
        coefficients=diffuse_coef,
    )

    # Informative config result
    informative_loo = LOOResult(
        loo=type("MockELPD", (), {
            "elpd_loo": -1240.1,
            "se": 44.8,
            "p_loo": 115.8,
            "pareto_k": np.array([0.2, 0.3]),
            "warning": None,
        })(),
        elpd_loo=-1240.1,
        se_elpd=44.8,
        p_loo=115.8,
        n_high_pareto_k=0,
        high_pareto_k_indices=np.array([]),
        warning=None,
    )

    informative_coef = mock_coefficients.copy()
    informative_coef["mean"] = [0.155, 0.085, 12.2]
    informative_coef["hdi_3%"] = [0.090, 0.045, 9.8]
    informative_coef["hdi_97%"] = [0.220, 0.125, 14.6]

    results["informative"] = SensitivityResult(
        name="informative",
        config={"priors": {"mu_artist_scale": 0.5}},
        idata=None,
        convergence=mock_convergence,
        loo=informative_loo,
        crps=None,
        coefficients=informative_coef,
    )

    return results


# ============================================================================
# Data Structure Tests
# ============================================================================


class TestSensitivityResult:
    """Tests for SensitivityResult dataclass."""

    def test_sensitivity_result_fields(self, mock_convergence, mock_loo_result, mock_coefficients):
        """Test that all fields are accessible."""
        result = SensitivityResult(
            name="test_variant",
            config={"threshold": 10},
            idata=None,
            convergence=mock_convergence,
            loo=mock_loo_result,
            crps=None,
            coefficients=mock_coefficients,
        )

        assert result.name == "test_variant"
        assert result.config == {"threshold": 10}
        assert result.idata is None
        assert result.convergence is mock_convergence
        assert result.loo is mock_loo_result
        assert result.crps is None
        assert result.coefficients is mock_coefficients

    def test_sensitivity_result_default_coefficients(self):
        """Test that coefficients default to empty DataFrame."""
        result = SensitivityResult(
            name="minimal",
            config={},
        )

        assert isinstance(result.coefficients, pd.DataFrame)
        assert result.coefficients.empty

    def test_sensitivity_result_all_none(self):
        """Test SensitivityResult with all optional fields as None."""
        result = SensitivityResult(
            name="none_test",
            config={"test": True},
            idata=None,
            convergence=None,
            loo=None,
            crps=None,
        )

        assert result.name == "none_test"
        assert result.convergence is None
        assert result.loo is None
        assert result.crps is None


class TestPriorConfigs:
    """Tests for prior configuration definitions."""

    def test_prior_configs_defined(self):
        """Test that PRIOR_CONFIGS has default, diffuse, and informative."""
        assert "default" in PRIOR_CONFIGS
        assert "diffuse" in PRIOR_CONFIGS
        assert "informative" in PRIOR_CONFIGS

    def test_prior_configs_count(self):
        """Test that we have exactly 3 prior configurations."""
        assert len(PRIOR_CONFIGS) == 3

    def test_diffuse_priors_wider(self):
        """Test that diffuse priors have larger scales than default."""
        default = PRIOR_CONFIGS["default"]
        diffuse = PRIOR_CONFIGS["diffuse"]

        assert diffuse.mu_artist_scale > default.mu_artist_scale
        assert diffuse.sigma_artist_scale > default.sigma_artist_scale
        assert diffuse.beta_scale > default.beta_scale

    def test_informative_priors_tighter(self):
        """Test that informative priors have smaller scales than default."""
        default = PRIOR_CONFIGS["default"]
        informative = PRIOR_CONFIGS["informative"]

        assert informative.mu_artist_scale < default.mu_artist_scale
        assert informative.sigma_artist_scale < default.sigma_artist_scale
        assert informative.beta_scale < default.beta_scale

    def test_prior_configs_all_positive_scales(self):
        """Test that all prior scales are positive."""
        for name, config in PRIOR_CONFIGS.items():
            assert config.mu_artist_scale > 0, f"{name}.mu_artist_scale"
            assert config.sigma_artist_scale > 0, f"{name}.sigma_artist_scale"
            assert config.beta_scale > 0, f"{name}.beta_scale"
            assert config.sigma_obs_scale > 0, f"{name}.sigma_obs_scale"


# ============================================================================
# Aggregation Function Tests
# ============================================================================


class TestAggregateSensitivityResults:
    """Tests for aggregate_sensitivity_results function."""

    def test_aggregate_sensitivity_results_returns_dataframe(self, mock_sensitivity_results):
        """Test that aggregate returns a DataFrame."""
        df = aggregate_sensitivity_results(mock_sensitivity_results, metric="elpd")
        assert isinstance(df, pd.DataFrame)

    def test_aggregate_sensitivity_results_columns_elpd(self, mock_sensitivity_results):
        """Test expected columns for ELPD aggregation."""
        df = aggregate_sensitivity_results(mock_sensitivity_results, metric="elpd")

        expected_columns = ["convergence_passed", "divergences", "rhat_max", "ess_bulk_min",
                           "elpd", "elpd_se", "p_loo", "n_high_pareto_k"]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_aggregate_sensitivity_results_rows(self, mock_sensitivity_results):
        """Test that all variants appear as rows."""
        df = aggregate_sensitivity_results(mock_sensitivity_results, metric="elpd")

        assert len(df) == 3
        assert "default" in df.index
        assert "diffuse" in df.index
        assert "informative" in df.index

    def test_aggregate_sensitivity_results_sorted_by_elpd(self, mock_sensitivity_results):
        """Test that results are sorted by ELPD (descending)."""
        df = aggregate_sensitivity_results(mock_sensitivity_results, metric="elpd")

        # Higher ELPD should come first
        elpd_values = df["elpd"].values
        assert elpd_values[0] >= elpd_values[1] >= elpd_values[2]

    def test_aggregate_sensitivity_results_convergence_metric(self, mock_sensitivity_results):
        """Test aggregation with convergence metric."""
        df = aggregate_sensitivity_results(mock_sensitivity_results, metric="convergence")

        assert "convergence_passed" in df.columns
        assert "divergences" in df.columns
        assert "rhat_max" in df.columns

    def test_aggregate_sensitivity_results_empty_dict(self):
        """Test aggregation with empty results dictionary."""
        df = aggregate_sensitivity_results({}, metric="elpd")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_aggregate_sensitivity_results_none_loo(self, mock_convergence, mock_coefficients):
        """Test aggregation when LOO is None."""
        results = {
            "no_loo": SensitivityResult(
                name="no_loo",
                config={},
                idata=None,
                convergence=mock_convergence,
                loo=None,
                crps=None,
                coefficients=mock_coefficients,
            )
        }

        df = aggregate_sensitivity_results(results, metric="elpd")
        assert df.loc["no_loo", "elpd"] is None


class TestCreateCoefficientComparisonDf:
    """Tests for create_coefficient_comparison_df function."""

    def test_create_coefficient_comparison_df_shape(self, mock_sensitivity_results):
        """Test correct shape for forest plot DataFrame."""
        params = ["user_rho", "user_beta[0]"]
        df = create_coefficient_comparison_df(mock_sensitivity_results, params)

        # Should have 3 variants * 2 params = 6 rows
        assert len(df) == 6

    def test_create_coefficient_comparison_df_columns(self, mock_sensitivity_results):
        """Test expected columns for forest plot data."""
        params = ["user_rho"]
        df = create_coefficient_comparison_df(mock_sensitivity_results, params)

        expected_columns = ["variant", "param", "mean", "lower", "upper"]
        assert list(df.columns) == expected_columns

    def test_create_coefficient_comparison_df_values(self, mock_sensitivity_results):
        """Test that values are correctly extracted."""
        params = ["user_rho"]
        df = create_coefficient_comparison_df(mock_sensitivity_results, params)

        # Check default row
        default_row = df[df["variant"] == "default"]
        assert len(default_row) == 1
        assert default_row.iloc[0]["mean"] == 0.15
        assert default_row.iloc[0]["lower"] == 0.09
        assert default_row.iloc[0]["upper"] == 0.21

    def test_create_coefficient_comparison_df_missing_param(self, mock_sensitivity_results):
        """Test handling of missing parameter names."""
        params = ["nonexistent_param"]
        df = create_coefficient_comparison_df(mock_sensitivity_results, params)

        # Should return empty DataFrame for missing params
        assert len(df) == 0

    def test_create_coefficient_comparison_df_empty_coefficients(self, mock_convergence, mock_loo_result):
        """Test handling of empty coefficients DataFrame."""
        results = {
            "empty_coef": SensitivityResult(
                name="empty_coef",
                config={},
                idata=None,
                convergence=mock_convergence,
                loo=mock_loo_result,
                crps=None,
                coefficients=pd.DataFrame(),  # Empty
            )
        }

        params = ["user_rho"]
        df = create_coefficient_comparison_df(results, params)
        assert len(df) == 0


# ============================================================================
# Integration Tests (Skip by default - require model fitting)
# ============================================================================


@pytest.mark.skip(reason="Integration test requires actual model fitting")
class TestRunPriorSensitivityIntegration:
    """Integration tests for run_prior_sensitivity (requires actual models)."""

    def test_run_prior_sensitivity_all_configs(self):
        """Test running sensitivity with all prior configs."""
        pass

    def test_run_prior_sensitivity_custom_configs(self):
        """Test running with custom prior configurations."""
        pass


@pytest.mark.skip(reason="Integration test requires actual model fitting")
class TestRunThresholdSensitivityIntegration:
    """Integration tests for run_threshold_sensitivity (requires data loading)."""

    def test_run_threshold_sensitivity_standard(self):
        """Test standard threshold sensitivity analysis."""
        pass


@pytest.mark.skip(reason="Integration test requires actual model fitting")
class TestRunFeatureAblationIntegration:
    """Integration tests for run_feature_ablation (requires actual models)."""

    def test_run_feature_ablation_standard(self):
        """Test standard feature ablation study."""
        pass
