"""Tests for posterior predictive evaluation helpers."""

import json
from dataclasses import dataclass
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from aoty_pred.pipelines.evaluate import _prepare_test_model_args, evaluate_models


@pytest.fixture
def mock_summary():
    """Minimal training summary for testing."""
    return {
        "artist_to_idx": {"Artist_A": 0, "Artist_B": 1, "Artist_C": 2},
        "n_artists": 3,
        "max_seq": 5,
        "max_albums": 10,
        "min_albums_filter": 2,
        "global_mean_score": 70.0,
        "feature_cols": ["feat_1", "feat_2"],
        "feature_scaler": {
            "mean": [1.0, 2.0],
            "std": [0.5, 1.0],
            "feature_cols": ["feat_1", "feat_2"],
        },
        "priors": {
            "mu_artist_loc": 0.0,
            "mu_artist_scale": 1.0,
            "sigma_artist_scale": 0.5,
            "sigma_rw_scale": 0.1,
            "rho_loc": 0.0,
            "rho_scale": 0.3,
            "beta_loc": 0.0,
            "beta_scale": 1.0,
            "sigma_obs_scale": 1.0,
            "sigma_ref_scale": 1.0,
            "n_exponent_alpha": 2.0,
            "n_exponent_beta": 4.0,
            "n_exponent_loc": 0.0,
            "n_exponent_scale": 1.0,
        },
        "n_exponent": 0.0,
        "learn_n_exponent": False,
        "n_exponent_prior": "logit-normal",
        "n_ref": None,
    }


@pytest.fixture
def mock_test_data():
    """Create aligned test_df and test_features."""
    test_df = pd.DataFrame(
        {
            "Artist": ["Artist_A", "Artist_A", "Artist_B", "Artist_D"],
            "User_Score": [75.0, 80.0, 65.0, 90.0],
            "User_Ratings": [100, 200, 50, 10],
        }
    )
    test_features = pd.DataFrame(
        {
            "feat_1": [1.5, 2.0, 0.5, 3.0],
            "feat_2": [3.0, 4.0, 1.0, 5.0],
            "n_reviews": [100, 200, 50, 10],
        },
        index=test_df.index,
    )
    return test_df, test_features


def test_drops_unknown_artists(mock_test_data, mock_summary):
    """Artists not in artist_to_idx are dropped."""
    test_df, test_features = mock_test_data
    model_args, y_true = _prepare_test_model_args(test_df, test_features, mock_summary)
    # Artist_D is not in mapping, so only 3 rows remain
    assert len(y_true) == 3
    assert model_args["X"].shape[0] == 3


def test_feature_standardization(mock_test_data, mock_summary):
    """Features are standardized using training scaler."""
    test_df, test_features = mock_test_data
    model_args, _ = _prepare_test_model_args(test_df, test_features, mock_summary)
    X = model_args["X"]
    # First feature: (val - 1.0) / 0.5
    # For Artist_A first album: (1.5 - 1.0) / 0.5 = 1.0
    assert abs(X[0, 0] - 1.0) < 1e-5


def test_y_is_none_for_prediction(mock_test_data, mock_summary):
    """model_args['y'] must be None for prediction mode."""
    test_df, test_features = mock_test_data
    model_args, _ = _prepare_test_model_args(test_df, test_features, mock_summary)
    assert model_args["y"] is None


def test_n_artists_from_summary(mock_test_data, mock_summary):
    """n_artists comes from training summary, not test data."""
    test_df, test_features = mock_test_data
    model_args, _ = _prepare_test_model_args(test_df, test_features, mock_summary)
    assert model_args["n_artists"] == 3  # From summary, not from test data


def test_prev_score_fill_with_global_mean(mock_test_data, mock_summary):
    """First album per artist gets global mean as prev_score."""
    test_df, test_features = mock_test_data
    model_args, _ = _prepare_test_model_args(test_df, test_features, mock_summary)
    # Artist_A first album should have prev_score = global_mean_score = 70.0
    assert abs(model_args["prev_score"][0] - 70.0) < 1e-5
    # Artist_B only album should also have global_mean_score
    assert abs(model_args["prev_score"][2] - 70.0) < 1e-5


def test_max_seq_from_summary(mock_test_data, mock_summary):
    """max_seq should come from training summary."""
    test_df, test_features = mock_test_data
    model_args, _ = _prepare_test_model_args(test_df, test_features, mock_summary)
    assert model_args["max_seq"] == 5


def test_n_reviews_as_int32(mock_test_data, mock_summary):
    """n_reviews should be cast to int32."""
    test_df, test_features = mock_test_data
    model_args, _ = _prepare_test_model_args(test_df, test_features, mock_summary)
    assert model_args["n_reviews"].dtype == np.int32


def test_heteroscedastic_config_from_summary(mock_test_data, mock_summary):
    """Heteroscedastic config should come from summary."""
    test_df, test_features = mock_test_data
    model_args, _ = _prepare_test_model_args(test_df, test_features, mock_summary)
    assert model_args["n_exponent"] == 0.0
    assert model_args["learn_n_exponent"] is False
    assert model_args["n_ref"] is None


# ---------------------------------------------------------------------------
# Fixtures / helpers for evaluate_models tests
# ---------------------------------------------------------------------------


class _MockDataArray:
    def __init__(self, values):
        self._values = values

    @property
    def values(self):
        return self._values


class _MockPosterior:
    def __init__(self, data_vars_dict):
        self._data_vars = data_vars_dict

    @property
    def data_vars(self):
        return self._data_vars

    def __getitem__(self, key):
        return _MockDataArray(self._data_vars[key])


class _MockIData:
    def __init__(self, posterior_dict):
        self.posterior = _MockPosterior(posterior_dict)


def _build_mock_idata():
    """Build a mock idata with posterior (2 chains, 5 draws)."""
    rng = np.random.RandomState(42)
    posterior_data = {
        "user_mu_artist": rng.randn(2, 5).astype(np.float32),
        "user_sigma_artist": np.abs(rng.randn(2, 5)).astype(np.float32) + 0.1,
        "user_beta": rng.randn(2, 5, 2).astype(np.float32),
        "user_rho": (rng.randn(2, 5) * 0.3).astype(np.float32),
        "user_sigma_obs": np.abs(rng.randn(2, 5)).astype(np.float32) + 0.1,
        "user_rw_effects": (rng.randn(2, 5, 3, 5) * 0.1).astype(np.float32),
        "user_sigma_rw": np.abs(rng.randn(2, 5) * 0.1).astype(np.float32) + 0.01,
    }
    return _MockIData(posterior_data)


def _build_eval_summary():
    """Build a training summary dict for evaluate_models tests."""
    return {
        "artist_to_idx": {"Artist_A": 0, "Artist_B": 1, "Artist_C": 2},
        "n_artists": 3,
        "max_seq": 5,
        "max_albums": 10,
        "min_albums_filter": 2,
        "global_mean_score": 70.0,
        "feature_cols": ["feat_1", "feat_2"],
        "feature_scaler": {
            "mean": [1.0, 2.0],
            "std": [0.5, 1.0],
            "feature_cols": ["feat_1", "feat_2"],
        },
        "priors": {
            "mu_artist_loc": 0.0,
            "mu_artist_scale": 1.0,
            "sigma_artist_scale": 0.5,
            "sigma_rw_scale": 0.1,
            "rho_loc": 0.0,
            "rho_scale": 0.3,
            "beta_loc": 0.0,
            "beta_scale": 1.0,
            "sigma_obs_scale": 1.0,
            "sigma_ref_scale": 1.0,
            "n_exponent_alpha": 2.0,
            "n_exponent_beta": 4.0,
            "n_exponent_loc": 0.0,
            "n_exponent_scale": 1.0,
        },
        "n_exponent": 0.0,
        "learn_n_exponent": False,
        "n_exponent_prior": "logit-normal",
        "n_ref": None,
    }


# ---------------------------------------------------------------------------
# Tests for evaluate_models
# ---------------------------------------------------------------------------


@dataclass
class _MockConvergenceDiagnostics:
    passed: bool = True
    rhat_max: float = 1.005
    ess_bulk_min: int = 800
    divergences: int = 0
    rhat_threshold: float = 1.01
    ess_threshold: int = 400


@dataclass
class _MockPointMetrics:
    rmse: float = 5.0
    mae: float = 4.0
    r2: float = 0.85
    mean_bias: float = 0.1
    n_observations: int = 3


@dataclass
class _MockCoverageResult:
    nominal: float = 0.90
    empirical: float = 0.87
    interval_width: float = 12.0


@dataclass
class _MockCrpsResult:
    mean_crps: float = 3.2
    n_obs: int = 3


class TestEvaluateModels:
    """Tests for evaluate_models function with mocked external deps."""

    def test_runs_prediction_loop(self, tmp_path):
        """evaluate_models returns dict with diagnostics and metrics."""
        mock_manifest = MagicMock()
        mock_manifest.current = {"user_score": "model.nc"}

        mock_idata = _build_mock_idata()
        summary = _build_eval_summary()

        # Test model args to return from _prepare_test_model_args
        y_true = np.array([75.0, 80.0, 65.0], dtype=np.float32)
        test_model_args = {
            "artist_idx": np.array([0, 0, 1], dtype=np.int32),
            "album_seq": np.array([1, 2, 1], dtype=np.int32),
            "prev_score": np.array([70.0, 75.0, 70.0], dtype=np.float32),
            "X": np.array([[1.0, 1.0], [2.0, 2.0], [0.0, -1.0]], dtype=np.float32),
            "y": None,
            "n_reviews": np.array([100, 200, 50], dtype=np.int32),
            "n_artists": 3,
            "max_seq": 5,
            "n_exponent": 0.0,
            "learn_n_exponent": False,
            "n_exponent_prior": "logit-normal",
            "n_ref": None,
            "priors": MagicMock(),
        }

        # Mock JAX
        mock_jax = MagicMock()
        mock_device = MagicMock()
        mock_jax.devices.return_value = [mock_device]
        mock_jax.default_device.return_value.__enter__ = MagicMock(return_value=None)
        mock_jax.default_device.return_value.__exit__ = MagicMock(return_value=False)

        # Mock Predictive
        mock_predictive_cls = MagicMock()
        rng = np.random.RandomState(99)
        mock_predictive_cls.return_value.return_value = {
            "user_y": rng.randn(10, 3).astype(np.float32)
        }

        # Mock random
        mock_random = MagicMock()
        mock_random.key.return_value = MagicMock()

        # Build mock context
        mock_ctx = MagicMock()

        # Use a real tmp_path for output
        output_dir = tmp_path / "outputs" / "evaluation"

        # Use individual patches instead of patch.multiple
        with (
            patch(
                "aoty_pred.pipelines.evaluate.load_manifest",
                return_value=mock_manifest,
            ),
            patch(
                "aoty_pred.pipelines.evaluate.load_model",
                return_value=mock_idata,
            ),
            patch(
                "aoty_pred.pipelines.evaluate.check_convergence",
                return_value=_MockConvergenceDiagnostics(),
            ),
            patch("aoty_pred.pipelines.evaluate.get_divergence_info"),
            patch(
                "aoty_pred.pipelines.evaluate._prepare_test_model_args",
                return_value=(test_model_args, y_true),
            ),
            patch("aoty_pred.pipelines.evaluate.Predictive", mock_predictive_cls),
            patch("aoty_pred.pipelines.evaluate.jax", mock_jax),
            patch("aoty_pred.pipelines.evaluate.random", mock_random),
            patch(
                "aoty_pred.pipelines.evaluate.compute_point_metrics",
                return_value=_MockPointMetrics(),
            ),
            patch(
                "aoty_pred.pipelines.evaluate.compute_coverage",
                return_value=_MockCoverageResult(),
            ),
            patch(
                "aoty_pred.pipelines.evaluate.compute_crps",
                return_value=_MockCrpsResult(),
            ),
            patch(
                "aoty_pred.pipelines.evaluate.pd.read_parquet",
                return_value=pd.DataFrame(
                    {
                        "Artist": ["Artist_A", "Artist_A", "Artist_B"],
                        "User_Score": [75.0, 80.0, 65.0],
                    }
                ),
            ),
            patch("builtins.open", mock_open(read_data=json.dumps(summary))),
            patch("aoty_pred.pipelines.evaluate.Path") as MockPath,
        ):
            # Make Path() return objects that support mkdir and / operator
            mock_model_dir = MagicMock()
            mock_model_dir.__truediv__ = lambda self, other: tmp_path / other

            mock_output_dir = MagicMock()
            mock_output_dir.mkdir = MagicMock()
            mock_output_dir.__truediv__ = lambda self, other: tmp_path / other

            def path_side_effect(p):
                if p == "models":
                    return mock_model_dir
                if p == "data/features":
                    return MagicMock()
                if p == "data/splits/within_artist_temporal":
                    return MagicMock()
                if p == "outputs/evaluation":
                    return mock_output_dir
                return MagicMock()

            MockPath.side_effect = path_side_effect

            result = evaluate_models(mock_ctx)

        assert isinstance(result, dict)
        assert "diagnostics" in result
        assert "metrics" in result
        assert result["diagnostics"]["passed"] is True
        assert "point_metrics" in result["metrics"]
        assert "calibration" in result["metrics"]
        assert "crps" in result["metrics"]

    def test_raises_without_manifest(self):
        """evaluate_models raises ValueError when no manifest found."""
        mock_ctx = MagicMock()

        with patch(
            "aoty_pred.pipelines.evaluate.load_manifest",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="No trained user_score model"):
                evaluate_models(mock_ctx)
