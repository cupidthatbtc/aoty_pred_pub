"""Tests for posterior predictive evaluation helpers."""

import numpy as np
import pandas as pd
import pytest

from aoty_pred.pipelines.evaluate import _prepare_test_model_args


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
