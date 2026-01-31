"""Model evaluation pipeline.

Runs model evaluation and diagnostics, generating metrics JSON files
for use in publication artifact generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import numpy as np
import pandas as pd
import structlog
from jax import random
from numpyro.infer import Predictive

from aoty_pred.evaluation.calibration import compute_coverage
from aoty_pred.evaluation.metrics import compute_crps, compute_point_metrics
from aoty_pred.models.bayes.diagnostics import check_convergence, get_divergence_info
from aoty_pred.models.bayes.io import load_manifest, load_model
from aoty_pred.models.bayes.model import user_score_model
from aoty_pred.models.bayes.priors import PriorConfig
from aoty_pred.pipelines.predict_next import _extract_posterior_samples
from aoty_pred.pipelines.train_bayes import _apply_max_albums_cap

if TYPE_CHECKING:
    from aoty_pred.pipelines.stages import StageContext

log = structlog.get_logger()


def _prepare_test_model_args(
    test_df: pd.DataFrame,
    test_features: pd.DataFrame,
    summary: dict,
    train_df: pd.DataFrame | None = None,
) -> tuple[dict, np.ndarray]:
    """Build model_args for test data using training summary metadata.

    Maps test observations to training artist indices, applies feature
    standardization using saved training scaler parameters, and constructs
    the model_args dict needed by Predictive.

    Args:
        test_df: Test split DataFrame with Artist, User_Score columns.
        test_features: Engineered features for test data (aligned index).
        summary: Training summary dict loaded from training_summary.json.
        train_df: Optional training DataFrame for seeding album sequences
            and previous scores from training history.

    Returns:
        Tuple of (test_model_args dict, y_true array for kept rows).
    """
    # Drop overlapping columns from test_df before join
    overlap_cols = list(set(test_df.columns) & set(test_features.columns))
    if overlap_cols:
        test_df = test_df.drop(columns=overlap_cols)

    # Validate alignment
    if len(test_df) != len(test_features):
        raise ValueError(
            f"DataFrame length mismatch: test_df={len(test_df)}, "
            f"test_features={len(test_features)}"
        )
    if not test_df.index.equals(test_features.index):
        raise ValueError(
            "Index mismatch between test_df and test_features. "
            f"test_df index: {test_df.index[:5].tolist()}..., "
            f"test_features index: {test_features.index[:5].tolist()}..."
        )

    # Join features
    test_df = test_df.join(test_features, how="left")

    # Map artists to training indices; drop unseen artists
    artist_to_idx = summary["artist_to_idx"]
    test_df = test_df.copy()
    test_df["_artist_idx"] = test_df["Artist"].map(artist_to_idx)

    n_before = len(test_df)
    test_df = test_df.dropna(subset=["_artist_idx"])
    n_dropped = n_before - len(test_df)
    if n_dropped > 0:
        log.info("test_unknown_artists_dropped", n_dropped=n_dropped, n_remaining=len(test_df))

    artist_idx = test_df["_artist_idx"].values.astype(np.int32)

    # Compute per-artist training history offsets
    if train_df is not None:
        train_artist_last_seq = (
            train_df.groupby("Artist").cumcount().groupby(train_df["Artist"]).last() + 1
        )  # 1-indexed
        train_artist_last_score = train_df.groupby("Artist")["User_Score"].last()
    else:
        train_artist_last_seq = pd.Series(dtype=int)
        train_artist_last_score = pd.Series(dtype=float)

    # Album sequence: offset by training count per artist
    raw_seq = test_df.groupby("Artist").cumcount() + 1
    train_offset = test_df["Artist"].map(train_artist_last_seq).fillna(0).astype(int)
    album_seq = (raw_seq + train_offset).values

    # Apply min_albums_filter clamping based on test-set counts
    min_albums_filter = summary.get("min_albums_filter", 2)
    artist_counts = test_df.groupby("Artist").size()
    below_threshold = test_df["Artist"].map(artist_counts < min_albums_filter).values
    album_seq = np.where(below_threshold, 1, album_seq)

    # Apply max_albums cap
    max_albums = summary.get("max_albums", 50)
    max_seq_train = summary["max_seq"]

    # Compute total (train+test) album counts per artist for max_albums cap.
    # album_seq already includes training offsets, so counts must match.
    test_counts = pd.Series(artist_idx).value_counts().sort_index()
    test_counts = test_counts.reindex(range(summary["n_artists"]), fill_value=0)
    if train_df is not None:
        artist_to_idx = summary["artist_to_idx"]
        train_idx = train_df["Artist"].map(artist_to_idx).dropna().astype(int)
        train_counts = train_idx.value_counts().sort_index()
        train_counts = train_counts.reindex(range(summary["n_artists"]), fill_value=0)
        artist_album_counts = train_counts + test_counts
    else:
        artist_album_counts = test_counts

    # Temporarily build a dict for _apply_max_albums_cap
    temp_args = {
        "artist_idx": artist_idx,
        "album_seq": album_seq,
    }
    temp_args = _apply_max_albums_cap(temp_args, max_albums, artist_album_counts)
    album_seq = temp_args["album_seq"]

    # Clamp album_seq to not exceed training max_seq
    album_seq = np.minimum(album_seq, max_seq_train).astype(np.int32)

    # Previous score: seed first test album with last training score
    global_mean = summary["global_mean_score"]
    test_df["_prev_score"] = test_df.groupby("Artist")["User_Score"].shift(1)
    # For first test album per artist, use last training score if available
    first_mask = test_df["_prev_score"].isna()
    train_last = test_df["Artist"].map(train_artist_last_score)
    test_df.loc[first_mask, "_prev_score"] = train_last[first_mask].fillna(global_mean)
    prev_score = test_df["_prev_score"].values.astype(np.float32)

    # Feature matrix: fill NaN, standardize using training scaler
    feature_cols = summary["feature_cols"]
    test_df[feature_cols] = test_df[feature_cols].fillna(0)
    X = test_df[feature_cols].values.astype(np.float32)

    scaler = summary["feature_scaler"]
    X_mean = np.array(scaler["mean"], dtype=np.float32)
    X_std = np.array(scaler["std"], dtype=np.float32)
    X = (X - X_mean) / X_std

    # Extract n_reviews
    if "n_reviews" in test_features.columns:
        n_reviews_raw = test_df["n_reviews"].values
    elif "User_Ratings" in test_df.columns:
        n_reviews_raw = test_df["User_Ratings"].values
    else:
        raise ValueError("No n_reviews or User_Ratings column found in test data")

    # Drop invalid n_reviews rows
    invalid_mask = pd.isna(n_reviews_raw) | (n_reviews_raw <= 0)
    valid_mask = ~invalid_mask

    if invalid_mask.sum() > 0:
        log.info("test_invalid_n_reviews_dropped", n_dropped=int(invalid_mask.sum()))
        artist_idx = artist_idx[valid_mask]
        album_seq = album_seq[valid_mask]
        prev_score = prev_score[valid_mask]
        X = X[valid_mask]
        n_reviews_raw = n_reviews_raw[valid_mask]
        test_df = test_df[valid_mask]

    n_reviews = n_reviews_raw.astype(np.int32)

    # Extract y_true
    y_true = test_df["User_Score"].values.astype(np.float32)

    # Build model_args for Predictive
    test_model_args = {
        "artist_idx": artist_idx,
        "album_seq": album_seq,
        "prev_score": prev_score,
        "X": X,
        "y": None,  # prediction mode
        "n_reviews": n_reviews,
        "n_artists": summary["n_artists"],
        "max_seq": max_seq_train,
        "n_exponent": summary.get("n_exponent", 0.0),
        "learn_n_exponent": summary.get("learn_n_exponent", False),
        "n_exponent_prior": summary.get("n_exponent_prior", "logit-normal"),
        "n_ref": summary.get("n_ref"),
        "priors": PriorConfig(**summary["priors"]),
    }

    return test_model_args, y_true


def evaluate_models(ctx: StageContext) -> dict:
    """Evaluate fitted models on test data.

    Computes convergence diagnostics, point prediction metrics,
    calibration metrics, and CRPS for the fitted model.

    Args:
        ctx: Stage context with run configuration.

    Returns:
        Dictionary with evaluation results.
    """
    log.info("evaluation_pipeline_start")

    # Load model manifest to find current model
    model_dir = Path("models")
    manifest = load_manifest(model_dir)

    if manifest is None or "user_score" not in manifest.current:
        raise ValueError("No trained user_score model found in models/manifest.json")

    model_filename = manifest.current["user_score"]
    model_path = model_dir / model_filename

    log.info("loading_model", path=str(model_path))
    idata = load_model(model_path)

    # Check convergence
    log.info("checking_convergence")
    diagnostics = check_convergence(idata)
    _ = get_divergence_info(idata)  # Used for side-effect logging

    diagnostics_result = {
        "passed": diagnostics.passed,
        "rhat_max": float(diagnostics.rhat_max),
        "ess_bulk_min": float(diagnostics.ess_bulk_min),
        "divergences": int(diagnostics.divergences),
        "rhat_threshold": float(diagnostics.rhat_threshold),
        "ess_threshold": int(diagnostics.ess_threshold),
    }

    log.info(
        "convergence_results",
        passed=diagnostics.passed,
        rhat_max=diagnostics.rhat_max,
        ess_bulk_min=diagnostics.ess_bulk_min,
    )

    # Load test data
    features_dir = Path("data/features")
    splits_dir = Path("data/splits/within_artist_temporal")

    test_features = pd.read_parquet(features_dir / "test_features.parquet")
    test_df = pd.read_parquet(splits_dir / "test.parquet")
    train_df = pd.read_parquet(splits_dir / "train.parquet")

    log.info("test_data_loaded", n_test=len(test_df))

    # Load training summary for model metadata
    summary_path = model_dir / "training_summary.json"
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    # Prepare test model_args using training summary metadata
    test_model_args, y_true = _prepare_test_model_args(
        test_df, test_features, summary, train_df=train_df
    )

    log.info("test_model_args_prepared", n_test_kept=len(y_true))

    # Extract posterior samples from idata into flat dict for Predictive
    posterior_samples = _extract_posterior_samples(idata)

    n_total_samples = next(iter(posterior_samples.values())).shape[0]
    log.info("posterior_samples_extracted", n_total_samples=n_total_samples)

    # Force CPU for all prediction work
    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        # Run Predictive in chunks to control memory
        batch_size = 500
        y_pred_chunks: list[np.ndarray] = []

        for start in range(0, n_total_samples, batch_size):
            end = min(start + batch_size, n_total_samples)
            batch_samples = {k: v[start:end] for k, v in posterior_samples.items()}

            predictive = Predictive(
                user_score_model,
                posterior_samples=batch_samples,
                batch_ndims=1,
            )
            rng_key = random.key(start)
            preds = predictive(rng_key, **test_model_args)

            # Find the y key (prefixed as "user_y")
            y_key = next(k for k in preds if k.endswith("_y"))
            y_pred_chunks.append(np.asarray(preds[y_key]))

        y_pred_samples = np.concatenate(y_pred_chunks, axis=0)

    log.info(
        "posterior_predictive_generated",
        shape=y_pred_samples.shape,
        n_samples=y_pred_samples.shape[0],
        n_obs=y_pred_samples.shape[1],
    )

    # Compute metrics
    y_pred_mean = np.mean(y_pred_samples, axis=0)

    log.info("computing_point_metrics")
    point_metrics = compute_point_metrics(y_true, y_pred_mean)

    metrics_result = {
        "rmse": float(point_metrics.rmse),
        "mae": float(point_metrics.mae),
        "r2": float(point_metrics.r2),
        "mean_bias": float(point_metrics.mean_bias),
        "n_observations": int(point_metrics.n_observations),
    }

    log.info(
        "point_metrics",
        rmse=point_metrics.rmse,
        mae=point_metrics.mae,
        r2=point_metrics.r2,
    )

    # Calibration metrics
    log.info("computing_calibration")
    coverage_90 = compute_coverage(y_true, y_pred_samples, prob=0.90)
    coverage_50 = compute_coverage(y_true, y_pred_samples, prob=0.50)

    calibration_result = {
        "coverage_90": float(coverage_90.empirical),
        "coverage_50": float(coverage_50.empirical),
        "expected_90": 0.90,
        "expected_50": 0.50,
        "interval_width_90": float(coverage_90.interval_width),
        "interval_width_50": float(coverage_50.interval_width),
    }

    log.info(
        "calibration_metrics",
        coverage_90=calibration_result["coverage_90"],
        coverage_50=calibration_result["coverage_50"],
    )

    # CRPS
    log.info("computing_crps")
    crps_result = compute_crps(y_true, y_pred_samples)
    log.info("crps_computed", mean_crps=crps_result.mean_crps)

    # Save results
    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save diagnostics
    diagnostics_path = output_dir / "diagnostics.json"
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics_result, f, indent=2)

    # Save metrics
    metrics_full = {
        "model": "user_score",
        "model_path": str(model_path),
        "n_test": len(y_true),
        "point_metrics": metrics_result,
        "calibration": calibration_result,
        "crps": {
            "mean_crps": float(crps_result.mean_crps),
            "n_obs": crps_result.n_obs,
        },
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_full, f, indent=2)

    log.info(
        "evaluation_pipeline_complete",
        diagnostics_path=str(diagnostics_path),
        metrics_path=str(metrics_path),
    )

    return {
        "diagnostics": diagnostics_result,
        "metrics": metrics_full,
    }
