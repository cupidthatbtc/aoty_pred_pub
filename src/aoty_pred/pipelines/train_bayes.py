"""Bayesian model training pipeline.

Fits NumPyro models on training data with configured MCMC parameters,
saves model artifacts, and handles convergence checking.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import structlog

from aoty_pred.models.bayes.fit import fit_model, MCMCConfig
from aoty_pred.models.bayes.io import save_model
from aoty_pred.models.bayes.model import user_score_model
from aoty_pred.models.bayes.priors import PriorConfig, get_default_priors
from aoty_pred.models.bayes.diagnostics import check_convergence
from aoty_pred.pipelines.errors import ConvergenceError
from aoty_pred.utils.hashing import hash_dataframe

if TYPE_CHECKING:
    from aoty_pred.pipelines.stages import StageContext

log = structlog.get_logger()


def prepare_model_data(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """Prepare data for NumPyro model fitting.

    Creates the arrays needed by the Bayesian model including artist indices,
    album sequences, and feature matrix.

    Args:
        train_df: Training data with features and target.
        feature_cols: List of feature column names.

    Returns:
        Dictionary with model arguments.
    """
    # Create artist index mapping
    artists = train_df["Artist"].unique()
    artist_to_idx = {a: i for i, a in enumerate(artists)}
    artist_idx = train_df["Artist"].map(artist_to_idx).values

    # Album sequence (within artist)
    album_seq = train_df.groupby("Artist").cumcount().values

    # Previous score (shifted within artist, using mean for first album)
    train_df = train_df.copy()
    train_df["prev_score"] = train_df.groupby("Artist")["User_Score"].shift(1)
    global_mean = train_df["User_Score"].mean()
    train_df["prev_score"] = train_df["prev_score"].fillna(global_mean)
    prev_score = train_df["prev_score"].values

    # Feature matrix
    X = train_df[feature_cols].values.astype(np.float32)

    # Target
    y = train_df["User_Score"].values.astype(np.float32)

    # Max sequence for JAX tracing, capped to limit rw_innovations tensor size.
    # This prevents memory issues for prolific artists. Albums beyond the cap
    # will use the same artist effect as the max position, which may slightly
    # reduce temporal resolution for those rare cases but keeps memory bounded.
    # Default cap is 50, configurable via --max-albums CLI option.
    uncapped_max_seq = int(album_seq.max()) + 1

    # Compute album counts per artist (indexed by artist_idx, not artist name)
    artist_album_counts = pd.Series(artist_idx).value_counts().sort_index()

    return {
        "artist_idx": artist_idx,
        "album_seq": album_seq,
        "prev_score": prev_score,
        "X": X,
        "y": y,
        "n_artists": len(artists),
        "uncapped_max_seq": uncapped_max_seq,
        "artist_album_counts": artist_album_counts,
    }


def _apply_max_albums_cap(
    model_args: dict,
    max_albums_cap: int,
    artist_album_counts: pd.Series,
) -> dict:
    """Apply max_albums cap to model arguments, keeping most recent albums.

    For artists with more than max_albums_cap albums, renumbers so that the
    most recent albums get distinct positions (0 to max_albums_cap-1) and
    older albums share position 0. This is appropriate because:
    - Recent albums are more predictive of future albums
    - No leakage since album_seq is calculated on training data only

    Args:
        model_args: Dictionary from prepare_model_data.
        max_albums_cap: Maximum albums per artist (from ctx.max_albums).
        artist_album_counts: Series mapping artist index to album count.

    Returns:
        Updated model_args with adjusted album_seq and max_seq.
    """
    uncapped_max_seq = model_args.pop("uncapped_max_seq")
    album_seq = model_args["album_seq"]
    artist_idx = model_args["artist_idx"]

    # For each artist, compute offset to shift album_seq so most recent albums
    # get positions 0 to max_albums_cap-1, and older albums share position 0
    # offset = max(0, n_albums - max_albums_cap)
    offsets = np.maximum(0, artist_album_counts.values - max_albums_cap)

    # Apply per-artist offset: new_seq = max(0, original_seq - offset[artist])
    artist_offsets = offsets[artist_idx]
    album_seq = np.maximum(0, album_seq - artist_offsets).astype(np.int32)

    max_seq = min(uncapped_max_seq, max_albums_cap)

    n_capped_artists = (artist_album_counts > max_albums_cap).sum()
    if n_capped_artists > 0:
        log.info(
            "max_albums_applied",
            max_albums=max_albums_cap,
            artists_capped=int(n_capped_artists),
            message=f"Using {max_albums_cap} most recent albums per artist; older albums share position 0",
        )

    model_args["album_seq"] = album_seq
    model_args["max_seq"] = max_seq
    return model_args


def train_models(ctx: "StageContext") -> dict:
    """Train Bayesian models on feature data.

    Fits the user score model using MCMC, checks convergence,
    and saves model artifacts.

    Args:
        ctx: Stage context with run configuration.

    Returns:
        Dictionary with training results and paths.

    Raises:
        ConvergenceError: If strict mode and divergences > 0.
    """
    log.info("train_pipeline_start", seed=ctx.seed, strict=ctx.strict, max_albums=ctx.max_albums)

    # Load feature data
    features_dir = Path("data/features")
    train_features = pd.read_parquet(features_dir / "train_features.parquet")

    # Load original training data for artist/target info
    splits_dir = Path("data/splits/within_artist_temporal")
    train_df = pd.read_parquet(splits_dir / "train.parquet")

    # Drop columns that overlap with engineered features
    overlap_cols = list(set(train_df.columns) & set(train_features.columns))
    if overlap_cols:
        train_df = train_df.drop(columns=overlap_cols)

    # Merge features with original data
    train_df = train_df.join(train_features, how="left")

    # Handle NaN values in features (fill with 0 for numeric stability)
    feature_cols = list(train_features.columns)
    nan_count = train_df[feature_cols].isna().sum().sum()
    if nan_count > 0:
        log.info("filling_nan_values", nan_count=nan_count)
        train_df[feature_cols] = train_df[feature_cols].fillna(0)

    log.info(
        "data_loaded",
        train_rows=len(train_df),
        n_features=len(train_features.columns),
    )

    # Prepare model data
    model_args = prepare_model_data(train_df, feature_cols)

    # Apply max_albums cap from CLI/config (uses most recent albums per artist)
    artist_album_counts = model_args.pop("artist_album_counts")
    model_args = _apply_max_albums_cap(model_args, ctx.max_albums, artist_album_counts)

    log.info(
        "model_data_prepared",
        n_artists=model_args["n_artists"],
        n_observations=len(model_args["y"]),
        n_features=model_args["X"].shape[1],
        max_seq=model_args["max_seq"],
    )

    # Configure MCMC
    mcmc_config = MCMCConfig(
        num_warmup=1000,
        num_samples=1000,
        num_chains=4,
        seed=ctx.seed,
        target_accept_prob=0.8,
        max_tree_depth=10,
    )

    # Get priors
    priors = get_default_priors()

    # Fit model
    log.info("fitting_model", model="user_score")
    fit_result = fit_model(
        model=user_score_model,
        model_args=model_args,
        config=mcmc_config,
        progress_bar=ctx.verbose,
    )

    log.info(
        "model_fitted",
        divergences=fit_result.divergences,
        runtime_seconds=fit_result.runtime_seconds,
        gpu_info=fit_result.gpu_info,
    )

    # Check convergence
    diagnostics = check_convergence(fit_result.idata)

    log.info(
        "convergence_check",
        passed=diagnostics.passed,
        rhat_max=diagnostics.rhat_max,
        ess_bulk_min=diagnostics.ess_bulk_min,
        divergences=diagnostics.divergences,
    )

    # Handle strict mode
    if ctx.strict and fit_result.divergences > 0:
        raise ConvergenceError(
            f"Model had {fit_result.divergences} divergent transitions. "
            "Re-run without --strict or increase target_accept_prob.",
            stage="train",
        )

    if ctx.strict and not diagnostics.passed:
        raise ConvergenceError(
            f"Convergence diagnostics failed: "
            f"rhat_max={diagnostics.rhat_max:.4f}, ess_bulk_min={diagnostics.ess_bulk_min:.0f}",
            stage="train",
        )

    # Compute data hash for reproducibility
    data_hash = hash_dataframe(train_df)

    # Save model
    model_dir = Path("models")
    model_path, manifest = save_model(
        fit_result=fit_result,
        model_type="user_score",
        priors=priors,
        data_hash=data_hash,
        output_dir=model_dir,
    )

    log.info("model_saved", path=str(model_path))

    # Save training summary
    summary = {
        "model_type": "user_score",
        "model_path": str(model_path),
        "mcmc_config": mcmc_config.to_dict(),
        "priors": asdict(priors),
        "data_hash": data_hash,
        "n_observations": len(model_args["y"]),
        "n_artists": model_args["n_artists"],
        "n_features": model_args["X"].shape[1],
        "divergences": fit_result.divergences,
        "runtime_seconds": fit_result.runtime_seconds,
        "diagnostics": {
            "passed": diagnostics.passed,
            "rhat_max": float(diagnostics.rhat_max),
            "ess_bulk_min": float(diagnostics.ess_bulk_min),
        },
    }

    summary_path = model_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info("train_pipeline_complete", summary_path=str(summary_path))

    return summary
