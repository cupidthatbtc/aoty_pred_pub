"""Bayesian model training pipeline.

Fits NumPyro models on training data with configured MCMC parameters,
saves model artifacts, and handles convergence checking.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import arviz as az
import jax.numpy as jnp
import numpy as np
import pandas as pd
import structlog

from aoty_pred.models.bayes.diagnostics import check_convergence
from aoty_pred.models.bayes.fit import MCMCConfig, fit_model
from aoty_pred.models.bayes.io import save_model
from aoty_pred.models.bayes.model import compute_sigma_scaled, user_score_model
from aoty_pred.models.bayes.priors import PriorConfig
from aoty_pred.pipelines.errors import ConvergenceError
from aoty_pred.utils.hashing import hash_dataframe

if TYPE_CHECKING:
    from aoty_pred.pipelines.stages import StageContext

log = structlog.get_logger()


def load_training_data(
    features_path: Path,
    splits_path: Path,
    min_albums_filter: int = 2,
) -> tuple[dict, list[str], pd.DataFrame]:
    """Load training data and prepare model arguments.

    Loads feature and split parquet files, merges them, fills NaN values,
    and prepares the model_args dictionary for MCMC fitting.

    Args:
        features_path: Path to train_features.parquet.
        splits_path: Path to train.parquet (splits).
        min_albums_filter: Minimum albums for dynamic effects.

    Returns:
        Tuple of (model_args dict, feature_cols list, merged train_df).
    """
    train_features = pd.read_parquet(features_path)
    train_df = pd.read_parquet(splits_path)

    # Drop columns that overlap with engineered features
    overlap_cols = list(set(train_df.columns) & set(train_features.columns))
    if overlap_cols:
        train_df = train_df.drop(columns=overlap_cols)

    # Validate DataFrame alignment before join
    if len(train_df) != len(train_features):
        raise ValueError(
            f"DataFrame length mismatch: train_df has {len(train_df)} rows, "
            f"train_features has {len(train_features)} rows. "
            "Ensure both files contain matching records."
        )
    if not train_df.index.equals(train_features.index):
        raise ValueError(
            "DataFrame index mismatch: train_df and train_features have different indices. "
            "Ensure both files are aligned before joining."
        )

    # Merge features with original data
    train_df = train_df.join(train_features, how="left")

    # Handle NaN values in features (fill with 0 for numeric stability)
    feature_cols = list(train_features.columns)
    train_df[feature_cols] = train_df[feature_cols].fillna(0)

    # Prepare model data
    model_args, valid_mask = prepare_model_data(
        train_df,
        feature_cols,
        min_albums_filter=min_albums_filter,
    )

    # Apply valid_mask to train_df so it matches the filtered model arrays
    train_df = train_df[valid_mask].copy()

    return model_args, feature_cols, train_df


def prepare_model_data(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    min_albums_filter: int = 2,
) -> tuple[dict, np.ndarray]:
    """Prepare data for NumPyro model fitting.

    Creates the arrays needed by the Bayesian model including artist indices,
    album sequences, and feature matrix.

    Args:
        train_df: Training data with features and target.
        feature_cols: List of feature column names.
        min_albums_filter: Minimum albums for dynamic effects. Artists with
            fewer albums have all their albums treated as sequence 1 (static effect only).

    Returns:
        Tuple of (model_args dict, valid_mask boolean array indicating retained rows).
    """
    # Create artist index mapping
    artists = train_df["Artist"].unique()
    artist_to_idx = {a: i for i, a in enumerate(artists)}
    artist_idx = train_df["Artist"].map(artist_to_idx).values

    # Album sequence (within artist, 1-indexed to match model expectations)
    album_seq = (train_df.groupby("Artist").cumcount() + 1).values

    # Apply min_albums_filter: artists below threshold get static effect only
    # by clamping their album_seq to 1
    artist_counts = train_df.groupby("Artist").size()
    below_threshold = train_df["Artist"].map(artist_counts < min_albums_filter).values
    album_seq = np.where(below_threshold, 1, album_seq)

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

    # Extract n_reviews for heteroscedastic noise
    # Keep as raw values (may be float with NaN) for proper NaN detection before int cast
    if "n_reviews" in train_df.columns:
        n_reviews_raw = train_df["n_reviews"].values
    else:
        # Fallback: if n_reviews not in features, try User_Ratings from source
        if "User_Ratings" in train_df.columns:
            n_reviews_raw = train_df["User_Ratings"].values
        else:
            raise ValueError(
                "n_reviews column not found. Feature parquet must include n_reviews "
                "or source data must include User_Ratings."
            )

    # Validate n_reviews: identify missing or invalid values BEFORE int cast
    # NaN cannot be represented in int32, so detection must happen on raw values
    invalid_mask = pd.isna(n_reviews_raw) | (n_reviews_raw <= 0)
    n_invalid = invalid_mask.sum()

    # Track which rows are valid (returned to caller for DataFrame filtering)
    valid_mask = ~invalid_mask

    if n_invalid > 0:
        invalid_pct = n_invalid / len(n_reviews_raw) * 100
        if invalid_pct > 50:
            raise ValueError(
                f"Too many invalid n_reviews: {n_invalid}/{len(n_reviews_raw)} "
                f"({invalid_pct:.1f}%). Check source data for missing User_Ratings."
            )
        # Log warning about rows that will be dropped
        log.warning(
            "n_reviews_invalid_rows",
            n_invalid=n_invalid,
            pct_invalid=round(invalid_pct, 1),
            action="dropping_invalid_rows",
        )
        # Filter out invalid rows from all arrays
        n_reviews_raw = n_reviews_raw[valid_mask]
        y = y[valid_mask]
        X = X[valid_mask]
        artist_idx = artist_idx[valid_mask]
        album_seq = album_seq[valid_mask]
        prev_score = prev_score[valid_mask]

    # Cast to int32 AFTER filtering (NaN-free at this point)
    n_reviews = n_reviews_raw.astype(np.int32)

    # Compute album counts per artist (indexed by artist_idx, not artist name)
    artist_album_counts = pd.Series(artist_idx).value_counts().sort_index()
    # Reindex to full range so _apply_max_albums_cap doesn't get IndexError
    artist_album_counts = artist_album_counts.reindex(range(len(artists)), fill_value=0)

    model_args = {
        "artist_idx": artist_idx,
        "album_seq": album_seq,
        "prev_score": prev_score,
        "X": X,
        "y": y,
        "n_reviews": n_reviews,
        "n_artists": len(artists),
        "artist_album_counts": artist_album_counts,
    }
    return model_args, valid_mask


def _apply_max_albums_cap(
    model_args: dict,
    max_albums_cap: int,
    artist_album_counts: pd.Series,
) -> dict:
    """Apply max_albums cap to model arguments, keeping most recent albums.

    For artists with more than max_albums_cap albums, renumbers so that the
    most recent albums get distinct positions (1 to max_albums_cap) and
    older albums share position 1. This is appropriate because:
    - Recent albums are more predictive of future albums
    - No leakage since album_seq is calculated on training data only

    Args:
        model_args: Dictionary from prepare_model_data.
        max_albums_cap: Maximum albums per artist (from ctx.max_albums).
        artist_album_counts: Series mapping artist index to album count.

    Returns:
        Updated model_args with adjusted album_seq and max_seq.
    """
    # Guard against non-positive max_albums_cap to ensure valid shapes
    max_albums_cap = max(1, int(max_albums_cap))

    album_seq = model_args["album_seq"]
    artist_idx = model_args["artist_idx"]

    # For each artist, compute offset to shift album_seq so most recent albums
    # get positions 1 to max_albums_cap, and older albums share position 1
    # offset = max(0, n_albums - max_albums_cap)
    offsets = np.maximum(0, artist_album_counts.values - max_albums_cap)

    # Apply per-artist offset: new_seq = max(1, original_seq - offset[artist])
    artist_offsets = offsets[artist_idx]
    album_seq = np.maximum(1, album_seq - artist_offsets).astype(np.int32)

    # Compute max_seq from actual capped album_seq values for consistency.
    # Since album_seq is 1-indexed and model converts to 0-indexed, max_seq = album_seq.max().
    max_seq = int(album_seq.max())

    n_capped_artists = (artist_album_counts > max_albums_cap).sum()
    if n_capped_artists > 0:
        log.info(
            "max_albums_applied",
            max_albums=max_albums_cap,
            artists_capped=int(n_capped_artists),
            message=f"Using {max_albums_cap} most recent albums per artist",
        )

    model_args["album_seq"] = album_seq
    model_args["max_seq"] = max_seq
    return model_args


def train_models(
    ctx: "StageContext",
    features_path: Path | None = None,
    splits_path: Path | None = None,
) -> dict:
    """Train Bayesian models on feature data.

    Fits the user score model using MCMC, checks convergence,
    and saves model artifacts.

    Args:
        ctx: Stage context with run configuration.
        features_path: Optional path to features parquet. Defaults to
            data/features/train_features.parquet.
        splits_path: Optional path to splits parquet. Defaults to
            data/splits/within_artist_temporal/train.parquet.

    Returns:
        Dictionary with training results and paths.

    Raises:
        ConvergenceError: If strict mode and divergences > 0.
    """
    log.info(
        "train_pipeline_start",
        seed=ctx.seed,
        strict=ctx.strict,
        max_albums=ctx.max_albums,
        min_albums_filter=ctx.min_albums_filter,
        num_chains=ctx.num_chains,
        num_samples=ctx.num_samples,
        num_warmup=ctx.num_warmup,
        target_accept=ctx.target_accept,
    )

    # Load training data using shared function
    features_path = features_path or Path("data/features/train_features.parquet")
    splits_path = splits_path or Path("data/splits/within_artist_temporal/train.parquet")

    model_args, feature_cols, train_df = load_training_data(
        features_path=features_path,
        splits_path=splits_path,
        min_albums_filter=ctx.min_albums_filter,
    )

    # Compute artists below threshold for metadata
    artist_counts = train_df.groupby("Artist").size()
    n_below_threshold = (artist_counts < ctx.min_albums_filter).sum()

    log.info(
        "data_loaded",
        train_rows=len(train_df),
        n_features=len(feature_cols),
    )

    # Apply max_albums cap from CLI/config (uses most recent albums per artist)
    artist_album_counts = model_args.pop("artist_album_counts")
    model_args = _apply_max_albums_cap(model_args, ctx.max_albums, artist_album_counts)

    # Log n_reviews statistics for diagnostics
    n_reviews = model_args["n_reviews"]
    log.info(
        "n_reviews_distribution",
        min=int(np.min(n_reviews)),
        max=int(np.max(n_reviews)),
        median=int(np.median(n_reviews)),
        mean=float(np.mean(n_reviews)),
    )

    log.info(
        "model_data_prepared",
        n_artists=model_args["n_artists"],
        n_observations=len(model_args["y"]),
        n_features=model_args["X"].shape[1],
        max_seq=model_args["max_seq"],
        n_reviews_shape=model_args["n_reviews"].shape,
    )

    # Add heteroscedastic noise configuration to model_args
    model_args["n_exponent"] = ctx.n_exponent
    model_args["learn_n_exponent"] = ctx.learn_n_exponent
    model_args["n_exponent_prior"] = ctx.n_exponent_prior

    # Log heteroscedastic mode
    if ctx.learn_n_exponent:
        if ctx.n_exponent_prior == "beta":
            log.info(
                "heteroscedastic_mode",
                mode="learned",
                prior_type="beta",
                prior_alpha=ctx.n_exponent_alpha,
                prior_beta=ctx.n_exponent_beta,
            )
        else:
            log.info(
                "heteroscedastic_mode",
                mode="learned",
                prior_type=ctx.n_exponent_prior,
            )
    elif ctx.n_exponent != 0.0:
        log.info("heteroscedastic_mode", mode="fixed", exponent=ctx.n_exponent)
    else:
        log.info("heteroscedastic_mode", mode="homoscedastic")

    # Configure MCMC from CLI args
    mcmc_config = MCMCConfig(
        num_warmup=ctx.num_warmup,
        num_samples=ctx.num_samples,
        num_chains=ctx.num_chains,
        seed=ctx.seed,
        target_accept_prob=ctx.target_accept,
        chain_method=ctx.chain_method,
        # max_tree_depth uses MCMCConfig default (12) for complex posterior geometry
    )

    # Get priors with heteroscedastic config from CLI
    priors = PriorConfig(
        n_exponent_alpha=ctx.n_exponent_alpha,
        n_exponent_beta=ctx.n_exponent_beta,
    )
    model_args["priors"] = priors

    # Fit model
    log.info("fitting_model", model="user_score")
    fit_result = fit_model(
        model=user_score_model,
        model_args=model_args,
        config=mcmc_config,
        progress_bar=True,  # Always show MCMC progress for real-time feedback
        exclude_from_idata=("user_rw_innovations",),  # Large tensor, exclude to prevent OOM
    )

    log.info(
        "model_fitted",
        divergences=fit_result.divergences,
        runtime_seconds=fit_result.runtime_seconds,
        gpu_info=fit_result.gpu_info,
    )

    # Check convergence using CLI-provided thresholds
    diagnostics = check_convergence(
        fit_result.idata,
        rhat_threshold=ctx.rhat_threshold,
        ess_threshold=ctx.ess_threshold,
        allow_divergences=ctx.allow_divergences,
    )

    log.info(
        "convergence_check",
        passed=diagnostics.passed,
        rhat_max=diagnostics.rhat_max,
        rhat_threshold=ctx.rhat_threshold,
        ess_bulk_min=diagnostics.ess_bulk_min,
        ess_threshold=ctx.ess_threshold,
        divergences=diagnostics.divergences,
        allow_divergences=ctx.allow_divergences,
    )

    # Handle strict mode
    # Note: allow_divergences is already passed to check_convergence above,
    # so diagnostics.passed accounts for it. But we need to check divergences
    # separately when strict=True and allow_divergences=False.
    if ctx.strict and fit_result.divergences > 0 and not ctx.allow_divergences:
        raise ConvergenceError(
            f"Model had {fit_result.divergences} divergent transitions. "
            "Re-run without --strict, use --allow-divergences, or increase --target-accept.",
            stage="train",
        )

    if ctx.strict and not diagnostics.passed:
        ess_thresh = ctx.ess_threshold * ctx.num_chains
        raise ConvergenceError(
            f"Convergence failed: rhat_max={diagnostics.rhat_max:.4f} "
            f"(thresh {ctx.rhat_threshold}), ess_min={diagnostics.ess_bulk_min:.0f} "
            f"(thresh {ess_thresh})",
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
        "convergence_thresholds": {
            "rhat_threshold": ctx.rhat_threshold,
            "ess_threshold": ctx.ess_threshold,
            "allow_divergences": ctx.allow_divergences,
        },
        "min_albums_filter": ctx.min_albums_filter,
        "n_artists_below_threshold": int(n_below_threshold),
        "priors": asdict(priors),
        "data_hash": data_hash,
        "n_observations": len(model_args["y"]),
        "n_artists": model_args["n_artists"],
        "n_features": model_args["X"].shape[1],
        "n_reviews_stats": {
            "min": int(np.min(model_args["n_reviews"])),
            "max": int(np.max(model_args["n_reviews"])),
            "median": int(np.median(model_args["n_reviews"])),
            "mean": float(np.mean(model_args["n_reviews"])),
        },
        "divergences": fit_result.divergences,
        "runtime_seconds": fit_result.runtime_seconds,
        "diagnostics": {
            "passed": diagnostics.passed,
            "rhat_max": float(diagnostics.rhat_max),
            "ess_bulk_min": float(diagnostics.ess_bulk_min),
            "rhat_threshold": float(diagnostics.rhat_threshold),
            "ess_threshold": int(diagnostics.ess_threshold),
        },
    }

    # Add heteroscedastic noise details to summary
    if ctx.learn_n_exponent:
        # Extract n_exponent posterior
        n_exp_samples = fit_result.idata.posterior["user_n_exponent"].values.flatten()
        n_exp_mean = float(np.mean(n_exp_samples))
        n_exp_std = float(np.std(n_exp_samples))

        # Compute 94% HDI
        hdi = az.hdi(fit_result.idata, var_names=["user_n_exponent"], hdi_prob=0.94)
        hdi_low = float(hdi["user_n_exponent"].values[0])
        hdi_high = float(hdi["user_n_exponent"].values[1])

        # Get ESS and R-hat for n_exponent
        n_exp_summary = az.summary(
            fit_result.idata, var_names=["user_n_exponent"], kind="diagnostics"
        )

        # Compute effective sigma range using posterior mean exponent
        n_reviews = model_args["n_reviews"]
        sigma_obs_mean = float(fit_result.idata.posterior["user_sigma_obs"].mean())
        # Min sigma at max n_reviews, max sigma at min n_reviews
        # Wrap numpy scalars in JAX arrays for compute_sigma_scaled compatibility
        sigma_at_max_n = float(
            compute_sigma_scaled(
                sigma_obs_mean, jnp.array(np.max(n_reviews)), jnp.array(n_exp_mean)
            )
        )
        sigma_at_min_n = float(
            compute_sigma_scaled(
                sigma_obs_mean, jnp.array(np.min(n_reviews)), jnp.array(n_exp_mean)
            )
        )

        # Reference scaling values for interpretation
        ref_sqrt = 0.5  # Square-root scaling
        ref_cube_root = 0.33  # Cube-root scaling
        interpretation = (
            "closer to cube-root scaling (0.33)"
            if abs(n_exp_mean - ref_cube_root) < abs(n_exp_mean - ref_sqrt)
            else "closer to square-root scaling (0.5)"
        )

        summary["heteroscedastic_mode"] = {
            "mode": "learned",
            "n_exponent_mean": n_exp_mean,
            "n_exponent_std": n_exp_std,
            "n_exponent_hdi_94": [hdi_low, hdi_high],
            "n_exponent_ess_bulk": int(n_exp_summary["ess_bulk"].values[0]),
            "n_exponent_r_hat": float(n_exp_summary["r_hat"].values[0]),
            "interpretation": interpretation,
            "reference_sqrt": ref_sqrt,
            "reference_cube_root": ref_cube_root,
            "sigma_scaled_range": {
                "min": sigma_at_max_n,
                "max": sigma_at_min_n,
                "at_n_reviews_max": int(np.max(n_reviews)),
                "at_n_reviews_min": int(np.min(n_reviews)),
                "base_sigma_obs": sigma_obs_mean,
            },
        }
        log.info(
            "heteroscedastic_summary",
            mode="learned",
            n_exponent_mean=round(n_exp_mean, 4),
            n_exponent_hdi_94=[round(hdi_low, 4), round(hdi_high, 4)],
            interpretation=interpretation,
            sigma_range=[round(sigma_at_max_n, 4), round(sigma_at_min_n, 4)],
        )
    elif ctx.n_exponent != 0.0:
        # Fixed heteroscedastic mode
        n_reviews = model_args["n_reviews"]
        sigma_obs_mean = float(fit_result.idata.posterior["user_sigma_obs"].mean())
        # Wrap numpy scalars in JAX arrays for compute_sigma_scaled compatibility
        sigma_at_max_n = float(
            compute_sigma_scaled(
                sigma_obs_mean, jnp.array(np.max(n_reviews)), jnp.array(ctx.n_exponent)
            )
        )
        sigma_at_min_n = float(
            compute_sigma_scaled(
                sigma_obs_mean, jnp.array(np.min(n_reviews)), jnp.array(ctx.n_exponent)
            )
        )

        summary["heteroscedastic_mode"] = {
            "mode": "fixed",
            "n_exponent": ctx.n_exponent,
            "sigma_scaled_range": {
                "min": sigma_at_max_n,
                "max": sigma_at_min_n,
                "at_n_reviews_max": int(np.max(n_reviews)),
                "at_n_reviews_min": int(np.min(n_reviews)),
                "base_sigma_obs": sigma_obs_mean,
            },
        }
        log.info(
            "heteroscedastic_summary",
            mode="fixed",
            n_exponent=ctx.n_exponent,
            sigma_range=[round(sigma_at_max_n, 4), round(sigma_at_min_n, 4)],
        )
    else:
        # Homoscedastic mode (default)
        summary["heteroscedastic_mode"] = {
            "mode": "homoscedastic",
        }

    summary_path = model_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info("train_pipeline_complete", summary_path=str(summary_path))

    return summary
