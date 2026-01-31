"""Next-album prediction pipeline.

Generates predictions for:
- Known artists (3 scenarios): next album using trained artist effects
- New/hypothetical artists (2 scenarios): using population distribution
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import structlog
from jax import random
from numpyro.infer import Predictive

from aoty_pred.models.bayes.io import load_manifest, load_model
from aoty_pred.models.bayes.model import user_score_model
from aoty_pred.models.bayes.predict import predict_new_artist
from aoty_pred.models.bayes.priors import PriorConfig

if TYPE_CHECKING:
    from aoty_pred.pipelines.stages import StageContext

log = structlog.get_logger()

# Scenario names
SCENARIOS_KNOWN = ["same", "population_mean", "artist_mean"]
SCENARIOS_NEW = ["population", "debut_defaults"]


def _extract_posterior_samples(idata: object) -> dict[str, jnp.ndarray]:
    """Flatten posterior samples from idata into a dict of JAX arrays."""
    posterior = idata.posterior  # type: ignore[attr-defined]
    posterior_samples: dict[str, jnp.ndarray] = {}
    for var_name in posterior.data_vars:
        vals = posterior[var_name].values
        n_chains, n_draws = vals.shape[:2]
        rest_shape = vals.shape[2:]
        flat = vals.reshape(n_chains * n_draws, *rest_shape)
        posterior_samples[var_name] = jnp.array(flat)
    return posterior_samples


def _predict_known_artists(
    posterior_samples: dict[str, jnp.ndarray],
    summary: dict,
    last_album_info: pd.DataFrame,
    artist_mean_features: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate next-album predictions for all known artists under 3 scenarios.

    Scenarios:
    - "same": Use the artist's last album's feature values
    - "population_mean": Use population mean features (zeros after z-scoring)
    - "artist_mean": Use the artist's mean feature values

    Args:
        posterior_samples: Flattened posterior samples dict.
        summary: Training summary dict.
        last_album_info: DataFrame with last album info per artist.
        artist_mean_features: DataFrame with mean feature values per artist.

    Returns:
        DataFrame with columns: artist, scenario, pred_mean, pred_std,
        pred_q05, pred_q25, pred_q50, pred_q75, pred_q95,
        last_score, n_training_albums.
    """
    artist_to_idx = summary["artist_to_idx"]
    max_seq = summary["max_seq"]
    feature_cols = summary["feature_cols"]
    scaler = summary["feature_scaler"]
    X_mean = np.array(scaler["mean"], dtype=np.float32)
    X_std = np.array(scaler["std"], dtype=np.float32)

    n_total_samples = next(iter(posterior_samples.values())).shape[0]
    batch_size = 500
    artist_batch_size = 50

    # Prepare per-artist metadata
    artists = list(artist_to_idx.keys())
    n_artists_total = len(artists)

    results = []

    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        # Process artists in batches
        for batch_start in range(0, n_artists_total, artist_batch_size):
            batch_end = min(batch_start + artist_batch_size, n_artists_total)
            batch_artists = artists[batch_start:batch_end]

            for scenario in SCENARIOS_KNOWN:
                # Build model_args for this batch of artists (one obs per artist)
                artist_idxs = []
                album_seqs = []
                prev_scores = []
                X_list = []
                n_reviews_list = []
                valid_artists = []
                last_scores = []
                n_training_albums_list = []

                for artist in batch_artists:
                    idx = artist_to_idx[artist]

                    if artist not in last_album_info.index:
                        continue

                    info = last_album_info.loc[artist]
                    last_seq = int(info["album_seq"])
                    next_seq = min(last_seq + 1, max_seq)
                    last_score = float(info["User_Score"])
                    median_n_reviews = int(info["median_n_reviews"])
                    n_albums = int(info["n_albums"])

                    artist_idxs.append(idx)
                    album_seqs.append(next_seq)
                    prev_scores.append(last_score)
                    n_reviews_list.append(median_n_reviews)
                    valid_artists.append(artist)
                    last_scores.append(last_score)
                    n_training_albums_list.append(n_albums)

                    # Feature vector depends on scenario
                    if scenario == "same":
                        raw_features = info[feature_cols].values.astype(np.float32)
                        X_list.append((raw_features - X_mean) / X_std)
                    elif scenario == "population_mean":
                        X_list.append(np.zeros(len(feature_cols), dtype=np.float32))
                    elif scenario == "artist_mean":
                        if artist in artist_mean_features.index:
                            raw_features = artist_mean_features.loc[
                                artist, feature_cols
                            ].values.astype(np.float32)
                            X_list.append((raw_features - X_mean) / X_std)
                        else:
                            X_list.append(np.zeros(len(feature_cols), dtype=np.float32))

                if not valid_artists:
                    continue

                artist_idx_arr = np.array(artist_idxs, dtype=np.int32)
                album_seq_arr = np.array(album_seqs, dtype=np.int32)
                prev_score_arr = np.array(prev_scores, dtype=np.float32)
                X_arr = np.stack(X_list).astype(np.float32)
                n_reviews_arr = np.array(n_reviews_list, dtype=np.int32)

                model_args = {
                    "artist_idx": artist_idx_arr,
                    "album_seq": album_seq_arr,
                    "prev_score": prev_score_arr,
                    "X": X_arr,
                    "y": None,
                    "n_reviews": n_reviews_arr,
                    "n_artists": summary["n_artists"],
                    "max_seq": max_seq,
                    "n_exponent": summary.get("n_exponent", 0.0),
                    "learn_n_exponent": summary.get("learn_n_exponent", False),
                    "n_exponent_prior": summary.get("n_exponent_prior", "logit-normal"),
                    "n_ref": summary.get("n_ref"),
                    "priors": PriorConfig(**summary["priors"]),
                }

                # Run Predictive in chunks -- create once, replace posterior_samples
                # per batch to preserve function identity and avoid JAX recompilation
                y_chunks: list[np.ndarray] = []
                first_batch_ps = {k: v[:batch_size] for k, v in posterior_samples.items()}
                predictive = Predictive(
                    user_score_model,
                    posterior_samples=first_batch_ps,
                    batch_ndims=1,
                )
                for start in range(0, n_total_samples, batch_size):
                    end = min(start + batch_size, n_total_samples)
                    batch_ps = {k: v[start:end] for k, v in posterior_samples.items()}
                    predictive.posterior_samples = batch_ps

                    rng_key = random.key(seed + start + batch_start * 1000)
                    preds = predictive(rng_key, **model_args)
                    y_key = next(k for k in preds if k.endswith("_y"))
                    y_chunks.append(np.asarray(preds[y_key]))

                # shape: (n_samples, n_artists_in_batch)
                y_pred = np.concatenate(y_chunks, axis=0)

                # Compute summary stats per artist
                for i, artist in enumerate(valid_artists):
                    samples = y_pred[:, i]
                    results.append(
                        {
                            "artist": artist,
                            "scenario": scenario,
                            "pred_mean": float(np.mean(samples)),
                            "pred_std": float(np.std(samples)),
                            "pred_q05": float(np.percentile(samples, 5)),
                            "pred_q25": float(np.percentile(samples, 25)),
                            "pred_q50": float(np.percentile(samples, 50)),
                            "pred_q75": float(np.percentile(samples, 75)),
                            "pred_q95": float(np.percentile(samples, 95)),
                            "last_score": last_scores[i],
                            "n_training_albums": n_training_albums_list[i],
                        }
                    )

            if batch_end % 200 == 0 or batch_end == n_artists_total:
                log.info(
                    "known_artist_progress",
                    processed=batch_end,
                    total=n_artists_total,
                )

    return pd.DataFrame(results)


def _predict_new_artists(
    posterior_samples: dict[str, jnp.ndarray],
    summary: dict,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate predictions for hypothetical new artists under 2 scenarios.

    Scenarios:
    - "population": Population mean features, median n_reviews
    - "debut_defaults": Population mean features, minimum n_reviews (debut-like)

    Args:
        posterior_samples: Flattened posterior samples dict (numpy-compatible).
        summary: Training summary dict.

    Returns:
        DataFrame with columns: scenario, pred_mean, pred_std,
        pred_q05, pred_q25, pred_q50, pred_q75, pred_q95.
    """
    n_features = len(summary["feature_cols"])
    n_reviews_median = summary["n_reviews_stats"]["median"]
    n_reviews_min = summary["n_reviews_stats"]["min"]

    # Determine if model uses heteroscedastic noise
    learn_n_exponent = summary.get("learn_n_exponent", False)
    n_exponent = summary.get("n_exponent", 0.0)
    has_hetero = learn_n_exponent or n_exponent != 0.0

    results = []

    scenarios = [
        ("population", n_reviews_median),
        ("debut_defaults", n_reviews_min),
    ]

    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        for scenario_name, n_reviews_val in scenarios:
            X_new = jnp.zeros(n_features)

            kwargs: dict = {
                "posterior_samples": {
                    k: jnp.array(np.asarray(v)) for k, v in posterior_samples.items()
                },
                "X_new": X_new,
                "prev_score": 0.0,
                "prefix": "user_",
                "seed": seed,
            }

            if has_hetero:
                kwargs["n_reviews_new"] = jnp.array([n_reviews_val])
                if not learn_n_exponent and n_exponent != 0.0:
                    kwargs["fixed_n_exponent"] = n_exponent

            pred = predict_new_artist(**kwargs)

            y_samples = np.asarray(pred["y"])
            results.append(
                {
                    "scenario": scenario_name,
                    "pred_mean": float(np.mean(y_samples)),
                    "pred_std": float(np.std(y_samples)),
                    "pred_q05": float(np.percentile(y_samples, 5)),
                    "pred_q25": float(np.percentile(y_samples, 25)),
                    "pred_q50": float(np.percentile(y_samples, 50)),
                    "pred_q75": float(np.percentile(y_samples, 75)),
                    "pred_q95": float(np.percentile(y_samples, 95)),
                }
            )

    return pd.DataFrame(results)


def predict_next_albums(ctx: StageContext) -> dict:
    """Generate next-album predictions for known and new artists.

    Known artists get 3 scenarios (same features, population mean, artist mean).
    New artists get 2 scenarios (population, debut defaults).

    Args:
        ctx: Stage context with run configuration.

    Returns:
        Dictionary with prediction summary and output paths.
    """
    log.info("predict_next_start")
    seed = ctx.seed

    # Load model
    model_dir = Path("models")
    manifest = load_manifest(model_dir)

    if manifest is None or "user_score" not in manifest.current:
        raise ValueError("No trained user_score model found in models/manifest.json")

    model_filename = manifest.current["user_score"]
    model_path = model_dir / model_filename

    log.info("loading_model", path=str(model_path))
    idata = load_model(model_path)

    # Load training summary
    summary_path = model_dir / "training_summary.json"
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    # Extract posterior samples
    posterior_samples = _extract_posterior_samples(idata)
    n_total_samples = next(iter(posterior_samples.values())).shape[0]
    log.info("posterior_samples_extracted", n_total_samples=n_total_samples)

    # Load training data to get per-artist last album info
    train_df = pd.read_parquet("data/splits/within_artist_temporal/train.parquet")
    train_features = pd.read_parquet("data/features/train_features.parquet")

    # Drop overlap and join
    overlap_cols = list(set(train_df.columns) & set(train_features.columns))
    if overlap_cols:
        train_df = train_df.drop(columns=overlap_cols)

    # Validate alignment before join
    if not train_df.index.equals(train_features.index):
        raise ValueError(
            "Index mismatch between train_df and train_features. "
            f"train_df index: {train_df.index[:5].tolist()}..., "
            f"train_features index: {train_features.index[:5].tolist()}..."
        )

    train_df = train_df.join(train_features, how="left")

    feature_cols = summary["feature_cols"]
    train_df[feature_cols] = train_df[feature_cols].fillna(0)

    # Compute album sequence within artist
    train_df = train_df.copy()
    train_df["album_seq"] = train_df.groupby("Artist").cumcount() + 1

    # Compute n_reviews column
    if "n_reviews" in train_df.columns:
        n_reviews_col = "n_reviews"
    elif "User_Ratings" in train_df.columns:
        n_reviews_col = "User_Ratings"
    else:
        n_reviews_col = None

    # Get last album info per artist (sort by album_seq, take last)
    train_df = train_df.sort_values(["Artist", "album_seq"])
    last_album_info = train_df.groupby("Artist").last()

    # Add n_albums and median n_reviews per artist
    artist_stats = train_df.groupby("Artist").agg(
        n_albums=("album_seq", "max"),
    )
    if n_reviews_col:
        artist_n_reviews = train_df.groupby("Artist")[n_reviews_col].median()
        artist_stats["median_n_reviews"] = artist_n_reviews
    else:
        artist_stats["median_n_reviews"] = summary["n_reviews_stats"]["median"]

    last_album_info = last_album_info.join(artist_stats[["n_albums", "median_n_reviews"]])

    # Compute artist mean features
    artist_mean_features = train_df.groupby("Artist")[feature_cols].mean()

    # Generate known artist predictions
    log.info("predicting_known_artists", n_artists=len(summary["artist_to_idx"]))
    known_df = _predict_known_artists(
        posterior_samples,
        summary,
        last_album_info,
        artist_mean_features,
        seed=seed,
    )
    log.info("known_predictions_complete", n_rows=len(known_df))

    # Generate new artist predictions
    log.info("predicting_new_artists")
    new_df = _predict_new_artists(posterior_samples, summary, seed=seed)
    log.info("new_predictions_complete", n_rows=len(new_df))

    # Save outputs
    output_dir = Path("outputs/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    known_df.to_csv(output_dir / "next_album_known_artists.csv", index=False)
    new_df.to_csv(output_dir / "next_album_new_artist.csv", index=False)

    pred_summary = {
        "n_known_artists": len(summary["artist_to_idx"]),
        "scenarios_known": SCENARIOS_KNOWN,
        "scenarios_new": SCENARIOS_NEW,
        "n_posterior_samples": int(n_total_samples),
        "batch_size": 500,
    }
    with open(output_dir / "prediction_summary.json", "w", encoding="utf-8") as f:
        json.dump(pred_summary, f, indent=2)

    log.info(
        "predict_next_complete",
        known_artists=len(summary["artist_to_idx"]),
        known_rows=len(known_df),
        new_rows=len(new_df),
    )

    return {
        "known_predictions_path": str(output_dir / "next_album_known_artists.csv"),
        "new_predictions_path": str(output_dir / "next_album_new_artist.csv"),
        "summary_path": str(output_dir / "prediction_summary.json"),
        "pred_summary": pred_summary,
    }
