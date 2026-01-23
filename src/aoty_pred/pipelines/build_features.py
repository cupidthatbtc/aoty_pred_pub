"""Feature matrix building pipeline.

Builds combined feature matrices from configured feature blocks for all splits
(train, validation, test) and saves them for reuse in model training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import structlog

from aoty_pred.features.base import FeatureContext
from aoty_pred.features.pipeline import FeaturePipeline
from aoty_pred.features.temporal import TemporalBlock
from aoty_pred.features.album_type import AlbumTypeBlock
from aoty_pred.features.artist import ArtistHistoryBlock
from aoty_pred.features.genre import GenreBlock
from aoty_pred.features.collaboration import CollaborationBlock

if TYPE_CHECKING:
    from aoty_pred.pipelines.stages import StageContext

log = structlog.get_logger()


def get_feature_blocks(
    enable_genre: bool = True,
    enable_artist: bool = True,
    enable_temporal: bool = True,
) -> list:
    """Get feature blocks filtered by enabled flags.

    Args:
        enable_genre: Include GenreBlock if True.
        enable_artist: Include ArtistHistoryBlock if True.
        enable_temporal: Include TemporalBlock if True.

    Returns:
        List of enabled feature blocks in dependency order.
    """
    blocks = []

    # Conditional blocks - temporal first for dependency order
    if enable_temporal:
        blocks.append(TemporalBlock({}))

    # Core blocks that are always included
    blocks.append(AlbumTypeBlock({}))

    # Conditional artist history block
    if enable_artist:
        blocks.append(ArtistHistoryBlock({}))

    # Conditional genre block
    if enable_genre:
        blocks.append(GenreBlock({"min_genre_count": 20, "n_components": 10}))

    # Core collaboration block always included
    blocks.append(CollaborationBlock({}))

    return blocks


def get_default_feature_blocks() -> list:
    """Get the default feature blocks for the pipeline.

    Legacy function for backward compatibility. Prefer get_feature_blocks()
    with explicit flags for new code.

    Returns:
        List of all feature blocks in dependency order.
    """
    return get_feature_blocks(
        enable_genre=True,
        enable_artist=True,
        enable_temporal=True,
    )


def build_features(ctx: "StageContext") -> dict:
    """Build feature matrices for all splits.

    Fits feature pipeline on training data only, then transforms all splits
    to prevent data leakage. Respects feature ablation flags from CLI.

    Args:
        ctx: Stage context with run configuration.

    Returns:
        Dictionary with paths to created feature matrices and metadata.
    """
    log.info(
        "feature_pipeline_start",
        seed=ctx.seed,
        enable_genre=ctx.enable_genre,
        enable_artist=ctx.enable_artist,
        enable_temporal=ctx.enable_temporal,
    )

    # Define paths
    splits_dir = Path("data/splits/within_artist_temporal")
    features_dir = Path("data/features")
    features_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    val_df = pd.read_parquet(splits_dir / "validation.parquet")
    test_df = pd.read_parquet(splits_dir / "test.parquet")

    log.info(
        "splits_loaded",
        train_rows=len(train_df),
        val_rows=len(val_df),
        test_rows=len(test_df),
    )

    # Create feature context
    feature_ctx = FeatureContext(
        config={},  # Using default configs
        random_state=ctx.seed,
    )

    # Build feature blocks based on ablation flags
    blocks = get_feature_blocks(
        enable_genre=ctx.enable_genre,
        enable_artist=ctx.enable_artist,
        enable_temporal=ctx.enable_temporal,
    )
    pipeline = FeaturePipeline(blocks)

    log.info(
        "fitting_features",
        blocks=[b.name for b in blocks],
        ablated={
            "genre": not ctx.enable_genre,
            "artist": not ctx.enable_artist,
            "temporal": not ctx.enable_temporal,
        },
    )
    pipeline.fit(train_df, feature_ctx)

    # Transform all splits
    log.info("transforming_train")
    train_output = pipeline.transform(train_df, feature_ctx)
    train_features = train_output.data

    log.info("transforming_validation")
    val_output = pipeline.transform(val_df, feature_ctx)
    val_features = val_output.data

    log.info("transforming_test")
    test_output = pipeline.transform(test_df, feature_ctx)
    test_features = test_output.data

    # Save feature matrices
    train_path = features_dir / "train_features.parquet"
    val_path = features_dir / "validation_features.parquet"
    test_path = features_dir / "test_features.parquet"

    train_features.to_parquet(train_path, index=True)
    val_features.to_parquet(val_path, index=True)
    test_features.to_parquet(test_path, index=True)

    log.info(
        "features_saved",
        train_path=str(train_path),
        train_shape=train_features.shape,
        val_shape=val_features.shape,
        test_shape=test_features.shape,
    )

    # Save manifest
    manifest = {
        "seed": ctx.seed,
        "blocks": [b.name for b in blocks],
        "feature_ablation": {
            "enable_genre": ctx.enable_genre,
            "enable_artist": ctx.enable_artist,
            "enable_temporal": ctx.enable_temporal,
        },
        "feature_names": train_output.feature_names,
        "splits": {
            "train": {
                "path": str(train_path),
                "rows": int(train_features.shape[0]),
                "cols": int(train_features.shape[1]),
            },
            "validation": {
                "path": str(val_path),
                "rows": int(val_features.shape[0]),
                "cols": int(val_features.shape[1]),
            },
            "test": {
                "path": str(test_path),
                "rows": int(test_features.shape[0]),
                "cols": int(test_features.shape[1]),
            },
        },
    }

    manifest_path = features_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log.info("feature_pipeline_complete", manifest_path=str(manifest_path))

    return manifest
