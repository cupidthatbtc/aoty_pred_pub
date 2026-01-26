"""End-to-end dataset preparation pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import structlog

from aoty_pred.data.cleaning import (
    CleaningConfig,
    clean_albums,
    filter_for_critic_score_model,
    filter_for_user_score_model,
)
from aoty_pred.data.ingest import LoadMetadata, load_raw_albums
from aoty_pred.data.lineage import AuditLogger

log = structlog.get_logger()


@dataclass
class PrepareConfig:
    """Configuration for dataset preparation."""

    raw_path: str = "data/raw/all_albums_full.csv"
    output_dir: str = "data/processed"
    audit_dir: str = "data/audit"
    min_ratings_thresholds: list[int] = field(default_factory=lambda: [5, 10, 25])
    min_critic_reviews: int = 1
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)


@dataclass
class PrepareResult:
    """Result of dataset preparation."""

    load_metadata: LoadMetadata
    datasets_created: dict[str, Path]
    audit_paths: dict[str, Path]
    summary: dict


def save_dataset(
    df: pd.DataFrame,
    output_dir: Path,
    name: str,
) -> dict[str, Path]:
    """Save dataset in both Parquet and CSV formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"{name}.parquet"
    csv_path = output_dir / f"{name}.csv"

    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
    df.to_csv(csv_path, index=False)

    log.info(
        "dataset_saved",
        name=name,
        rows=len(df),
        parquet=str(parquet_path),
        csv=str(csv_path),
    )

    return {"parquet": parquet_path, "csv": csv_path}


def prepare_datasets(config: Optional[PrepareConfig] = None) -> PrepareResult:
    """
    Prepare all cleaned datasets from raw CSV.

    Creates:
    - User score datasets at multiple rating thresholds (5, 10, 25)
    - Critic score datasets
    - Audit logs with all exclusions

    Args:
        config: Pipeline configuration

    Returns:
        PrepareResult with paths to all created files
    """
    config = config or PrepareConfig()
    output_dir = Path(config.output_dir)
    audit_dir = Path(config.audit_dir)

    # Initialize audit logger
    logger = AuditLogger(output_dir=audit_dir)

    log.info("pipeline_start", raw_path=config.raw_path)

    # Step 1: Load raw data
    raw_df, load_meta = load_raw_albums(config.raw_path)
    log.info(
        "raw_loaded",
        rows=load_meta.row_count,
        hash=load_meta.file_hash[:16] + "...",
    )

    # Step 2: Apply cleaning transformations
    cleaned_df = clean_albums(raw_df, config=config.cleaning, logger=logger)
    log.info("cleaning_complete", rows=len(cleaned_df))

    # Step 3: Generate user score datasets at multiple thresholds
    datasets_created: dict[str, Path] = {}

    for threshold in config.min_ratings_thresholds:
        # Create a fresh logger section for this threshold
        user_df = filter_for_user_score_model(
            cleaned_df.copy(),
            min_ratings=threshold,
            logger=logger,
        )

        # Save dataset
        name = f"user_score_minratings_{threshold}"
        paths = save_dataset(user_df, output_dir, name)
        datasets_created[name] = paths["parquet"]

        log.info(
            "user_dataset_created",
            threshold=threshold,
            rows=len(user_df),
            unique_artists=user_df["Artist"].nunique(),
        )

    # Step 4: Generate critic score dataset
    critic_df = filter_for_critic_score_model(
        cleaned_df.copy(),
        min_reviews=config.min_critic_reviews,
        logger=logger,
    )

    name = "critic_score"
    paths = save_dataset(critic_df, output_dir, name)
    datasets_created[name] = paths["parquet"]

    log.info(
        "critic_dataset_created",
        rows=len(critic_df),
        unique_artists=critic_df["Artist"].nunique(),
    )

    # Step 5: Save full cleaned dataset (before score filtering)
    name = "cleaned_all"
    paths = save_dataset(cleaned_df, output_dir, name)
    datasets_created[name] = paths["parquet"]

    # Step 6: Save audit logs
    audit_paths = logger.save()

    # Build summary
    summary = {
        "raw_rows": load_meta.row_count,
        "raw_hash": load_meta.file_hash,
        "cleaned_rows": len(cleaned_df),
        "datasets": {
            name: {"path": str(path), "rows": pd.read_parquet(path).shape[0]}
            for name, path in datasets_created.items()
        },
        "exclusions": logger.get_summary(),
    }

    log.info("pipeline_complete", datasets=len(datasets_created))

    return PrepareResult(
        load_metadata=load_meta,
        datasets_created=datasets_created,
        audit_paths=audit_paths,
        summary=summary,
    )


def main():
    """CLI entry point for dataset preparation."""
    result = prepare_datasets()

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nRaw data: {result.load_metadata.row_count:,} rows")
    print(f"File hash: {result.load_metadata.file_hash[:32]}...")
    print("\nDatasets created:")
    for name, path in result.datasets_created.items():
        rows = result.summary["datasets"][name]["rows"]
        print(f"  - {name}: {rows:,} rows")
    print(f"\nAudit log: {result.audit_paths.get('summary')}")
    print(f"Total exclusions: {result.summary['exclusions']['total_exclusions']:,}")
    print("\nExclusions by reason:")
    for reason, count in list(result.summary["exclusions"]["exclusions_by_reason"].items())[:10]:
        print(f"  - {reason}: {count:,}")


if __name__ == "__main__":
    main()
