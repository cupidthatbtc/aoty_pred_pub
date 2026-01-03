"""Split manifest schema and I/O for reproducibility."""

import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd


@dataclass
class SplitAssignment:
    """Per-row split assignment with reasoning."""
    original_row_id: int
    split: str  # "train", "validation", or "test"
    reason: str  # e.g., "last_album_for_artist", "artist_in_test_group"


@dataclass
class SplitStats:
    """Statistics for a single split."""
    row_count: int
    unique_artists: int
    sha256: str


@dataclass
class SplitManifest:
    """
    Complete manifest for a split operation.

    Records all metadata needed to reproduce and audit the split.
    """
    version: str
    created_at: str
    split_type: str  # "within_artist_temporal" or "artist_disjoint"
    parameters: Dict[str, Any]
    source_dataset: Dict[str, Any]  # path, sha256, row_count, unique_artists
    splits: Dict[str, SplitStats]  # train, validation, test stats
    assignments: List[SplitAssignment] = field(default_factory=list)
    content_hash: str = ""  # Combined hash of all splits

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert SplitStats and SplitAssignment dataclasses
        d["splits"] = {k: asdict(v) for k, v in self.splits.items()}
        d["assignments"] = [asdict(a) for a in self.assignments]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SplitManifest":
        """Create from dictionary."""
        d["splits"] = {k: SplitStats(**v) for k, v in d["splits"].items()}
        d["assignments"] = [SplitAssignment(**a) for a in d["assignments"]]
        return cls(**d)


def generate_manifest_filename(version: str, content_hash: str) -> str:
    """
    Generate manifest filename with version, timestamp, and hash.

    Format: split_{version}_{timestamp}_{hash_prefix}.json
    Example: split_v1_20260118_abc123de.json
    """
    timestamp = datetime.now().strftime("%Y%m%d")
    hash_prefix = content_hash[:8]
    return f"split_{version}_{timestamp}_{hash_prefix}.json"


def save_manifest(manifest: SplitManifest, output_dir: Path) -> Path:
    """
    Save manifest to JSON file.

    Args:
        manifest: SplitManifest to save
        output_dir: Directory to save manifest in

    Returns:
        Path to saved manifest file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = generate_manifest_filename(manifest.version, manifest.content_hash)
    filepath = output_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, default=str)

    return filepath


def load_manifest(filepath: Path) -> SplitManifest:
    """
    Load manifest from JSON file.

    Args:
        filepath: Path to manifest JSON file

    Returns:
        SplitManifest object
    """
    with open(filepath, "r", encoding="utf-8") as f:
        d = json.load(f)
    return SplitManifest.from_dict(d)


def create_split_assignments(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_type: str,
    artist_col: str = "Artist",
) -> List[SplitAssignment]:
    """
    Create per-row split assignments with reasoning.

    Args:
        train_df, val_df, test_df: Split DataFrames
        split_type: "within_artist_temporal" or "artist_disjoint"
        artist_col: Column name for artist

    Returns:
        List of SplitAssignment objects
    """
    assignments = []

    if split_type == "within_artist_temporal":
        # For temporal splits, reason includes album position
        for _, row in test_df.iterrows():
            assignments.append(SplitAssignment(
                original_row_id=int(row["original_row_id"]),
                split="test",
                reason=f"last_album_for_{row[artist_col][:50]}"
            ))
        for _, row in val_df.iterrows():
            assignments.append(SplitAssignment(
                original_row_id=int(row["original_row_id"]),
                split="validation",
                reason=f"second_last_album_for_{row[artist_col][:50]}"
            ))
        for _, row in train_df.iterrows():
            assignments.append(SplitAssignment(
                original_row_id=int(row["original_row_id"]),
                split="train",
                reason=f"earlier_album_for_{row[artist_col][:50]}"
            ))
    else:  # artist_disjoint
        test_artists = set(test_df[artist_col])
        val_artists = set(val_df[artist_col])

        for _, row in test_df.iterrows():
            assignments.append(SplitAssignment(
                original_row_id=int(row["original_row_id"]),
                split="test",
                reason="artist_in_test_group"
            ))
        for _, row in val_df.iterrows():
            assignments.append(SplitAssignment(
                original_row_id=int(row["original_row_id"]),
                split="validation",
                reason="artist_in_validation_group"
            ))
        for _, row in train_df.iterrows():
            assignments.append(SplitAssignment(
                original_row_id=int(row["original_row_id"]),
                split="train",
                reason="artist_in_train_group"
            ))

    return assignments
