"""Data lineage and audit logging."""

import json
import structlog
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import pandas as pd


@dataclass
class ExclusionRecord:
    """Record of a single row exclusion."""
    original_row_id: int
    artist: str
    album: str
    reason: str
    value: Any = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class FilterStats:
    """Statistics for a single filter application."""
    filter_name: str
    rows_before: int
    rows_excluded: int
    rows_after: int

    @property
    def exclusion_rate(self) -> float:
        if self.rows_before == 0:
            return 0.0
        return self.rows_excluded / self.rows_before


class AuditLogger:
    """
    Logger for tracking row exclusions during data cleaning.

    Usage:
        logger = AuditLogger(output_dir="data/audit")
        logger.log_exclusion(row_id=42, artist="...", album="...", reason="...")
        logger.log_filter_stats("missing_score", before=1000, excluded=50, after=950)
        logger.save()  # Writes JSONL and summary
    """

    def __init__(self, output_dir: str | Path = "data/audit", run_id: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exclusions: list[ExclusionRecord] = []
        self.filter_stats: list[FilterStats] = []

        # Configure structlog for console output
        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
        )
        self.log = structlog.get_logger()

    def log_exclusion(
        self,
        row_id: int,
        artist: str,
        album: str,
        reason: str,
        value: Any = None,
    ) -> None:
        """Log a single row exclusion."""
        record = ExclusionRecord(
            original_row_id=row_id,
            artist=str(artist)[:100],  # Truncate long names
            album=str(album)[:100],
            reason=reason,
            value=value,
        )
        self.exclusions.append(record)

    def log_exclusions_bulk(
        self,
        df: pd.DataFrame,
        reason: str,
        value_col: Optional[str] = None,
    ) -> None:
        """Log exclusions for all rows in a DataFrame."""
        for _, row in df.iterrows():
            value = row.get(value_col) if value_col else None
            self.log_exclusion(
                row_id=int(row["original_row_id"]),
                artist=row["Artist"],
                album=row["Album"],
                reason=reason,
                value=value,
            )

    def log_filter_stats(
        self,
        filter_name: str,
        rows_before: int,
        rows_excluded: int,
        rows_after: int,
    ) -> None:
        """Log statistics for a filter application."""
        stats = FilterStats(
            filter_name=filter_name,
            rows_before=rows_before,
            rows_excluded=rows_excluded,
            rows_after=rows_after,
        )
        self.filter_stats.append(stats)

        # Also log to console
        self.log.info(
            "filter_applied",
            filter=filter_name,
            before=rows_before,
            excluded=rows_excluded,
            after=rows_after,
            rate=f"{stats.exclusion_rate:.1%}",
        )

    def save(self) -> dict[str, Path]:
        """
        Save audit logs to files.

        Returns:
            Dict with paths to saved files
        """
        paths = {}

        # Save exclusions as JSONL
        exclusions_path = self.output_dir / f"exclusions_{self.run_id}.jsonl"
        with open(exclusions_path, "w", encoding="utf-8") as f:
            for record in self.exclusions:
                f.write(json.dumps(asdict(record), default=str) + "\n")
        paths["exclusions"] = exclusions_path

        # Save summary as JSON
        summary = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "total_exclusions": len(self.exclusions),
            "exclusions_by_reason": self._count_by_reason(),
            "filter_stats": [asdict(s) for s in self.filter_stats],
        }
        summary_path = self.output_dir / f"summary_{self.run_id}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        paths["summary"] = summary_path

        self.log.info(
            "audit_saved",
            exclusions=len(self.exclusions),
            exclusions_path=str(exclusions_path),
            summary_path=str(summary_path),
        )

        return paths

    def _count_by_reason(self) -> dict[str, int]:
        """Count exclusions by reason."""
        counts: dict[str, int] = {}
        for record in self.exclusions:
            counts[record.reason] = counts.get(record.reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def get_summary(self) -> dict:
        """Get summary statistics without saving."""
        return {
            "total_exclusions": len(self.exclusions),
            "exclusions_by_reason": self._count_by_reason(),
            "filter_stats": [asdict(s) for s in self.filter_stats],
        }


def record_lineage(step_name: str, inputs: dict, outputs: dict) -> None:
    """
    Record lineage for a processing step (legacy stub compatibility).

    For full lineage tracking, use AuditLogger directly.
    """
    log = structlog.get_logger()
    log.info(
        "lineage_record",
        step=step_name,
        inputs=inputs,
        outputs=outputs,
    )
