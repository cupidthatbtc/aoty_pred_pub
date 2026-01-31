"""Smoke tests for dashboard data loading from outputs/ directory."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from aoty_pred.visualization.dashboard import DashboardData
from aoty_pred.visualization.server import _find_project_root, load_dashboard_data


class TestFindProjectRoot:
    """Tests for _find_project_root helper."""

    def test_finds_project_root(self) -> None:
        """Project root should contain pyproject.toml or .git."""
        root = _find_project_root()
        assert (root / "pyproject.toml").exists() or (root / ".git").exists()

    def test_returns_path_object(self) -> None:
        root = _find_project_root()
        assert isinstance(root, Path)


class TestLoadDashboardData:
    """Smoke tests for load_dashboard_data."""

    def test_returns_dashboard_data(self) -> None:
        """load_dashboard_data should return a DashboardData instance."""
        data = load_dashboard_data()
        assert isinstance(data, DashboardData)

    def test_has_eval_metrics_fields(self) -> None:
        """DashboardData should have eval_metrics, known_predictions, new_predictions."""
        data = DashboardData()
        assert hasattr(data, "eval_metrics")
        assert hasattr(data, "known_predictions")
        assert hasattr(data, "new_predictions")

    def test_loads_eval_metrics_from_outputs(self, tmp_path: Path) -> None:
        """Should load metrics.json when outputs/evaluation/ exists."""
        # Create mock directory structure
        eval_dir = tmp_path / "outputs" / "evaluation"
        eval_dir.mkdir(parents=True)

        metrics = {
            "model": "user_score",
            "n_test": 100,
            "point_metrics": {"rmse": 10.0, "mae": 7.0, "r2": 0.5},
            "calibration": {"coverage_90": 0.88, "coverage_50": 0.45},
        }
        with open(eval_dir / "metrics.json", "w") as f:
            json.dump(metrics, f)

        # Mock _find_project_root to return tmp_path
        with patch(
            "aoty_pred.visualization.server._find_project_root",
            return_value=tmp_path,
        ):
            data = load_dashboard_data()

        assert data.eval_metrics is not None
        assert data.eval_metrics["n_test"] == 100
        assert data.eval_metrics["point_metrics"]["rmse"] == 10.0

    def test_loads_known_predictions_from_outputs(self, tmp_path: Path) -> None:
        """Should load next_album_known_artists.csv when available."""
        pred_dir = tmp_path / "outputs" / "predictions"
        pred_dir.mkdir(parents=True)

        known_df = pd.DataFrame(
            {
                "artist": ["Test Artist"],
                "scenario": ["same"],
                "pred_mean": [75.0],
                "pred_std": [5.0],
                "pred_q05": [65.0],
                "pred_q25": [72.0],
                "pred_q50": [75.0],
                "pred_q75": [78.0],
                "pred_q95": [85.0],
                "last_score": [80.0],
                "n_training_albums": [10],
            }
        )
        known_df.to_csv(pred_dir / "next_album_known_artists.csv", index=False)

        with patch(
            "aoty_pred.visualization.server._find_project_root",
            return_value=tmp_path,
        ):
            data = load_dashboard_data()

        assert data.known_predictions is not None
        assert len(data.known_predictions) == 1
        assert data.known_predictions.iloc[0]["artist"] == "Test Artist"

    def test_loads_new_predictions_from_outputs(self, tmp_path: Path) -> None:
        """Should load next_album_new_artist.csv when available."""
        pred_dir = tmp_path / "outputs" / "predictions"
        pred_dir.mkdir(parents=True)

        new_df = pd.DataFrame(
            {
                "scenario": ["population", "debut_defaults"],
                "pred_mean": [70.0, 65.0],
                "pred_std": [10.0, 12.0],
                "pred_q05": [55.0, 45.0],
                "pred_q25": [65.0, 58.0],
                "pred_q50": [70.0, 65.0],
                "pred_q75": [75.0, 72.0],
                "pred_q95": [85.0, 82.0],
            }
        )
        new_df.to_csv(pred_dir / "next_album_new_artist.csv", index=False)

        with patch(
            "aoty_pred.visualization.server._find_project_root",
            return_value=tmp_path,
        ):
            data = load_dashboard_data()

        assert data.new_predictions is not None
        assert len(data.new_predictions) == 2
        assert set(data.new_predictions["scenario"]) == {"population", "debut_defaults"}

    def test_handles_missing_outputs_gracefully(self, tmp_path: Path) -> None:
        """Should return DashboardData with None fields when outputs/ missing."""
        with patch(
            "aoty_pred.visualization.server._find_project_root",
            return_value=tmp_path,
        ):
            data = load_dashboard_data()

        # Fields should be None (not raise)
        assert data.eval_metrics is None
        assert data.known_predictions is None
        assert data.new_predictions is None
