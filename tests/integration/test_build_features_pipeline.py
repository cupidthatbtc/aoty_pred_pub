import json
from pathlib import Path

import pandas as pd
import pytest

from aoty_pred.data.cleaning import clean_raw_data
from aoty_pred.pipelines import build_features


def test_build_features_end_to_end(tmp_path: Path):
    pytest.importorskip("pyarrow")

    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "raw_all_albums_full.csv"
    raw_df = pd.read_csv(fixture_path, encoding="utf-8-sig")
    processed_df = clean_raw_data(raw_df)

    processed_path = tmp_path / "processed.csv"
    processed_df.to_csv(processed_path, index=False)

    features_dir = tmp_path / "features"
    manifest_path = features_dir / "features_manifest.json"

    config_path = tmp_path / "config.yaml"
    # Use forward slashes in YAML paths (works on all platforms)
    fixture_path_str = str(fixture_path).replace("\\", "/")
    processed_path_str = str(processed_path).replace("\\", "/")
    features_dir_str = str(features_dir).replace("\\", "/")
    manifest_path_str = str(manifest_path).replace("\\", "/")
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                f"  raw_csv: \"{fixture_path_str}\"",
                "  encoding: \"utf-8-sig\"",
                "  min_ratings: 30",
                "splits:",
                "  seed: 42",
                "model:",
                "  sampler: \"pymc\"",
                "  tune: 10",
                "  draws: 10",
                "  chains: 1",
                "outputs:",
                f"  processed_path: \"{processed_path_str}\"",
                f"  features_dir: \"{features_dir_str}\"",
                f"  features_manifest: \"{manifest_path_str}\"",
                "features:",
                "  blocks:",
                "    - name: core_numeric",
                "      params: {}",
            ]
        ),
        encoding="utf-8",
    )

    build_features.run([str(config_path)])

    matrix_path = features_dir / "feature_matrix.parquet"
    assert matrix_path.exists()
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["rows"] == len(processed_df)
    assert manifest["blocks"][0]["block"] == "core_numeric"
