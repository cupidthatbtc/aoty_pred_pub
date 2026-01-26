import tempfile
from pathlib import Path

from aoty_pred.config.loader import load_config


def test_load_config_merges_overrides(monkeypatch):
    base = """
dataset:
  raw_csv: "${AOTY_DATASET_PATH}"
splits:
  seed: 1
model:
  tune: 1000
  draws: 1000
  chains: 2
"""
    override = """
model:
  tune: 2000
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("AOTY_DATASET_PATH", "env.csv")
        base_path = Path(tmpdir) / "base.yaml"
        override_path = Path(tmpdir) / "override.yaml"
        base_path.write_text(base, encoding="utf-8")
        override_path.write_text(override, encoding="utf-8")

        cfg = load_config([base_path, override_path])
        assert cfg.dataset.raw_csv == "env.csv"
        assert cfg.model.tune == 2000
        assert cfg.model.draws == 1000
