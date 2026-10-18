"""PANnuke configuration checks (pytest port of test_pannuke_config.py).

Needs the real data checkout under data/processed/; skips cleanly without it.
"""

from pathlib import Path

import pandas as pd
import pytest


def _require(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        pytest.skip(f"needs data checkout: {path} not found")
    return p


def test_pannuke_csvs_and_images_present():
    train_df = pd.read_csv(_require("data/processed/train.csv"))
    test_df = pd.read_csv(_require("data/processed/test.csv"))
    assert len(train_df) > 0
    assert len(test_df) > 0

    pannuke_path = _require("data/processed/PANnuke")
    assert list(pannuke_path.glob("*.png")), "no PANnuke PNGs found"

    # Spot-check that CSV-listed files exist on disk.
    sample_files = train_df["Image"].head(5).tolist()
    missing = [f for f in sample_files if not (pannuke_path / f).exists()]
    assert not missing, f"{len(missing)} CSV-listed files missing: {missing[:3]}"


def test_training_script_configured_for_pannuke():
    script = _require("src/pl_training_with_analysis_and_optimization.py").read_text()
    assert "'pannuke': 'PANnuke'" in script  # CLI value maps to histodata name
    assert "--dataset" in script


def test_histodata_supports_dataset_selection():
    content = _require("src/histodata.py").read_text()
    assert 'dataset_name: str = "HeparUnifiedPNG"' in content
