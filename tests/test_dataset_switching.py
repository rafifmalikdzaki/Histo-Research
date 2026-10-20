"""Dataset-switching checks (pytest port of test_dataset_switching.py).

Needs the real data checkout under data/processed/; skips cleanly without it.
"""

from pathlib import Path

import pandas as pd
import pytest

TRAINING_SCRIPT = Path("src/histodae/cli/train.py")

CSV_FILES = [
    "data/processed/train_pannuke.csv",
    "data/processed/test_pannuke.csv",
    "data/processed/train_hepar.csv",
    "data/processed/test_hepar.csv",
]


def _require(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        pytest.skip(f"needs data checkout: {path} not found")
    return p


def test_csv_files_exist_and_nonempty():
    for csv_path in CSV_FILES:
        df = pd.read_csv(_require(csv_path))
        assert len(df) > 0, f"{csv_path} is empty"


def test_dataset_directories_exist():
    pannuke_dir = _require("data/processed/PANnuke")
    hepar_dir = _require("data/processed/HeparUnifiedPNG")
    assert list(pannuke_dir.glob("*.png")), "no PANnuke PNGs found"
    assert list(hepar_dir.rglob("*.png")), "no Hepar PNGs found"


def test_training_script_accepts_dataset_flag():
    script = _require(str(TRAINING_SCRIPT)).read_text()
    assert "--dataset" in script
    assert "choices=['pannuke', 'hepar']" in script


def test_histodata_loads_both_datasets():
    for csv_path in CSV_FILES:
        _require(csv_path)
    from histodae.data.datasets import create_dataset

    for dataset_name in ("PANnuke", "HeparUnifiedPNG"):
        X, y = create_dataset("train", dataset_name=dataset_name)
        assert len(X) > 0, f"no samples loaded for {dataset_name}"
