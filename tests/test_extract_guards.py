"""Edge guards for src/extract_embeddings_for_clustering.extract.

Runs the real function on CPU with a randomly-initialised baseline model and
tiny synthetic tiles (no checkpoint, no dataset checkout): missing image
files are dropped with outputs kept aligned, and empty/missing-everything
inputs fail fast with a clear error instead of a cryptic concatenate crash
or an inscrutable DataLoader worker traceback.
"""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src import extract_embeddings_for_clustering as ext


def _tiles(tmp_path, n):
    paths = []
    for i in range(n):
        p = tmp_path / f"tile_{i}.png"
        Image.fromarray(
            np.zeros((64, 64, 3), dtype=np.uint8)).save(p)
        paths.append(str(p))
    return paths


def _frame(paths):
    df = pd.DataFrame({
        "specimen_id": [f"s{i}" for i in range(len(paths))],
        "treatment_group": ["g1"] * len(paths),
        "diet": ["d1"] * len(paths),
        "agent": ["a1"] * len(paths),
        "magnification": ["20x"] * len(paths),
        "tile_id": [f"t{i}" for i in range(len(paths))],
        "label": [0] * len(paths),
        "Image": [f"img{i}" for i in range(len(paths))],
        "path": list(paths),
    })
    return df


def _run(monkeypatch, tmp_path, df):
    monkeypatch.setattr(ext, "load_metadata", lambda *a, **k: df)
    out = tmp_path / "out"
    rc = ext.extract(
        model_name="baseline", dataset="whatever", phase="test",
        checkpoint=None, image_root=tmp_path, output_dir=out,
        batch_size=2, num_workers=0, image_size=64, device="cpu")
    assert rc == 0
    return out


def test_missing_files_dropped_outputs_aligned(monkeypatch, tmp_path, capsys):
    paths = _tiles(tmp_path, 3) + [str(tmp_path / "ghost.png")]
    out = _run(monkeypatch, tmp_path, _frame(paths))
    err = capsys.readouterr().err
    assert "Dropping 1" in err
    emb = np.load(out / "embeddings" / "test_embeddings.npy")
    meta = pd.read_csv(out / "embeddings" / "test_metadata.csv")
    assert len(emb) == len(meta) == 3
    assert "ghost" not in meta["path"].iloc[0]


def test_empty_frame_fails_fast(monkeypatch, tmp_path):
    monkeypatch.setattr(ext, "load_metadata", lambda *a, **k: _frame([]))
    with pytest.raises(RuntimeError, match="No test rows"):
        ext.extract(
            model_name="baseline", dataset="whatever", phase="test",
            checkpoint=None, image_root=tmp_path,
            output_dir=tmp_path / "out",
            batch_size=2, num_workers=0, image_size=64, device="cpu")


def test_all_files_missing_fails_fast(monkeypatch, tmp_path):
    df = _frame([str(tmp_path / "ghost.png")])
    monkeypatch.setattr(ext, "load_metadata", lambda *a, **k: df)
    with pytest.raises(RuntimeError, match="All test image files are missing"):
        ext.extract(
            model_name="baseline", dataset="whatever", phase="test",
            checkpoint=None, image_root=tmp_path,
            output_dir=tmp_path / "out",
            batch_size=2, num_workers=0, image_size=64, device="cpu")
