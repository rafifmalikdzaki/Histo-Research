"""Seed provenance for src/run_clustering_analysis.extract_seed.

Guards the stats-stage pairing fix: the seed must come from the training
config when present (never the -1 sentinel), fall back to a seed<N> dirname
fragment, and be None when unknown.
"""

import json
from pathlib import Path

import yaml

from src.run_clustering_analysis import extract_seed
from src.run_corrected_pipeline import _stamp_run_seed


def _run_dir(tmp_path: Path, name: str) -> Path:
    d = tmp_path / name
    d.mkdir()
    return d


def test_yaml_config_seed_wins(tmp_path):
    d = _run_dir(tmp_path, "baseline_baseline_20251005_100438")
    cfg = d / "configs"
    cfg.mkdir()
    (cfg / "experiment.yaml").write_text("seed: 123\nmodel_name: baseline\n")
    assert extract_seed(d) == 123


def test_json_config_seed(tmp_path):
    d = _run_dir(tmp_path, "baseline_baseline_20251005_100438")
    cfg = d / "configs"
    cfg.mkdir()
    (cfg / "experiment.json").write_text(json.dumps({"seed": 456}))
    assert extract_seed(d) == 456


def test_dirname_fallback(tmp_path):
    assert extract_seed(_run_dir(tmp_path, "bam_only_seed42_hepar")) == 42


def test_unknown_seed_is_none_not_sentinel(tmp_path):
    assert extract_seed(_run_dir(tmp_path, "baseline_baseline_20251005_100438")) is None


def test_pipeline_stamp_round_trips_through_extract_seed(tmp_path):
    d = _run_dir(tmp_path, "baseline_baseline_20251005_100438")
    _stamp_run_seed(d, "baseline", 456, "hepar")
    with open(d / "configs" / "experiment.yaml") as f:
        meta = yaml.safe_load(f)
    assert meta["seed"] == 456
    assert extract_seed(d) == 456


def test_pipeline_stamp_preserves_existing_config(tmp_path):
    d = _run_dir(tmp_path, "baseline_baseline_20251005_100438")
    cfg = d / "configs"
    cfg.mkdir()
    (cfg / "experiment.yaml").write_text("model_name: baseline\nlr: 0.002\n")
    _stamp_run_seed(d, "baseline", 42, "hepar")
    with open(cfg / "experiment.yaml") as f:
        meta = yaml.safe_load(f)
    assert (meta["seed"], meta["model_name"], meta["lr"]) == (42, "baseline", 0.002)
