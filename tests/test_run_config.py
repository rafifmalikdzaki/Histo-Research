"""Shared best-effort config readers (src/utils/reproducibility).

Covers the merged run-config helper consumed by the clustering stage and the
single-file parser backing the evaluate-stats dataset scan.
"""

import json

from src.utils.reproducibility import read_config_file, read_run_config


def _run_dir(tmp_path, name="run_a"):
    d = tmp_path / name
    (d / "configs").mkdir(parents=True)
    return d


def test_read_config_file_formats(tmp_path):
    (tmp_path / "a.yaml").write_text("seed: 42\n")
    (tmp_path / "b.json").write_text(json.dumps({"seed": 7}))
    assert read_config_file(tmp_path / "a.yaml") == {"seed": 42}
    assert read_config_file(tmp_path / "b.json") == {"seed": 7}


def test_read_config_file_never_raises(tmp_path):
    assert read_config_file(tmp_path / "missing.yaml") == {}
    bad = tmp_path / "bad.yaml"
    bad.write_text("seed: [unclosed")
    assert read_config_file(bad) == {}
    lst = tmp_path / "list.yaml"
    lst.write_text("- a\n- b\n")
    assert read_config_file(lst) == {}
    txt = tmp_path / "notes.txt"
    txt.write_text("seed: 1")
    assert read_config_file(txt) == {}


def test_read_run_config_merges_json_over_yaml(tmp_path):
    d = _run_dir(tmp_path)
    (d / "configs" / "experiment.yaml").write_text("seed: 42\nmodel_name: m\n")
    (d / "configs" / "experiment.json").write_text(json.dumps({"seed": 123}))
    assert read_run_config(d) == {"seed": 123, "model_name": "m"}


def test_read_run_config_missing_is_empty(tmp_path):
    assert read_run_config(_run_dir(tmp_path)) == {}
    assert read_run_config(tmp_path / "no_such_dir") == {}
