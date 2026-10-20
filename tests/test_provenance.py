"""Run provenance: every experiment dir must be self-describing.

A results table that cannot be traced back to a commit and a split is not
trustworthy, so each run directory carries a ``manifest.json``.
"""

import json
from pathlib import Path

from histodae.utils.provenance import data_fingerprint, git_commit, write_run_manifest


def test_write_run_manifest_records_provenance(tmp_path):
    path = write_run_manifest(
        tmp_path,
        seed=42,
        model_name="dae_kan_attention",
        dataset="hepar",
        hyperparameters={"latent_dim": 128},
    )
    assert path == tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())

    assert manifest["seed"] == 42
    assert manifest["model_name"] == "dae_kan_attention"
    assert manifest["dataset"] == "hepar"
    assert manifest["hyperparameters"]["latent_dim"] == 128
    assert manifest["created_at"]
    assert manifest["python_version"]


def test_manifest_records_the_git_commit(tmp_path):
    commit = git_commit()
    assert commit is None or len(commit) == 40
    manifest = json.loads(write_run_manifest(tmp_path).read_text())
    assert manifest["git_commit"] == commit


def test_data_fingerprint_hashes_split_csvs(tmp_path):
    csv = tmp_path / "train_hepar.csv"
    csv.write_text("specimen_id,label\ns1,0\n")
    fingerprint = data_fingerprint([csv, tmp_path / "missing.csv"])
    assert list(fingerprint["csv_sha256"]) == [str(csv)]
    assert len(fingerprint["csv_sha256"][str(csv)]) == 64


def test_data_fingerprint_tolerates_missing_files(tmp_path):
    fingerprint = data_fingerprint([tmp_path / "nope.csv"])
    assert fingerprint["csv_sha256"] == {}


def test_experiment_creation_writes_a_manifest(tmp_path):
    from histodae.utils.experiment import create_experiment_config

    exp_dir, _config = create_experiment_config(
        "prov_test",
        custom_params={"seed": 7, "model_name": "baseline", "dataset": "hepar"},
        base_dir=str(tmp_path),
    )
    manifest_path = Path(exp_dir) / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["seed"] == 7
    assert manifest["model_name"] == "baseline"
