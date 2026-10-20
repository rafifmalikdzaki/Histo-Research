"""Regression tests for src/evaluate_stats.py statistics wiring.

Guards the real bugs found 2026-10-18: the dataset filter that matched
nothing, seed-unaware test pairing, silent duplicate-seed overwrite,
independent-groups Cohen's d on paired data, and population std reporting.
"""

import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from histodae.cli.evaluate import (
    aggregate_multi_seed_results,
    compute_cohens_d,
    compute_statistical_tests,
)
from histodae.io.run_discovery import find_experiment_runs


def _make_run(base: Path, name: str, seed: int, dataset: str,
              silhouette: float) -> Path:
    d = base / name
    (d / "configs").mkdir(parents=True)
    with open(d / "configs" / "experiment.yaml", "w") as f:
        yaml.safe_dump({"dataset": dataset, "seed": seed}, f)
    with open(d / "clustering_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "silhouette", "davies_bouldin"])
        w.writerow([0, silhouette, 2.0])
    return d


@pytest.fixture()
def runs_dir(tmp_path):
    base = tmp_path / "outputs"
    base.mkdir()
    return base


def test_dataset_filter_matches_config_tag(runs_dir):
    _make_run(runs_dir, "exp_20261018_000001_dae_seed42", 42, "hepar", 0.5)
    _make_run(runs_dir, "exp_20261018_000002_dae_seed43", 43, "pannuke", 0.6)
    found = find_experiment_runs(str(runs_dir), model_name="dae",
                                 dataset="HeparUnifiedPNG")
    assert [p.name for p in found] == ["exp_20261018_000001_dae_seed42"]


def test_dataset_filter_matches_dir_name(runs_dir):
    _make_run(runs_dir, "exp_foo_HeparUnifiedPNG_seed42", 42, "hepar", 0.5)
    found = find_experiment_runs(str(runs_dir), dataset="hepar")
    assert len(found) == 1


def test_aggregate_sorts_by_seed_and_warns_on_duplicates(runs_dir, capsys):
    _make_run(runs_dir, "exp_b_dae_seed123", 123, "hepar", 0.6)
    _make_run(runs_dir, "exp_a_dae_seed42", 42, "hepar", 0.5)
    _make_run(runs_dir, "exp_c_dae_seed42_dup", 42, "hepar", 0.9)
    agg = aggregate_multi_seed_results(
        find_experiment_runs(str(runs_dir), model_name="dae"))
    assert agg["silhouette"]["seeds"] == [42, 123]
    assert agg["silhouette"]["values"] == pytest.approx([0.5, 0.6])
    assert "Duplicate seed 42" in capsys.readouterr().out


def test_aggregate_uses_sample_std(runs_dir):
    _make_run(runs_dir, "exp_a_seed42", 42, "hepar", 0.5)
    _make_run(runs_dir, "exp_b_seed123", 123, "hepar", 0.7)
    _make_run(runs_dir, "exp_c_seed456", 456, "hepar", 0.6)
    agg = aggregate_multi_seed_results(
        find_experiment_runs(str(runs_dir)))
    assert agg["silhouette"]["std"] == pytest.approx(np.std([0.5, 0.7, 0.6], ddof=1))


def _agg(values, seeds):
    return {"values": list(values), "seeds": list(seeds)}


def test_tests_align_pairs_by_seed_not_position():
    a = {"silhouette": _agg([0.5, 0.6, 0.9], [42, 123, 456])}
    b = {"silhouette": _agg([0.1, 0.2], [456, 42])}
    res = compute_statistical_tests(a, b, test_type="ttest")["silhouette"]
    assert res["n_paired"] == 2
    assert res["paired_seeds"] == [42, 456]
    # mean_diff uses seed-aligned pairs: (0.5-0.2) and (0.9-0.1)
    assert res["mean_diff"] == pytest.approx(0.55)


def test_tests_refuse_untagged_unequal_lengths():
    a = {"silhouette": {"values": [0.5, 0.6, 0.9]}}
    b = {"silhouette": {"values": [0.1, 0.2]}}
    res = compute_statistical_tests(a, b)["silhouette"]
    assert res["statistic"] is None
    assert "cannot be paired" in res["note"]


def test_paired_cohens_d_uses_difference_sd():
    d = compute_cohens_d([1.0, 2.0, 3.0], [1.5, 2.5, 2.0])
    diffs = np.array([-0.5, -0.5, 1.0])
    assert d == pytest.approx(diffs.mean() / diffs.std(ddof=1))
    assert compute_cohens_d([1.0], [2.0]) == 0.0
    assert compute_cohens_d([1.0, 1.0], [1.0, 1.0]) == 0.0
