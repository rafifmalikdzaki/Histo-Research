"""Regression tests for src/compute_statistical_significance.py wiring.

Mirrors the evaluate_stats fixes in the summary-CSV path: seed-aligned
pairing (never positional truncation), duplicate-seed averaging, safe cell
parsing, paired Cohen's d, and sample std reporting.
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.seed_pairing import _align_seed_pairs
from src.utils.stats import compute_cohens_d
from src.utils.stats_tables import (
    _parse_literal_list,
    compute_summary_statistics,
    run_statistical_tests,
)


def _row(model, method, values, seeds=None):
    row = {"model": model, "clustering_method": method, "k": 6,
           "n_samples": 100, "silhouette_values": str(list(values)),
           "silhouette_mean": float(np.mean(list(values)))}
    if seeds is not None:
        row["silhouette_seeds"] = str(list(seeds))
    return row


def test_parse_literal_list_never_executes():
    assert _parse_literal_list("[0.1, 0.2]") == [0.1, 0.2]
    assert _parse_literal_list("[]") == []
    assert _parse_literal_list("not a list") == []
    assert _parse_literal_list("__import__('os').system('x')") == []
    assert _parse_literal_list([1]) == [1]


def test_align_pairs_by_seed_with_duplicates_averaged():
    a, b = _align_seed_pairs("m", "base", "kmeans", "silhouette",
                             [0.5, 0.7, 0.9], [42, 42, 123],
                             [0.1, 0.2], [42, 123])
    assert a == pytest.approx([0.6, 0.9])
    assert b == pytest.approx([0.1, 0.2])


def test_align_refuses_untagged_unequal():
    a, b = _align_seed_pairs("m", "base", "kmeans", "silhouette",
                             [0.5, 0.6, 0.9], [], [0.1, 0.2], [])
    assert (a, b) == ([], [])


def test_align_keeps_equal_legacy_arms():
    a, b = _align_seed_pairs("m", "base", "kmeans", "silhouette",
                             [0.5, 0.6], [], [0.1, 0.2], [])
    assert (a, b) == ([0.5, 0.6], [0.1, 0.2])


def test_align_treats_missing_sentinels_as_untagged():
    # Legacy -1 sentinels, None (new runs), and NaN (CSV round-trip) must
    # never pair rows onto one bogus shared seed: fully-untagged arms fall
    # back to positional pairing at equal length.
    for missing in ([-1, -1], [None, None], [float("nan"), float("nan")]):
        a, b = _align_seed_pairs("m", "base", "kmeans", "silhouette",
                                 [0.5, 0.6], missing, [0.1, 0.2], missing)
        assert (a, b) == ([0.5, 0.6], [0.1, 0.2])


def test_align_drops_unknown_seeds_but_keeps_real_pairs():
    a, b = _align_seed_pairs("m", "base", "kmeans", "silhouette",
                             [0.5, 0.9, 0.7], [-1, 42, 123],
                             [0.2, 0.4], [42, 123])
    assert a == pytest.approx([0.9, 0.7])
    assert b == pytest.approx([0.2, 0.4])


def test_align_skips_when_only_one_arm_tagged():
    a, b = _align_seed_pairs("m", "base", "kmeans", "silhouette",
                             [0.5, 0.6], [42, 123], [0.1, 0.2], [-1, -1])
    assert (a, b) == ([], [])


def test_align_matches_integral_float_seeds():
    a, b = _align_seed_pairs("m", "base", "kmeans", "silhouette",
                             [0.5, 0.6], [42.0, 123.0], [0.1, 0.2], [42, 123])
    assert (a, b) == ([0.5, 0.6], [0.1, 0.2])


def test_run_tests_pairs_by_seed_end_to_end():
    df = pd.DataFrame([
        _row("proposed", "kmeans", [0.5, 0.7, 0.9], [42, 123, 456]),
        _row("baseline", "kmeans", [0.2, 0.4], [42, 456]),
    ])
    out = run_statistical_tests(df, "baseline", metrics=["silhouette"])
    assert len(out) == 1
    row = out.iloc[0]
    # aligned pairs: (0.5, 0.2), (0.9, 0.4) -> mean diff 0.4
    assert row["mean_diff"] == pytest.approx(0.4)
    assert row["n_seeds_model"] == 2
    assert row["n_seeds_baseline"] == 2


def test_paired_cohens_d():
    d = compute_cohens_d(np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.5, 2.0]))
    diffs = np.array([-0.5, -0.5, 1.0])
    assert d == pytest.approx(diffs.mean() / diffs.std(ddof=1))
    assert compute_cohens_d(np.array([1.0]), np.array([2.0, 3.0])) == 0.0


def test_summary_reports_sample_std_and_seed_tags():
    df = pd.DataFrame([
        {"model": "m", "method": "k", "model_col": 1, "clustering_method": "kmeans",
         "k": 6, "n_samples": 10, "seed": 42, "silhouette": 0.5},
        {"model": "m", "method": "k", "model_col": 1, "clustering_method": "kmeans",
         "k": 6, "n_samples": 10, "seed": 123, "silhouette": 0.7},
        {"model": "m", "method": "k", "model_col": 1, "clustering_method": "kmeans",
         "k": 6, "n_samples": 10, "seed": 456, "silhouette": 0.6},
    ])
    out = compute_summary_statistics(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["silhouette_std"] == pytest.approx(np.std([0.5, 0.7, 0.6], ddof=1))
    assert row["silhouette_seeds"] == str([42, 123, 456])
