"""Shared summary/tests CSV loader (src/utils/reporting.load_data)."""

import pandas as pd

from src.utils.reporting import load_data


def test_loads_both_csvs(tmp_path):
    summary = tmp_path / "summary.csv"
    tests = tmp_path / "tests.csv"
    pd.DataFrame([{"model": "m", "k": 6}]).to_csv(summary, index=False)
    pd.DataFrame([{"model": "m", "p": 0.1}]).to_csv(tests, index=False)
    summary_df, tests_df = load_data(str(summary), str(tests))
    assert len(summary_df) == 1 and len(tests_df) == 1


def test_missing_tests_yields_none(tmp_path):
    summary = tmp_path / "summary.csv"
    pd.DataFrame([{"model": "m", "k": 6}]).to_csv(summary, index=False)
    summary_df, tests_df = load_data(str(summary), str(tmp_path / "nope.csv"))
    assert len(summary_df) == 1 and tests_df is None
    _, tests_df = load_data(str(summary))
    assert tests_df is None
