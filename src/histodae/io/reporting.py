"""Shared loaders for the report generators (plots + LaTeX tables).

Both report stages consume the same two CSVs — the per-model summary and the
optional statistical-tests table — so the loader lives here instead of being
duplicated in each script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

__all__ = ["load_data", "load_raw_metrics"]


def load_raw_metrics(input_path: str) -> pd.DataFrame:
    """Load a raw per-run clustering-metrics CSV, exiting when missing."""
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path)
    print(f"✓ Loaded {len(df)} results from {input_path}")
    return df


def load_data(
    summary_path: str,
    tests_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load summary and statistical-tests CSVs.

    The tests table is optional: a missing path (or ``None``) yields ``None``
    so both generators degrade to summary-only output (e.g. seedless legacy
    runs that produce no paired tests).
    """
    summary_df = pd.read_csv(summary_path)
    print(f"✓ Loaded summary: {len(summary_df)} rows")

    tests_df = None
    if tests_path and Path(tests_path).exists():
        tests_df = pd.read_csv(tests_path)
        print(f"✓ Loaded statistical tests: {len(tests_df)} rows")

    return summary_df, tests_df
