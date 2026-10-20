"""Statistics layer: effect sizes, seed pairing, and result tables."""

from .metrics import compute_cohens_d, format_mean_std, summarize_values
from .tables import (
    METRICS,
    apply_holm_correction,
    compute_summary_statistics,
    generate_statistical_tests_table,
    generate_summary_table,
    run_statistical_tests,
    save_latex_table,
)

__all__ = [
    "METRICS",
    "apply_holm_correction",
    "compute_cohens_d",
    "compute_summary_statistics",
    "format_mean_std",
    "generate_statistical_tests_table",
    "generate_summary_table",
    "run_statistical_tests",
    "save_latex_table",
    "summarize_values",
]
