"""Shared seed-aligned statistics helpers (single source of truth).

Both stats entry points — ``evaluate_stats`` (per-run CSV path) and
``compute_statistical_significance`` (summary-CSV path) — historically carried
their own copy of the paired effect size and already drifted once (a NaN guard
present in only one copy). Import from here instead of duplicating.
"""

import numpy as np

__all__ = ["compute_cohens_d", "format_mean_std"]


def compute_cohens_d(group1, group2) -> float:
    """Compute the paired Cohen's d effect size (d_z).

    The callers always pass seed-aligned pairs, so the correct estimator is
    the mean paired difference over the standard deviation of the
    differences — not the independent-groups pooled standard deviation.

    Args:
        group1: First group values (paired with group2 by seed).
        group2: Second group values (paired with group1 by seed).

    Returns:
        Paired Cohen's d value, or 0.0 when the arms cannot be paired
        (unequal lengths, fewer than 2 pairs) or carry no signal
        (zero/NaN spread of differences).
    """
    arr1 = np.asarray(group1, dtype=float)
    arr2 = np.asarray(group2, dtype=float)
    if len(arr1) != len(arr2) or len(arr1) < 2:
        # Unequal arms cannot be paired elementwise (broadcasting would
        # silently compute garbage); callers must align by seed first.
        return 0.0
    diffs = arr1 - arr2
    sd = np.std(diffs, ddof=1)

    if sd == 0 or np.isnan(sd):
        return 0.0

    return float(np.mean(diffs) / sd)


def format_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    """Format mean ± std as a string; "N/A" for missing values.

    Accepts both ``None`` and ``NaN`` as missing (one historical copy
    crashed with ``TypeError`` on ``None`` via a bare ``np.isnan`` check).
    """
    if mean is None or std is None:
        return "N/A"
    if np.isnan(mean) or np.isnan(std):
        return "N/A"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"
