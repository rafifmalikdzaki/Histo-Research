"""Guards the single-source paired Cohen's d in src/utils/stats.py.

Both stats entry points must share one identical function object so the
estimator cannot drift again (the NaN guard once existed in only one copy).
"""

import numpy as np
import pytest

from src import compute_statistical_significance as sig
from src import evaluate_stats as ev

# NOTE: import via the installed top-level path — the same path the scripts
# themselves use (`from utils.stats import ...`) — so `is` compares one
# shared module object, not the `src.utils.stats` namespace-package twin.
from utils import stats as shared


def test_both_entry_points_share_one_function_object():
    assert ev.compute_cohens_d is shared.compute_cohens_d
    assert sig.compute_cohens_d is shared.compute_cohens_d


def test_paired_d_z_value():
    d = shared.compute_cohens_d([1.0, 2.0, 3.0], [1.5, 2.5, 2.0])
    diffs = np.array([-0.5, -0.5, 1.0])
    assert d == float(np.mean(diffs) / np.std(diffs, ddof=1))


def test_degenerate_arms_return_zero():
    assert shared.compute_cohens_d([1.0], [2.0]) == 0.0  # < 2 pairs
    assert shared.compute_cohens_d([1.0, 2.0], [1.0]) == 0.0  # unequal arms
    assert shared.compute_cohens_d([1.0, 1.0], [1.0, 1.0]) == 0.0  # zero spread


def test_summarize_values_kernel():
    s = shared.summarize_values([1.0, 2.0, 3.0, 4.0])
    assert s["mean"] == 2.5
    assert s["std"] == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0], ddof=1))
    assert (s["min"], s["max"], s["n"]) == (1.0, 4.0, 4)


def test_summarize_values_single_and_empty():
    s = shared.summarize_values([5.0])
    assert (s["mean"], s["std"], s["n"]) == (5.0, 0.0, 1)
    with pytest.raises(ValueError):
        shared.summarize_values([])


def test_format_mean_std_shared_and_none_safe():
    assert ev.format_mean_std is shared.format_mean_std
    assert sig.format_mean_std is shared.format_mean_std
    assert shared.format_mean_std(0.5, 0.12345) == "0.5000 ± 0.1235"
    assert shared.format_mean_std(0.5, 0.12345, 2) == "0.50 ± 0.12"
    assert shared.format_mean_std(None, 0.1) == "N/A"
    assert shared.format_mean_std(0.5, None) == "N/A"
    assert shared.format_mean_std(float("nan"), 0.1) == "N/A"
    assert shared.format_mean_std(0.5, float("nan")) == "N/A"
