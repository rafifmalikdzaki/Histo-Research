"""Sanity checks for src/clustering_metrics.py (pytest port of test_metrics.py).

These guard against the regressions that made the original metrics untrustworthy:
  * Xie-Beni returning inf, and CH/silhouette returning 0/NaN, on well-separated data.
  * Metrics not ordering well-separated clusters above overlapping ones.
  * External metrics not recognising a clustering that matches ground truth.
"""

import numpy as np

from histodae.clustering.metrics import (
    external_metrics,
    internal_metrics,
    normalize_embeddings,
    xie_beni,
)

CENTERS = [[0, 0], [10, 10], [0, 10], [10, 0]]


def _blobs(centers, spread, n_per=200, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for i, c in enumerate(centers):
        xs.append(rng.normal(c, spread, size=(n_per, len(c))))
        ys.append(np.full(n_per, i))
    return np.vstack(xs), np.concatenate(ys)


def test_xie_beni_finite_and_low_on_separated():
    Xsep, ysep = _blobs(CENTERS, spread=0.5)
    msep = internal_metrics(Xsep, ysep)
    assert np.isfinite(msep["xie_beni"])
    assert msep["xie_beni"] < 0.5


def test_internal_metrics_order_separated_above_overlapping():
    Xsep, ysep = _blobs(CENTERS, spread=0.5)
    Xover, yover = _blobs(CENTERS, spread=6.0)
    msep = internal_metrics(Xsep, ysep)
    mover = internal_metrics(Xover, yover)
    assert msep["silhouette"] > 0.7
    assert msep["calinski_harabasz"] > 0
    assert msep["silhouette"] > mover["silhouette"]
    assert msep["davies_bouldin"] < mover["davies_bouldin"]
    assert msep["xie_beni"] < mover["xie_beni"]


def test_single_cluster_returns_nan_not_crash():
    Xsep, _ = _blobs(CENTERS, spread=0.5)
    single = internal_metrics(Xsep, np.zeros(len(Xsep), dtype=int))
    assert np.isnan(single["silhouette"])
    assert np.isnan(single["xie_beni"])


def test_noncontiguous_labels():
    Xsep, ysep = _blobs(CENTERS, spread=0.5)
    noncontig = ysep.copy()
    noncontig[noncontig == 1] = 5
    noncontig[noncontig == 2] = 9
    assert np.isfinite(xie_beni(Xsep, noncontig))


def test_normalize_preserves_shape():
    Xsep, _ = _blobs(CENTERS, spread=0.5)
    assert normalize_embeddings(Xsep, "standardize_l2").shape == Xsep.shape


def test_external_perfect_match_scores_one():
    _, ysep = _blobs(CENTERS, spread=0.5)
    perfect = external_metrics(ysep, ysep, prefix="t_")
    assert perfect["t_ami"] > 0.99
    assert perfect["t_ari"] > 0.99


def test_external_random_labels_score_zero():
    _, ysep = _blobs(CENTERS, spread=0.5)
    rng = np.random.default_rng(1)
    rand_pred = rng.integers(0, 4, size=len(ysep))
    assert external_metrics(ysep, rand_pred, prefix="t_")["t_ami"] < 0.05
