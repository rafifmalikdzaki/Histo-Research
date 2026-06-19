#!/usr/bin/env python3
"""
Sanity checks for src/clustering_metrics.py.

Run: python test_metrics.py   (exits non-zero on failure)

These guard against the regressions that made the original metrics untrustworthy:
  * Xie-Beni returning inf, and CH/silhouette returning 0/NaN, on well-separated data.
  * Metrics not ordering well-separated clusters above overlapping ones.
  * External metrics not recognising a clustering that matches ground truth.
"""

import sys
import numpy as np

from src.clustering_metrics import (
    normalize_embeddings, internal_metrics, external_metrics, xie_beni,
)


def _blobs(centers, spread, n_per=200, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for i, c in enumerate(centers):
        xs.append(rng.normal(c, spread, size=(n_per, len(c))))
        ys.append(np.full(n_per, i))
    return np.vstack(xs), np.concatenate(ys)


def main() -> int:
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
        print(("PASS" if cond else "FAIL"), "-", msg)

    centers = [[0, 0], [10, 10], [0, 10], [10, 0]]

    # Well-separated vs overlapping.
    Xsep, ysep = _blobs(centers, spread=0.5)
    Xover, yover = _blobs(centers, spread=6.0)

    msep = internal_metrics(Xsep, ysep)
    mover = internal_metrics(Xover, yover)

    check(np.isfinite(msep["xie_beni"]), "Xie-Beni is finite on separated blobs")
    check(msep["xie_beni"] < 0.5, f"Xie-Beni low on separated blobs (got {msep['xie_beni']:.4f})")
    check(msep["silhouette"] > 0.7, f"Silhouette high on separated blobs (got {msep['silhouette']:.3f})")
    check(msep["calinski_harabasz"] > 0, "Calinski-Harabasz positive (not 0) on separated blobs")
    check(msep["silhouette"] > mover["silhouette"], "Silhouette: separated > overlapping")
    check(msep["davies_bouldin"] < mover["davies_bouldin"], "Davies-Bouldin: separated < overlapping")
    check(msep["xie_beni"] < mover["xie_beni"], "Xie-Beni: separated < overlapping")

    # Degenerate: a single cluster -> NaN (not a misleading number), no crash.
    single = internal_metrics(Xsep, np.zeros(len(Xsep), dtype=int))
    check(np.isnan(single["silhouette"]), "Single-cluster silhouette -> NaN")
    check(np.isnan(single["xie_beni"]), "Single-cluster Xie-Beni -> NaN")

    # Non-contiguous labels (e.g. cluster ids {0, 2, 5}) handled without crash.
    noncontig = ysep.copy()
    noncontig[noncontig == 1] = 5
    noncontig[noncontig == 2] = 9
    check(np.isfinite(xie_beni(Xsep, noncontig)), "Xie-Beni finite with non-contiguous labels")

    # Normalization preserves shape and ordering of separability.
    Xn = normalize_embeddings(Xsep, "standardize_l2")
    check(Xn.shape == Xsep.shape, "normalize_embeddings preserves shape")

    # External metrics: perfect match -> ~1; random -> ~0.
    perfect = external_metrics(ysep, ysep, prefix="t_")
    check(perfect["t_ami"] > 0.99 and perfect["t_ari"] > 0.99, "External AMI/ARI ~1 on perfect match")
    rng = np.random.default_rng(1)
    rand_pred = rng.integers(0, 4, size=len(ysep))
    randm = external_metrics(ysep, rand_pred, prefix="t_")
    check(randm["t_ami"] < 0.05, f"External AMI ~0 on random labels (got {randm['t_ami']:.3f})")

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print("  -", f)
        return 1
    print("\nAll metric checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
