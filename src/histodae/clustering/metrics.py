"""
Robust clustering metrics + embedding normalization for the histodae analysis.

Why this module exists
----------------------
The original metric code (in run_clustering_analysis.py) clustered raw, unnormalized
autoencoder features and computed a Xie-Beni index that assumed cluster labels were a
contiguous 0..k-1 range. In practice GMM/empty clusters produced non-contiguous labels,
empty-slice means (NaN centers), and zero separation -> the index returned `inf` for
~95% of rows and Calinski-Harabasz collapsed to 0. That made every reported internal
index untrustworthy.

This module provides:
  * normalize_embeddings: standardize and/or L2-normalize before clustering,
  * internal_metrics: silhouette / davies_bouldin / calinski_harabasz / xie_beni,
    all robust to degenerate (single / empty) clusters,
  * external_metrics: AMI / ARI / NMI / homogeneity / completeness / purity of a
    clustering against a categorical ground-truth label (e.g. treatment_group).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
)

NORMALIZERS = ("none", "standardize", "l2", "standardize_l2")


def normalize_embeddings(emb: np.ndarray, method: str = "standardize_l2") -> np.ndarray:
    """Normalize an (n_samples, n_features) embedding matrix.

    standardize    : zero-mean, unit-variance per feature (guards zero-variance dims).
    l2             : scale each row to unit L2 norm.
    standardize_l2 : standardize then L2 (recommended default for AE features —
                     removes per-feature scale, then puts samples on a sphere so
                     Euclidean clustering reflects direction not magnitude).
    """
    if method not in NORMALIZERS:
        raise ValueError(f"Unknown normalize method {method!r}; choose from {NORMALIZERS}")
    x = np.asarray(emb, dtype=np.float64)
    if method in ("standardize", "standardize_l2"):
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        x = (x - mu) / sd
    if method in ("l2", "standardize_l2"):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        x = x / norms
    return x


def xie_beni(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Xie-Beni index (lower is better), robust to non-contiguous / empty clusters.

    XB = (1/n) * sum_i ||x_i - c_{label_i}||^2 / (n * min_{a<b} ||c_a - c_b||^2)
    Returns NaN when fewer than 2 non-empty clusters exist (index undefined).
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    uniq = np.unique(labels)
    center_list = []
    for u in uniq:
        pts = embeddings[labels == u]
        if len(pts) == 0:
            continue
        center_list.append(pts.mean(axis=0))
    centers = np.asarray(center_list)
    if len(centers) < 2:
        return float("nan")

    n = len(embeddings)
    compactness = 0.0
    for u, c in zip(uniq, centers):
        pts = embeddings[labels == u]
        compactness += np.sum(np.linalg.norm(pts - c, axis=1) ** 2)

    # Minimum squared distance between distinct cluster centers.
    min_sep = np.inf
    for a in range(len(centers)):
        for b in range(a + 1, len(centers)):
            d = float(np.sum((centers[a] - centers[b]) ** 2))
            if d < min_sep:
                min_sep = d
    eps = 1e-12
    return float(compactness / (n * max(min_sep, eps)))


def internal_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """Internal cluster-validity indices. Degenerate clusterings (a single
    populated cluster) yield NaN for indices that require >=2 clusters rather
    than a misleading 0."""
    labels = np.asarray(labels)
    n_clusters = len(np.unique(labels))
    out: dict[str, Any] = {"n_effective_clusters": int(n_clusters)}

    if n_clusters < 2 or n_clusters >= len(embeddings):
        return {**out, "silhouette": np.nan, "davies_bouldin": np.nan,
                "calinski_harabasz": np.nan, "xie_beni": np.nan}

    try:
        out["silhouette"] = float(silhouette_score(embeddings, labels))
    except Exception:
        out["silhouette"] = np.nan
    try:
        out["davies_bouldin"] = float(davies_bouldin_score(embeddings, labels))
    except Exception:
        out["davies_bouldin"] = np.nan
    try:
        out["calinski_harabasz"] = float(calinski_harabasz_score(embeddings, labels))
    except Exception:
        out["calinski_harabasz"] = np.nan
    out["xie_beni"] = xie_beni(embeddings, labels)
    return out


def purity_score(y_true, y_pred) -> float:
    """Cluster purity: fraction of samples assigned to the majority true class
    of their cluster."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    total = 0
    for c in np.unique(y_pred):
        mask = y_pred == c
        if mask.sum() == 0:
            continue
        vals, counts = np.unique(y_true[mask], return_counts=True)
        total += counts.max()
    return float(total / len(y_true))


def external_metrics(y_true, y_pred, prefix: str = "") -> dict:
    """Agreement of a clustering (y_pred) with categorical ground truth (y_true).

    `prefix` namespaces the keys (e.g. 'treatment_' or 'diet_'/'mag_') so multiple
    label sets can be reported side by side.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p = prefix
    return {
        f"{p}ami": float(adjusted_mutual_info_score(y_true, y_pred)),
        f"{p}ari": float(adjusted_rand_score(y_true, y_pred)),
        f"{p}nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        f"{p}homogeneity": float(homogeneity_score(y_true, y_pred)),
        f"{p}completeness": float(completeness_score(y_true, y_pred)),
        f"{p}purity": purity_score(y_true, y_pred),
    }
