"""
Shared clustering entry point + canonical option lists for the histodae CLI.

Every CLI command that clusters embeddings (``histodae cluster``, ``biology``,
``per-magnification``) previously carried its own near-duplicate
``run_clustering`` and its own ``--method`` / ``--normalize`` choices. Those
copies drifted apart (e.g. ``gmm`` was accepted by ``biology`` but not
``per-magnification``) and some commands failed to validate ``--normalize`` at
parse time, letting an invalid value surface only later as a ValueError deep
inside the clustering code.

This module is the single source of truth for that shared behaviour. Commands
pull the canonical method list from ``CLUSTERING_METHODS`` (the canonical
normalizer list already lives in :mod:`histodae.clustering.metrics` as
``NORMALIZERS``), and route all clustering through :func:`run_clustering`, so a
fix made here applies to every command at once.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.mixture import GaussianMixture

# Canonical set of clustering methods accepted by every CLI command. Keep this
# the single source of truth for ``--method`` / ``--methods`` choices. The order
# matches the historical default ordering of ``histodae cluster --methods`` so
# existing invocations keep producing identical result rows.
CLUSTERING_METHODS = ("bisecting_kmeans", "kmeans", "gmm")


def run_clustering(embeddings: np.ndarray, method: str, k: int = 6) -> np.ndarray:
    """Run a clustering algorithm on embeddings, returning cluster labels.

    Args:
        embeddings: (n_samples, n_features) array to cluster.
        method: one of ``CLUSTERING_METHODS``.
        k: number of clusters / mixture components.

    Returns:
        Integer label array of shape ``(n_samples,)``.
    """
    if method == "bisecting_kmeans":
        model = BisectingKMeans(n_clusters=k, random_state=42, n_init=10)
    elif method == "kmeans":
        model = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    elif method == "gmm":
        model = GaussianMixture(n_components=k, random_state=42, n_init=5)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    return model.fit_predict(embeddings)
