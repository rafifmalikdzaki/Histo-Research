#!/usr/bin/env python3
"""
Per-magnification clustering: measure treatment signal with magnification fixed.

The pooled analysis (cluster_biology_analysis.py) found the unsupervised clusters
associate with imaging magnification MORE than with treatment arm (Cramér's V_mag >
V_treat). That makes a pooled "treatment phenotype" claim unsafe — the clusters
could be separating 10x/20x/40x rather than biology.

This script controls the confound directly: it clusters the test embeddings
*within each magnification stratum* and reports treatment AMI/ARI per stratum. If
treatment agreement survives within strata, the biological signal is real; if it
collapses toward 0, the pooled signal was magnification-driven.

Usage:
    python per_magnification_analysis.py \
        --embeddings <run>/embeddings/test_embeddings.npy \
        --metadata   <run>/embeddings/test_metadata.csv \
        --output results/per_magnification_<tag>.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from clustering_metrics import normalize_embeddings, external_metrics, internal_metrics  # noqa: E402


def run(embeddings_path: Path, metadata_path: Path, output: Path,
        normalize: str, method: str, min_per_stratum: int) -> int:
    from sklearn.cluster import KMeans, BisectingKMeans

    emb = np.load(embeddings_path)
    meta = pd.read_csv(metadata_path)
    if len(emb) != len(meta):
        print(f"❌ length mismatch {len(emb)} vs {len(meta)}")
        return 1

    rows = []
    # Pooled reference (k = number of treatment groups overall).
    X_all = normalize_embeddings(emb, normalize)
    n_treat_all = meta["treatment_group"].nunique()
    for tag, X, sub, k in _iter_strata(X_all, meta, min_per_stratum):
        if k < 2 or len(X) <= k:
            print(f"  skip {tag}: too few treatments/samples")
            continue
        if method == "bisecting_kmeans":
            labels = BisectingKMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        else:
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        im = internal_metrics(X, labels)
        ext = external_metrics(sub["treatment_group"].to_numpy(), labels, prefix="treat_")
        diet = external_metrics(sub["diet"].to_numpy(), labels, prefix="diet_") \
            if sub["diet"].nunique() > 1 else {}
        row = {
            "stratum": tag, "n_tiles": len(X), "k": k,
            "n_treatments": sub["treatment_group"].nunique(),
            "silhouette": im["silhouette"],
            "treat_ami": ext["treat_ami"], "treat_ari": ext["treat_ari"],
            "treat_nmi": ext["treat_nmi"], "treat_purity": ext["treat_purity"],
        }
        row.update({k2: v for k2, v in diet.items() if k2 in ("diet_ami", "diet_ari")})
        rows.append(row)
        print(f"  {tag:<14} n={len(X):<5} k={k} treat_AMI={ext['treat_ami']:.3f} "
              f"treat_ARI={ext['treat_ari']:.3f}")

    out = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"✓ wrote {output}")

    # Verdict: does within-stratum treatment AMI hold up vs the pooled value?
    strat = out[out["stratum"] != "POOLED"]
    pooled = out[out["stratum"] == "POOLED"]
    if not strat.empty and not pooled.empty:
        mean_strat = strat["treat_ami"].mean()
        pooled_ami = pooled["treat_ami"].iloc[0]
        verdict = ("treatment signal SURVIVES magnification control"
                   if mean_strat >= 0.5 * pooled_ami else
                   "treatment signal largely COLLAPSES within magnification strata "
                   "(pooled signal was confound-driven)")
        print(f"  -> mean within-stratum treat_AMI={mean_strat:.3f} vs "
              f"pooled={pooled_ami:.3f}: {verdict}")
    return 0


def _iter_strata(X_all, meta, min_per_stratum):
    # Pooled first.
    yield "POOLED", X_all, meta, int(meta["treatment_group"].nunique())
    for mag, idx in meta.groupby("magnification").groups.items():
        idx = np.asarray(idx)
        sub = meta.loc[idx]
        if len(idx) < min_per_stratum:
            continue
        k = int(sub["treatment_group"].nunique())
        yield f"mag_{mag}", X_all[idx], sub.reset_index(drop=True), k


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--normalize", default="standardize_l2")
    ap.add_argument("--method", default="kmeans", choices=["kmeans", "bisecting_kmeans"])
    ap.add_argument("--min-per-stratum", type=int, default=100)
    args = ap.parse_args()
    return run(args.embeddings, args.metadata, args.output, args.normalize,
               args.method, args.min_per_stratum)


if __name__ == "__main__":
    sys.exit(main())
