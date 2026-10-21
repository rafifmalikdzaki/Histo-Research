#!/usr/bin/env python3
"""
Regenerate per-epoch embedding visualisations offline (no GPU, no training).

When training is run with `--epoch-viz dump` it saves the per-epoch embeddings and
metadata but skips the expensive rendering (t-SNE/UMAP/clustering plots), cutting
~40% off each epoch. This script rebuilds the same information from those dumps:

  * per-epoch internal clustering metrics -> epoch_metric_evolution.csv
  * metric-evolution line plot           -> epoch_metric_evolution.png
  * 2D projection scatter (PCA, optional t-SNE) coloured by treatment_group, for
    the first and last epoch                -> projection_<phase>_epoch_<n>.png

Everything here is a pure function of the saved (embeddings, metadata); no model
weights or GPU are needed. Run it once after a `--epoch-viz dump` run.

Usage:
    histodae regenerate-viz --run-dir auto_analysis/<run> [--phase val] \
        [--k 6] [--tsne]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from histodae.clustering.metrics import (
    NORMALIZERS,
    external_metrics,
    internal_metrics,
    normalize_embeddings,
)


def _epoch_files(emb_dir: Path, phase: str):
    files = sorted(glob.glob(str(emb_dir / f"{phase}_embeddings_epoch_*.npy")))
    out = []
    for f in files:
        ep = int(os.path.basename(f).split("_")[-1].split(".")[0])
        meta = f.replace("_embeddings_epoch_", "_metadata_epoch_").replace(".npy", ".csv")
        out.append((ep, Path(f), Path(meta)))
    return out


def _project(X: np.ndarray, use_tsne: bool) -> np.ndarray:
    if use_tsne:
        from sklearn.manifold import TSNE
        n = min(30, max(5, len(X) // 4))
        return TSNE(n_components=2, perplexity=n, random_state=42, init="pca").fit_transform(X)
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=42).fit_transform(X)


def _scatter(X2: np.ndarray, labels, title: str, out: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    cats = pd.Series(labels).astype("category")
    for code, name in enumerate(cats.cat.categories):
        m = cats.cat.codes.to_numpy() == code
        ax.scatter(X2[m, 0], X2[m, 1], s=6, alpha=0.6, label=str(name))
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def run(run_dir: Path, phase: str, k: int, normalize: str, use_tsne: bool) -> int:
    emb_dir = run_dir / "embeddings"
    if not emb_dir.exists():
        print(f"❌ no embeddings dir in {run_dir}")
        return 1
    epochs = _epoch_files(emb_dir, phase)
    if not epochs:
        print(f"❌ no per-epoch dumps ({phase}_embeddings_epoch_*.npy) in {emb_dir}. "
              "Was training run with --epoch-viz dump/render?")
        return 1

    from sklearn.cluster import KMeans

    out_dir = emb_dir / "regenerated_viz"
    out_dir.mkdir(exist_ok=True)
    rows = []
    for ep, ef, mf in epochs:
        X = normalize_embeddings(np.load(ef), normalize)
        meta = pd.read_csv(mf) if mf.exists() else None
        kk = min(k, max(2, len(X) - 1))
        labels = KMeans(n_clusters=kk, random_state=42, n_init=10).fit_predict(X)
        row = {"epoch": ep, "n": len(X), "k": kk}
        row.update(internal_metrics(X, labels))
        if meta is not None and "treatment_group" in meta.columns and meta["treatment_group"].notna().any():
            row.update(external_metrics(meta["treatment_group"].to_numpy(), labels, prefix="treat_"))
        rows.append(row)
        print(f"  epoch {ep:>3}: sil={row.get('silhouette', float('nan')):.4f}"
              f" treat_ami={row.get('treat_ami', float('nan')):.4f}")

    df = pd.DataFrame(rows).sort_values("epoch")
    csv = out_dir / f"epoch_metric_evolution_{phase}.csv"
    df.to_csv(csv, index=False)
    print(f"✓ {csv}")

    # Metric-evolution line plot.
    metrics = [m for m in ("silhouette", "davies_bouldin", "calinski_harabasz",
                           "xie_beni", "treat_ami", "treat_ari") if m in df.columns]
    if metrics:
        fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 3.2))
        if len(metrics) == 1:
            axes = [axes]
        for ax, m in zip(axes, metrics):
            ax.plot(df["epoch"], df[m], marker="o")
            ax.set_title(m)
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
        fig.suptitle(f"Embedding metric evolution ({phase})")
        fig.tight_layout()
        p = out_dir / f"epoch_metric_evolution_{phase}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        print(f"✓ {p}")

    # First/last epoch projection scatter, coloured by treatment group.
    for ep, ef, mf in (epochs[0], epochs[-1]):
        X = normalize_embeddings(np.load(ef), normalize)
        meta = pd.read_csv(mf) if mf.exists() else None
        labels = (meta["treatment_group"].to_numpy()
                  if meta is not None and "treatment_group" in meta.columns
                  else np.zeros(len(X)))
        X2 = _project(X, use_tsne)
        tag = "tsne" if use_tsne else "pca"
        p = out_dir / f"projection_{phase}_epoch_{ep:03d}_{tag}.png"
        _scatter(X2, labels, f"{phase} epoch {ep} ({tag.upper()}) by treatment", p)
        print(f"✓ {p}")

    print(f"✓ regenerated viz -> {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--phase", default="val", choices=["train", "val"])
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--normalize", default="standardize_l2", choices=list(NORMALIZERS))
    ap.add_argument("--tsne", action="store_true", help="Use t-SNE instead of PCA (slower).")
    args = ap.parse_args(argv)
    return run(args.run_dir, args.phase, args.k, args.normalize, args.tsne)


if __name__ == "__main__":
    sys.exit(main())
