#!/usr/bin/env python3
"""
Cluster <-> biology association analysis + pathologist-review materials.

Answers Reviewer 2's "is there a treatment-induced phenotype?" objection
quantitatively, and produces the materials a pathologist needs to validate the
clusters morphologically.

Given an embeddings .npy and its aligned biology metadata sidecar (written by
extract_embeddings_for_clustering.py), this script:

  1. normalises embeddings and clusters them (reusing the trusted metric path),
  2. builds contingency tables cluster x {treatment_group, diet, magnification},
  3. tests association with chi-square and reports Cramer's V (effect size),
  4. computes per-cluster treatment ENRICHMENT (observed/expected odds),
  5. exports per-cluster tile montages (centroid-nearest + random) + attention
     overlays placeholder hooks, and a blank pathologist scoring template,

into <output-dir>/ (default: pathologist_review/).

The magnification association is a CONFOUND check: we want clusters to track
treatment_group (high Cramer's V) far more than magnification (low Cramer's V).

Usage:
    histodae biology \
        --embeddings auto_analysis/<run>/embeddings/test_embeddings.npy \
        --metadata   auto_analysis/<run>/embeddings/test_metadata.csv \
        --k 6 --method kmeans --output-dir pathologist_review/<run>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from histodae.clustering import CLUSTERING_METHODS, run_clustering
from histodae.clustering.metrics import NORMALIZERS, normalize_embeddings

ASSOC_COLUMNS = ["treatment_group", "diet", "magnification"]


def cramers_v(contingency: np.ndarray) -> tuple[float, float, int]:
    """Bias-corrected Cramer's V for a contingency table.

    Returns (V, chi2_pvalue, dof). V in [0,1]; ~0 = independent, 1 = perfect
    association. Bias correction follows Bergsma (2013) so small tables are not
    inflated.
    """
    from scipy.stats import chi2_contingency

    contingency = np.asarray(contingency, dtype=float)
    if contingency.size == 0 or contingency.sum() == 0:
        return float("nan"), float("nan"), 0
    # Drop all-zero rows/cols so chi2 is defined.
    contingency = contingency[contingency.sum(1) > 0][:, contingency.sum(0) > 0]
    if min(contingency.shape) < 2:
        return float("nan"), float("nan"), 0
    chi2, p, dof, _ = chi2_contingency(contingency)
    n = contingency.sum()
    phi2 = chi2 / n
    r, k = contingency.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    v = float(np.sqrt(phi2corr / denom)) if denom > 0 else float("nan")
    return v, float(p), int(dof)


def enrichment_table(labels: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """Per (cluster, group) observed/expected ratio under independence.

    ratio > 1 => the group is over-represented in that cluster. This is the
    'cluster X is enriched for treatment Y' evidence.
    """
    ct = pd.crosstab(labels, groups)
    total = ct.values.sum()
    row = ct.sum(1).values[:, None]
    col = ct.sum(0).values[None, :]
    expected = row * col / total
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(expected > 0, ct.values / expected, np.nan)
    return pd.DataFrame(ratio, index=ct.index, columns=ct.columns)


def export_montages(meta: pd.DataFrame, labels: np.ndarray, embeddings: np.ndarray,
                    out_dir: Path, n_per_cluster: int = 16, tile_px: int = 128) -> None:
    """Per-cluster montage of representative tiles (centroid-nearest + random).

    Requires a `path` column in metadata pointing at the tile images. Silently
    skips clusters whose tiles cannot be read.
    """
    from PIL import Image

    if "path" not in meta.columns:
        print("⚠️  no `path` column in metadata; skipping montages")
        return
    mont_dir = out_dir / "montages"
    mont_dir.mkdir(parents=True, exist_ok=True)

    grid = int(np.ceil(np.sqrt(n_per_cluster)))
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        centroid = embeddings[idx].mean(0)
        d = np.linalg.norm(embeddings[idx] - centroid, axis=1)
        nearest = idx[np.argsort(d)[: n_per_cluster // 2]]
        rng = np.random.default_rng(42)
        rand = rng.choice(idx, size=min(n_per_cluster - len(nearest), len(idx)),
                          replace=False)
        chosen = list(dict.fromkeys(list(nearest) + list(rand)))[:n_per_cluster]

        canvas = Image.new("RGB", (grid * tile_px, grid * tile_px), (255, 255, 255))
        placed = 0
        for j, row_i in enumerate(chosen):
            p = meta.iloc[row_i].get("path")
            try:
                img = Image.open(p).convert("RGB").resize((tile_px, tile_px))
            except Exception:
                continue
            r, col = divmod(j, grid)
            canvas.paste(img, (col * tile_px, r * tile_px))
            placed += 1
        if placed:
            canvas.save(mont_dir / f"cluster_{int(c):02d}_montage.png")
    print(f"✓ Montages -> {mont_dir}")


def write_scoring_template(out_dir: Path, clusters: np.ndarray) -> None:
    """Blank CSV for the pathologist to annotate morphology per cluster."""
    rows = [{
        "cluster": int(c),
        "dominant_morphology": "",
        "steatosis_grade_0_3": "",
        "inflammation_present_y_n": "",
        "fibrosis_present_y_n": "",
        "confidence_1_5": "",
        "notes": "",
    } for c in np.unique(clusters)]
    path = out_dir / "pathologist_scoring_template.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"✓ Scoring template -> {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--method", default="kmeans",
                    choices=list(CLUSTERING_METHODS))
    ap.add_argument("--normalize", default="standardize_l2", choices=list(NORMALIZERS))
    ap.add_argument("--output-dir", type=Path, default=Path("pathologist_review"))
    ap.add_argument("--no-montages", action="store_true")
    ap.add_argument("--n-per-cluster", type=int, default=16)
    args = ap.parse_args(argv)

    emb = np.load(args.embeddings)
    meta = pd.read_csv(args.metadata)
    if len(emb) != len(meta):
        print(f"❌ length mismatch: {len(emb)} embeddings vs {len(meta)} metadata rows")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    X = normalize_embeddings(emb, args.normalize)
    labels = run_clustering(X, args.method, args.k)

    summary_rows = []
    for col in ASSOC_COLUMNS:
        if col not in meta.columns or meta[col].notna().sum() == 0:
            continue
        groups = meta[col].astype(str).to_numpy()
        ct = pd.crosstab(pd.Series(labels, name="cluster"),
                         pd.Series(groups, name=col))
        ct.to_csv(args.output_dir / f"contingency_cluster_x_{col}.csv")

        v, p, dof = cramers_v(ct.values)
        enrichment_table(labels, groups).to_csv(
            args.output_dir / f"enrichment_cluster_x_{col}.csv")
        summary_rows.append({
            "variable": col, "cramers_v": v, "chi2_pvalue": p, "dof": dof,
            "is_confound_check": col == "magnification",
        })
        print(f"  {col:<16} Cramer's V={v:.3f}  chi2 p={p:.2e}  dof={dof}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "association_summary.csv", index=False)

    # Interpretation hint for the rebuttal.
    if {"treatment_group", "magnification"}.issubset(set(summary["variable"])):
        v_treat = summary.loc[summary.variable == "treatment_group", "cramers_v"].iloc[0]
        v_mag = summary.loc[summary.variable == "magnification", "cramers_v"].iloc[0]
        if np.isnan(v_mag):
            verdict = "magnification not testable (single value); no confound check"
        elif np.isnan(v_treat):
            verdict = "treatment association not testable"
        elif v_treat > v_mag:
            verdict = "clusters track TREATMENT more than magnification (good)"
        else:
            verdict = "WARNING: magnification confound dominates treatment signal"
        print(f"  -> {verdict} (V_treat={v_treat:.3f} vs V_mag={v_mag:.3f})")

    write_scoring_template(args.output_dir, labels)
    if not args.no_montages:
        export_montages(meta, labels, X, args.output_dir,
                        n_per_cluster=args.n_per_cluster)

    # Persist the per-tile assignments so montages / overlays can be regenerated.
    out = meta.copy()
    out["cluster"] = labels
    out.to_csv(args.output_dir / "tile_cluster_assignments.csv", index=False)
    print(f"✓ Wrote analysis to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
