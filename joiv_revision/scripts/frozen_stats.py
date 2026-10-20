#!/usr/bin/env python3
"""Paired stats from frozen_metrics.csv: dae_kan vs baseline (recon),
dae_kan DEC vs recon. Paired by run_index (1..3 = seeds 42/123/456).
Holm-Bonferroni across each comparison family."""
import csv
import statistics as st
from pathlib import Path

from scipy import stats

R = Path(__file__).resolve().parent.parent / "results"
rows = list(csv.DictReader(open(R / "frozen_metrics.csv")))
MET = ["silhouette", "davies_bouldin", "calinski_harabasz", "xie_beni",
       "treatment_ami", "treatment_ari", "diet_ami", "diet_ari"]


def get(arm, model, method):
    sel = [r for r in rows if r["arm"] == arm and r["model"] == model
           and r["clustering_method"] == method]
    return {int(r["run_index"]): r for r in sel}


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    m = len(pvals)
    run_max = 0.0
    for rank, i in enumerate(order):
        run_max = max(run_max, (m - rank) * pvals[i])
        adj[i] = min(1.0, run_max)
    return adj


out = []
for label, (a_arm, a_mod), (b_arm, b_mod) in [
    ("dae_kan_vs_baseline_recon", ("recon", "dae_kan_attention"), ("recon", "baseline")),
    ("dae_kan_dec_vs_recon", ("dec_on", "dae_kan_attention"), ("recon", "dae_kan_attention")),
]:
    fam = []
    for method in ["kmeans", "gmm", "bisecting_kmeans"]:
        A, B = get(a_arm, a_mod, method), get(b_arm, b_mod, method)
        for met in MET:
            pairs = [(float(A[i][met]), float(B[i][met])) for i in sorted(set(A) & set(B))
                     if A[i][met] not in ("", None) and B[i][met] not in ("", None)]
            if len(pairs) < 2:
                continue
            diffs = [a - b for a, b in pairs]
            t, p = stats.ttest_rel([a for a, _ in pairs], [b for _, b in pairs])
            d = st.mean(diffs) / (st.stdev(diffs) if len(diffs) > 1 and st.stdev(diffs) > 0 else 1.0)
            fam.append({"comparison": label, "method": method, "metric": met,
                        "mean_a": round(st.mean([a for a, _ in pairs]), 4),
                        "mean_b": round(st.mean([b for _, b in pairs]), 4),
                        "mean_diff": round(st.mean(diffs), 4),
                        "ttest_p": float(p), "cohens_d": round(float(d), 3),
                        "n_pairs": len(pairs)})
    adj = holm([f["ttest_p"] for f in fam])
    for f, pa in zip(fam, adj):
        f["ttest_p_holm"] = round(pa, 4)
        f["significant_holm"] = pa < 0.05
    out += fam

with open(R / "frozen_stats.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
sig = [o for o in out if o["significant_holm"]]
print(f"wrote {len(out)} tests; Holm-significant: {len(sig)}")
for s in sig:
    print(" SIG", s["comparison"], s["method"], s["metric"],
          s["mean_a"], "vs", s["mean_b"], "d=", s["cohens_d"], "p_holm=", s["ttest_p_holm"])
