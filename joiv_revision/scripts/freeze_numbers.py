#!/usr/bin/env python3
"""Freeze JOIV numbers: merge shard raw + dae_kan recon + DEC raw into one table.

Inputs (in joiv_revision/results/):
  gpukc_pull/raw_clustering_metrics_corrected.csv  (6 shard variants, 3 seeds)
  dae_kan_recon_metrics.csv                        (scoped aggregate)
  dec_metrics.csv                                  (scoped aggregate, when ready)
Output: frozen_metrics.csv (all rows) + frozen_summary.md (mean±SD tables).
Seeds are encoder as -1 by the aggregate (run order 42/123/456); we record
run_index 1..3 within (model, method) and document it in the summary header.
"""
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
KEYS = ["model", "clustering_method"]


def load(name, arm):
    p = R / name
    if not p.exists():
        print(f"skip {name} (missing)")
        return []
    rows = list(csv.DictReader(open(p)))
    out = []
    for r in rows:
        rn = {}
        for k, v in r.items():
            k2 = ("treatment_" + k[6:]) if k.startswith("treat_") else k
            rn[k2] = v
        rn["arm"] = arm
        out.append(rn)
    return out


rows = load("gpukc_pull/raw_clustering_metrics_corrected.csv", "recon")
rows += load("dae_kan_recon_metrics.csv", "recon")
# Fresh scoped DEC aggregate (code_dec test sidecars); the bundled
# code_dec/results CSV is stale scouting output and is never used.
rows += load("dec_metrics_fresh.csv", "dec_on")
assert rows, "no inputs"

# run index within (arm, model, method) in file order = seed 42,123,456
seen = defaultdict(int)
for r in rows:
    seen[(r["arm"], r["model"], r["clustering_method"])] += 1
    r["run_index"] = seen[(r["arm"], r["model"], r["clustering_method"])]

fields = []
for r in rows:
    for k in r:
        if k not in fields:
            fields.append(k)
with open(R / "frozen_metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

MET = ["silhouette", "davies_bouldin", "calinski_harabasz", "xie_beni",
       "treatment_ami", "treatment_ari", "diet_ami", "diet_ari", "mag_ami"]


def ms(vals):
    vals = [float(v) for v in vals if v not in ("", None)]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{st.mean(vals):.3f}±{st.stdev(vals):.3f}"


out = ["# Frozen numbers (3 runs = seeds 42/123/456, 10 epochs, k = 6)\n"]
for arm in ["recon", "dec_on"]:
    out.append(f"\n## Arm: {arm}\n")
    out.append("| Model | Method | Sil↑ | DBI↓ | CHI↑ | XB↓ | treat-AMI↑ | treat-ARI↑ | diet-AMI↑ |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    cells = defaultdict(list)
    for r in rows:
        if r["arm"] == arm:
            cells[(r["model"], r["clustering_method"])].append(r)
    for (m, meth), rs in sorted(cells.items()):
        out.append("| " + " | ".join(
            [m, meth] + [ms([r[c] for r in rs]) for c in MET[:4]] +
            [ms([r[c] for r in rs]) for c in MET[4:7]]) + " |")
(R / "frozen_summary.md").write_text("\n".join(out) + "\n")
print(f"frozen: {len(rows)} rows -> {R}/frozen_metrics.csv + frozen_summary.md")
