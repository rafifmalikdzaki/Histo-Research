#!/usr/bin/env python3
"""Evaluate DEC-arm models on their 128-d latent (projection-head) embeddings.

The sweep evaluated the 4096-d bottleneck z for every arm. DEC training,
however, optimises centroids in the 128-d unit-norm latent. This script
rebuilds LatentProjectionHead from each DEC checkpoint (keys latent_head.*),
dumps test latents, and evaluates them identically (standardize+L2? No -
latents are already unit-norm; we still apply standardize_l2 for comparability
AND raw, reporting both).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_package_on_path  # noqa: E402

ensure_package_on_path()
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

from histodae.cli.extract import TileDataset, _load_state_dict  # noqa: E402
from histodae.clustering.dec import LatentProjectionHead  # noqa: E402
from histodae.clustering.metrics import (  # noqa: E402
    external_metrics,
    internal_metrics,
    normalize_embeddings,
)
from histodae.data.datasets import load_metadata  # noqa: E402
from histodae.models.factory import get_model  # noqa: E402

RUNS = sys.argv[1:]  # run dirs with checkpoints/last.ckpt
device = "cuda"
out_rows = []
for run in RUNS:
    run = Path(run)
    ckpt = run / "checkpoints/last.ckpt"
    df = load_metadata("test", image_path="./data/processed", dataset_name="HeparUnifiedPNG")
    ds = TileDataset(df["path"].tolist(), image_size=128)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    model = get_model("dae_kan_attention")().to(device).eval()
    sd = _load_state_dict(ckpt)
    model.load_state_dict(sd, strict=False)
    head = LatentProjectionHead(128).to(device).eval()
    head_sd = {k[len("latent_head."):]: v for k, v in sd.items() if k.startswith("latent_head.")}
    zs, lats = [], []
    with torch.no_grad():
        for x in dl:
            x = x.to(device)
            _, _, z = model(x)
            zs.append(z.reshape(z.size(0), -1).cpu())
            lats.append(head(z).cpu().numpy())
    # build head net with real dim then load weights
    head._build(zs[0].size(1), torch.device(device), torch.float32)
    miss = head.load_state_dict({k: v for k, v in head_sd.items()}, strict=False)
    print(run.name, "head keys:", len(head_sd), "missing:", miss.missing_keys)
    lats = []
    with torch.no_grad():
        for x in dl:
            x = x.to(device)
            _, _, z = model(x)
            lats.append(head(z).cpu().numpy())
    L = np.concatenate(lats).astype(np.float32)
    np.save(f"/tmp/declat_{run.name}.npy", L)
    print(run.name, "latent:", L.shape)
    for tag, X in [("latent_raw", L), ("latent_std_l2", normalize_embeddings(L, "standardize_l2"))]:
        for name, mdl in [("kmeans", KMeans(n_clusters=6, random_state=42, n_init=10)),
                          ("gmm", GaussianMixture(n_components=6, random_state=42, n_init=5))]:
            labels = mdl.fit_predict(X)
            row = {"run": run.name, "space": tag, "clustering_method": name,
                   **internal_metrics(X, labels),
                   **external_metrics(df["treatment_group"].values, labels, prefix="treatment_"),
                   **external_metrics(df["diet"].values, labels, prefix="diet_")}
            out_rows.append(row)
            print(tag, name, f"sil={row['silhouette']:.3f} dbi={row['davies_bouldin']:.2f} "
                            f"tAMI={row['treatment_ami']:.3f}")
pd.DataFrame(out_rows).to_csv("/tmp/dec_latent_metrics.csv", index=False)
print("DONE -> /tmp/dec_latent_metrics.csv")
