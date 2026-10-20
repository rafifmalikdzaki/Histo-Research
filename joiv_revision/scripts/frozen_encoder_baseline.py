#!/usr/bin/env python3
"""Frozen ImageNet ResNet50 encoder + clustering baseline (A6 minimum).
No training: forward-pass features on held-out test tiles, standardize+L2,
KMeans/Bisecting/GMM k=6, internal + external metrics via src/clustering_metrics.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from histodae.clustering.metrics import (  # noqa: E402
    external_metrics,
    internal_metrics,
    normalize_embeddings,
)
from sklearn.cluster import KMeans, BisectingKMeans  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

PROJ = Path("/mnt/home/dzakirm/histodae_joiv")
meta = pd.read_csv(PROJ / "hub_data/test_hepar.csv")
files = [PROJ / "tiles_hepar" / f for f in meta["Image"]]
print(f"tiles: {len(files)}")

weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
model = torchvision.models.resnet50(weights=weights)
model.fc = torch.nn.Identity()
model.eval().cuda()
tf = weights.transforms()

feats = []
with torch.no_grad():
    for i in range(0, len(files), 256):
        batch = torch.stack([tf(Image.open(f).convert("RGB")) for f in files[i:i + 256]]).cuda()
        feats.append(model(batch).float().cpu().numpy())
        print(f"{min(i + 256, len(files))}/{len(files)}", flush=True)
X = np.concatenate(feats)
np.save("/tmp/frozen_resnet50_test_embeddings.npy", X)
print("features:", X.shape)

Xn = normalize_embeddings(X, "standardize_l2")
rows = []
for name, mdl in [("kmeans", KMeans(n_clusters=6, random_state=42, n_init=10)),
                  ("bisecting_kmeans", BisectingKMeans(n_clusters=6, random_state=42, n_init=10)),
                  ("gmm", GaussianMixture(n_components=6, random_state=42, n_init=5))]:
    labels = mdl.fit_predict(Xn)
    im = internal_metrics(Xn, labels)
    em = external_metrics(meta["treatment_group"].values, labels, prefix="treatment_")
    em.update(external_metrics(meta["diet"].values, labels, prefix="diet_"))
    row = {"model": "frozen_resnet50", "clustering_method": name, **im, **em}
    rows.append(row)
    print(row)
pd.DataFrame(rows).to_csv("/tmp/frozen_resnet50_metrics.csv", index=False)
print("DONE")
