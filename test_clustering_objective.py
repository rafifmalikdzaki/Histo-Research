#!/usr/bin/env python3
"""
CPU sanity checks for src/clustering_objective.py.

Run: python test_clustering_objective.py   (exits non-zero on failure)

Verifies the latent projection head and DEC loss behave correctly without
needing a GPU or a full training run.
"""

import sys
import numpy as np
import torch

from src.clustering_objective import LatentProjectionHead, DECClusteringLoss


def main() -> int:
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
        print(("PASS" if cond else "FAIL"), "-", msg)

    torch.manual_seed(0)

    # --- LatentProjectionHead ---
    head = LatentProjectionHead(latent_dim=128)
    z_conv = torch.randn(8, 16, 8, 8)          # mimic conv bottleneck
    lat = head(z_conv)
    check(lat.shape == (8, 128), f"projection -> (8,128) (got {tuple(lat.shape)})")
    norms = lat.norm(dim=1)
    check(torch.allclose(norms, torch.ones(8), atol=1e-5), "latents are unit-norm")
    # lazy build infers in_dim and is reused
    lat2 = head(torch.randn(4, 16, 8, 8))
    check(lat2.shape == (4, 128), "head reuses inferred input dim")

    # --- DECClusteringLoss ---
    dec = DECClusteringLoss(n_clusters=4, latent_dim=128)
    check(not dec.initialised, "DEC starts uninitialised")

    # build well-separated latents from 4 groups
    groups = []
    centers_true = torch.randn(4, 128)
    for c in centers_true:
        groups.append(c + 0.05 * torch.randn(50, 128))
    latents = torch.cat(groups, 0)
    latents = torch.nn.functional.normalize(latents, dim=1)

    dec.init_centers(latents)
    check(dec.initialised, "DEC initialised after init_centers")
    check(dec.centers.shape == (4, 128), "centers shape (4,128)")

    q = dec.soft_assign(latents)
    check(torch.allclose(q.sum(1), torch.ones(len(latents)), atol=1e-5),
          "soft assignments are row-stochastic")

    p = dec.target_distribution(q)
    check(torch.allclose(p.sum(1), torch.ones(len(latents)), atol=1e-5),
          "target distribution is row-stochastic")

    loss = dec(latents)
    check(torch.isfinite(loss), "DEC loss is finite")
    check(loss.item() >= 0, "KL loss is non-negative")

    # gradient flows to centers
    loss.backward()
    check(dec.centers.grad is not None and torch.isfinite(dec.centers.grad).all(),
          "gradient flows to centers")

    # soft_assign responds to centroid quality: on tight, well-separated clusters
    # in a low-dim space, centers placed ON the true means produce far more
    # confident (peaked) assignments than randomly-placed centers.
    d = 8
    true_centers = torch.eye(4, d) * 5.0          # well separated
    tight = torch.cat([c + 0.01 * torch.randn(40, d) for c in true_centers], 0)
    dec_lo = DECClusteringLoss(n_clusters=4, latent_dim=d)
    dec_lo.centers.data.copy_(true_centers)
    dec_lo._initialised.fill_(True)
    conf_good = dec_lo.soft_assign(tight).max(dim=1).values.mean().item()

    dec_rand = DECClusteringLoss(n_clusters=4, latent_dim=d)
    dec_rand.centers.data.copy_(torch.randn(4, d) * 0.1)  # collapsed/poor centers
    dec_rand._initialised.fill_(True)
    conf_bad = dec_rand.soft_assign(tight).max(dim=1).values.mean().item()
    check(conf_good > conf_bad,
          f"on-cluster centers give more confident assignments "
          f"({conf_good:.3f} > {conf_bad:.3f})")

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print("  -", f)
        return 1
    print("\nAll clustering-objective checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
