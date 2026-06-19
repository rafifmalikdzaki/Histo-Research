"""
Clustering-friendly latent bottleneck + DEC/IDEC objective.

Why this exists
---------------
The reviewers' decisive criticism is that the proposed model does not produce
better-separated clusters than the baseline. The current training optimises only
pixel-MSE reconstruction; nothing in the objective encourages the latent space to
be clusterable. Moreover the "latent" used for clustering is the raw 16x8x8=1024
conv feature map, not the `latent_dim` the config advertises.

This module supplies the two missing pieces, kept out of the model files so the
SAME objective applies uniformly to all 7 ablation variants (the Lightning module
wraps any of them):

  * LatentProjectionHead: flatten the conv bottleneck `z` and project to a true
    `latent_dim` vector, L2-normalised so clustering reflects direction.
  * DECClusteringLoss: a DEC/IDEC-style KL clustering loss over Student-t soft
    assignments to learnable cluster centroids, with a target distribution that
    sharpens high-confidence assignments.

Default behaviour is OFF: with cluster_weight=0 the Lightning module never builds
these, so existing pure-reconstruction training is byte-for-byte unchanged. The
clustering objective is strictly opt-in (--cluster-weight > 0).

References: Xie, Girshick, Farhadi, "Unsupervised Deep Embedding for Clustering
Analysis" (DEC, ICML 2016); Guo et al., "Improved Deep Embedded Clustering with
Local Structure Preservation" (IDEC, IJCAI 2017).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentProjectionHead(nn.Module):
    """Project a (flattened) bottleneck tensor to a unit-norm latent vector.

    The input dim is inferred lazily on the first forward call, so callers do not
    need to know the conv bottleneck size in advance (it differs per variant).
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int | None = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self._net: nn.Module | None = None

    def _build(self, in_dim: int, device, dtype) -> None:
        if self.hidden_dim:
            net = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.ELU(inplace=True),
                nn.Linear(self.hidden_dim, self.latent_dim),
            )
        else:
            net = nn.Linear(in_dim, self.latent_dim)
        self._net = net.to(device=device, dtype=dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        flat = z.reshape(z.size(0), -1)
        if self._net is None:
            self._build(flat.size(1), flat.device, flat.dtype)
        latent = self._net(flat)
        # L2-normalise: clustering on the unit sphere reflects direction not scale.
        return F.normalize(latent, p=2, dim=1)


class DECClusteringLoss(nn.Module):
    """DEC/IDEC clustering loss with learnable centroids.

    Soft assignment uses a Student-t kernel (1 dof):
        q_ij = (1 + ||z_i - mu_j||^2 / alpha)^-(alpha+1)/2  (row-normalised)
    Target distribution sharpens and balances clusters:
        p_ij = (q_ij^2 / f_j) / sum_k (q_ik^2 / f_k),  f_j = sum_i q_ij
    Loss = KL(P || Q).

    Centroids must be initialised once (typically via KMeans on warmed-up latents)
    with `init_centers` before the loss contributes.
    """

    def __init__(self, n_clusters: int, latent_dim: int, alpha: float = 1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.centers = nn.Parameter(torch.zeros(n_clusters, latent_dim))
        self.register_buffer("_initialised", torch.tensor(False))

    @property
    def initialised(self) -> bool:
        return bool(self._initialised.item())

    @torch.no_grad()
    def init_centers(self, latents: torch.Tensor) -> None:
        """KMeans-initialise centroids from a batch/pool of latent vectors."""
        from sklearn.cluster import KMeans

        x = latents.detach().cpu().numpy()
        km = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        km.fit(x)
        centers = torch.as_tensor(km.cluster_centers_, dtype=self.centers.dtype,
                                  device=self.centers.device)
        self.centers.data.copy_(centers)
        self._initialised.fill_(True)

    def soft_assign(self, z: torch.Tensor) -> torch.Tensor:
        # ||z_i - mu_j||^2 via cdist; shape (N, K).
        dist2 = torch.cdist(z, self.centers) ** 2
        q = (1.0 + dist2 / self.alpha).pow(-(self.alpha + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return q

    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        weight = q.pow(2) / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        return weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        q = self.soft_assign(z)
        p = self.target_distribution(q).detach()
        # KL(P || Q), batchmean to keep scale comparable to MSE.
        return F.kl_div(q.clamp_min(1e-12).log(), p, reduction="batchmean")
