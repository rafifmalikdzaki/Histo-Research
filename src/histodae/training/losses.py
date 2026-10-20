"""Loss computation for the dual autoencoder.

The reconstruction objective is plain MSE. The clustering objective is
opt-in: with ``cluster_weight == 0`` the loss is exactly ``recon_weight *
MSE``, so reconstruction-only runs are unchanged.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

__all__ = ["compute_training_loss", "compute_validation_loss"]


def compute_training_loss(
    module: Any, x: torch.Tensor, decoded: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return ``(total_loss, mse_loss, cluster_loss_value)`` for one batch.

    The composite objective is::

        loss = recon_weight * MSE + cluster_weight * DEC-KL

    where the DEC term only applies after the warmup, once the centroids
    have been KMeans-initialised from a pool of latents.
    """
    mse_loss = nn.functional.mse_loss(x, decoded)
    loss = module.recon_weight * mse_loss
    cluster_loss_val = 0.0

    if module.cluster_weight > 0 and module.latent_head is not None:
        from histodae.training.optim import ensure_head_registered

        ensure_head_registered(module)
        latent = module.latent_head(z)
        # Maintain a rolling latent pool for centroid initialisation.
        if not module.dec_loss.initialised:
            module._latent_pool.append(latent.detach().cpu())
            pooled = sum(t.size(0) for t in module._latent_pool)
            if pooled > module._latent_pool_cap:
                module._latent_pool.pop(0)
            past_warmup = module.current_epoch >= module.cluster_warmup_epochs
            enough = pooled >= max(10 * module.n_clusters, module.n_clusters + 1)
            if past_warmup and enough:
                module.dec_loss.init_centers(torch.cat(module._latent_pool, 0))
                module._latent_pool = []
                print(f"✓ DEC centroids initialised at epoch {module.current_epoch}")
        if module.dec_loss.initialised:
            cluster_loss = module.dec_loss(latent)
            loss = loss + module.cluster_weight * cluster_loss
            cluster_loss_val = cluster_loss.item()

    return loss, mse_loss, cluster_loss_val


def compute_validation_loss(module: Any, x: torch.Tensor, decoded: torch.Tensor) -> torch.Tensor:
    """Return the validation reconstruction loss (MSE) for one batch."""
    return nn.functional.mse_loss(x, decoded)
