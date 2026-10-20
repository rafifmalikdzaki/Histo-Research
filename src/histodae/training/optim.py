"""Optimizer construction and the lazily-built-head safety net.

Kept apart from the Lightning module so both can be reasoned about (and
unit-tested) without instantiating a trainer.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "build_optimizer",
    "build_scheduler",
    "ensure_head_registered",
    "optimizer_config",
]


def build_optimizer(module: Any) -> torch.optim.Optimizer:
    """Build the training optimizer.

    ``module.parameters()`` covers the autoencoder *and* the projection head
    plus the DEC centroids when the clustering objective is enabled.
    """
    return torch.optim.Adam(
        module.parameters(),
        lr=module.learning_rate,
        weight_decay=1e-5,
    )


def build_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Build the plateau scheduler used for the learning rate."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=2, min_lr=1e-5
    )


def optimizer_config(module: Any) -> dict[str, Any]:
    """Return the dict Lightning expects from ``configure_optimizers``."""
    optimizer = build_optimizer(module)
    scheduler = build_scheduler(optimizer)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val/loss",
        },
    }


def ensure_head_registered(module: Any) -> None:
    """Register lazily-built head params with the optimizer (safety net).

    The projection head is normally pre-built in ``__init__`` (probed with
    a zeros batch) before ``configure_optimizers()`` runs, making this a
    no-op. If pre-build failed, the head materialises on its first real
    forward — after the optimizer exists — and without this its weights
    would silently never train. Retried every step until registered, so a
    missing trainer (e.g. manual calls) simply defers.
    """
    if module._head_optimised or module.latent_head is None:
        return
    if not list(module.latent_head.parameters()):
        return  # head not built yet; nothing to register
    trainer = getattr(module, "trainer", None)
    optimizers = list(getattr(trainer, "optimizers", None) or [])
    if not optimizers:
        return
    for opt in optimizers:
        known = {id(p) for group in opt.param_groups for p in group["params"]}
        missing = [p for p in module.latent_head.parameters() if id(p) not in known]
        if missing:
            opt.add_param_group({"params": missing})
    module._head_optimised = True
