"""Safety net: lazily-built LatentProjectionHead params must join the optimizer.

The head is normally pre-built in ``__init__`` before ``configure_optimizers``.
If that probe fails, the head materialises on its first real forward — after
the optimizer exists — and without registration its weights would silently
never train. These tests drive the unbound net method with a stub self, so no
Lightning trainer is needed.
"""

from types import SimpleNamespace

import torch

from histodae.cli.train import (
    OptimizedMainModelWithAnalysis,
)
from histodae.clustering.dec import LatentProjectionHead


def _stub(head):
    return SimpleNamespace(latent_head=head, _head_optimised=False, trainer=None)


def test_unbuilt_head_is_a_noop():
    stub = _stub(LatentProjectionHead(latent_dim=8))
    OptimizedMainModelWithAnalysis._ensure_head_optimised(stub)
    assert stub._head_optimised is False  # nothing built, nothing to register


def test_built_head_registers_exactly_once():
    head = LatentProjectionHead(latent_dim=8)
    head(torch.randn(2, 16))  # materialise the lazy Linear
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    stub = _stub(head)
    stub.trainer = SimpleNamespace(optimizers=[opt])

    OptimizedMainModelWithAnalysis._ensure_head_optimised(stub)
    assert stub._head_optimised is True
    n_groups = len(opt.param_groups)
    assert n_groups == 2

    OptimizedMainModelWithAnalysis._ensure_head_optimised(stub)
    assert len(opt.param_groups) == n_groups  # idempotent, no duplicate group

    head_ids = {id(p) for p in head.parameters()}
    known_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert head_ids <= known_ids
