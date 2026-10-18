"""CPU sanity checks for src/clustering_objective.py (pytest port).

Verifies the latent projection head and DEC loss behave correctly without
needing a GPU or a full training run.
"""

import torch

from src.clustering_objective import DECClusteringLoss, LatentProjectionHead


def test_projection_shape_and_unit_norm():
    torch.manual_seed(0)
    head = LatentProjectionHead(latent_dim=128)
    lat = head(torch.randn(8, 16, 8, 8))  # mimic conv bottleneck
    assert lat.shape == (8, 128)
    assert torch.allclose(lat.norm(dim=1), torch.ones(8), atol=1e-5)


def test_head_reuses_inferred_input_dim():
    torch.manual_seed(0)
    head = LatentProjectionHead(latent_dim=128)
    head(torch.randn(8, 16, 8, 8))
    assert head(torch.randn(4, 16, 8, 8)).shape == (4, 128)


def _initialised_dec(seed=0, n_clusters=4, latent_dim=128, n_per=50):
    torch.manual_seed(seed)
    dec = DECClusteringLoss(n_clusters=n_clusters, latent_dim=latent_dim)
    assert not dec.initialised
    centers_true = torch.randn(n_clusters, latent_dim)
    groups = [c + 0.05 * torch.randn(n_per, latent_dim) for c in centers_true]
    latents = torch.nn.functional.normalize(torch.cat(groups, 0), dim=1)
    dec.init_centers(latents)
    assert dec.initialised
    assert dec.centers.shape == (n_clusters, latent_dim)
    return dec, latents


def test_soft_assign_and_target_are_row_stochastic():
    dec, latents = _initialised_dec()
    q = dec.soft_assign(latents)
    assert torch.allclose(q.sum(1), torch.ones(len(latents)), atol=1e-5)
    p = dec.target_distribution(q)
    assert torch.allclose(p.sum(1), torch.ones(len(latents)), atol=1e-5)


def test_dec_loss_finite_nonnegative_with_gradients():
    dec, latents = _initialised_dec()
    loss = dec(latents)
    assert torch.isfinite(loss)
    assert loss.item() >= 0
    loss.backward()
    assert dec.centers.grad is not None
    assert torch.isfinite(dec.centers.grad).all()


def test_on_cluster_centers_give_confident_assignments():
    torch.manual_seed(0)
    d = 8
    true_centers = torch.eye(4, d) * 5.0  # well separated
    tight = torch.cat([c + 0.01 * torch.randn(40, d) for c in true_centers], 0)

    dec_lo = DECClusteringLoss(n_clusters=4, latent_dim=d)
    dec_lo.centers.data.copy_(true_centers)
    dec_lo._initialised.fill_(True)
    conf_good = dec_lo.soft_assign(tight).max(dim=1).values.mean().item()

    dec_rand = DECClusteringLoss(n_clusters=4, latent_dim=d)
    dec_rand.centers.data.copy_(torch.randn(4, d) * 0.1)  # collapsed centers
    dec_rand._initialised.fill_(True)
    conf_bad = dec_rand.soft_assign(tight).max(dim=1).values.mean().item()

    assert conf_good > conf_bad
