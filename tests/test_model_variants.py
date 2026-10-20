"""Regression tests for the unified dual-autoencoder family (src/models/dae.py).

The seven ablation variants share one parametrised implementation; these tests
pin the structure so refactors cannot silently change it:

* every factory name builds a distinct configuration,
* seeded construction is deterministic (same weights + same outputs twice),
* total parameter counts match the audited table below.
"""

import pytest
import torch

from histodae.models.dae import VARIANT_CONFIGS, DualAutoencoder
from histodae.models.factory import MODELS, get_model

# Audited 2026-10-18 against the pre-unification per-variant files
# (bit-identical state_dict keys and forward outputs on CUDA).
EXPECTED_PARAMS = {
    "dae_kan_attention": 5230015,
    "baseline": 4897403,
    "bam_only": 4978846,
    "kan_only": 5190502,
    "no_bam": 5227980,
    "no_kan": 4788004,
    "no_eka": 4796598,
}


def _count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def test_factory_covers_all_variant_configs():
    assert set(MODELS) == set(VARIANT_CONFIGS) == set(EXPECTED_PARAMS)
    for name in MODELS:
        assert get_model(name) is MODELS[name]
        assert get_model(name.upper()) is MODELS[name]


def test_variant_param_counts():
    for name, cls in MODELS.items():
        torch.manual_seed(0)
        assert _count_params(cls(device="cpu")) == EXPECTED_PARAMS[name], name


def test_seeded_construction_is_deterministic():
    # NOTE: KAN spline init goes through torch.linalg.lstsq, which varies by
    # ~1e-9 run-to-run (multithreaded LAPACK), so exact equality is impossible
    # even with identical seeds. allclose at 1e-6 is the honest bound.
    for name, cls in MODELS.items():
        torch.manual_seed(0)
        first = cls(device="cpu").eval()
        sd_first = {k: v.clone() for k, v in first.state_dict().items()}
        torch.manual_seed(0)
        second = cls(device="cpu").eval()
        for key, value in second.state_dict().items():
            assert torch.allclose(value, sd_first[key], atol=1e-6), (name, key)


def test_forward_shapes_and_determinism():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 128, 128)
    for name, cls in MODELS.items():
        torch.manual_seed(0)
        model = cls(device="cpu").eval()
        with torch.no_grad():
            encoded, decoded, z = model(x)
            encoded2, decoded2, z2 = model(x)
        assert encoded.shape == (1, 384, 64, 64), name
        assert decoded.shape == (1, 3, 128, 128), name
        assert z.shape == (1, 16, 16, 16), name
        assert torch.equal(encoded, encoded2)
        assert torch.equal(decoded, decoded2)
        assert torch.equal(z, z2)


def test_variant_configs_differ():
    assert len({repr(sorted(c.items())) for c in VARIANT_CONFIGS.values()}) == len(
        VARIANT_CONFIGS
    )
    assert issubclass(MODELS["dae_kan_attention"], DualAutoencoder)


def test_kan_blocks_honor_device_arg():
    """KAN conv layers must live on the requested device at construction.

    The `device` argument used to be silently ignored (params stayed on CPU
    until an outer .to() moved them).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(0)
    model = MODELS["dae_kan_attention"](device="cuda")
    for name, p in model.named_parameters():
        assert p.device.type == "cuda", name
    for name, b in model.named_buffers():
        assert b.device.type == "cuda", name


def test_decoder_stage1_skips_are_live():
    """Stage-1 residuals must reach the stage-2 decoder output.

    Regression test for the wired-up skip connections (the original code left
    `+ residualEnc1/2` commented out, so the taps were silently dropped):
    zeroing or perturbing a tap must move the reconstruction.
    """
    torch.manual_seed(0)
    model = MODELS["dae_kan_attention"](device="cpu").eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        encoded, r1, r2 = model.ae_encoder(x)
        stage_out, _ = model.bottleneck(encoded)
        assert r1.shape == (1, 384, 64, 64)
        assert r2.shape == (1, 128, 32, 32)
        y_skips = model.ae_decoder(stage_out, r1, r2)
        y_zeros = model.ae_decoder(
            stage_out, torch.zeros_like(r1), torch.zeros_like(r2))
        y_pert = model.ae_decoder(stage_out, r1 + 1.0, r2)
    assert y_skips.shape == (1, 3, 128, 128)
    assert not torch.allclose(y_skips, y_zeros, atol=1e-6)
    assert not torch.allclose(y_skips, y_pert, atol=1e-6)
