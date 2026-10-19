"""Unified dual-autoencoder model family.

Before this module, each ablation variant lived in its own ~270-line file
(``model.py``, ``model_baseline.py``, ...), ~90% duplicated. The variants
differ only in two axes (see :mod:`src.models.blocks`):

===================  =================  ================  =============
variant              kan block          bottleneck attn   trailing BAM
===================  =================  ================  =============
dae_kan_attention    kan                bam               no
baseline             conv               plain             no
bam_only             conv_bam           bam               yes
kan_only             kan                none              no
no_bam               kan                gate              no
no_kan               conv               bam               no
no_eka               efficient          bam               no
===================  =================  ================  =============

``DualAutoencoder`` is parametrised over those axes; the seven subclasses
below pin the historical configs so ``get_model(name)``, checkpoints, and
published numbers are unaffected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from .attention_mechanisms.bam import BAM
from .blocks import (
    ATTN_MODES,
    KAN_MODES,
    AttnMode,
    KanKind,
    conv_bn_elu,
    deconv_bn_elu,
    make_bottleneck_attn,
    make_kan_block,
)


@dataclass(frozen=True)
class VariantConfig:
    """One ablation variant's wiring over the ``kan`` x ``bottleneck_attn`` axes.

    Frozen so configs cannot drift at runtime; ``__post_init__`` fails fast
    on typos. ``VARIANT_CONFIGS`` below derives plain dicts from these so
    every existing consumer (subclasses, tests) sees identical values.
    """

    kan: KanKind = "kan"
    bottleneck_attn: AttnMode = "bam"
    trailing_bam: bool = False

    def __post_init__(self) -> None:
        if self.kan not in KAN_MODES:
            raise ValueError(f"Unknown kan block kind: {self.kan!r} (expected one of {KAN_MODES})")
        if self.bottleneck_attn not in ATTN_MODES:
            raise ValueError(
                f"Unknown bottleneck attention mode: {self.bottleneck_attn!r} "
                f"(expected one of {ATTN_MODES})"
            )


_VARIANTS: dict[str, VariantConfig] = {
    "dae_kan_attention": VariantConfig(kan="kan", bottleneck_attn="bam"),
    "baseline": VariantConfig(kan="conv", bottleneck_attn="plain"),
    "bam_only": VariantConfig(kan="conv_bam", bottleneck_attn="bam", trailing_bam=True),
    "kan_only": VariantConfig(kan="kan", bottleneck_attn="none"),
    "no_bam": VariantConfig(kan="kan", bottleneck_attn="gate"),
    "no_kan": VariantConfig(kan="conv", bottleneck_attn="bam"),
    "no_eka": VariantConfig(kan="efficient", bottleneck_attn="bam"),
}

VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    name: asdict(cfg) for name, cfg in _VARIANTS.items()
}


def _maybe_trailing_bam(block: nn.Sequential, channels: int,
                        trailing_bam: bool) -> nn.Sequential:
    """Append a BAM gate for the ``bam_only`` variant; identity otherwise."""
    if trailing_bam:
        block.append(BAM(channels))
    return block


class DualEncoder(nn.Module):
    """First autoencoder stage: downsample 128px tiles to a 64-channel map.

    Forward shapes (for a ``[B, 3, 128, 128]`` input) return::

        out:        [B, 384, 64, 64]   # decoded stage-1 output
        residual1:  [B, 384, 64, 64]   # skip tap after encoder1
        residual2:  [B, 128, 32, 32]   # skip tap after encoder2
    """

    def __init__(self, kan: KanKind = "kan", trailing_bam: bool = False,
                 device: str = "cpu"):
        super().__init__()
        self.kan = make_kan_block(kan, 64, 64, kernel_size=3, padding=1,
                                  grid_size=3, spline_order=2, device=device)
        self.encoder1 = _maybe_trailing_bam(conv_bn_elu(3, 384, stride=2), 384,
                                            trailing_bam)
        self.encoder2 = _maybe_trailing_bam(conv_bn_elu(384, 128, stride=2), 128,
                                            trailing_bam)
        self.encoder3 = _maybe_trailing_bam(conv_bn_elu(128, 64, stride=2), 64,
                                            trailing_bam)
        self.decoder1 = _maybe_trailing_bam(
            nn.Sequential(
                nn.BatchNorm2d(64),
                nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(128),
                nn.ELU(True),
            ),
            128, trailing_bam,
        )
        self.decoder2 = _maybe_trailing_bam(deconv_bn_elu(128, 384), 384,
                                            trailing_bam)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, 3, 128, 128]. encoder1/encoder2 are pure functions of x, so
        # the skip taps reuse the main-path tensors instead of recomputing
        # them (~1/3 less encoder work). Values are bit-identical; gradients
        # match to autograd summation-order noise (~6e-6 relative on the
        # two-path early layers — below multithreaded run-to-run variance).
        enc1 = self.encoder1(x)  # [B, 384, 64, 64]
        enc2 = self.encoder2(enc1)  # [B, 128, 32, 32]
        enc3 = self.encoder3(enc2)  # [B, 64, 16, 16]

        out = self.kan(enc3)  # [B, 64, 16, 16]
        out = self.decoder1(out) + enc2  # [B, 128, 32, 32]
        out = self.decoder2(out) + enc1  # [B, 384, 64, 64]

        return out, enc1, enc2


class DualBottleneck(nn.Module):
    """Compress the stage-1 map to the clustered latent ``z``.

    Forward shapes (for a ``[B, 384, 64, 64]`` input) return::

        x:  [B, 128, 32, 32]   # decoded bottleneck map fed to stage 2
        z:  [B, 16, 16, 16]    # latent map that is later clustered
    """

    def __init__(self, attn: AttnMode = "bam", trailing_bam: bool = False):
        super().__init__()
        self._attn_mode = attn
        self.encoder1 = conv_bn_elu(384, 384, stride=2)
        self.attn1 = make_bottleneck_attn(attn, 384)
        self.encoder2 = conv_bn_elu(384, 16, stride=2)
        self.attn2 = make_bottleneck_attn(attn, 16)
        # NOTE: only the bam_only variant appends BAM to this decoder.
        self.decoder = _maybe_trailing_bam(deconv_bn_elu(16, 128), 128,
                                           trailing_bam)

    def _apply_attn(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, C, H, W]. "gate" multiplies by the module's
        # [0, 1] mask; "none" is the identity (module is nn.Identity).
        if self._attn_mode == "gate":
            return x * module(x)
        if self._attn_mode == "none":
            return x
        return module(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, 384, 64, 64].
        x = self.encoder1(x)  # [B, 384, 32, 32]
        x = self._apply_attn(self.attn1, x)  # [B, 384, 32, 32]
        x = self.encoder2(x)  # [B, 16, 16, 16]
        z = self._apply_attn(self.attn2, x)  # [B, 16, 16, 16]
        x = self.decoder(z)  # [B, 128, 32, 32]

        return x, z


class DualDecoder(nn.Module):
    """Second autoencoder stage: rebuild the 128px tile from the bottleneck map.

    Forward shapes (for an ``x`` of ``[B, 128, 32, 32]``) return::

        out:  [B, 3, 128, 128]   # reconstructed tile

    The stage-1 skip taps ``residualEnc1`` (``[B, 384, 64, 64]``) and
    ``residualEnc2`` (``[B, 128, 32, 32]``) are added after ``decoder2`` and
    ``decoder1`` respectively — the wiring the original implementation left
    commented out ("skip ... for now"). This intentionally changes
    reconstruction numerics versus the skip-less builds.
    """

    def __init__(self, kan: KanKind = "kan", trailing_bam: bool = False,
                 device: str = "cpu"):
        super().__init__()
        self.kan = make_kan_block(kan, 64, 64, kernel_size=3, padding=1,
                                  grid_size=3, spline_order=2, device=device)
        self.encoder1 = _maybe_trailing_bam(deconv_bn_elu(128, 128), 128,
                                            trailing_bam)
        self.encoder2 = _maybe_trailing_bam(conv_bn_elu(128, 128, stride=2), 128,
                                            trailing_bam)
        self.encoder3 = _maybe_trailing_bam(conv_bn_elu(128, 64, stride=2), 64,
                                            trailing_bam)
        self.decoder1 = _maybe_trailing_bam(
            nn.Sequential(
                nn.BatchNorm2d(64),
                nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(128),
                nn.ELU(True),
            ),
            128, trailing_bam,
        )
        self.decoder2 = _maybe_trailing_bam(deconv_bn_elu(128, 384), 384,
                                            trailing_bam)
        self.decoder3 = _maybe_trailing_bam(deconv_bn_elu(384, 384), 384,
                                            trailing_bam)
        self.Output_Layer = conv_bn_elu(384, 3, kernel_size=3, stride=1,
                                        padding=1)
        # NOTE: the "conv_bam" kind already appends BAM(3) flat inside the
        # block, mirroring the original bam_only layout (flat indices keep
        # old checkpoints loading with strict=True).
        self.reconstruction = make_kan_block(kan, 3, 3, kernel_size=1,
                                             padding=0, grid_size=2,
                                             spline_order=1, device=device)

    def forward(self, x: torch.Tensor, residualEnc1: torch.Tensor,
                residualEnc2: torch.Tensor) -> torch.Tensor:
        # x: [B, 128, 32, 32] -> [B, 3, 128, 128]. Stage-1 skips re-inject
        # high-resolution features U-Net style (shapes match exactly).
        out = self.encoder1(x)  # [B, 128, 32, 32] -> [B, 128, 64, 64]
        out = self.encoder2(out)  # [B, 128, 64, 64] -> [B, 128, 32, 32]
        out = self.encoder3(out)  # [B, 128, 32, 32] -> [B, 64, 16, 16]

        out = self.kan(out)  # [B, 64, 16, 16] -> [B, 64, 16, 16]

        out = self.decoder1(out) + residualEnc2  # [B, 128, 32, 32]
        out = self.decoder2(out) + residualEnc1  # [B, 384, 64, 64]
        out = self.decoder3(out)  # [B, 384, 64, 64] -> [B, 384, 128, 128]
        out = self.Output_Layer(out)  # [B, 384, 128, 128] -> [B, 3, 128, 128]
        out = self.reconstruction(out)  # [B, 3, 128, 128] -> [B, 3, 128, 128]

        return out


def _default_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class DualAutoencoder(nn.Module):
    """Dual-stage autoencoder parametrised over the ablation axes.

    Forward shapes (for a ``[B, 3, 128, 128]`` tile batch) return::

        encoded:  [B, 384, 64, 64]   # stage-1 code (reconstruction target ref)
        decoded:  [B, 3, 128, 128]   # reconstructed tile
        z:        [B, 16, 16, 16]    # bottleneck latent map (clustered downstream)
    """

    def __init__(self, kan: KanKind = "kan", bottleneck_attn: AttnMode = "bam",
                 trailing_bam: bool = False, device: str | None = None):
        super().__init__()
        device = _default_device(device)
        self.device = device
        self.ae_encoder = DualEncoder(kan=kan, trailing_bam=trailing_bam,
                                      device=device)
        self.bottleneck = DualBottleneck(attn=bottleneck_attn,
                                         trailing_bam=trailing_bam)
        self.ae_decoder = DualDecoder(kan=kan, trailing_bam=trailing_bam,
                                      device=device)
        # Honor `device` for every submodule (plain conv/BN blocks take no
        # device argument, so without this only the KAN blocks would move).
        # Values are untouched — placement only, keys unchanged.
        self.to(device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, 3, 128, 128].
        encoded, residual1, residual2 = self.ae_encoder(x)
        decoded, z = self.bottleneck(encoded)
        decoded = self.ae_decoder(decoded, residual1, residual2)

        return encoded, decoded, z


class DAE_KAN_Attention(DualAutoencoder):
    """Full model: KAN layers + BAM attention."""

    def __init__(self, device: str | None = None):
        super().__init__(**VARIANT_CONFIGS["dae_kan_attention"], device=device)


class BaselineDAE_KAN_Attention(DualAutoencoder):
    """Baseline: standard convolutions, no KAN, no BAM."""

    def __init__(self, device: str | None = None):
        super().__init__(**VARIANT_CONFIGS["baseline"], device=device)


class BAMOnly_DAE_KAN_Attention(DualAutoencoder):
    """BAM attention after every block, no KAN layers."""

    def __init__(self, device: str | None = None):
        super().__init__(**VARIANT_CONFIGS["bam_only"], device=device)


class KANOnly_DAE_KAN_Attention(DualAutoencoder):
    """KAN layers only, all attention removed."""

    def __init__(self, device: str | None = None):
        super().__init__(**VARIANT_CONFIGS["kan_only"], device=device)


class NoBAM_DAE_KAN_Attention(DualAutoencoder):
    """KAN layers kept, BAM replaced by 1x1 conv gates."""

    def __init__(self, device: str | None = None):
        super().__init__(**VARIANT_CONFIGS["no_bam"], device=device)


class NoKAN_DAE_KAN_Attention(DualAutoencoder):
    """BAM kept, KAN layers replaced by standard convolutions."""

    def __init__(self, device: str | None = None):
        super().__init__(**VARIANT_CONFIGS["no_kan"], device=device)


class NoEKA_DAE_KAN_Attention(DualAutoencoder):
    """BAM kept, KAN layers replaced by efficient standard convolutions."""

    def __init__(self, device: str | None = None):
        super().__init__(**VARIANT_CONFIGS["no_eka"], device=device)
