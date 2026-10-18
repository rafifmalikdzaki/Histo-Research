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

import torch
import torch.nn as nn

from .attention_mechanisms.bam import BAM
from .blocks import (
    conv_bn_elu,
    deconv_bn_elu,
    make_bottleneck_attn,
    make_kan_block,
)

VARIANT_CONFIGS = {
    "dae_kan_attention": {"kan": "kan", "bottleneck_attn": "bam"},
    "baseline": {"kan": "conv", "bottleneck_attn": "plain"},
    "bam_only": {"kan": "conv_bam", "bottleneck_attn": "bam", "trailing_bam": True},
    "kan_only": {"kan": "kan", "bottleneck_attn": "none"},
    "no_bam": {"kan": "kan", "bottleneck_attn": "gate"},
    "no_kan": {"kan": "conv", "bottleneck_attn": "bam"},
    "no_eka": {"kan": "efficient", "bottleneck_attn": "bam"},
}


def _maybe_trailing_bam(block: nn.Sequential, channels: int,
                        trailing_bam: bool) -> nn.Sequential:
    if trailing_bam:
        block.append(BAM(channels))
    return block


class DualEncoder(nn.Module):
    def __init__(self, kan: str = "kan", trailing_bam: bool = False,
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

    def forward(self, x: torch.Tensor):
        out = self.encoder1(x)
        out = self.encoder2(out)
        out = self.encoder3(out)

        residual1 = self.encoder1(x)
        residual2 = self.encoder2(residual1)

        out = self.kan(out)
        out = self.decoder1(out) + residual2
        out = self.decoder2(out) + residual1

        return out, residual1, residual2


class DualBottleneck(nn.Module):
    def __init__(self, attn: str = "bam", trailing_bam: bool = False):
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
        if self._attn_mode == "gate":
            return x * module(x)
        if self._attn_mode == "none":
            return x
        return module(x)

    def forward(self, x: torch.Tensor):
        x = self.encoder1(x)
        x = self._apply_attn(self.attn1, x)
        x = self.encoder2(x)
        z = self._apply_attn(self.attn2, x)
        x = self.decoder(z)

        return x, z


class DualDecoder(nn.Module):
    def __init__(self, kan: str = "kan", trailing_bam: bool = False,
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
                residualEnc2: torch.Tensor):
        out = self.encoder1(x)
        out = self.encoder2(out)
        out = self.encoder3(out)

        out = self.kan(out)

        out = self.decoder1(out)
        out = self.decoder2(out)
        out = self.decoder3(out)
        out = self.Output_Layer(out)
        out = self.reconstruction(out)

        return out


def _default_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class DualAutoencoder(nn.Module):
    """Dual-stage autoencoder parametrised over the ablation axes."""

    def __init__(self, kan: str = "kan", bottleneck_attn: str = "bam",
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

    def forward(self, x: torch.Tensor):
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
