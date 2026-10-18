"""Shared building blocks for the dual-autoencoder model family.

The seven ablation variants (see :mod:`src.models.factory`) differ only in two
axes, factored out here so each block has a single source of truth:

* ``kan`` — the ``self.kan`` / ``self.reconstruction`` block, one of
  ``"kan"`` (KAN convolution), ``"conv"`` (plain Conv-BN-ELU),
  ``"conv_bam"`` (plain + trailing BAM), ``"efficient"`` (two-layer
  standard conv matching KAN capacity).
* ``bottleneck_attn`` — the bottleneck ``attn1``/``attn2`` modules, one of
  ``"bam"``, ``"gate"`` (1x1 conv gate, applied multiplicatively),
  ``"plain"`` (1x1 conv stack, applied directly), ``"none"``.

Module attribute names and sub-layer order intentionally mirror the original
per-variant files, so existing checkpoints keep loading with ``strict=True``.
"""

import torch
import torch.nn as nn

from .attention_mechanisms.bam import BAM
from .kan_convolutional.efficient_kan_conv_fixed import (
    KAN_Convolutional_Layer as KANCL,
)

KAN_MODES = ("kan", "conv", "conv_bam", "efficient")
ATTN_MODES = ("bam", "gate", "plain", "none")


def conv_bn_elu(in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1,
                padding: int = 1) -> nn.Sequential:
    """Standard Conv-BatchNorm-ELU stack used throughout encoder/decoder."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                  padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ELU(True),
    )


def deconv_bn_elu(in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 2,
                  padding: int = 1, output_padding: int = 1) -> nn.Sequential:
    """Standard ConvTranspose-BatchNorm-ELU stack."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                           padding=padding, output_padding=output_padding),
        nn.BatchNorm2d(out_ch),
        nn.ELU(True),
    )


class EfficientConvBlock(nn.Module):
    """Two-layer standard conv matching KAN expressiveness (the no_eka block)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, dilation: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      dilation),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def make_kan_block(kind: str, in_ch: int, out_ch: int, kernel_size: int = 3,
                   padding: int = 1, grid_size: int = 3, spline_order: int = 2,
                   device: str = "cpu") -> nn.Module:
    """Build the interchangeable ``kan`` / ``reconstruction`` block."""
    if kind == "kan":
        return KANCL(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            padding=padding,
            grid_size=grid_size,
            spline_order=spline_order,
            device=device,
        )
    if kind == "efficient":
        return EfficientConvBlock(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            padding=padding,
        )
    block = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ELU(True),
    )
    if kind == "conv_bam":
        block.append(BAM(out_ch))
    elif kind != "conv":
        raise ValueError(f"Unknown kan block kind: {kind!r} (expected one of {KAN_MODES})")
    return block


def make_bottleneck_attn(mode: str, channels: int) -> nn.Module:
    """Build a bottleneck attention module (``attn1``/``attn2``)."""
    if mode == "bam":
        return BAM(channels)
    if mode == "gate":
        return nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.BatchNorm2d(channels // 8),
            nn.ELU(True),
            nn.Conv2d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid(),
        )
    if mode == "plain":
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ELU(True),
        )
    if mode == "none":
        return nn.Identity()
    raise ValueError(f"Unknown bottleneck attention mode: {mode!r} (expected one of {ATTN_MODES})")
