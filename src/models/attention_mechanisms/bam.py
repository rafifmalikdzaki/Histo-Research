"""
PyTorch implementation of Bam: Bottleneck attention module

As described in http://bmvc2018.org/contents/papers/0092.pdf

Given a 3D feature map, BAM produces a 3D attention map to emphasize important elements. BAM
decomposes the process of inferring a 3D attention map in two streams , so that the
computational and parametric overhead are significantly reduced. As the channels of feature
maps can be regarded as feature detectors, the two branches (spatial and channel) explicitly
learn "what" and "where" to focus on.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ChannelGate(nn.Module):
    """Squeeze-and-excite style channel mask: ``[B, C, H, W] -> [B, C, 1, 1]``."""

    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        # Guard against tiny channel counts (e.g. the 3-channel reconstruction
        # stage in the bam_only variant): channel // reduction underflows to 0,
        # which builds a 0-width layer and crashes. Floor the bottleneck at 1.
        hidden = max(1, channel // reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channel, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel),
        )
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, H, W].
        b, c, _, _ = x.shape
        y = self.avgpool(x).view(b, c)  # [B, C]
        y = self.mlp(y)  # [B, C]
        y = self.bn(y).view(b, c, 1, 1)  # [B, C, 1, 1]
        return y.expand_as(x)


class SpatialGate(nn.Module):
    """Dilated-conv spatial mask: ``[B, C, H, W] -> [B, 1, H, W]``."""

    def __init__(self, channel: int, reduction: int = 16, kernel_size: int = 3,
                 dilation_val: int = 4):
        super().__init__()
        # Floor the reduced width at 1 (see ChannelGate) so small-channel inputs
        # don't build a 0-channel conv.
        hidden = max(1, channel // reduction)
        self.conv1 = nn.Conv2d(channel, hidden, kernel_size=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size,
                padding=dilation_val,
                dilation=dilation_val,
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size,
                padding=dilation_val,
                dilation=dilation_val,
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Conv2d(hidden, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, H, W] -> [B, 1, H, W] broadcast mask.
        y = self.conv1(x)  # [B, h, H, W]
        y = self.conv2(y)  # [B, h, H, W]
        y = self.conv3(y)  # [B, 1, H, W]
        y = self.bn(y)  # [B, 1, H, W]
        return y.expand_as(x)


class BAM(nn.Module):
    """Bottleneck Attention Module: residual channel-spatial recalibration.

    Maps ``[B, C, H, W] -> [B, C, H, W]`` as ``x + x * sigmoid(c(x) + s(x))``.
    """

    def __init__(self, channel: int):
        super().__init__()
        self.channel_attn = ChannelGate(channel)
        self.spatial_attn = SpatialGate(channel)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, H, W]; both gates broadcast to x before the sum.
        attn = F.sigmoid(self.channel_attn(x) + self.spatial_attn(x))  # [B, C, H, W]
        return x + x * attn


if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    attn = BAM(64)
    y = attn(x)
    print(y.shape)
