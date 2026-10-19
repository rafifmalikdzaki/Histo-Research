"""Efficient 2D KAN convolution: patch extraction + a single shared KANLinear."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .efficient_kan import KANLinear


class EfficientKANConv2D(nn.Module):
    """2D KAN convolution applying one KANLinear to all extracted patches.

    Forward maps ``[B, C_in, H, W] -> [B, C_out, H', W']`` with ``H'``/``W'``
    from the usual conv arithmetic. The patch extractor is built once in
    ``__init__`` (``nn.Unfold`` is stateless and parameter-free, so hoisting
    it changes neither numerics nor ``state_dict`` keys).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        grid_size: int = 3,
        spline_order: int = 2,
        device: str = "cpu",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.unfold = nn.Unfold(
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )
        # Single KANLinear shared by all output channels (far cheaper than
        # one layer per channel). Placed on `device` at construction (the arg
        # used to be silently ignored, leaving params on CPU until an outer
        # .to() moved them).
        self.kan_layer = KANLinear(
            in_features=in_channels * kernel_size * kernel_size,
            out_features=out_channels,
            grid_size=grid_size,
            spline_order=spline_order,
        ).to(device)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C_in, H, W].
        batch_size, _, height, width = x.shape

        # [B, C_in*K*K, L] with L = H'*W' patch positions.
        patches = self.unfold(x)

        out_h = (height + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        out_w = (width + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        num_patches = out_h * out_w

        # [B*L, C_in*K*K]: one batched KANLinear call for every patch and
        # every output channel at once.
        patches_flat = patches.transpose(1, 2).contiguous().view(-1, patches.size(1))
        out = self.kan_layer(patches_flat)  # [B*L, C_out]

        # [B*L, C_out] -> [B, C_out, L] -> [B, C_out, H', W'].
        output = out.view(batch_size, num_patches, self.out_channels).transpose(1, 2)
        output = output.view(batch_size, self.out_channels, out_h, out_w)

        return output


# Alias for compatibility with existing code
KAN_Convolutional_Layer = EfficientKANConv2D
