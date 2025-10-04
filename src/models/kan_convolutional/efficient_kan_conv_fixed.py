import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .efficient_kan import KANLinear


class EfficientKANConv2D(nn.Module):
    """
    Efficient 2D KAN Convolution Layer using KANLinear on extracted patches
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

        # Optimized: Use a single KANLinear for all output channels
        # This is much more efficient than separate layers
        self.kan_layer = KANLinear(
            in_features=in_channels * kernel_size * kernel_size,
            out_features=out_channels,
            grid_size=grid_size,
            spline_order=spline_order,
        )

    def forward(self, x):
        batch_size, _, height, width = x.shape

        # Extract patches using unfold
        unfold = nn.Unfold(
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride
        )

        # Shape: (batch_size, in_channels * kernel_size * kernel_size, out_h * out_w)
        patches = unfold(x)

        # Calculate output dimensions
        out_h = (height + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        out_w = (width + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        num_patches = out_h * out_w

        # Optimized: Process all patches and output channels at once
        # Transpose to (batch_size * num_patches, in_features) for batch processing
        patches_flat = patches.transpose(1, 2).contiguous().view(-1, patches.size(1))
        # Shape: (batch_size * num_patches, in_features)

        # Single KANLinear call for all output channels
        out = self.kan_layer(patches_flat)  # (batch_size * num_patches, out_channels)

        # Reshape back to spatial dimensions
        output = out.view(batch_size, num_patches, self.out_channels).transpose(1, 2)
        # Shape: (batch_size, out_channels, num_patches)

        # Reshape to final output shape
        output = output.view(batch_size, self.out_channels, out_h, out_w)

        return output


# Alias for compatibility with existing code
KAN_Convolutional_Layer = EfficientKANConv2D