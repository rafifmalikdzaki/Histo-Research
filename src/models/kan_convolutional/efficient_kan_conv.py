import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class EfficientKANConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: Tuple[float, float] = [-1, 1],
    ):
        super(EfficientKANConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Number of input features for each kernel element
        self.in_features = kernel_size[0] * kernel_size[1]

        # Create grid for B-splines
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(self.in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # Base weights (equivalent to standard conv weights)
        self.base_weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *kernel_size)
        )

        # Spline weights for learnable activation functions
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *kernel_size, grid_size + spline_order)
        )

        # Optional scale for spline functions
        self.spline_scaler = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *kernel_size)
        )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize base weights like standard convolution
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        # Initialize spline weights
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features,
                              self.out_channels, self.in_channels, *self.kernel_size)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            # Simplified initialization for spline weights
            self.spline_weight.data.uniform_(-0.1, 0.1)

        # Initialize spline scaler
        nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute B-spline bases efficiently
        x shape: (batch_size, channels, height, width, in_features)
        """
        grid = self.grid  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)  # (..., in_features, 1)

        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)] + 1e-8)
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)] + 1e-8)
                * bases[:, :, 1:]
            )

        return bases.contiguous()

    def forward(self, x: torch.Tensor):
        batch_size, channels, height, width = x.shape

        # Efficient convolution using unfold
        # Extract sliding windows
        unfold = nn.Unfold(
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride
        )

        # Shape: (batch_size, channels * kernel_size * kernel_size, output_height * output_width)
        unfolded = unfold(x)
        output_height = int((height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        output_width = int((width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        # Reshape for processing
        # (batch_size, channels, kernel_size*kernel_size, output_height * output_width)
        unfolded = unfolded.view(batch_size, channels, self.in_features, output_height * output_width)

        # Base convolution (standard convolution)
        # Reshape base_weight for matrix multiplication
        base_weight_flat = self.base_weight.view(self.out_channels, self.in_channels, -1)
        base_output = F.conv2d(x, self.base_weight, stride=self.stride,
                              padding=self.padding, dilation=self.dilation)

        # Spline computation
        spline_output = torch.zeros(batch_size, self.out_channels,
                                   output_height * output_width, device=x.device)

        for ch in range(channels):
            # Extract patches for this channel
            channel_patches = unfolded[:, ch, :, :]  # (batch_size, in_features, num_patches)

            # Compute B-spline bases
            spline_bases = self.b_splines(channel_patches.transpose(1, 2))
            # shape: (batch_size, num_patches, in_features, grid_size + spline_order)

            # Combine with spline weights
            spline_weights_ch = self.spline_weight[:, ch, :, :, :]  # (out_channels, in_features, kernel_h, kernel_w, grid_size + spline_order)
            spline_weights_ch = spline_weights_ch.view(self.out_channels, self.in_features, -1)

            # Compute spline activation
            spline_bases_flat = spline_bases.view(batch_size, output_height * output_width, -1)
            spline_activation = torch.einsum('bpi,opi->bo', spline_bases_flat, spline_weights_ch)

            spline_output += spline_activation

        # Reshape spline output
        spline_output = spline_output.view(batch_size, self.out_channels, output_height, output_width)

        # Apply spline scaling
        scaled_spline_output = spline_output * self.spline_scaler.mean(dim=(2, 3), keepdim=True)

        # Combine base and spline outputs
        output = base_output + scaled_spline_output

        return output


class EfficientKANConvLayer(nn.Module):
    """Efficient KAN Convolutional Layer that can handle multiple input/output channels"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
        grid_size: int = 5,
        spline_order: int = 3,
        **kwargs
    ):
        super(EfficientKANConvLayer, self).__init__()

        self.conv = EfficientKANConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            grid_size=grid_size,
            spline_order=spline_order,
            **kwargs
        )

    def forward(self, x: torch.Tensor):
        return self.conv(x)