import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EfficientKANConv2D(nn.Module):
    """
    Efficient 2D KAN Convolution Layer

    This implementation avoids expanding intermediate tensors by pre-computing
    B-spline basis activations and using matrix multiplication for linear combinations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=nn.SiLU,
        grid_range: tuple = (-1, 1),
        device: str = "cpu",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Standard convolution weights for base activation
        self.base_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

        # Learnable spline coefficients
        self.spline_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, grid_size + spline_order))

        # Grid for B-splines
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        self.register_buffer('grid', grid)

        self.base_activation = base_activation()
        self.scale_base = scale_base
        self.scale_spline = scale_spline

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5) * self.scale_spline)

    def compute_bspline_basis(self, x):
        """
        Compute B-spline basis functions efficiently
        x: input tensor of shape (..., in_features)
        Returns: basis functions of shape (..., in_features, grid_size + spline_order)
        """
        grid = self.grid
        x = x.unsqueeze(-1)

        # Initialize basis functions
        basis = ((x >= grid[:-1]) & (x < grid[1:])).float()

        # Recursive formula for B-splines
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:-(k + 1)]) / (grid[k:-1] - grid[:-(k + 1)] + 1e-8)
            right = (grid[k + 1:] - x) / (grid[k + 1:] - grid[1:-k] + 1e-8)
            basis = left * basis[:, :, :-1] + right * basis[:, :, 1:]

        return basis

    def forward(self, x):
        batch_size, _, height, width = x.shape

        # Standard convolution with base activation
        base_conv = F.conv2d(x, self.base_weight, stride=self.stride,
                           padding=self.padding, dilation=self.dilation)
        base_out = self.base_activation(base_conv)

        # Extract patches for spline computation
        unfold = nn.Unfold(kernel_size=self.kernel_size,
                         dilation=self.dilation,
                         padding=self.padding,
                         stride=self.stride)

        # Shape: (batch_size, in_channels * kernel_size * kernel_size, out_h * out_w)
        patches = unfold(x)

        # Reshape for processing
        out_h = (height + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        out_w = (width + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        patches = patches.view(batch_size, self.in_channels, self.kernel_size * self.kernel_size, out_h * out_w)

        # Compute spline activations
        spline_out = torch.zeros(batch_size, self.out_channels, out_h * out_w, device=x.device)

        for ich in range(self.in_channels):
            for kh in range(self.kernel_size):
                for kw in range(self.kernel_size):
                    idx = kh * self.kernel_size + kw
                    input_vals = patches[:, ich, idx, :]  # (batch_size, out_h * out_w)

                    # Compute B-spline basis
                    basis = self.compute_bspline_basis(input_vals)  # (batch_size, out_h * out_w, grid_size + spline_order)

                    # Apply spline weights
                    spline_coeffs = self.spline_weight[:, ich, kh, kw, :]  # (out_channels, grid_size + spline_order)
                    spline_contrib = torch.einsum('bnc,oc->bno', basis, spline_coeffs)

                    spline_out += spline_contrib

        # Reshape spline output
        spline_out = spline_out.view(batch_size, self.out_channels, out_h, out_w)

        # Combine base and spline outputs
        return base_out + self.scale_spline * spline_out


# Alias for compatibility with existing code
KAN_Convolutional_Layer = EfficientKANConv2D