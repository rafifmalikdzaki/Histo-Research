import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class OptimizedKANConv2D(nn.Module):
    """
    Highly optimized 2D KAN Convolution Layer with:
    - Vectorized B-spline computation
    - Eliminated nested loops
    - Memory-efficient operations
    - Better GPU utilization
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
        grid_range: Tuple[float, float] = (-1, 1),
        device: str = "cuda",
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
        self.base_weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )

        # Learnable spline coefficients - flattened for efficiency
        self.spline_weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels * kernel_size * kernel_size,
                grid_size + spline_order
            )
        )

        # Spline scaler for each output channel
        self.spline_scaler = nn.Parameter(torch.ones(out_channels))

        # Precompute grid and B-spline denominators
        self._setup_bspline_parameters(grid_range)

        self.base_activation = base_activation()
        self.scale_base = scale_base
        self.scale_spline = scale_spline

        self.reset_parameters()

    def _setup_bspline_parameters(self, grid_range: Tuple[float, float]):
        """Setup B-spline grid parameters and precompute denominators"""
        # Create grid for B-splines
        h = (grid_range[1] - grid_range[0]) / self.grid_size
        grid = torch.arange(
            -self.spline_order,
            self.grid_size + self.spline_order + 1,
            dtype=torch.float32
        ) * h + grid_range[0]
        self.register_buffer('grid', grid)

        # Precompute B-spline denominators to avoid division in forward pass
        denominators = []
        for k in range(1, self.spline_order + 1):
            left_den = grid[k:-1] - grid[:-(k + 1)]
            right_den = grid[k + 1 :] - grid[1:(-k)]
            # Clamp to avoid division by zero
            left_den = left_den.clamp(min=1e-8)
            right_den = right_den.clamp(min=1e-8)
            denominators.append((left_den, right_den))

        self.register_buffer("bspline_denominators", denominators)

    def reset_parameters(self):
        """Initialize parameters with better initialization"""
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5) * self.scale_spline)

    def compute_bspline_bases_vectorized(self, x: torch.Tensor):
        """
        Vectorized B-spline basis computation for batch processing

        Args:
            x: Input tensor of shape (batch_size, num_points)

        Returns:
            B-spline bases of shape (batch_size, num_points, grid_size + spline_order)
        """
        grid = self.grid
        x_expanded = x.unsqueeze(-1)  # (batch_size, num_points, 1)

        # Initialize basis functions
        bases = ((x_expanded >= grid[:-1]) & (x_expanded < grid[1:])).float()

        # Recursive formula for B-splines - fully vectorized
        for k in range(1, self.spline_order + 1):
            left_den, right_den = self.bspline_denominators[k-1]

            left_term = (
                (x_expanded - grid[:-(k + 1)]) / left_den * bases[:, :, :-1]
            )
            right_term = (
                (grid[k + 1 :] - x_expanded) / right_den * bases[:, :, 1:]
            )
            bases = left_term + right_term

        return bases

    def forward(self, x: torch.Tensor):
        """
        Optimized forward pass with vectorized operations and no nested loops
        """
        batch_size, _, height, width = x.shape

        # Standard convolution with base activation
        base_conv = F.conv2d(
            x,
            self.base_weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )
        base_output = self.base_activation(base_conv)

        # Extract patches for spline computation - vectorized
        unfold = nn.Unfold(
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride
        )

        # Shape: (batch_size, in_channels * kernel_size^2, out_h * out_w)
        patches = unfold(x)

        # Calculate output dimensions
        out_h = (height + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        out_w = (width + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        num_patches = out_h * out_w

        # Reshape patches for batch processing
        # (batch_size * num_patches, in_channels * kernel_size^2)
        patches_flat = patches.transpose(1, 2).contiguous().view(-1, patches.size(1))

        # Vectorized B-spline computation for all patches at once
        # Compute B-spline bases for all input values
        spline_bases = self.compute_bspline_bases_vectorized(patches_flat)
        # Shape: (batch_size * num_patches, in_channels * kernel_size^2, grid_size + spline_order)

        # Vectorized spline weight application
        # Use einsum for efficient batch matrix multiplication
        spline_output_flat = torch.einsum(
            'bpf,opf->bpo',
            spline_bases,  # (batch_size * num_patches, in_features, grid_size + spline_order)
            self.spline_weight.view(
                self.out_channels,
                self.in_channels * self.kernel_size * self.kernel_size,
                -1
            )  # (out_channels, in_features, grid_size + spline_order)
        )
        # Shape: (batch_size * num_patches, out_channels)

        # Apply spline scaling
        spline_output_flat = spline_output_flat * self.spline_scaler

        # Reshape back to spatial dimensions
        spline_output = spline_output_flat.view(
            batch_size, num_patches, self.out_channels
        ).transpose(1, 2).view(
            batch_size, self.out_channels, out_h, out_w
        )

        # Combine base and spline outputs
        return base_output + self.scale_spline * spline_output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """Compute regularization loss for spline weights"""
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / (regularization_loss_activation + 1e-8)
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class OptimizedKANConvLayer(nn.Module):
    """
    Wrapper class for compatibility with existing code
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
        **kwargs
    ):
        super().__init__()
        self.conv = OptimizedKANConv2D(
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

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return self.conv.regularization_loss(regularize_activation, regularize_entropy)


# Alias for compatibility with existing code
KAN_Convolutional_Layer = OptimizedKANConvLayer