import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class OptimizedKANLinear(nn.Module):
    """
    Optimized KAN Linear implementation with vectorized B-spline computation
    and memory-efficient operations for better GPU utilization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation=nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: Tuple[float, float] = [-1, 1],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Precompute grid parameters for B-splines
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        # Precompute B-spline denominators for efficiency
        self._precompute_bspline_denominators()

        # Initialize weights
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )

        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def _precompute_bspline_denominators(self):
        """Precompute B-spline denominator values to avoid division in forward pass"""
        grid = self.grid  # (in_features, grid_size + 2 * spline_order + 1)
        denominators = []

        for k in range(1, self.spline_order + 1):
            # Left denominator: (grid[:, k:-1] - grid[:, :-(k + 1)])
            left_den = grid[:, k:-1] - grid[:, :-(k + 1)]
            # Right denominator: (grid[:, k + 1 :] - grid[:, 1:(-k)])
            right_den = grid[:, k + 1 :] - grid[:, 1:(-k)]

            denominators.append((left_den, right_den))

        self.register_buffer("bspline_denominators", denominators)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )

            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines_vectorized(self, x: torch.Tensor):
        """
        Vectorized B-spline computation for better GPU utilization

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid = self.grid  # (in_features, grid_size + 2 * spline_order + 1)
        x_expanded = x.unsqueeze(-1)  # (batch_size, in_features, 1)

        # Initialize basis functions - vectorized
        bases = ((x_expanded >= grid[:, :-1]) & (x_expanded < grid[:, 1:])).to(x.dtype)

        # Vectorized B-spline recursion
        for k in range(1, self.spline_order + 1):
            left_den, right_den = self.bspline_denominators[k-1]

            # Add small epsilon to avoid division by zero
            left_den = left_den.clamp(min=1e-8)
            right_den = right_den.clamp(min=1e-8)

            # Vectorized operations - no loops
            left_term = (x_expanded - grid[:, : -(k + 1)]) / left_den * bases[:, :, :-1]
            right_term = (grid[:, k + 1 :] - x_expanded) / right_den * bases[:, :, 1:]
            bases = left_term + right_term

        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """Compute B-spline coefficients using least squares"""
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines_vectorized(x).transpose(0, 1)  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)

        # Use torch.linalg.lstsq for numerical stability
        solution = torch.linalg.lstsq(A, B).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(2, 0, 1)  # (out_features, in_features, grid_size + spline_order)

        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0
        )

    def forward(self, x: torch.Tensor):
        """Optimized forward pass with reduced memory allocations"""
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        # Base activation - optimized
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # Spline output - vectorized computation
        spline_bases = self.b_splines_vectorized(x)  # (batch_size, in_features, grid_size + spline_order)
        spline_bases_flat = spline_bases.view(x.size(0), -1)  # (batch_size, in_features * (grid_size + spline_order))

        scaled_weight = self.scaled_spline_weight.view(self.out_features, -1)
        spline_output = F.linear(spline_bases_flat, scaled_weight)

        # Combine outputs
        output = base_output + spline_output
        output = output.reshape(*original_shape[:-1], self.out_features)

        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        """Update grid based on input distribution"""
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        # Compute current spline outputs
        splines = self.b_splines_vectorized(x).permute(1, 0, 2)  # (in_features, batch_size, coeff)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)  # (in_features, coeff, out_features)
        unreduced_spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)

        # Adaptive grid computation
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        # Combine adaptive and uniform grids
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        # Update grid and recompute spline weights
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

        # Re-precompute denominators for new grid
        self._precompute_bspline_denominators()

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


class OptimizedKAN(nn.Module):
    """Optimized KAN layer using OptimizedKANLinear"""

    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                OptimizedKANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )