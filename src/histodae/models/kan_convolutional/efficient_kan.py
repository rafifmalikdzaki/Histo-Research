from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F


class KANLinear(torch.nn.Module):
    """Linear layer with a learnable B-spline residual on a fixed grid.

    Forward maps ``[..., in_features] -> [..., out_features]`` as
    ``base_activation(x) @ base_weight + spline(x) @ spline_weight``.
    """

    grid: torch.Tensor
    base_weight: torch.Tensor
    spline_weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 3,
        spline_order: int = 2,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: Callable[[], torch.nn.Module] = torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: list[float] | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        if grid_range is None:
            grid_range = [-1, 1]

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            # noise: [grid_size + 1, in_features, out_features], drawn from the
            # global RNG (seeded construction stays reproducible).
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            with torch.no_grad():
                self.spline_weight.copy_(
                    (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                    * self.curve2coeff(
                        self.grid.T[self.spline_order : -self.spline_order],
                        noise,
                    )
                )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """B-spline basis expansion: ``[B, in] -> [B, in, grid+spline]``."""
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid  # [in, grid + 2*spline + 1]
        x = x.unsqueeze(-1)  # [B, in, 1]
        # Order-0 indicator bases: [B, in, grid + 2*spline].
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        # Cox-de Boor recursion over the spline order (typically 2 steps).
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Least-squares spline coefficients: ``[B, in]`` x ``[B, in, out]``.

        Returns ``[out, in, grid+spline]``. Uses ``torch.linalg.lstsq``, whose
        ~1e-9 nondeterminism is why seeded construction is only approximately
        reproducible (see the ``allclose(atol=1e-6)`` parity note in tests).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)  # [in, B, grid+spline]
        B = y.transpose(0, 1)  # [in, B, out]
        solution = torch.linalg.lstsq(A, B).solution  # [in, grid+spline, out]
        result = solution.permute(2, 0, 1)  # [out, in, grid+spline]

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        """``[out, in, grid+spline]`` spline weights with the per-edge scaler."""
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[..., in] -> [..., out]``: base activation path + spline path."""
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)  # [B', in]

        base_output = F.linear(self.base_activation(x), self.base_weight)  # [B', out]
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),  # [B', in*(grid+spline)]
            self.scaled_spline_weight.view(self.out_features, -1),  # [out, in*(grid+spline)]
        )  # [B', out]
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01) -> None:
        """Adapt the spline grid to the data distribution (inference-time op)."""
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # [B, in, coeff]
        splines = splines.permute(1, 0, 2)  # [in, B, coeff]
        orig_coeff = self.scaled_spline_weight  # [out, in, coeff]
        orig_coeff = orig_coeff.permute(1, 2, 0)  # [in, coeff, out]
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # [in, B, out]
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)  # [B, in, out]

        # Sort each channel individually to collect the data distribution.
        # Index/constant tensors inherit the input's device (never hardcoded).
        x_sorted = torch.sort(x, dim=0)[0]  # [B, in]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]  # [grid+1, in]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        with torch.no_grad():
            self.grid.copy_(grid.T)
            self.spline_weight.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(
        self,
        regularize_activation: float = 1.0,
        regularize_entropy: float = 1.0,
    ) -> torch.Tensor:
        """
        L1 + entropy regularization over spline weights.

        This approximates the original paper's sample-based L1 (which needs the
        expanded [B, in, out] intermediate hidden behind F.linear) with the
        mean absolute spline weight plus the authors' own weight term.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    """Stack of KANLinear layers over a hidden-dims list (unused by the conv
    family, which uses :class:`KANLinear` directly; kept for API stability)."""

    layers: torch.nn.ModuleList

    def __init__(
        self,
        layers_hidden: list[int],
        grid_size: int = 3,
        spline_order: int = 2,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation: Callable[[], torch.nn.Module] = torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: list[float] | None = None,
    ):
        super().__init__()
        if grid_range is None:
            grid_range = [-1, 1]
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
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

    def forward(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        for layer in self.layers:
            kan_layer = cast("KANLinear", layer)
            if update_grid:
                kan_layer.update_grid(x)
            x = kan_layer(x)
        return x

    def regularization_loss(
        self,
        regularize_activation: float = 1.0,
        regularize_entropy: float = 1.0,
    ) -> torch.Tensor:
        first = cast("KANLinear", self.layers[0])
        total = torch.zeros((), device=first.base_weight.device)
        for layer in self.layers:
            kan_layer = cast("KANLinear", layer)
            total = total + kan_layer.regularization_loss(
                regularize_activation, regularize_entropy
            )
        return total
