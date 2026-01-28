"""Stochastic duration predictor for VITS."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules import DDSConv, ElementwiseAffine, Flip, LayerNorm, Log


class ConvFlow(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        n_layers: int,
        num_bins: int = 10,
        tail_bound: float = 5.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2

        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.0)
        # Output: num_bins widths + num_bins heights + (num_bins + 1) derivatives
        self.proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 + 1), 1)
        self.proj.weight.data.zero_()
        if self.proj.bias is not None:
            self.proj.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None = None,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        x0, x1 = torch.split(x, [self.half_channels, self.half_channels], 1)
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask

        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)

        unnormalized_widths = h[..., : self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_derivatives = h[..., 2 * self.num_bins :]  # num_bins + 1 derivatives

        x1, logabsdet = self._piecewise_rational_quadratic_transform(
            x1, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, reverse=reverse
        )

        x = torch.cat([x0, x1], 1) * x_mask
        if not reverse:
            logdet = torch.sum(logabsdet * x_mask, [1, 2])
            return x, logdet
        return x

    def _piecewise_rational_quadratic_transform(
        self,
        inputs: torch.Tensor,
        unnormalized_widths: torch.Tensor,
        unnormalized_heights: torch.Tensor,
        unnormalized_derivatives: torch.Tensor,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        min_bin_width = 1e-3
        min_bin_height = 1e-3
        min_derivative = 1e-3

        widths = F.softmax(unnormalized_widths, dim=-1)
        widths = min_bin_width + (1 - min_bin_width * self.num_bins) * widths
        cumwidths = torch.cumsum(widths, dim=-1)
        cumwidths = F.pad(cumwidths, (1, 0), mode="constant", value=0.0)
        cumwidths = (cumwidths - 0.5) * 2 * self.tail_bound
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]

        heights = F.softmax(unnormalized_heights, dim=-1)
        heights = min_bin_height + (1 - min_bin_height * self.num_bins) * heights
        cumheights = torch.cumsum(heights, dim=-1)
        cumheights = F.pad(cumheights, (1, 0), mode="constant", value=0.0)
        cumheights = (cumheights - 0.5) * 2 * self.tail_bound
        heights = cumheights[..., 1:] - cumheights[..., :-1]

        derivatives = min_derivative + F.softplus(unnormalized_derivatives)

        return self._rational_quadratic_spline(
            inputs,
            widths,
            heights,
            derivatives,
            cumwidths,
            cumheights,
            reverse=reverse,
        )

    def _rational_quadratic_spline(
        self,
        inputs: torch.Tensor,
        widths: torch.Tensor,
        heights: torch.Tensor,
        derivatives: torch.Tensor,
        cumwidths: torch.Tensor,
        cumheights: torch.Tensor,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = torch.zeros_like(inputs)
        logabsdet = torch.zeros_like(inputs)

        if reverse:
            bin_idx = self._searchsorted(cumheights, inputs)
        else:
            bin_idx = self._searchsorted(cumwidths, inputs)

        # Clamp bin_idx to valid range for all tensors
        num_bins = widths.shape[-1]
        bin_idx = bin_idx.clamp(0, num_bins - 1)
        bin_idx_expanded = bin_idx[..., None]

        input_cumwidths = cumwidths.gather(-1, bin_idx_expanded)[..., 0]
        input_bin_widths = widths.gather(-1, bin_idx_expanded)[..., 0]
        input_cumheights = cumheights.gather(-1, bin_idx_expanded)[..., 0]
        input_heights = heights.gather(-1, bin_idx_expanded)[..., 0]
        input_delta = input_heights / input_bin_widths.clamp(min=1e-5)
        input_derivatives = derivatives.gather(-1, bin_idx_expanded)[..., 0]

        # For derivatives_plus_one, use bin_idx + 1 clamped
        bin_idx_plus_one = (bin_idx + 1).clamp(0, derivatives.shape[-1] - 1)
        input_derivatives_plus_one = derivatives.gather(-1, bin_idx_plus_one[..., None])[..., 0]

        if reverse:
            a = (inputs - input_cumheights) * (
                input_derivatives + input_derivatives_plus_one - 2 * input_delta
            ) + input_heights * (input_delta - input_derivatives)
            b = input_heights * input_derivatives - (inputs - input_cumheights) * (
                input_derivatives + input_derivatives_plus_one - 2 * input_delta
            )
            c = -input_delta * (inputs - input_cumheights)
            discriminant = b.pow(2) - 4 * a * c
            discriminant = discriminant.clamp(min=0)
            root = (2 * c) / (-b - torch.sqrt(discriminant + 1e-6))
            outputs = root * input_bin_widths + input_cumwidths

            theta_one_minus_theta = root * (1 - root)
            denominator = input_delta + (
                (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
            )
            derivative_numerator = input_delta.pow(2) * (
                input_derivatives_plus_one * root.pow(2)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * (1 - root).pow(2)
            )
            logabsdet = -torch.log(derivative_numerator.clamp(min=1e-6)) + 2 * torch.log(denominator.clamp(min=1e-6))
        else:
            theta = (inputs - input_cumwidths) / input_bin_widths.clamp(min=1e-5)
            theta = theta.clamp(0, 1)
            theta_one_minus_theta = theta * (1 - theta)

            numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
            denominator = input_delta + (
                (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
            )
            outputs = input_cumheights + numerator / denominator.clamp(min=1e-6)

            derivative_numerator = input_delta.pow(2) * (
                input_derivatives_plus_one * theta.pow(2)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * (1 - theta).pow(2)
            )
            logabsdet = torch.log(derivative_numerator.clamp(min=1e-6)) - 2 * torch.log(denominator.clamp(min=1e-6))

        return outputs, logabsdet

    def _searchsorted(self, bin_locations: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        idx = torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1
        return idx.clamp(min=0, max=bin_locations.shape[-1] - 2)


class StochasticDurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        n_flows: int = 4,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        filter_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = Log()
        self.flows = nn.ModuleList()
        self.flows.append(ElementwiseAffine(2))
        for _ in range(n_flows):
            self.flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.post_flows = nn.ModuleList()
        self.post_flows.append(ElementwiseAffine(2))
        for _ in range(4):
            self.post_flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        w: torch.Tensor | None = None,
        g: torch.Tensor | None = None,
        reverse: bool = False,
        noise_scale: float = 1.0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = torch.randn(w.size(0), 2, w.size(2), device=x.device, dtype=x.dtype) * x_mask
            z_q = e_q

            for flow in self.post_flows:
                z_q, _ = flow(z_q, x_mask, g=(x + h_w))
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q = 0.0
            logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2])
            logw, logdet_q = self.log_flow(z0, x_mask)
            logdet_tot_q += logdet_q
            logdet_tot_q += torch.sum((z1**2 + math.log(2 * math.pi)) * x_mask, [1, 2]) * -0.5

            z = torch.cat([logw, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot_q -= logdet

            nll = torch.sum(0.5 * (math.log(2 * math.pi) + z**2) * x_mask, [1, 2]) - logdet_tot_q
            return nll / torch.sum(x_mask)

        flows = list(reversed(self.flows))
        flows = flows[:-2] + [flows[-1]]
        z = torch.randn(x.size(0), 2, x.size(2), device=x.device, dtype=x.dtype) * noise_scale
        for flow in flows:
            z = flow(z, x_mask, g=x, reverse=reverse)
        z0, _ = torch.split(z, [1, 1], 1)
        logw = z0
        return logw


class DurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = self.norm_1(x)
        x = F.relu(x)
        x = F.dropout(x, self.p_dropout, training=self.training)
        x = self.conv_2(x * x_mask)
        x = self.norm_2(x)
        x = F.relu(x)
        x = F.dropout(x, self.p_dropout, training=self.training)
        x = self.proj(x * x_mask)
        return x * x_mask
