"""Posterior encoder for VITS.

Encodes mel spectrograms into latent space with stable variance computation.
"""

import torch
import torch.nn as nn

from src.models.modules import WN


class PosteriorEncoder(nn.Module):
    """Posterior encoder q(z|x) with numerical stability improvements.

    Uses clamped log-variance and stable sampling for training from scratch.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        g: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_mask = torch.unsqueeze(
            self._sequence_mask(x_lengths, x.size(2)), 1
        ).to(x.dtype)

        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)

        # Clamp log-variance to prevent exp overflow/underflow
        # Range [-7, 7] means variance in [~0.001, ~1100]
        logs = torch.clamp(logs, min=-7.0, max=7.0)

        # Stable reparameterization sampling
        # z = m + std * noise, where std = exp(logs)
        noise = torch.randn_like(m)
        std = torch.exp(logs)
        z = (m + std * noise) * x_mask

        return z, m, logs, x_mask

    def _sequence_mask(self, length: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
        if max_length is None:
            max_length = int(length.max().item())
        ids = torch.arange(max_length, device=length.device)
        return ids.unsqueeze(0) < length.unsqueeze(1)
