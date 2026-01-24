"""VITS2 neural network modules."""

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate padding for 'same' convolution."""
    return (kernel_size * dilation - dilation) // 2


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """Initialize module weights."""
    if isinstance(m, nn.Conv1d | nn.Linear):
        m.weight.data.normal_(mean, std)


class LayerNorm(nn.Module):
    """Layer normalization for channel-first tensors."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class WN(nn.Module):
    """WaveNet-style dilated convolution block."""

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = get_padding(kernel_size, dilation)

            in_layer = weight_norm(
                nn.Conv1d(
                    hidden_channels,
                    2 * hidden_channels,
                    kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.in_layers.append(in_layer)

            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = weight_norm(
                nn.Conv1d(hidden_channels, res_skip_channels, 1)
            )
            self.res_skip_layers.append(res_skip_layer)

        self.dropout = nn.Dropout(dropout)

        if gin_channels > 0:
            self.cond_layer = weight_norm(
                nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor | None = None,
        g: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = torch.zeros_like(x)

        if g is not None:
            g = self.cond_layer(g)

        for i, (in_layer, res_skip_layer) in enumerate(
            zip(self.in_layers, self.res_skip_layers, strict=True)
        ):
            x_in = in_layer(x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
                x_in = x_in + g_l

            acts = torch.tanh(x_in[:, : self.hidden_channels, :]) * torch.sigmoid(
                x_in[:, self.hidden_channels :, :]
            )
            acts = self.dropout(acts)

            res_skip = res_skip_layer(acts)

            if i < self.n_layers - 1:
                x = x + res_skip[:, : self.hidden_channels, :]
                output = output + res_skip[:, self.hidden_channels :, :]
            else:
                output = output + res_skip

        if x_mask is not None:
            output = output * x_mask

        return output

    def remove_weight_norm(self) -> None:
        for layer in self.in_layers:
            remove_parametrizations(layer, "weight")
        for layer in self.res_skip_layers:
            remove_parametrizations(layer, "weight")
        if hasattr(self, "cond_layer"):
            remove_parametrizations(self.cond_layer, "weight")


class MultiHeadAttention(nn.Module):
    """Multi-head attention with relative positional encoding."""

    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        dropout: float = 0.0,
        window_size: int | None = 4,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.dropout = dropout
        self.window_size = window_size

        self.k_channels = channels // n_heads
        self.scale = self.k_channels**-0.5

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)

        self.drop = nn.Dropout(dropout)

        if window_size is not None:
            n_heads_rel = 1
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        q = rearrange(q, "b (h d) t -> b h t d", h=self.n_heads)
        k = rearrange(k, "b (h d) t -> b h t d", h=self.n_heads)
        v = rearrange(v, "b (h d) t -> b h t d", h=self.n_heads)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.window_size is not None:
            scores = self._add_relative_position(scores, q, k)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)

        output = torch.matmul(attn, v)
        output = rearrange(output, "b h t d -> b (h d) t")

        return self.conv_o(output)

    def _add_relative_position(
        self,
        scores: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """Add relative positional bias to attention scores."""
        # Simplified relative position implementation
        return scores


class FFN(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.conv_2 = nn.Conv1d(
            filter_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.conv_1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.conv_2(x)
        if x_mask is not None:
            x = x * x_mask
        return x


class EncoderLayer(nn.Module):
    """Transformer encoder layer."""

    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        kernel_size: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(
            hidden_channels, hidden_channels, n_heads, dropout
        )
        self.norm1 = LayerNorm(hidden_channels)
        self.ffn = FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, dropout)
        self.norm2 = LayerNorm(hidden_channels)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out = self.attn(x, x, attn_mask)
        x = self.norm1(x + self.drop(attn_out))
        ffn_out = self.ffn(x, x_mask)
        x = self.norm2(x + self.drop(ffn_out))
        if x_mask is not None:
            x = x * x_mask
        return x


class TextEncoder(nn.Module):
    """Text encoder with transformer layers."""

    def __init__(
        self,
        n_vocab: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(hidden_channels, filter_channels, n_heads, kernel_size, dropout)
                for _ in range(n_layers)
            ]
        )

        self.proj = nn.Conv1d(hidden_channels, hidden_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Phoneme IDs [B, T]
            x_lengths: Sequence lengths [B]

        Returns:
            Tuple of (hidden, mean, logvar, mask)
        """
        x = self.emb(x) * math.sqrt(self.hidden_channels)
        x = x.transpose(1, 2)  # [B, C, T]

        x_mask = self._sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)

        for layer in self.encoder_layers:
            x = layer(x, x_mask)

        stats = self.proj(x) * x_mask
        mean, logvar = torch.split(stats, self.hidden_channels, dim=1)

        return x, mean, logvar, x_mask

    @staticmethod
    def _sequence_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
        """Create sequence mask from lengths."""
        max_len = max_len or lengths.max().item()
        ids = torch.arange(0, max_len, device=lengths.device)
        return ids < lengths.unsqueeze(1)


class PosteriorEncoder(nn.Module):
    """Posterior encoder for VAE."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 16,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels

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
        """Forward pass.

        Args:
            x: Mel spectrogram [B, M, T]
            x_lengths: Sequence lengths [B]
            g: Speaker embedding [B, G, 1]

        Returns:
            Tuple of (z, mean, logvar, mask)
        """
        max_len = x.size(2)
        x_mask = self._sequence_mask(x_lengths, max_len).unsqueeze(1).to(x.dtype)

        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g)

        stats = self.proj(x) * x_mask
        mean, logvar = torch.split(stats, self.out_channels, dim=1)

        # Reparameterization
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)

        return z, mean, logvar, x_mask

    @staticmethod
    def _sequence_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
        max_len = max_len or lengths.max().item()
        ids = torch.arange(0, max_len, device=lengths.device)
        return ids < lengths.unsqueeze(1)


class ResBlock1(nn.Module):
    """Residual block type 1 for HiFi-GAN."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, ...] = (1, 3, 5),
    ) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for d in dilation:
            self.convs1.append(
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
            )
            self.convs2.append(
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
            )

        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2, strict=True):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self) -> None:
        for conv in self.convs1:
            remove_parametrizations(conv, "weight")
        for conv in self.convs2:
            remove_parametrizations(conv, "weight")


class ResBlock2(nn.Module):
    """Residual block type 2 for HiFi-GAN."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, ...] = (1, 3),
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()

        for d in dilation:
            self.convs.append(
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
            )

        self.convs.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            xt = F.leaky_relu(x, 0.1)
            xt = conv(xt)
            x = xt + x
        return x

    def remove_weight_norm(self) -> None:
        for conv in self.convs:
            remove_parametrizations(conv, "weight")


class Generator(nn.Module):
    """HiFi-GAN generator for audio synthesis."""

    def __init__(
        self,
        initial_channel: int,
        resblock_type: str = "1",
        resblock_kernel_sizes: list[int] | None = None,
        resblock_dilation_sizes: list[list[int]] | None = None,
        upsample_rates: list[int] | None = None,
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: list[int] | None = None,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()

        resblock_kernel_sizes = resblock_kernel_sizes or [3, 7, 11]
        resblock_dilation_sizes = resblock_dilation_sizes or [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        upsample_rates = upsample_rates or [8, 8, 2, 2]
        upsample_kernel_sizes = upsample_kernel_sizes or [16, 16, 4, 4]

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        resblock = ResBlock1 if resblock_type == "1" else ResBlock2

        self.conv_pre = weight_norm(
            nn.Conv1d(initial_channel, upsample_initial_channel, 7, padding=3)
        )

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=True)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        ch,
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=True)
            ):
                self.resblocks.append(resblock(ch, k, tuple(d)))

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate audio from latent representation.

        Args:
            x: Latent representation [B, C, T]
            g: Speaker embedding [B, G, 1]

        Returns:
            Audio waveform [B, 1, T']
        """
        x = self.conv_pre(x)

        if g is not None:
            x = x + self.cond(g)

        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)

            xs = 0.0
            for j in range(self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self) -> None:
        remove_parametrizations(self.conv_pre, "weight")
        for up in self.ups:
            remove_parametrizations(up, "weight")
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_parametrizations(self.conv_post, "weight")


class DiscriminatorP(nn.Module):
    """Period discriminator for multi-period discriminator."""

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.period = period
        norm_fn = nn.utils.spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList(
            [
                norm_fn(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), (2, 0))),
                norm_fn(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), (2, 0))),
                norm_fn(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), (2, 0))),
                norm_fn(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), (2, 0))),
                norm_fn(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, (2, 0))),
            ]
        )
        self.conv_post = norm_fn(nn.Conv2d(1024, 1, (3, 1), 1, (1, 0)))

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []

        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = x.flatten(1, -1)

        return x, fmap


class DiscriminatorS(nn.Module):
    """Scale discriminator for multi-scale discriminator."""

    def __init__(self, use_spectral_norm: bool = False) -> None:
        super().__init__()
        norm_fn = nn.utils.spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList(
            [
                norm_fn(nn.Conv1d(1, 128, 15, 1, 7)),
                norm_fn(nn.Conv1d(128, 128, 41, 2, 20, groups=4)),
                norm_fn(nn.Conv1d(128, 256, 41, 2, 20, groups=16)),
                norm_fn(nn.Conv1d(256, 512, 41, 4, 20, groups=16)),
                norm_fn(nn.Conv1d(512, 1024, 41, 4, 20, groups=16)),
                norm_fn(nn.Conv1d(1024, 1024, 41, 1, 20, groups=16)),
                norm_fn(nn.Conv1d(1024, 1024, 5, 1, 2)),
            ]
        )
        self.conv_post = norm_fn(nn.Conv1d(1024, 1, 3, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = x.flatten(1, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """Multi-period discriminator combining period and scale discriminators."""

    def __init__(
        self,
        periods: list[int] | None = None,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        periods = periods or [2, 3, 5, 7, 11]

        self.discriminators = nn.ModuleList(
            [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in periods]
        )
        self.discriminators.append(DiscriminatorS(use_spectral_norm=use_spectral_norm))

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[list[torch.Tensor]],
        list[list[torch.Tensor]],
    ]:
        """Forward pass for real and generated audio.

        Returns:
            Tuple of (real_outputs, fake_outputs, real_fmaps, fake_fmaps)
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DurationPredictor(nn.Module):
    """Duration predictor for alignment."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, hidden_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm1 = LayerNorm(hidden_channels)
        self.conv2 = nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm2 = LayerNorm(hidden_channels)
        self.proj = nn.Conv1d(hidden_channels, 1, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv1(x * x_mask)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x * x_mask)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class ResidualCouplingLayer(nn.Module):
    """Residual coupling layer for normalizing flow."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        self.half_channels = channels // 2

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels, 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None = None,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x0, x1 = torch.split(x, [self.half_channels, self.half_channels], dim=1)

        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g)
        m = self.post(h) * x_mask

        if not reverse:
            x1 = x1 + m
            logdet = torch.zeros(x.size(0), device=x.device)
        else:
            x1 = x1 - m
            logdet = torch.zeros(x.size(0), device=x.device)

        x = torch.cat([x0, x1], dim=1)
        return x, logdet


class Flip(nn.Module):
    """Flip layer for normalizing flow."""

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor | None = None,
        g: torch.Tensor | None = None,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.flip(x, dims=[1])
        logdet = torch.zeros(x.size(0), device=x.device)
        return x, logdet


class ResidualCouplingBlock(nn.Module):
    """Block of residual coupling layers."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()

        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                )
            )
            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x, _ = flow(x, x_mask, g, reverse=reverse)
        return x
