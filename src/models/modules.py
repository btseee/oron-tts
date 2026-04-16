"""DiT building blocks: RMSNorm, SinusoidalEmbedding, TimestepEmbedding,
RotaryEmbedding, ConvPositionEmbedding, GRN, ConvNeXtV2Block,
AdaLayerNorm (6-param gated), Attention, FeedForward, DiTBlock.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Normalization ─────────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.to(self.weight.dtype)
        return x * self.weight


# ── Timestep embeddings ──────────────────────────────────────────────────────


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal position embedding (no MLP)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: float = 1000.0) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding + MLP for timestep conditioning."""

    def __init__(self, dim: int, freq_embed_dim: int = 256) -> None:
        super().__init__()
        self.time_embed = SinusoidalEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        time_hidden = self.time_embed(timestep)
        return self.time_mlp(time_hidden.to(timestep.dtype))


# ── Rotary Positional Embedding (RoPE) ────────────────────────────────────────


class RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, dim: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._cached_cos: torch.Tensor | None = None
        self._cached_sin: torch.Tensor | None = None
        self._cached_len: int = 0
        self._cached_device: torch.device | None = None

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        if seq_len <= self._cached_len and device == self._cached_device:
            return
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cached_cos = emb.cos()[None, None, :, :]
        self._cached_sin = emb.sin()[None, None, :, :]
        self._cached_len = seq_len
        self._cached_device = device

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]
        self._build_cache(seq_len, q.device)
        assert self._cached_cos is not None and self._cached_sin is not None
        cos = self._cached_cos[:, :, :seq_len, :]
        sin = self._cached_sin[:, :, :seq_len, :]
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ── Convolutional Position Embedding ─────────────────────────────────────────


class ConvPositionEmbedding(nn.Module):
    """Convolutional position embedding (from official F5-TTS)."""

    def __init__(self, dim: int, kernel_size: int = 31, groups: int = 16) -> None:
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # [B, N, D] → [B, D, N]
        mask_3d: torch.Tensor | None = None
        if mask is not None:
            mask_3d = mask.unsqueeze(1)
            x = x.masked_fill(~mask_3d, 0.0)
        x = self.conv1d(x)
        if mask_3d is not None:
            x = x.masked_fill(~mask_3d, 0.0)
        return x.permute(0, 2, 1)


# ── Global Response Normalization (ConvNeXt V2) ──────────────────────────────


class GRN(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=1, keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


# ── ConvNeXt V2 Block ────────────────────────────────────────────────────────


class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, dilation: int = 1) -> None:
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


# ── Precompute sinusoidal freqs for text encoder ─────────────────────────────


def precompute_freqs_cis(dim: int, end: int) -> torch.Tensor:
    """Precompute sinusoidal position frequencies for text encoder."""
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    return torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)


# ── Adaptive Layer Norm (6-parameter with gating) ────────────────────────────


class AdaLayerNorm(nn.Module):
    """6-parameter AdaLN: shift, scale, gate for both attention and FFN paths."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(
            emb, 6, dim=1
        )
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormFinal(nn.Module):
    """Final layer AdaLN: only scale + shift, no gating."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


# ── Attention ─────────────────────────────────────────────────────────────────


class Attention(nn.Module):
    def __init__(
        self, dim: int, heads: int, dim_head: int = 64, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.heads = heads
        self.inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim), nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope: RotaryEmbedding | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        head_dim = self.inner_dim // self.heads

        q = self.to_q(x).view(B, T, self.heads, head_dim).transpose(1, 2)
        k = self.to_k(x).view(B, T, self.heads, head_dim).transpose(1, 2)
        v = self.to_v(x).view(B, T, self.heads, head_dim).transpose(1, 2)

        if rope is not None:
            q, k = rope(q, k)

        attn_mask = None
        if mask is not None:
            # [B, 1, 1, T] — broadcastable key-padding mask.
            # NOT expanded to [B, h, T, T] so SDPA can use the memory-efficient backend.
            attn_mask = mask[:, None, None, :]

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, T, self.inner_dim)
        out = self.to_out(out)

        if mask is not None:
            out = out.masked_fill(~mask.unsqueeze(-1), 0.0)

        return out


# ── FeedForward (GELU, matching official F5-TTS) ─────────────────────────────


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        self.ff = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


# ── DiT Block ─────────────────────────────────────────────────────────────────


class DiTBlock(nn.Module):
    """DiT block with 6-parameter AdaLN modulation and gating."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope: RotaryEmbedding | None = None,
    ) -> torch.Tensor:
        # Pre-norm & modulation
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # Attention with gating
        attn_output = self.attn(norm, mask=mask, rope=rope)
        x = x + gate_msa.unsqueeze(1) * attn_output

        # FFN with modulation and gating
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x
