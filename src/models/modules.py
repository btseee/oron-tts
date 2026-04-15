"""DiT building blocks for F5-TTS: RMSNorm, RoPE, AdaLN, Attention, FFN, DiTBlock."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Normalization ─────────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard RMSNorm: x / sqrt(mean(x^2)) * gamma
        rms = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (rms + self.eps) * self.gamma


# ── Timestep embedding ────────────────────────────────────────────────────────


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal timestep embedding → MLP → conditioning vector."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] values in [0, 1]
        device = t.device
        freqs = torch.exp(
            -math.log(10000) * torch.arange(self.half_dim, device=device) / (self.half_dim - 1)
        )
        emb = t[:, None] * freqs[None, :]  # [B, half_dim]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # [B, dim]
        return self.mlp(emb)


# ── Rotary Positional Embedding (RoPE) ────────────────────────────────────────


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096) -> None:
        super().__init__()
        assert dim % 2 == 0
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
        freqs = torch.outer(t, self.inv_freq.to(device))  # type: ignore[arg-type]
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cached_cos = emb.cos()[None, None, :, :]  # [1,1,T,D]
        self._cached_sin = emb.sin()[None, None, :, :]
        self._cached_len = seq_len
        self._cached_device = device

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k: [B, heads, T, head_dim]
        seq_len = q.shape[2]
        self._build_cache(seq_len, q.device)
        assert self._cached_cos is not None and self._cached_sin is not None
        cos = self._cached_cos[:, :, :seq_len, :]
        sin = self._cached_sin[:, :, :seq_len, :]
        q_rot = q * cos + self.rotate_half(q) * sin
        k_rot = k * cos + self.rotate_half(k) * sin
        return q_rot, k_rot


# ── Adaptive Layer Norm (AdaLN) ───────────────────────────────────────────────


class AdaLN(nn.Module):
    """Adaptive Layer Norm: scale & shift conditioned on timestep embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.proj = nn.Linear(dim, dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], cond: [B, D]
        scale, shift = self.proj(cond).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale) + shift


# ── Attention ─────────────────────────────────────────────────────────────────


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, rope: RotaryEmbedding) -> None:
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.rope = rope

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each [B, H, T, D_h]

        q, k = self.rope(q, k)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T, T]

        if mask is not None:
            # mask: [B, T] bool, True = valid; expand for broadcasting
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(out)


# ── FeedForward (SwiGLU) ──────────────────────────────────────────────────────


class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_mult: int = 2) -> None:
        super().__init__()
        inner_dim = dim * ff_mult
        self.gate_proj = nn.Linear(dim, inner_dim, bias=False)
        self.up_proj = nn.Linear(dim, inner_dim, bias=False)
        self.down_proj = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── DiT Block ─────────────────────────────────────────────────────────────────


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int, rope: RotaryEmbedding) -> None:
        super().__init__()
        self.adaLN_attn = AdaLN(dim)
        self.attn = Attention(dim, heads, rope)
        self.adaLN_ff = AdaLN(dim)
        self.ff = FeedForward(dim, ff_mult)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.adaLN_attn(x, t_emb), mask)
        x = x + self.ff(self.adaLN_ff(x, t_emb))
        return x
