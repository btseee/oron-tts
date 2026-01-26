"""Attention mechanisms with Flash Attention 2 and RoPE."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for attention.

    Encodes relative position through rotation of query/key pairs.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10000,
    ) -> None:
        """Initialize RoPE.

        Args:
            dim: Head dimension (must be even).
            max_seq_len: Maximum sequence length.
            base: Base for frequency computation.
        """
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build cos/sin cache for positions."""
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)

        # Interleave cos/sin for complex rotation
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def forward(self, x: Tensor, offset: int = 0) -> tuple[Tensor, Tensor]:
        """Get cos/sin for positions.

        Args:
            x: Input tensor to get sequence length. Shape: (B, T, H, D).
            offset: Position offset for incremental decoding.

        Returns:
            (cos, sin) tensors for rotation.
        """
        seq_len = x.size(1)

        # Extend cache if needed
        if seq_len + offset > self.max_seq_len:
            self._build_cache(seq_len + offset)

        return (
            self.cos_cache[offset : offset + seq_len],
            self.sin_cache[offset : offset + seq_len],
        )


def apply_rotary_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply rotary embeddings to query and key tensors.

    Args:
        q: Query tensor. Shape: (B, T, H, D).
        k: Key tensor. Shape: (B, T, H, D).
        cos: Cosine rotation. Shape: (T, D).
        sin: Sine rotation. Shape: (T, D).

    Returns:
        Rotated (q, k) tensors.
    """

    def rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    # Broadcast cos/sin to (1, T, 1, D)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin

    return q_rot, k_rot


class FlashMultiHeadAttention(nn.Module):
    """Multi-head attention with Flash Attention 2 support.

    Uses F.scaled_dot_product_attention with Flash Attention backend
    when available (CUDA, fp16/bf16).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        use_flash: bool = True,
    ) -> None:
        """Initialize attention.

        Args:
            dim: Model dimension.
            num_heads: Number of attention heads.
            dropout: Attention dropout probability.
            qkv_bias: Use bias in QKV projection.
            use_flash: Use Flash Attention when available.
        """
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = dropout
        self.use_flash = use_flash

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: Tensor,
        rope: RotaryPositionalEmbedding | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass with optional RoPE and masking.

        Args:
            x: Input features. Shape: (B, T, D).
            rope: Optional rotary position embedding.
            mask: Optional padding mask. Shape: (B, T).

        Returns:
            Attention output. Shape: (B, T, D).
        """
        B, T, D = x.shape

        # Compute QKV
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b t (three h d) -> three b t h d", three=3, h=self.num_heads)
        q, k, v = qkv.unbind(dim=0)

        # Apply RoPE if provided
        if rope is not None:
            cos, sin = rope(q)
            q, k = apply_rotary_emb(q, k, cos, sin)

        # Reshape for attention: (B, H, T, D)
        q = rearrange(q, "b t h d -> b h t d")
        k = rearrange(k, "b t h d -> b h t d")
        v = rearrange(v, "b t h d -> b h t d")

        # Build attention mask
        attn_mask = None
        if mask is not None:
            # Convert padding mask to attention mask
            # mask: (B, T) -> attn_mask: (B, 1, 1, T)
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, self.num_heads, T, -1)
            attn_mask = attn_mask.bool()

        # Use PyTorch's optimized attention (Flash Attention 2 backend)
        if self.use_flash and self._can_use_flash(x):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            # Manual attention for compatibility
            out = self._manual_attention(q, k, v, attn_mask)

        # Reshape back
        out = rearrange(out, "b h t d -> b t (h d)")
        out = self.proj(out)

        return out

    def _can_use_flash(self, x: Tensor) -> bool:
        """Check if Flash Attention can be used."""
        return (
            x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16)
            and hasattr(F, "scaled_dot_product_attention")
        )

    def _manual_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None,
    ) -> Tensor:
        """Manual attention computation for fallback."""
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        return torch.matmul(attn, v)


class CrossAttention(nn.Module):
    """Cross-attention for conditioning on reference audio/text.

    Used in F5-TTS for voice cloning from reference mel-spectrograms.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_flash: bool = True,
    ) -> None:
        """Initialize cross-attention.

        Args:
            dim: Query dimension.
            context_dim: Context (key/value) dimension.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            use_flash: Use Flash Attention when available.
        """
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = dropout
        self.use_flash = use_flash

        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(context_dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Query tensor. Shape: (B, T_q, D).
            context: Key/value context. Shape: (B, T_kv, D_ctx).
            context_mask: Context padding mask. Shape: (B, T_kv).

        Returns:
            Cross-attention output. Shape: (B, T_q, D).
        """
        B, T_q, _ = x.shape
        T_kv = context.size(1)

        # Compute Q, K, V
        q = self.q_proj(x)
        kv = self.kv_proj(context)
        k, v = kv.chunk(2, dim=-1)

        # Reshape: (B, T, D) -> (B, H, T, head_dim)
        q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.num_heads)

        # Build mask
        attn_mask = None
        if context_mask is not None:
            attn_mask = context_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, self.num_heads, T_q, -1).bool()

        # Attention
        if self.use_flash and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                attn = attn.masked_fill(~attn_mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            out = torch.matmul(attn, v)

        # Reshape back
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.out_proj(out)
