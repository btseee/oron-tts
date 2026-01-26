"""Diffusion Transformer (DiT) backbone for F5-TTS.

Implements a transformer with adaptive layer normalization (adaLN)
for conditioning on timestep and speaker embeddings.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from src.modules.attention import FlashMultiHeadAttention, RotaryPositionalEmbedding


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization conditioned on timestep/speaker."""

    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, dim * 2)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Apply adaptive layer norm.

        Args:
            x: Input features. Shape: (B, T, D).
            cond: Conditioning vector. Shape: (B, D_cond).

        Returns:
            Normalized and modulated features.
        """
        x = self.norm(x)
        scale, shift = self.proj(cond).unsqueeze(1).chunk(2, dim=-1)
        return x * (1 + scale) + shift


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(
        self,
        dim: int,
        mult: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DiTBlock(nn.Module):
    """Single Diffusion Transformer block with adaLN conditioning."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        ff_mult: float = 4.0,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = AdaLayerNorm(dim, cond_dim)
        self.attn = FlashMultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            use_flash=use_flash_attn,
        )
        self.norm2 = AdaLayerNorm(dim, cond_dim)
        self.ff = FeedForward(dim, mult=ff_mult, dropout=dropout)

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        rope: RotaryPositionalEmbedding | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass with residual connections.

        Args:
            x: Input features. Shape: (B, T, D).
            cond: Conditioning (timestep + speaker). Shape: (B, D_cond).
            rope: Optional rotary position embedding.
            mask: Optional attention mask. Shape: (B, T).

        Returns:
            Output features. Shape: (B, T, D).
        """
        x = x + self.attn(self.norm1(x, cond), rope=rope, mask=mask)
        x = x + self.ff(self.norm2(x, cond))
        return x


class DiTBackbone(nn.Module):
    """Diffusion Transformer backbone for mel-spectrogram generation.

    Architecture follows F5-TTS: phoneme + mel conditioning with
    masked prediction objective.
    """

    def __init__(
        self,
        mel_dim: int = 100,
        phoneme_vocab_size: int = 256,
        dim: int = 1024,
        depth: int = 22,
        num_heads: int = 16,
        ff_mult: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
        num_speakers: int = 1,
        speaker_dim: int = 256,
        use_flash_attn: bool = True,
    ) -> None:
        """Initialize DiT backbone.

        Args:
            mel_dim: Mel-spectrogram feature dimension.
            phoneme_vocab_size: Size of phoneme vocabulary.
            dim: Model hidden dimension.
            depth: Number of transformer blocks.
            num_heads: Number of attention heads.
            ff_mult: FFN expansion factor.
            dropout: Dropout probability.
            max_seq_len: Maximum sequence length for RoPE.
            num_speakers: Number of speakers for embedding table.
            speaker_dim: Speaker embedding dimension.
            use_flash_attn: Use Flash Attention 2.
        """
        super().__init__()
        self.dim = dim
        self.depth = depth

        # Input projections
        self.mel_proj = nn.Linear(mel_dim, dim)
        self.phoneme_embed = nn.Embedding(phoneme_vocab_size, dim)

        # Timestep embedding (sinusoidal)
        self.time_embed = TimestepEmbedding(dim)

        # Speaker embedding
        self.speaker_embed = nn.Embedding(num_speakers, speaker_dim)
        self.cond_dim = dim + speaker_dim  # timestep + speaker

        # Conditioning projection
        self.cond_proj = nn.Linear(self.cond_dim, self.cond_dim)

        # Rotary positional embedding
        self.rope = RotaryPositionalEmbedding(dim // num_heads, max_seq_len)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=dim,
                num_heads=num_heads,
                cond_dim=self.cond_dim,
                ff_mult=ff_mult,
                dropout=dropout,
                use_flash_attn=use_flash_attn,
            )
            for _ in range(depth)
        ])

        # Output projection
        self.norm_out = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier/Kaiming schemes."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

        # Zero-init output projection for stable training
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        phonemes: Tensor,
        speaker_ids: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass predicting velocity field.

        Args:
            x_t: Noisy mel at time t. Shape: (B, T, mel_dim).
            t: Timesteps. Shape: (B,).
            phonemes: Phoneme indices. Shape: (B, T).
            speaker_ids: Speaker indices. Shape: (B,).
            mask: Padding mask. Shape: (B, T).

        Returns:
            Predicted velocity. Shape: (B, T, mel_dim).
        """
        batch_size = x_t.size(0)
        device = x_t.device

        # Project inputs
        mel_emb = self.mel_proj(x_t)
        phoneme_emb = self.phoneme_embed(phonemes)

        # Upsample phoneme embeddings to match mel length
        mel_len = mel_emb.size(1)
        phoneme_len = phoneme_emb.size(1)
        if phoneme_len != mel_len:
            # Simple linear interpolation upsampling
            phoneme_emb = F.interpolate(
                phoneme_emb.transpose(1, 2),  # (B, dim, T_phone)
                size=mel_len,
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)  # (B, T_mel, dim)

        # Combine mel and phoneme (F5-TTS style concatenation)
        x = mel_emb + phoneme_emb

        # Build conditioning vector
        time_emb = self.time_embed(t)  # (B, dim)

        if speaker_ids is None:
            speaker_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        speaker_emb = self.speaker_embed(speaker_ids)  # (B, speaker_dim)

        cond = torch.cat([time_emb, speaker_emb], dim=-1)
        cond = self.cond_proj(cond)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, cond, rope=self.rope, mask=mask)

        # Output projection
        x = self.norm_out(x)
        v_pred = self.proj_out(x)

        return v_pred


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding with MLP projection."""

    def __init__(self, dim: int, max_period: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """Embed timesteps.

        Args:
            t: Timesteps in [0, 1]. Shape: (B,).

        Returns:
            Embeddings. Shape: (B, dim).
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=t.device, dtype=torch.float32)
            / half_dim
        )
        args = t[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))

        return self.mlp(embedding)
