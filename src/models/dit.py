"""Diffusion Transformer (DiT) backbone for F5-TTS.

Architecture summary:
  - Input: noisy mel [B, T, n_mels] + text token IDs zero-padded to T [B, T]
  - Both are projected to model dim and *summed* (following F5-TTS paper)
  - Processed by N DiTBlocks (self-attention + FFN, each conditioned on timestep via AdaLN)
  - Output: velocity field [B, T, n_mels] for the CFM objective

Compatible with F5-TTS Base pretrained checkpoint layout when:
  dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
"""

import torch
import torch.nn as nn

from src.models.encoder import TextConvEmbed
from src.models.modules import DiTBlock, RMSNorm, RotaryEmbedding, SinusoidalEmbedding


class DiT(nn.Module):
    def __init__(
        self,
        n_mels: int = 100,
        vocab_size: int = 67,
        dim: int = 1024,
        depth: int = 22,
        heads: int = 16,
        ff_mult: int = 2,
        text_dim: int = 512,
        conv_layers: int = 4,
        p_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.dim = dim

        # ── Embeddings ────────────────────────────────────────────────────────
        self.mel_proj = nn.Linear(n_mels, dim)
        self.text_encoder = TextConvEmbed(
            vocab_size=vocab_size,
            text_dim=text_dim,
            model_dim=dim,
            conv_layers=conv_layers,
            p_dropout=p_dropout,
        )
        self.t_emb = SinusoidalEmbedding(dim)
        self.drop = nn.Dropout(p_dropout)

        # ── Transformer ───────────────────────────────────────────────────────
        rope = RotaryEmbedding(dim // heads)
        self.blocks = nn.ModuleList([DiTBlock(dim, heads, ff_mult, rope) for _ in range(depth)])
        self.norm_out = RMSNorm(dim)
        self.out_proj = nn.Linear(dim, n_mels)

        # Zero-init output projection for stable training
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict velocity field.

        Args:
            x: Noisy mel frames [B, T, n_mels].
            t: Diffusion timestep [B], values in [0, 1].
            text_ids: Token IDs zero-padded to length T [B, T].
            mask: Valid frame mask [B, T], True = real frame.

        Returns:
            Predicted velocity [B, T, n_mels].
        """
        # Timestep conditioning
        t_cond = self.t_emb(t)  # [B, dim]

        # Mel + text combined embedding
        x = self.mel_proj(x)  # [B, T, dim]
        text_embed = self.text_encoder(text_ids)  # [B, T, dim]
        x = self.drop(x + text_embed)

        for block in self.blocks:
            x = block(x, t_cond, mask)

        x = self.norm_out(x)
        return self.out_proj(x)  # [B, T, n_mels]
