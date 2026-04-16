"""Diffusion Transformer (DiT) backbone for F5-TTS.

Architecture (matching official SWivid/F5-TTS):
  - TextEmbedding: token IDs → text_dim via ConvNeXtV2 blocks
  - InputEmbedding: concat [noised_mel, cond_mel, text_embed] → proj → conv_pos_embed
  - N × DiTBlock: self-attention + FFN, conditioned via 6-param AdaLN with gating
  - AdaLayerNormFinal + linear → output velocity field [B, T, mel_dim]

Supports classifier-free guidance via cfg_infer mode (double-batch cond+uncond).
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as ckpt_fn

from src.models.encoder import TextEmbedding
from src.models.modules import (
    AdaLayerNormFinal,
    ConvPositionEmbedding,
    DiTBlock,
    RotaryEmbedding,
    TimestepEmbedding,
)


class InputEmbedding(nn.Module):
    """Combines noised mel, conditioning mel, and text embedding.

    Concatenates [x, cond, text_embed] along feature dim and projects to model dim.
    Adds convolutional position embedding.
    """

    def __init__(self, mel_dim: int, text_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        text_embed: torch.Tensor,
        drop_audio_cond: bool = False,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Noised mel [B, N, mel_dim].
            cond: Conditioning mel [B, N, mel_dim].
            text_embed: Text embeddings [B, N, text_dim].
            drop_audio_cond: Zero out conditioning audio (for CFG).
            mask: Valid frame mask [B, N].
        """
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x, mask=mask) + x
        return x


class DiT(nn.Module):
    """Diffusion Transformer backbone for F5-TTS flow matching."""

    def __init__(
        self,
        *,
        dim: int = 1024,
        depth: int = 22,
        heads: int = 16,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.1,
        mel_dim: int = 100,
        vocab_size: int = 65,
        text_dim: int = 512,
        conv_layers: int = 4,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.gradient_checkpointing = gradient_checkpointing

        # Timestep conditioning
        self.time_embed = TimestepEmbedding(dim)

        # Text embedding (ConvNeXtV2-based)
        self.text_embed = TextEmbedding(
            vocab_size=vocab_size,
            text_dim=text_dim,
            conv_layers=conv_layers,
        )
        self.text_cond: torch.Tensor | None = None
        self.text_uncond: torch.Tensor | None = None

        # Input embedding (concat x + cond + text → dim)
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        # Rotary position embedding
        self.rotary_embed = RotaryEmbedding(dim_head)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        # Final output
        self.norm_out = AdaLayerNormFinal(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Zero-init AdaLN and output layers for stable training start."""
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def _get_input_embed(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        text: torch.Tensor,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cache: bool = True,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute input embeddings with optional text caching."""
        if self.text_uncond is None or self.text_cond is None or not cache:
            seq_len = x.shape[1]
            text_embed = self.text_embed(text, seq_len=seq_len, drop_text=drop_text)
            if cache:
                if drop_text:
                    self.text_uncond = text_embed
                else:
                    self.text_cond = text_embed

        if cache:
            text_embed = self.text_uncond if drop_text else self.text_cond
        assert text_embed is not None  # type: ignore[possibly-undefined]

        return self.input_embed(
            x, cond, text_embed, drop_audio_cond=drop_audio_cond, mask=mask
        )

    def clear_cache(self) -> None:
        """Clear cached text embeddings (call after inference)."""
        self.text_cond = None
        self.text_uncond = None

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        text: torch.Tensor,
        time: torch.Tensor,
        mask: torch.Tensor | None = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
    ) -> torch.Tensor:
        """Predict velocity field.

        Args:
            x: Noised mel frames [B, N, mel_dim].
            cond: Conditioning mel (unmasked region) [B, N, mel_dim].
            text: Token IDs [B, Nt].
            time: Diffusion timestep [B] or scalar.
            mask: Valid frame mask [B, N], True = real frame.
            drop_audio_cond: Zero out audio conditioning (for CFG).
            drop_text: Zero out text (for CFG).
            cfg_infer: If True, pack cond+uncond in batch dim for CFG inference.
            cache: If True, cache text embeddings for repeated calls.

        Returns:
            Predicted velocity [B, N, mel_dim] (or [2B, N, mel_dim] if cfg_infer).
        """
        batch = x.shape[0]
        if time.ndim == 0:
            time = time.repeat(batch)

        # Timestep conditioning
        t = self.time_embed(time)

        if cfg_infer:
            # Double-batch: conditioned + unconditioned
            x_cond = self._get_input_embed(
                x, cond, text,
                drop_audio_cond=False, drop_text=False,
                cache=cache, mask=mask,
            )
            x_uncond = self._get_input_embed(
                x, cond, text,
                drop_audio_cond=True, drop_text=True,
                cache=cache, mask=mask,
            )
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            if mask is not None:
                mask = torch.cat((mask, mask), dim=0)
        else:
            x = self._get_input_embed(
                x, cond, text,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                cache=cache, mask=mask,
            )

        # RoPE (shared across all blocks)
        rope = self.rotary_embed

        for block in self.transformer_blocks:
            if self.gradient_checkpointing and self.training:
                x = ckpt_fn(block, x, t, mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope)

        x = self.norm_out(x, t)
        return self.proj_out(x)
