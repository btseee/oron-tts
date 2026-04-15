"""Text character encoder for F5-TTS: embedding lookup + conv layers + projection."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextConvEmbed(nn.Module):
    """Converts token IDs to a fixed-dim embedding via lookup + depthwise conv stack.

    Architecture follows F5-TTS: char embed → N×conv1d (with residual) → linear proj.
    The output is in model `dim` space so it can be directly summed with the mel embed.
    """

    def __init__(
        self,
        vocab_size: int,
        text_dim: int = 512,
        model_dim: int = 1024,
        conv_layers: int = 4,
        conv_kernel: int = 5,
        p_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, text_dim, padding_idx=0)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(conv_layers):
            self.convs.append(
                nn.Conv1d(
                    text_dim,
                    text_dim,
                    conv_kernel,
                    padding=conv_kernel // 2,
                    groups=text_dim,  # depthwise
                )
            )
            self.norms.append(nn.LayerNorm(text_dim))

        self.proj = nn.Linear(text_dim, model_dim)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [B, L] long tensor (zero-padded to mel length T outside).
        Returns:
            text_embed: [B, L, model_dim]
        """
        x = self.embed(token_ids)  # [B, L, text_dim]

        # Conv stack expects [B, C, L]
        x = x.transpose(1, 2)
        for conv, norm in zip(self.convs, self.norms, strict=False):
            residual = x
            x = conv(x)
            x = x.transpose(1, 2)
            x = norm(x)
            x = F.gelu(x)
            x = self.drop(x)
            x = x.transpose(1, 2)
            x = x + residual
        x = x.transpose(1, 2)  # [B, L, text_dim]

        return self.proj(x)  # [B, L, model_dim]
