"""Text embedding for F5-TTS: token lookup + ConvNeXtV2 blocks.

Matches the official F5-TTS TextEmbedding architecture. Token IDs are
offset by +1 internally so that 0 serves as the filler/padding token.
Output is in text_dim space; the InputEmbedding in dit.py handles projection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules import ConvNeXtV2Block, precompute_freqs_cis


class TextEmbedding(nn.Module):
    """Text token embedding with optional ConvNeXtV2 processing.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        text_dim: Embedding dimension for text.
        conv_layers: Number of ConvNeXtV2 blocks (0 = embed only).
        conv_mult: Intermediate dim multiplier for ConvNeXtV2.
    """

    def __init__(
        self,
        vocab_size: int,
        text_dim: int,
        conv_layers: int = 0,
        conv_mult: int = 2,
    ) -> None:
        super().__init__()
        # +1 so 0 is the filler token (original IDs are offset by +1 in forward)
        self.text_embed = nn.Embedding(vocab_size + 1, text_dim)

        if conv_layers > 0:
            self.extra_modeling = True
            max_pos = 8192
            self.register_buffer(
                "freqs_cis",
                precompute_freqs_cis(text_dim, max_pos),
                persistent=False,
            )
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(
        self,
        text: torch.Tensor,
        seq_len: int | torch.Tensor,
        drop_text: bool = False,
    ) -> torch.Tensor:
        """Embed and process text tokens.

        Args:
            text: Token IDs [B, Nt].
            seq_len: Target sequence length (scalar int or [B] tensor).
            drop_text: If True, zero out text embedding (for CFG training).

        Returns:
            Text embeddings [B, max_seq_len, text_dim].
        """
        text = text + 1  # offset: 0 becomes filler token

        max_seq_len = int(seq_len.max().item()) if torch.is_tensor(seq_len) else int(seq_len)

        # Curtail or pad to match mel length
        text = text[:, :max_seq_len]
        text = F.pad(text, (0, max_seq_len - text.shape[1]), value=0)

        # Identify padding/filler positions
        text_mask = text == 0  # True where padding

        if drop_text:
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # [B, N, text_dim]

        if self.extra_modeling:
            # Sinusoidal position encoding
            freqs = self.freqs_cis[:max_seq_len, :].to(text.device)
            text = text + freqs

            # ConvNeXtV2 blocks with padding masking
            mask_expanded = text_mask.unsqueeze(-1).expand_as(text)
            text = text.masked_fill(mask_expanded, 0.0)
            for block in self.text_blocks:
                text = block(text)
                text = text.masked_fill(mask_expanded, 0.0)

        return text
