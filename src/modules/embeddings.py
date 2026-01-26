"""Embedding modules for TTS."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding from Transformer.

    Fixed (non-learnable) position encoding using sin/cos functions.
    """

    def __init__(
        self,
        dim: int,
        max_len: int = 4096,
        dropout: float = 0.0,
    ) -> None:
        """Initialize positional embedding.

        Args:
            dim: Embedding dimension.
            max_len: Maximum sequence length.
            dropout: Dropout probability.
        """
        super().__init__()
        self.dim = dim
        self.dropout = nn.Dropout(p=dropout)

        # Precompute embeddings
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional embedding to input.

        Args:
            x: Input tensor. Shape: (B, T, D).

        Returns:
            Input with positional embedding added.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embedding."""

    def __init__(
        self,
        dim: int,
        max_len: int = 4096,
        dropout: float = 0.0,
    ) -> None:
        """Initialize learned positional embedding.

        Args:
            dim: Embedding dimension.
            max_len: Maximum sequence length.
            dropout: Dropout probability.
        """
        super().__init__()
        self.embed = nn.Embedding(max_len, dim)
        self.dropout = nn.Dropout(p=dropout)

        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        """Add positional embedding.

        Args:
            x: Input tensor. Shape: (B, T, D).
            offset: Position offset for incremental decoding.

        Returns:
            Input with positional embedding added.
        """
        seq_len = x.size(1)
        positions = torch.arange(offset, offset + seq_len, device=x.device)
        x = x + self.embed(positions)
        return self.dropout(x)


class SpeakerEmbedding(nn.Module):
    """Speaker embedding with optional conditioning projection.

    Supports both lookup table and external speaker embeddings.
    """

    def __init__(
        self,
        num_speakers: int,
        dim: int,
        external_dim: int | None = None,
    ) -> None:
        """Initialize speaker embedding.

        Args:
            num_speakers: Number of speakers in lookup table.
            dim: Output embedding dimension.
            external_dim: Dimension of external embeddings (e.g., x-vector).
        """
        super().__init__()
        self.num_speakers = num_speakers
        self.dim = dim
        self.external_dim = external_dim

        # Lookup table for training speakers
        self.embed = nn.Embedding(num_speakers, dim)
        nn.init.normal_(self.embed.weight, std=0.02)

        # Projection for external embeddings (zero-shot)
        if external_dim is not None:
            self.external_proj = nn.Linear(external_dim, dim)
        else:
            self.external_proj = None

    def forward(
        self,
        speaker_ids: Tensor | None = None,
        external_emb: Tensor | None = None,
    ) -> Tensor:
        """Get speaker embedding.

        Args:
            speaker_ids: Speaker indices. Shape: (B,).
            external_emb: External speaker embedding. Shape: (B, external_dim).

        Returns:
            Speaker embedding. Shape: (B, dim).
        """
        if external_emb is not None:
            if self.external_proj is None:
                raise ValueError("External embeddings provided but no projection layer")
            return self.external_proj(external_emb)

        if speaker_ids is None:
            raise ValueError("Must provide speaker_ids or external_emb")

        return self.embed(speaker_ids)


class ConvPositionalEmbedding(nn.Module):
    """Convolutional positional embedding from wav2vec 2.0.

    Uses grouped convolution to encode relative positions,
    which generalizes better to longer sequences.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 128,
        groups: int = 16,
    ) -> None:
        """Initialize conv positional embedding.

        Args:
            dim: Embedding dimension.
            kernel_size: Convolution kernel size.
            groups: Number of convolution groups.
        """
        super().__init__()
        self.conv = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )

        # Weight normalization for stable training
        self.conv = nn.utils.parametrizations.weight_norm(self.conv, name="weight", dim=2)

        nn.init.normal_(self.conv.weight, mean=0, std=2 / (kernel_size * dim) ** 0.5)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional information via convolution.

        Args:
            x: Input tensor. Shape: (B, T, D).

        Returns:
            Input with positional information added.
        """
        # (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        # Remove extra padding
        return x[:, :-1]


class TextEmbedding(nn.Module):
    """Text/phoneme embedding with optional language embedding."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_languages: int = 1,
        padding_idx: int = 0,
        dropout: float = 0.0,
    ) -> None:
        """Initialize text embedding.

        Args:
            vocab_size: Vocabulary size.
            dim: Embedding dimension.
            num_languages: Number of languages for multilingual support.
            padding_idx: Padding token index.
            dropout: Dropout probability.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(p=dropout)

        if num_languages > 1:
            self.lang_embed = nn.Embedding(num_languages, dim)
        else:
            self.lang_embed = None

        nn.init.normal_(self.embed.weight, std=0.02)
        if padding_idx is not None:
            nn.init.zeros_(self.embed.weight[padding_idx])

    def forward(
        self,
        tokens: Tensor,
        language_ids: Tensor | None = None,
    ) -> Tensor:
        """Embed tokens with optional language embedding.

        Args:
            tokens: Token indices. Shape: (B, T).
            language_ids: Language indices. Shape: (B,).

        Returns:
            Token embeddings. Shape: (B, T, D).
        """
        x = self.embed(tokens)

        if language_ids is not None and self.lang_embed is not None:
            lang = self.lang_embed(language_ids)  # (B, D)
            x = x + lang.unsqueeze(1)

        return self.dropout(x)
