"""Duration prediction module for non-autoregressive TTS."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DurationPredictor(nn.Module):
    """Duration predictor for phoneme-to-mel alignment.

    Predicts duration (number of mel frames) for each phoneme,
    used for length regulation in non-autoregressive synthesis.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        kernel_size: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initialize duration predictor.

        Args:
            dim: Input dimension.
            hidden_dim: Hidden layer dimension.
            kernel_size: Convolution kernel size.
            num_layers: Number of conv layers.
            dropout: Dropout probability.
        """
        super().__init__()

        layers = []
        in_dim = dim

        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim
            layers.extend([
                nn.Conv1d(
                    in_dim, out_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim

        self.layers = nn.Sequential(*layers)
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Predict durations for each input token.

        Args:
            x: Input features. Shape: (B, T, D).
            mask: Padding mask. Shape: (B, T).

        Returns:
            Predicted durations (log scale). Shape: (B, T).
        """
        # (B, T, D) -> (B, D, T) for conv
        x = x.transpose(1, 2)

        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                x = x.transpose(1, 2)
                x = layer(x)
                x = x.transpose(1, 2)
            else:
                x = layer(x)

        # (B, D, T) -> (B, T, D) -> (B, T)
        x = x.transpose(1, 2)
        duration = self.proj(x).squeeze(-1)

        # Apply mask
        if mask is not None:
            duration = duration.masked_fill(~mask.bool(), 0.0)

        return duration

    def inference(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        speed: float = 1.0,
    ) -> Tensor:
        """Get integer durations for inference.

        Args:
            x: Input features. Shape: (B, T, D).
            mask: Padding mask. Shape: (B, T).
            speed: Speed factor (>1 = faster, <1 = slower).

        Returns:
            Integer durations. Shape: (B, T).
        """
        log_duration = self.forward(x, mask)
        duration = torch.exp(log_duration) / speed
        duration = torch.round(duration).long()
        duration = duration.clamp(min=1)

        if mask is not None:
            duration = duration * mask.long()

        return duration


class LengthRegulator(nn.Module):
    """Expand phoneme features according to predicted durations.

    Used in FastSpeech-style models to regulate output length.
    """

    def __init__(self, expand_max_len: int = 10000) -> None:
        """Initialize length regulator.

        Args:
            expand_max_len: Maximum total expanded length.
        """
        super().__init__()
        self.expand_max_len = expand_max_len

    def forward(
        self,
        x: Tensor,
        durations: Tensor,
        max_len: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Expand input according to durations.

        Args:
            x: Input features. Shape: (B, T_in, D).
            durations: Duration for each position. Shape: (B, T_in).
            max_len: Optional maximum output length.

        Returns:
            Expanded features and lengths: (output, lengths).
            Output shape: (B, T_out, D).
        """
        batch_size = x.size(0)
        device = x.device

        # Compute output lengths
        out_lengths = durations.sum(dim=1)  # (B,)
        max_out_len = out_lengths.max().item()

        if max_len is not None:
            max_out_len = min(max_out_len, max_len)

        max_out_len = min(max_out_len, self.expand_max_len)

        # Expand each sample
        outputs = []
        for i in range(batch_size):
            expanded = self._expand_single(x[i], durations[i], max_out_len)
            outputs.append(expanded)

        return torch.stack(outputs), out_lengths

    def _expand_single(
        self,
        x: Tensor,
        durations: Tensor,
        max_len: int,
    ) -> Tensor:
        """Expand a single sequence.

        Args:
            x: Input features. Shape: (T_in, D).
            durations: Durations. Shape: (T_in,).
            max_len: Maximum output length.

        Returns:
            Expanded sequence. Shape: (max_len, D).
        """
        device = x.device
        dim = x.size(-1)

        # Create expansion indices
        expanded = []
        for i, dur in enumerate(durations):
            dur = dur.item()
            if dur > 0:
                expanded.extend([x[i]] * dur)

        if len(expanded) == 0:
            return torch.zeros(max_len, dim, device=device)

        expanded = torch.stack(expanded[:max_len])

        # Pad if needed
        if expanded.size(0) < max_len:
            padding = torch.zeros(max_len - expanded.size(0), dim, device=device)
            expanded = torch.cat([expanded, padding])

        return expanded


class MonotonicAligner(nn.Module):
    """Monotonic alignment search for duration extraction.

    Computes hard alignment between text and mel using
    dynamic programming to find the most probable path.
    """

    @staticmethod
    @torch.jit.script
    def _maximum_path(neg_log_probs: Tensor) -> Tensor:
        """Find most probable monotonic alignment.

        Args:
            neg_log_probs: Negative log probabilities. Shape: (T_mel, T_text).

        Returns:
            Binary alignment matrix. Shape: (T_mel, T_text).
        """
        T_mel, T_text = neg_log_probs.shape
        device = neg_log_probs.device

        # DP cost matrix
        cost = torch.full_like(neg_log_probs, float("inf"))
        cost[0, 0] = neg_log_probs[0, 0]

        for i in range(1, T_mel):
            cost[i, 0] = cost[i - 1, 0] + neg_log_probs[i, 0]

        for j in range(1, T_text):
            for i in range(j, T_mel):
                cost[i, j] = neg_log_probs[i, j] + min(
                    cost[i - 1, j - 1],  # Diagonal
                    cost[i - 1, j],  # Vertical
                )

        # Backtrack
        path = torch.zeros_like(neg_log_probs)
        i, j = T_mel - 1, T_text - 1
        path[i, j] = 1

        while i > 0 or j > 0:
            if j == 0:
                i -= 1
            elif i == 0:
                j -= 1
            elif cost[i - 1, j - 1] <= cost[i - 1, j]:
                i -= 1
                j -= 1
            else:
                i -= 1
            path[i, j] = 1

        return path

    def forward(
        self,
        text_emb: Tensor,
        mel_emb: Tensor,
        text_mask: Tensor | None = None,
        mel_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute alignment between text and mel.

        Args:
            text_emb: Text embeddings. Shape: (B, T_text, D).
            mel_emb: Mel embeddings. Shape: (B, T_mel, D).
            text_mask: Text padding mask. Shape: (B, T_text).
            mel_mask: Mel padding mask. Shape: (B, T_mel).

        Returns:
            Alignment matrix. Shape: (B, T_mel, T_text).
        """
        batch_size = text_emb.size(0)

        # Compute similarity (negative distance as log prob)
        # (B, T_mel, D) @ (B, D, T_text) -> (B, T_mel, T_text)
        log_probs = torch.bmm(mel_emb, text_emb.transpose(1, 2))

        # Apply masks
        if text_mask is not None:
            log_probs = log_probs.masked_fill(
                ~text_mask.unsqueeze(1), float("-inf")
            )
        if mel_mask is not None:
            log_probs = log_probs.masked_fill(
                ~mel_mask.unsqueeze(2), float("-inf")
            )

        # Find optimal alignment for each sample
        alignments = []
        for i in range(batch_size):
            neg_log_prob = -log_probs[i]
            path = self._maximum_path(neg_log_prob)
            alignments.append(path)

        return torch.stack(alignments)

    def extract_durations(self, alignment: Tensor) -> Tensor:
        """Extract durations from alignment matrix.

        Args:
            alignment: Binary alignment. Shape: (B, T_mel, T_text).

        Returns:
            Durations. Shape: (B, T_text).
        """
        return alignment.sum(dim=1)
