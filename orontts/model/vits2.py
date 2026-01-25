"""VITS2 main model implementation."""

import math
from typing import Any

import torch
import torch.nn as nn

from orontts.model.config import VITS2Config
from orontts.model.modules import (
    DurationPredictor,
    Generator,
    MultiPeriodDiscriminator,
    PosteriorEncoder,
    ResidualCouplingBlock,
    TextEncoder,
)
from orontts.utils import rand_slice_segments


@torch.jit.script
def maximum_path(value: torch.Tensor, mask: torch.Tensor, max_neg_val: float = -1e9) -> torch.Tensor:
    """Monotonic alignment search (JIT optimized).
    
    Args:
        value: Log probabilities [B, T_x, T_y]
        mask: Mask [B, T_x, T_y]
        max_neg_val: Value for invalid transitions

    Returns:
        One-hot alignment path [B, T_x, T_y]
    """
    value = value * mask
    device = value.device
    dtype = value.dtype
    b, t_x, t_y = value.shape

    direction = torch.zeros(b, t_x, t_y, dtype=torch.long, device=device)
    v = torch.zeros(b, t_x, t_y, dtype=dtype, device=device)

    # Initialize
    for i in range(b):
        v[i, 0, 0] = value[i, 0, 0]
        for y in range(1, t_y):
            v[i, 0, y] = max_neg_val

    # DP
    for x in range(1, t_x):
        for i in range(b):
            for y in range(t_y):
                current_val = value[i, x, y]
                v_prev = v[i, x - 1, y]
                
                if y > 0:
                    v_prev_diag = v[i, x - 1, y - 1]
                    if v_prev >= v_prev_diag:
                        v[i, x, y] = v_prev + current_val
                        direction[i, x, y] = 0
                    else:
                        v[i, x, y] = v_prev_diag + current_val
                        direction[i, x, y] = 1
                else:
                    v[i, x, y] = v_prev + current_val
                    direction[i, x, y] = 0

    # Backtrack
    path = torch.zeros(b, t_x, t_y, dtype=dtype, device=device)
    for i in range(b):
        index = t_y - 1
        for x in range(t_x - 1, -1, -1):
            path[i, x, index] = 1
            if index > 0 and direction[i, x, index] == 1:
                index -= 1
                
    return path * mask


class VITS2(nn.Module):
    """VITS2: Variational Inference with adversarial learning for end-to-end TTS.

    This implementation includes:
    - Text encoder with transformer
    - Posterior encoder (VAE)
    - Normalizing flow
    - Duration predictor with MAS
    - HiFi-GAN generator
    - Multi-period discriminator

    Attributes:
        config: Model configuration.
    """

    def __init__(self, config: VITS2Config) -> None:
        """Initialize VITS2 model.

        Args:
            config: Complete model configuration.
        """
        super().__init__()
        self.config = config

        # Calculate gin_channels for speaker conditioning
        gin_channels = config.speaker_embedding_dim if config.n_speakers > 1 else 0

        # Text encoder
        self.text_encoder = TextEncoder(
            n_vocab=config.text_encoder.n_vocab,
            hidden_channels=config.text_encoder.hidden_channels,
            filter_channels=config.text_encoder.filter_channels,
            n_heads=config.text_encoder.n_heads,
            n_layers=config.text_encoder.n_layers,
            kernel_size=config.text_encoder.kernel_size,
            dropout=config.text_encoder.dropout,
        )

        # Posterior encoder
        self.posterior_encoder = PosteriorEncoder(
            in_channels=config.audio.n_mels,
            hidden_channels=config.posterior_encoder.hidden_channels,
            out_channels=config.posterior_encoder.out_channels,
            kernel_size=config.posterior_encoder.kernel_size,
            dilation_rate=config.posterior_encoder.dilation_rate,
            n_layers=config.posterior_encoder.n_layers,
            gin_channels=gin_channels,
        )

        # Normalizing flow
        self.flow = ResidualCouplingBlock(
            channels=config.flow.hidden_channels,
            hidden_channels=config.flow.hidden_channels,
            kernel_size=config.flow.kernel_size,
            dilation_rate=config.flow.dilation_rate,
            n_layers=config.flow.n_layers,
            n_flows=config.flow.n_flows,
            gin_channels=gin_channels,
        )

        # Duration predictor
        self.duration_predictor = DurationPredictor(
            in_channels=config.text_encoder.hidden_channels,
            hidden_channels=config.duration_predictor.hidden_channels,
            kernel_size=config.duration_predictor.kernel_size,
            dropout=config.duration_predictor.dropout,
        )

        # Generator (HiFi-GAN)
        self.generator = Generator(
            initial_channel=config.generator.initial_channel,
            resblock_type=config.generator.resblock_type,
            resblock_kernel_sizes=config.generator.resblock_kernel_sizes,
            resblock_dilation_sizes=config.generator.resblock_dilation_sizes,
            upsample_rates=config.generator.upsample_rates,
            upsample_initial_channel=config.generator.upsample_initial_channel,
            upsample_kernel_sizes=config.generator.upsample_kernel_sizes,
            gin_channels=gin_channels,
        )

        # Speaker embedding
        if config.n_speakers > 1:
            self.speaker_embedding = nn.Embedding(
                config.n_speakers, config.speaker_embedding_dim
            )
        else:
            self.speaker_embedding = None

        # Projection for latent matching
        self.proj_m = nn.Conv1d(
            config.text_encoder.hidden_channels,
            config.posterior_encoder.out_channels,
            1,
        )
        self.proj_s = nn.Conv1d(
            config.text_encoder.hidden_channels,
            config.posterior_encoder.out_channels,
            1,
        )

    def forward(
        self,
        phoneme_ids: torch.Tensor,
        phoneme_lengths: torch.Tensor,
        mel: torch.Tensor,
        mel_lengths: torch.Tensor,
        speaker_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for training.

        Args:
            phoneme_ids: Phoneme ID tensor [B, T_text]
            phoneme_lengths: Text lengths [B]
            mel: Mel spectrogram [B, M, T_mel]
            mel_lengths: Mel lengths [B]
            speaker_ids: Speaker IDs [B] (optional)

        Returns:
            Dictionary with all outputs needed for loss computation.
        """
        # Get speaker embedding
        g = None
        if self.speaker_embedding is not None and speaker_ids is not None:
            g = self.speaker_embedding(speaker_ids).unsqueeze(-1)  # [B, G, 1]

        # Text encoding
        x, x_m, x_logs, x_mask = self.text_encoder(phoneme_ids, phoneme_lengths)

        # Posterior encoding
        z, z_m, z_logs, z_mask = self.posterior_encoder(mel, mel_lengths, g)

        # Project text encoder output
        x_m_proj = self.proj_m(x) * x_mask
        x_logs_proj = self.proj_s(x) * x_mask

        # Duration prediction
        log_duration_pred = self.duration_predictor(x.detach(), x_mask)

        # Compute alignment with MAS (monotonic alignment search)
        with torch.no_grad():
            # Simplified alignment - in practice use proper MAS
            attn_mask = x_mask.unsqueeze(2) * z_mask.unsqueeze(-1)
            s_p_sq_r = torch.exp(-2 * x_logs_proj)
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs_proj, dim=1, keepdim=True)
            neg_cent2 = torch.matmul(
                -0.5 * (z**2).transpose(1, 2), s_p_sq_r
            )
            neg_cent3 = torch.matmul(z.transpose(1, 2), x_m_proj * s_p_sq_r)
            neg_cent4 = torch.sum(-0.5 * (x_m_proj**2) * s_p_sq_r, dim=1, keepdim=True)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            # Monotonic Alignment Search
            attn_mask = z_mask.unsqueeze(-1) * x_mask.unsqueeze(2)
            attn_mask = attn_mask.squeeze(1)  # [B, T_y, T_x]
            
            # Run MAS
            attn = maximum_path(neg_cent, attn_mask).unsqueeze(1).detach() 

        # Duration targets
        duration_target = attn.sum(dim=2)

        # Expand text encoder output
            # resulting [B, 1, T_x, T_y].
            
            # We need to match neg_cent [B, T_y, T_x].
            # So we need [B, 1, T_y, T_x].
            # z_mask: [B, 1, T_y] -> unsqueeze(-1) -> [B, 1, T_y, 1]
            # x_mask: [B, 1, T_x] -> unsqueeze(2) -> [B, 1, 1, T_x]
            # Product: [B, 1, T_y, T_x].
            
            attn_mask = z_mask.unsqueeze(-1) * x_mask.unsqueeze(2)
            attn_mask = attn_mask.squeeze(1) # [B, T_y, T_x]
            
            # Run MAS
            # Note: maximum_path implementation usually expects [B, T_text, T_audio] or [B, T_audio, T_text]?
            # The one I added loops over 'x' then 'y'.
            # If we pass [B, T_y, T_x], it aligns Audio (x) to Text (y).
            # This allows multiple audio frames per text phoneme. This makes sense.
            # Text is "slower" (fewer items), Audio is "faster".
            # Alignment path is monotonic.
            
            attn = maximum_path(neg_cent, attn_mask).unsqueeze(1).detach() 
            # attn: [B, 1, T_y, T_x]
            
            # If neg_cent is [B, T_y, T_x], attn is [B, 1, T_y, T_x].
            
            # Leter: 
            # duration_target = attn.sum(dim=2)
            # dim 2 is T_x (Text).
            # Summing over T_x (Text) gives duration for each Audio frame??
            # No. Duration of a Phoneme = Number of Audio frames assigned to it.
            # So we sum over Audio frames (T_y).
            
            # If attn is [B, 1, T_y, T_x].
            # Sum over dim=2 (T_y)? No, T_y is dim 2 (indices 0, 1, 2, 3).
            # Indices: B=0, 1=1, Ty=2, Tx=3.
            # Sum over dim=2 (Ty) gives [B, 1, Tx]. This is duration per phoneme. Correct.
            
            # So attn must be [B, 1, Ty, Tx] ?
            
            # Let's check original code:
            # attn_squeezed = attn.squeeze(1) # [B, T_y, T_x]
            # x_m_expanded = torch.matmul(attn_squeezed, x_m_proj.transpose(1, 2)).transpose(1, 2)
            # x_m_proj.transpose(1, 2) -> [B, T_x, hidden].
            # [B, T_y, T_x] @ [B, T_x, hidden] -> [B, T_y, hidden]. 
            # Transpose(1,2) -> [B, hidden, T_y].
            # This matches z (Audio) shape. Correct.
            
            # So attn MUST be [B, 1, T_y, T_x]. (Audio, Text).

            # So my attn_mask calculation:
            # attn_mask = z_mask.unsqueeze(-1) * x_mask.unsqueeze(2)
            # z_mask [B, 1, T_y] -> [B, 1, T_y, 1]
            # x_mask [B, 1, T_x] -> [B, 1, 1, T_x]
            # Result [B, 1, T_y, T_x]. Correct.


        # Duration targets
        duration_target = attn.sum(dim=2)

        # Expand text encoder output
        # attn is [B, 1, mel_len, text_len], squeeze to [B, mel_len, text_len]
        # x_m_proj is [B, hidden, text_len], transpose to [B, text_len, hidden]
        # matmul: [B, mel_len, text_len] @ [B, text_len, hidden] -> [B, mel_len, hidden]
        # then transpose back to [B, hidden, mel_len]
        attn_squeezed = attn.squeeze(1)  # [B, mel_len, text_len]
        x_m_expanded = torch.matmul(attn_squeezed, x_m_proj.transpose(1, 2)).transpose(1, 2)
        x_logs_expanded = torch.matmul(attn_squeezed, x_logs_proj.transpose(1, 2)).transpose(1, 2)

        # Flow
        z_p = self.flow(z, z_mask, g)

        # Random slice for efficient training
        z_slice, ids_slice = rand_slice_segments(
            z, mel_lengths, self.config.training.segment_size // self.config.audio.hop_length
        )

        # Generate audio
        audio = self.generator(z_slice, g)

        return {
            "audio": audio,
            "z": z,
            "z_p": z_p,
            "z_m": z_m,
            "z_logs": z_logs,
            "x_m": x_m_expanded,
            "x_logs": x_logs_expanded,
            "z_mask": z_mask,
            "x_mask": x_mask,
            "log_duration_pred": log_duration_pred,
            "duration_target": duration_target,
            "attn": attn,
            "ids_slice": ids_slice,
        }

    @torch.inference_mode()
    def infer(
        self,
        phoneme_ids: torch.Tensor,
        phoneme_lengths: torch.Tensor,
        speaker_ids: torch.Tensor | None = None,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        noise_scale_w: float = 0.8,
    ) -> torch.Tensor:
        """Inference for synthesis.

        Args:
            phoneme_ids: Phoneme ID tensor [B, T]
            phoneme_lengths: Sequence lengths [B]
            speaker_ids: Speaker IDs [B] (optional)
            noise_scale: Noise scale for sampling
            length_scale: Duration scaling factor
            noise_scale_w: Noise scale for duration

        Returns:
            Generated audio waveform [B, 1, T]
        """
        # Get speaker embedding
        g = None
        if self.speaker_embedding is not None and speaker_ids is not None:
            g = self.speaker_embedding(speaker_ids).unsqueeze(-1)

        # Text encoding
        x, x_m, x_logs, x_mask = self.text_encoder(phoneme_ids, phoneme_lengths)

        # Duration prediction
        log_duration = self.duration_predictor(x, x_mask)
        duration = torch.exp(log_duration) * x_mask * length_scale
        duration = torch.ceil(duration).long().squeeze(1)

        # Generate alignment from duration
        y_lengths = duration.sum(dim=1)
        y_mask = self._sequence_mask(y_lengths).unsqueeze(1).to(x.dtype)

        # Expand with duration
        attn = self._generate_path(duration, x_mask.squeeze(1))

        # Project and expand
        x_m_proj = self.proj_m(x) * x_mask
        x_logs_proj = self.proj_s(x) * x_mask

        x_m_expanded = torch.matmul(attn.transpose(1, 2), x_m_proj.transpose(1, 2)).transpose(1, 2)
        x_logs_expanded = torch.matmul(attn.transpose(1, 2), x_logs_proj.transpose(1, 2)).transpose(1, 2)

        # Sample from prior
        z_p = x_m_expanded + torch.randn_like(x_m_expanded) * torch.exp(x_logs_expanded) * noise_scale

        # Inverse flow
        z = self.flow(z_p, y_mask, g, reverse=True)

        # Generate audio
        audio = self.generator(z * y_mask, g)

        return audio

    @staticmethod
    def _sequence_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
        max_len = max_len or lengths.max().item()
        ids = torch.arange(0, max_len, device=lengths.device)
        return ids < lengths.unsqueeze(1)

    @staticmethod
    def _generate_path(
        duration: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Generate alignment path from duration."""
        device = duration.device
        batch_size, t_x = duration.shape
        t_y = duration.sum(dim=1).max().item()

        path = torch.zeros(batch_size, t_x, int(t_y), device=device)

        for b in range(batch_size):
            col = 0
            for i in range(t_x):
                dur = duration[b, i].item()
                if dur > 0 and col < t_y:
                    path[b, i, col : col + int(dur)] = 1.0
                    col += int(dur)

        return path

    def remove_weight_norm(self) -> None:
        """Remove weight normalization for inference optimization."""
        self.generator.remove_weight_norm()


class VITS2Discriminator(nn.Module):
    """Discriminator wrapper for VITS2 training."""

    def __init__(self, config: VITS2Config) -> None:
        super().__init__()
        self.discriminator = MultiPeriodDiscriminator(
            periods=config.discriminator.periods,
            use_spectral_norm=config.discriminator.use_spectral_norm,
        )

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[list[torch.Tensor]],
        list[list[torch.Tensor]],
    ]:
        return self.discriminator(y, y_hat)
