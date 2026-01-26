"""F5-TTS model combining DiT backbone with CFM training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor

from src.core.cfm import CFMLoss, ConditionalFlowMatcher
from src.core.dit import DiTBackbone
from src.core.ode import CFGSampler


@dataclass
class F5TTSConfig:
    """F5-TTS model configuration."""

    # Audio
    mel_dim: int = 100
    sample_rate: int = 24000

    # Text
    phoneme_vocab_size: int = 256

    # Model architecture
    dim: int = 1024
    depth: int = 22
    num_heads: int = 16
    ff_mult: float = 4.0
    dropout: float = 0.1
    max_seq_len: int = 4096

    # Multi-speaker
    num_speakers: int = 1
    speaker_dim: int = 256

    # Training
    sigma_min: float = 1e-4

    # Inference
    use_flash_attn: bool = True

    @classmethod
    def light(cls) -> F5TTSConfig:
        """Lightweight config for faster inference."""
        return cls(
            dim=512,
            depth=12,
            num_heads=8,
            ff_mult=2.0,
            dropout=0.0,
            speaker_dim=128,
        )

    @classmethod
    def high_quality(cls) -> F5TTSConfig:
        """High-quality config for maximum prosody."""
        return cls(
            dim=1024,
            depth=22,
            num_heads=16,
            ff_mult=4.0,
            dropout=0.1,
            speaker_dim=256,
        )


class F5TTS(nn.Module):
    """F5-TTS: Flow Matching TTS with Diffusion Transformer.

    Combines conditional flow matching with a DiT backbone for
    high-quality Mongolian speech synthesis.
    """

    def __init__(self, config: F5TTSConfig) -> None:
        """Initialize F5-TTS model.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config

        # Flow matching components
        self.cfm = ConditionalFlowMatcher(sigma_min=config.sigma_min)
        self.criterion = CFMLoss()

        # DiT backbone
        self.backbone = DiTBackbone(
            mel_dim=config.mel_dim,
            phoneme_vocab_size=config.phoneme_vocab_size,
            dim=config.dim,
            depth=config.depth,
            num_heads=config.num_heads,
            ff_mult=config.ff_mult,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
            num_speakers=config.num_speakers,
            speaker_dim=config.speaker_dim,
            use_flash_attn=config.use_flash_attn,
        )

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
        return self.backbone(x_t, t, phonemes, speaker_ids, mask)

    def compute_loss(
        self,
        mel: Tensor,
        phonemes: Tensor,
        speaker_ids: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute CFM training loss.

        Args:
            mel: Target mel-spectrogram. Shape: (B, T, mel_dim).
            phonemes: Phoneme indices. Shape: (B, T).
            speaker_ids: Speaker indices. Shape: (B,).
            mask: Padding mask. Shape: (B, T).

        Returns:
            Dictionary with 'loss' and optional auxiliary losses.
        """
        # Sample CFM training data
        x_t, t, u_t, _ = self.cfm(mel, mask)

        # Predict velocity
        v_pred = self(x_t, t, phonemes, speaker_ids, mask)

        # Compute loss
        loss = self.criterion(v_pred, u_t, mask)

        return {"loss": loss}

    @torch.inference_mode()
    def synthesize(
        self,
        phonemes: Tensor,
        speaker_ids: Tensor | None = None,
        cfg_scale: float = 2.0,
        num_steps: int = 32,
        method: Literal["euler", "midpoint", "rk4"] = "euler",
    ) -> Tensor:
        """Synthesize mel-spectrogram from phonemes.

        Args:
            phonemes: Phoneme indices. Shape: (B, T).
            speaker_ids: Speaker indices. Shape: (B,).
            cfg_scale: Classifier-free guidance scale.
            num_steps: ODE integration steps.
            method: ODE solver method.

        Returns:
            Synthesized mel-spectrogram. Shape: (B, T, mel_dim).
        """
        sampler = CFGSampler(
            model=self,
            cfg_scale=cfg_scale,
            method=method,
            num_steps=num_steps,
        )
        return sampler.sample(phonemes, speaker_ids)

    def save_pretrained(self, path: str | Path) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "config": self.config.__dict__,
                "state_dict": self.state_dict(),
            },
            path / "model.pt",
        )

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
    ) -> F5TTS:
        """Load model from checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path / "model.pt", map_location=device, weights_only=False)

        config = F5TTSConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])

        return model.to(device)

    @property
    def num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
