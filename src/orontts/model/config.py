"""VITS2 model configuration using Pydantic."""

import json
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, Field

from orontts.constants import (
    HOP_LENGTH,
    MEL_FMAX,
    MEL_FMIN,
    N_FFT,
    N_MELS,
    PHONEME_SYMBOLS,
    SAMPLE_RATE,
    WIN_LENGTH,
)
from orontts.exceptions import ConfigurationError


class AudioConfig(BaseModel):
    """Audio processing configuration."""

    sample_rate: int = SAMPLE_RATE
    n_fft: int = N_FFT
    n_mels: int = N_MELS
    hop_length: int = HOP_LENGTH
    win_length: int = WIN_LENGTH
    mel_fmin: float = MEL_FMIN
    mel_fmax: float = MEL_FMAX


class TextEncoderConfig(BaseModel):
    """Text encoder configuration."""

    n_vocab: int = len(PHONEME_SYMBOLS)
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    dropout: float = 0.1


class PosteriorEncoderConfig(BaseModel):
    """Posterior encoder configuration."""

    in_channels: int = N_MELS
    hidden_channels: int = 192
    out_channels: int = 192
    kernel_size: int = 5
    dilation_rate: int = 1
    n_layers: int = 16


class GeneratorConfig(BaseModel):
    """HiFi-GAN generator configuration."""

    initial_channel: int = 192
    resblock_type: Literal["1", "2"] = "1"
    resblock_kernel_sizes: list[int] = Field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list[list[int]] = Field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    upsample_rates: list[int] = Field(default_factory=lambda: [8, 8, 2, 2])
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: list[int] = Field(default_factory=lambda: [16, 16, 4, 4])


class DiscriminatorConfig(BaseModel):
    """Discriminator configuration."""

    periods: list[int] = Field(default_factory=lambda: [2, 3, 5, 7, 11])
    use_spectral_norm: bool = False


class FlowConfig(BaseModel):
    """Normalizing flow configuration."""

    hidden_channels: int = 192
    kernel_size: int = 5
    dilation_rate: int = 1
    n_layers: int = 4
    n_flows: int = 4


class DurationPredictorConfig(BaseModel):
    """Duration predictor configuration."""

    hidden_channels: int = 192
    kernel_size: int = 3
    dropout: float = 0.5


class TrainingConfig(BaseModel):
    """Training hyperparameters."""

    learning_rate: float = 2e-4
    betas: tuple[float, float] = (0.8, 0.99)
    eps: float = 1e-9
    lr_decay: float = 0.999875
    segment_size: int = 8192
    batch_size: int = 16
    epochs: int = 1000
    seed: int = 42
    fp16: bool = True
    grad_clip: float = 1.0

    # Loss weights
    kl_weight: float = 1.0
    mel_weight: float = 45.0
    duration_weight: float = 1.0
    adversarial_weight: float = 1.0
    feature_matching_weight: float = 2.0


class VITS2Config(BaseModel):
    """Complete VITS2 model configuration."""

    model_name: str = "vits2_mongolian"
    n_speakers: int = 1
    speaker_embedding_dim: int = 256

    audio: AudioConfig = Field(default_factory=AudioConfig)
    text_encoder: TextEncoderConfig = Field(default_factory=TextEncoderConfig)
    posterior_encoder: PosteriorEncoderConfig = Field(
        default_factory=PosteriorEncoderConfig
    )
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    discriminator: DiscriminatorConfig = Field(default_factory=DiscriminatorConfig)
    flow: FlowConfig = Field(default_factory=FlowConfig)
    duration_predictor: DurationPredictorConfig = Field(
        default_factory=DurationPredictorConfig
    )
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load configuration from JSON file."""
        if not path.exists():
            raise ConfigurationError(f"Config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        return cls.model_validate(data)

    @classmethod
    def from_preset(cls, preset: Literal["light", "hq"]) -> Self:
        """Create configuration from preset.

        Args:
            preset: Either "light" for fast/small or "hq" for quality.

        Returns:
            VITS2Config instance.
        """
        if preset == "light":
            return cls(
                model_name="vits2_mongolian_light",
                text_encoder=TextEncoderConfig(
                    hidden_channels=128,
                    filter_channels=512,
                    n_heads=2,
                    n_layers=4,
                ),
                posterior_encoder=PosteriorEncoderConfig(
                    hidden_channels=128,
                    out_channels=128,
                    n_layers=12,
                ),
                generator=GeneratorConfig(
                    initial_channel=128,
                    upsample_initial_channel=256,
                ),
                flow=FlowConfig(
                    hidden_channels=128,
                    n_layers=3,
                    n_flows=3,
                ),
                duration_predictor=DurationPredictorConfig(
                    hidden_channels=128,
                ),
                training=TrainingConfig(
                    batch_size=32,
                    segment_size=4096,
                ),
            )
        elif preset == "hq":
            return cls(
                model_name="vits2_mongolian_hq",
                text_encoder=TextEncoderConfig(
                    hidden_channels=256,
                    filter_channels=1024,
                    n_heads=4,
                    n_layers=8,
                ),
                posterior_encoder=PosteriorEncoderConfig(
                    hidden_channels=256,
                    out_channels=256,
                    n_layers=20,
                ),
                generator=GeneratorConfig(
                    initial_channel=256,
                    upsample_initial_channel=512,
                ),
                flow=FlowConfig(
                    hidden_channels=256,
                    n_layers=6,
                    n_flows=6,
                ),
                duration_predictor=DurationPredictorConfig(
                    hidden_channels=256,
                ),
                training=TrainingConfig(
                    batch_size=8,
                    segment_size=16384,
                ),
            )
        else:
            raise ConfigurationError(f"Unknown preset: {preset}")


def load_config(path: str | Path) -> VITS2Config:
    """Convenience function to load configuration.

    Args:
        path: Path to config JSON file.

    Returns:
        VITS2Config instance.
    """
    return VITS2Config.load(Path(path))
