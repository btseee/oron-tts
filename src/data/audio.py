"""Audio processing pipeline with DeepFilterNet denoising.

Handles audio loading, resampling, mel-spectrogram extraction,
and enhancement of non-professional recordings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore[import-untyped]
import torch
import torchaudio  # type: ignore[import-untyped]
from torch import Tensor


@dataclass
class AudioConfig:
    """Audio processing configuration."""

    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 100
    fmin: float = 0.0
    fmax: float | None = None  # None = sample_rate / 2
    power: float = 1.0  # 1 for energy, 2 for power
    normalized: bool = False
    center: bool = True
    pad_mode: str = "reflect"

    # DeepFilterNet settings
    denoise: bool = True
    denoise_atten_lim: float = 100.0  # Max attenuation in dB

    # Normalization
    normalize_audio: bool = True
    normalize_mel: bool = True
    mel_mean: float = -4.0  # Target mel mean (log scale)
    mel_std: float = 4.0  # Target mel std


class AudioProcessor:
    """Audio loading, processing, and mel-spectrogram extraction.

    Features:
    - DeepFilterNet noise reduction for non-studio recordings
    - Configurable mel-spectrogram parameters
    - Proper resampling with anti-aliasing
    - Normalization for stable training
    """

    def __init__(self, config: AudioConfig | None = None) -> None:
        """Initialize audio processor.

        Args:
            config: Audio processing configuration.
        """
        self.config = config or AudioConfig()
        self._mel_transform: torchaudio.transforms.MelSpectrogram | None = None
        self._denoiser: DeepFilterNetWrapper | None = None

    @property
    def mel_transform(self) -> torchaudio.transforms.MelSpectrogram:
        """Lazy-initialized mel-spectrogram transform."""
        if self._mel_transform is None:
            self._mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.config.sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.win_length,
                n_mels=self.config.n_mels,
                f_min=self.config.fmin,
                f_max=self.config.fmax,
                power=self.config.power,
                normalized=self.config.normalized,
                center=self.config.center,
                pad_mode=self.config.pad_mode,
            )
        return self._mel_transform

    @property
    def denoiser(self) -> DeepFilterNetWrapper | None:
        """Lazy-initialized DeepFilterNet denoiser."""
        if self.config.denoise and self._denoiser is None:
            self._denoiser = DeepFilterNetWrapper(
                atten_lim=self.config.denoise_atten_lim,
            )
        return self._denoiser

    def load_audio(
        self,
        path: str | Path,
        start_sec: float | None = None,
        duration_sec: float | None = None,
    ) -> Tensor:
        """Load and preprocess audio file.

        Args:
            path: Path to audio file.
            start_sec: Optional start time in seconds.
            duration_sec: Optional duration in seconds.

        Returns:
            Audio waveform. Shape: (1, samples).
        """
        path = Path(path)

        # Compute frame offsets
        info = torchaudio.info(str(path))
        original_sr = info.sample_rate

        frame_offset = 0
        num_frames = -1

        if start_sec is not None:
            frame_offset = int(start_sec * original_sr)
        if duration_sec is not None:
            num_frames = int(duration_sec * original_sr)

        # Load audio
        waveform, sr = torchaudio.load(
            str(path),
            frame_offset=frame_offset,
            num_frames=num_frames,
        )

        # Convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.config.sample_rate,
            )
            waveform = resampler(waveform)

        # Normalize amplitude
        if self.config.normalize_audio:
            waveform = self._normalize_audio(waveform)

        return waveform

    def denoise_audio(self, waveform: Tensor) -> Tensor:
        """Apply DeepFilterNet noise reduction.

        Args:
            waveform: Input audio. Shape: (1, samples) or (samples,).

        Returns:
            Denoised audio.
        """
        if self.denoiser is None:
            return waveform

        return self.denoiser(waveform, self.config.sample_rate)

    def extract_mel(self, waveform: Tensor) -> Tensor:
        """Extract log mel-spectrogram.

        Args:
            waveform: Audio waveform. Shape: (1, samples).

        Returns:
            Log mel-spectrogram. Shape: (T, n_mels).
        """
        # Ensure 3D input: (batch, channels, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        # Extract mel spectrogram
        mel = self.mel_transform(waveform)  # (B, n_mels, T)

        # Convert to log scale
        mel = torch.log(mel.clamp(min=1e-5))

        # Remove batch and channel dims, transpose
        mel = mel.squeeze(0).squeeze(0)  # (n_mels, T)
        mel = mel.transpose(0, 1)  # (T, n_mels)

        # Normalize
        if self.config.normalize_mel:
            mel = (mel - self.config.mel_mean) / self.config.mel_std

        return mel

    def process_array(
        self,
        audio_array: Any,
        sample_rate: int,
        denoise: bool | None = None,
    ) -> Tensor:
        """Process audio from numpy array (for HuggingFace datasets).

        Args:
            audio_array: Audio as numpy array or similar.
            sample_rate: Sample rate of the audio.
            denoise: Override config denoise setting.

        Returns:
            Log mel-spectrogram. Shape: (T, n_mels).
        """

        # Convert to tensor
        if isinstance(audio_array, np.ndarray):
            waveform = torch.from_numpy(audio_array).float()
        else:
            waveform = torch.tensor(audio_array, dtype=torch.float32)

        # Ensure 1D
        if waveform.dim() == 0:
            waveform = waveform.unsqueeze(0)

        # Resample if necessary
        if sample_rate != self.config.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.config.sample_rate,
            )

        # Normalize
        if self.config.normalize_audio:
            waveform = self._normalize_audio(waveform)

        # Denoise if requested
        should_denoise = denoise if denoise is not None else self.config.denoise
        if should_denoise:
            waveform = self.denoise_audio(waveform)

        # Extract mel
        mel = self.extract_mel(waveform)
        return mel

    def process(
        self,
        path: str | Path,
        denoise: bool | None = None,
    ) -> Tensor:
        """Full processing pipeline: load → denoise → mel.

        Args:
            path: Path to audio file.
            denoise: Override config denoise setting.

        Returns:
            Log mel-spectrogram. Shape: (T, n_mels).
        """
        waveform = self.load_audio(path)

        should_denoise = denoise if denoise is not None else self.config.denoise
        if should_denoise:
            waveform = self.denoise_audio(waveform)

        mel = self.extract_mel(waveform)
        return mel

    def _normalize_audio(self, waveform: Tensor) -> Tensor:
        """Normalize audio to [-1, 1] range."""
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val * 0.95
        return waveform

    def mel_to_audio(
        self,
        mel: Tensor,
        vocoder: torch.nn.Module | None = None,
    ) -> Tensor:
        """Convert mel-spectrogram back to audio.

        Args:
            mel: Log mel-spectrogram. Shape: (T, n_mels).
            vocoder: Neural vocoder (e.g., HiFi-GAN).

        Returns:
            Audio waveform.
        """
        # Denormalize
        if self.config.normalize_mel:
            mel = mel * self.config.mel_std + self.config.mel_mean

        if vocoder is not None:
            # Neural vocoder
            mel = mel.transpose(0, 1).unsqueeze(0)  # (1, n_mels, T)
            with torch.inference_mode():
                audio = vocoder(mel)
            return audio.squeeze()

        # Griffin-Lim fallback (lower quality)
        mel = mel.transpose(0, 1)  # (n_mels, T)
        mel = torch.exp(mel)  # Convert from log

        # Approximate inverse mel
        inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=self.config.n_fft // 2 + 1,
            n_mels=self.config.n_mels,
            sample_rate=self.config.sample_rate,
        )
        spec = inverse_mel(mel.unsqueeze(0))

        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            power=self.config.power,
        )
        audio = griffin_lim(spec)

        return audio.squeeze()


class DeepFilterNetWrapper:
    """Wrapper for DeepFilterNet audio enhancement.

    Provides noise reduction for non-professional recordings.
    """

    def __init__(self, atten_lim: float = 100.0) -> None:
        """Initialize DeepFilterNet.

        Args:
            atten_lim: Maximum noise attenuation in dB.
        """
        self.atten_lim = atten_lim
        self._model = None
        self._df_state = None

    def _load_model(self) -> None:
        """Lazy-load DeepFilterNet model."""
        if self._model is None:
            try:
                from df.enhance import init_df  # type: ignore[import-not-found,import-untyped]

                self._model, self._df_state, _ = init_df()
            except ImportError as err:
                raise ImportError(
                    "DeepFilterNet not installed. Run: pip install deepfilternet"
                ) from err

    def __call__(self, waveform: Tensor, sample_rate: int) -> Tensor:
        """Denoise audio waveform.

        Args:
            waveform: Input audio. Shape: (1, samples) or (samples,).
            sample_rate: Audio sample rate.

        Returns:
            Denoised audio with same shape.
        """
        self._load_model()

        from df.enhance import enhance  # type: ignore[import-untyped]

        # Ensure correct shape
        squeeze_output = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze_output = True

        # DeepFilterNet expects (samples, channels) numpy array
        audio_np = waveform.squeeze(0).numpy()

        # Resample to 48kHz if needed (DeepFilterNet native rate)
        if sample_rate != 48000:
            waveform_48k = torchaudio.functional.resample(waveform, sample_rate, 48000)
            audio_np = waveform_48k.squeeze(0).numpy()

        # Enhance
        enhanced = enhance(
            self._model,
            self._df_state,
            audio_np,
            atten_lim_db=self.atten_lim,
        )

        # Convert back to tensor
        enhanced_tensor = torch.from_numpy(enhanced).unsqueeze(0)

        # Resample back if needed
        if sample_rate != 48000:
            enhanced_tensor = torchaudio.functional.resample(enhanced_tensor, 48000, sample_rate)

        if squeeze_output:
            enhanced_tensor = enhanced_tensor.squeeze(0)

        return enhanced_tensor
