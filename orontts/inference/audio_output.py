"""Audio output wrapper for synthesis results."""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import soundfile as sf
import torch


@dataclass
class AudioOutput:
    """Container for synthesized audio with utility methods.

    Attributes:
        audio: Audio waveform as numpy array or torch tensor.
        sample_rate: Sample rate in Hz.
        speaker_id: Speaker ID used for synthesis.
        text: Original input text.
    """

    audio: np.ndarray | torch.Tensor
    sample_rate: int
    speaker_id: int | None = None
    text: str | None = None

    def __post_init__(self) -> None:
        """Convert tensor to numpy if needed."""
        if isinstance(self.audio, torch.Tensor):
            self.audio = self.audio.cpu().numpy()

        # Ensure 1D
        if self.audio.ndim > 1:
            self.audio = self.audio.squeeze()

    def save(
        self,
        path: str | Path,
        format: str | None = None,
        subtype: str | None = None,
    ) -> Path:
        """Save audio to file.

        Args:
            path: Output file path.
            format: Audio format (wav, flac, ogg, etc.). Inferred from extension if None.
            subtype: Audio subtype (PCM_16, PCM_24, etc.).

        Returns:
            Path to saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(
            path,
            self.audio,
            self.sample_rate,
            format=format,
            subtype=subtype or "PCM_16",
        )

        return path

    def to_wav_bytes(self) -> bytes:
        """Convert to WAV bytes for streaming/API responses."""
        import io

        buffer = io.BytesIO()
        sf.write(buffer, self.audio, self.sample_rate, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        return buffer.read()

    def resample(self, target_sr: int) -> Self:
        """Resample audio to target sample rate.

        Args:
            target_sr: Target sample rate.

        Returns:
            New AudioOutput with resampled audio.
        """
        if target_sr == self.sample_rate:
            return self

        import librosa

        resampled = librosa.resample(
            self.audio,
            orig_sr=self.sample_rate,
            target_sr=target_sr,
        )

        return AudioOutput(
            audio=resampled,
            sample_rate=target_sr,
            speaker_id=self.speaker_id,
            text=self.text,
        )

    def normalize(self, peak: float = 0.95) -> Self:
        """Normalize audio amplitude.

        Args:
            peak: Target peak amplitude.

        Returns:
            New AudioOutput with normalized audio.
        """
        max_val = np.abs(self.audio).max()
        if max_val > 0:
            normalized = self.audio / max_val * peak
        else:
            normalized = self.audio

        return AudioOutput(
            audio=normalized,
            sample_rate=self.sample_rate,
            speaker_id=self.speaker_id,
            text=self.text,
        )

    def trim_silence(self, top_db: float = 20.0) -> Self:
        """Trim leading/trailing silence.

        Args:
            top_db: Threshold in dB.

        Returns:
            New AudioOutput with trimmed audio.
        """
        import librosa

        trimmed, _ = librosa.effects.trim(self.audio, top_db=top_db)

        return AudioOutput(
            audio=trimmed,
            sample_rate=self.sample_rate,
            speaker_id=self.speaker_id,
            text=self.text,
        )

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self.audio) / self.sample_rate

    @property
    def numpy(self) -> np.ndarray:
        """Get audio as numpy array."""
        if isinstance(self.audio, torch.Tensor):
            return self.audio.cpu().numpy()
        return self.audio

    @property
    def tensor(self) -> torch.Tensor:
        """Get audio as torch tensor."""
        if isinstance(self.audio, np.ndarray):
            return torch.from_numpy(self.audio)
        return self.audio

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.audio)

    def __repr__(self) -> str:
        return (
            f"AudioOutput(duration={self.duration:.2f}s, "
            f"sample_rate={self.sample_rate}, "
            f"speaker_id={self.speaker_id})"
        )
