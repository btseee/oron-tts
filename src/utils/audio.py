"""Audio processing utilities for F5-TTS.

Mel spectrogram computation matches Vocos vocoder exactly:
  torchaudio MelSpectrogram(power=1, center=True) + log(clamp(x, min=1e-5))
This ensures generated mels are directly decodable by pretrained Vocos
(charactr/vocos-mel-24khz uses the same clip_val=1e-5).
"""

import logging
from pathlib import Path
from typing import Final

import numpy as np
import soundfile as sf
import torch
import torchaudio

_logger = logging.getLogger(__name__)

# F5-TTS default audio settings
DEFAULT_SAMPLE_RATE: Final[int] = 24000
DEFAULT_N_MELS: Final[int] = 100
DEFAULT_N_FFT: Final[int] = 1024
DEFAULT_HOP_LENGTH: Final[int] = 256
DEFAULT_WIN_LENGTH: Final[int] = 1024


def _safe_log(x: torch.Tensor, clip_val: float = 1e-5) -> torch.Tensor:
    """Log with clipping — matches Vocos safe_log (clip_val=1e-5)."""
    return torch.log(torch.clamp(x, min=clip_val))


class AudioProcessor:
    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        win_length: int = DEFAULT_WIN_LENGTH,
        n_mels: int = DEFAULT_N_MELS,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels

        # Use torchaudio MelSpectrogram — matches Vocos feature extractor exactly.
        # Do NOT pass f_min/f_max: Vocos uses torchaudio defaults (0, sr/2).
        self._mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            center=True,
            power=1,  # magnitude, not power — matches Vocos
        )

    def load_audio(self, path: str | Path) -> tuple[torch.Tensor, int]:
        waveform, sr = torchaudio.load(str(path))
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0), self.sample_rate

    def save_audio(self, path: str | Path, audio: torch.Tensor | np.ndarray) -> None:
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        sf.write(str(path), audio, self.sample_rate)

    def normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        max_val = audio.abs().max()
        if max_val < 1e-8:  # Silent audio
            return audio
        return torch.clamp(audio / (max_val + 1e-7), -1.0, 1.0)

    def trim_silence(
        self,
        audio: torch.Tensor,
        top_db: float = 20.0,
        frame_length: int = 2048,
        hop_length: int = 512,
    ) -> torch.Tensor:
        import librosa  # lazy import — only needed if trim_silence is called

        audio_np = audio.cpu().numpy()
        trimmed, _ = librosa.effects.trim(
            audio_np, top_db=top_db, frame_length=frame_length, hop_length=hop_length
        )
        return torch.from_numpy(trimmed)

    def mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute log-mel spectrogram matching Vocos format.

        Args:
            audio: Waveform [T] or [1, T].

        Returns:
            Log-mel spectrogram [n_mels, T_frames].
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        device = audio.device
        mel_transform = self._mel_transform.to(device)
        mel = mel_transform(audio)  # [1, n_mels, T]
        log_mel = _safe_log(mel)
        return log_mel.squeeze(0)  # [n_mels, T]

    def get_audio_duration(self, audio: torch.Tensor) -> float:
        return len(audio) / self.sample_rate


