"""Audio processing utilities for VITS TTS."""

from pathlib import Path
from typing import Final

import librosa
import numpy as np
import scipy.signal as signal
import soundfile as sf
import torch
import torchaudio

MEL_FMIN: Final[float] = 0.0
MEL_FMAX: Final[float] = 8000.0


class AudioProcessor:
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        fmin: float = MEL_FMIN,
        fmax: float = MEL_FMAX,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

        self._mel_basis: torch.Tensor | None = None
        self._hann_window: dict[str, torch.Tensor] = {}

    def _get_mel_basis(self, device: torch.device) -> torch.Tensor:
        if self._mel_basis is None or self._mel_basis.device != device:
            mel_np = librosa.filters.mel(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax,
            )
            self._mel_basis = torch.from_numpy(mel_np).float().to(device)
        return self._mel_basis

    def _get_hann_window(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        if key not in self._hann_window:
            self._hann_window[key] = torch.hann_window(self.win_length).to(device)
        return self._hann_window[key]

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
        audio_np = audio.cpu().numpy()
        trimmed, _ = librosa.effects.trim(
            audio_np, top_db=top_db, frame_length=frame_length, hop_length=hop_length
        )
        return torch.from_numpy(trimmed)

    def spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        device = audio.device
        window = self._get_hann_window(device)

        pad_amount = (self.n_fft - self.hop_length) // 2
        audio = torch.nn.functional.pad(audio, (pad_amount, pad_amount), mode="reflect")

        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=False,
            return_complex=True,
        )
        spec = torch.abs(spec)
        return spec.squeeze(0)

    def mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        spec = self.spectrogram(audio)
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)

        mel_basis = self._get_mel_basis(audio.device)
        mel = torch.matmul(mel_basis, spec)
        mel = self._amp_to_db(mel)
        return mel.squeeze(0)

    def _amp_to_db(self, x: torch.Tensor, min_level: float = 1e-5) -> torch.Tensor:
        x_clamped = torch.clamp(x, min=min_level)
        log_spec = torch.log(x_clamped)
        # Check for NaN/Inf in spectrogram
        if torch.isnan(log_spec).any() or torch.isinf(log_spec).any():
            log_spec = torch.nan_to_num(log_spec, nan=0.0, posinf=0.0, neginf=-11.5)
        return log_spec

    def _db_to_amp(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)

    def get_audio_duration(self, audio: torch.Tensor) -> float:
        return len(audio) / self.sample_rate

    def apply_preemphasis(self, audio: torch.Tensor, coef: float = 0.97) -> torch.Tensor:
        return torch.cat([audio[:1], audio[1:] - coef * audio[:-1]])

    def remove_preemphasis(self, audio: torch.Tensor, coef: float = 0.97) -> torch.Tensor:
        result = signal.lfilter([1], [1, -coef], audio.cpu().numpy())
        return torch.from_numpy(result).float()
