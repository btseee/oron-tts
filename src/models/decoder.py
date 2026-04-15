"""Vocos-style iSTFT vocoder: mel spectrogram → waveform."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # [B, T, C]
        x = self.norm(x)
        x = F.gelu(self.pwconv1(x))
        x = self.pwconv2(x)
        return x.transpose(1, 2) + residual  # [B, C, T]


class VocosDecoder(nn.Module):
    """Vocos: mel → waveform via ConvNeXt backbone + iSTFT head.

    Can optionally load weights from the official Vocos checkpoint
    (hubert-type key mapping handled in CheckpointManager).

    Reference: Siuzdak 2023, https://arxiv.org/abs/2306.00814
    """

    def __init__(
        self,
        n_mels: int = 100,
        dim: int = 512,
        n_layers: int = 8,
        intermediate_dim: int = 1536,
        n_fft: int = 1024,
        hop_length: int = 256,
        sample_rate: int = 24000,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        # Mel projection
        self.input_proj = nn.Conv1d(n_mels, dim, kernel_size=7, padding=3)
        self.norm_pre = nn.LayerNorm(dim)

        # ConvNeXt backbone
        self.layers = nn.ModuleList([ConvNeXtBlock(dim, intermediate_dim) for _ in range(n_layers)])
        self.norm_post = nn.LayerNorm(dim)

        # iSTFT head: predict magnitude + phase (real + imag) for all STFT bins
        self.istft_head = nn.Linear(dim, (n_fft // 2 + 1) * 2)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram to waveform.

        Args:
            mel: Log mel spectrogram [B, n_mels, T].

        Returns:
            Waveform [B, T * hop_length].
        """
        x = self.input_proj(mel)  # [B, dim, T]
        x = x.transpose(1, 2)
        x = self.norm_pre(x)
        x = x.transpose(1, 2)

        for layer in self.layers:
            x = layer(x)

        x = x.transpose(1, 2)
        x = self.norm_post(x)  # [B, T, dim]

        # Predict complex STFT coefficients
        out = self.istft_head(x)  # [B, T, (n_fft//2+1) * 2]
        B, T, _ = out.shape
        n_bins = self.n_fft // 2 + 1
        out = out.view(B, T, n_bins, 2)

        real = out[..., 0].transpose(1, 2)  # [B, n_bins, T]
        imag = out[..., 1].transpose(1, 2)

        # Pad for iSTFT
        spec = torch.complex(real, imag)
        window = torch.hann_window(self.n_fft, device=mel.device)
        waveform = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            normalized=True,
            onesided=True,
        )
        return waveform
