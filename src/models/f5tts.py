"""F5TTS: top-level model combining DiT + CFM + Vocos vocoder.

Usage (inference):
    model = F5TTS.from_config(config)
    wav = model.synthesize(
        text="Сайн байна уу",
        lang="mn",
        ref_audio_path="voices/mongolian/female_young.wav",
    )

Usage (training):
    model = F5TTS.from_config(config)
    loss = model(mel, text_ids, mask)
    loss.backward()
"""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.models.decoder import VocosDecoder
from src.models.dit import DiT
from src.models.flow import CFM
from src.utils.audio import AudioProcessor
from src.utils.text_cleaner import TextCleaner


def _pad_text_to_len(token_ids: list[int], target_len: int, pad_id: int = 0) -> list[int]:
    """Zero-pad (or truncate) a token list to exactly target_len."""
    if len(token_ids) >= target_len:
        return token_ids[:target_len]
    return token_ids + [pad_id] * (target_len - len(token_ids))


class F5TTS(nn.Module):
    """Full F5-TTS model with flow-matching training and reference-audio inference."""

    def __init__(
        self,
        n_mels: int = 100,
        vocab_size: int = 67,
        # DiT size
        dim: int = 1024,
        depth: int = 22,
        heads: int = 16,
        ff_mult: int = 2,
        text_dim: int = 512,
        conv_layers: int = 4,
        p_dropout: float = 0.1,
        # Vocoder
        vocos_dim: int = 512,
        vocos_layers: int = 8,
        vocos_intermediate: int = 1536,
        # Audio
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length

        backbone = DiT(
            n_mels=n_mels,
            vocab_size=vocab_size,
            dim=dim,
            depth=depth,
            heads=heads,
            ff_mult=ff_mult,
            text_dim=text_dim,
            conv_layers=conv_layers,
            p_dropout=p_dropout,
        )
        self.cfm = CFM(backbone)

        self.vocoder = VocosDecoder(
            n_mels=n_mels,
            dim=vocos_dim,
            n_layers=vocos_layers,
            intermediate_dim=vocos_intermediate,
            n_fft=n_fft,
            hop_length=hop_length,
            sample_rate=sample_rate,
        )

    def forward(
        self,
        mel: torch.Tensor,
        text_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Training forward pass — returns CFM loss.

        Args:
            mel: Log-mel spectrograms [B, n_mels, T].
            text_ids: Token IDs zero-padded to T [B, T].
            mask: Valid frame mask [B, T].
        """
        return self.cfm(mel, text_ids, mask)

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        lang: str = "mn",
        attr_tokens: list[str] | None = None,
        ref_audio_path: str | Path | None = None,
        ref_text: str | None = None,
        n_steps: int = 32,
        target_duration_s: float | None = None,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Synthesize speech from text with optional reference audio.

        Args:
            text: Input Cyrillic text.
            lang: "mn" or "kz".
            attr_tokens: Optional style tags, e.g. ["[FEMALE]", "[YOUNG]"].
            ref_audio_path: Path to 3-10 s reference WAV for voice cloning.
            ref_text: Transcript of the reference audio clip.
            n_steps: ODE integration steps (more = slower but better quality).
            target_duration_s: Override target duration.  Defaults to estimate.
            device: Inference device.

        Returns:
            Waveform tensor [T_samples] on CPU.
        """
        self.eval()
        self.to(device)

        cleaner = TextCleaner()
        audio_proc = AudioProcessor(
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        # ── Encode target text ────────────────────────────────────────────────
        target_ids = cleaner.text_to_sequence(text, lang=lang, attr_tokens=attr_tokens)

        # ── Load & encode reference audio ─────────────────────────────────────
        ref_mel: torch.Tensor | None = None
        ref_len: int = 0
        ref_ids: list[int] = []

        if ref_audio_path is not None:
            ref_wav, _ = audio_proc.load_audio(ref_audio_path)
            ref_wav = audio_proc.normalize_audio(ref_wav).to(device)
            ref_mel = audio_proc.mel_spectrogram(ref_wav).unsqueeze(0)  # [1, n_mels, T_ref]
            ref_len = ref_mel.shape[-1]
            if ref_text is not None:
                ref_ids = cleaner.text_to_sequence(ref_text, lang=lang)

        # ── Estimate target length ────────────────────────────────────────────
        if target_duration_s is not None:
            target_len = int(target_duration_s * self.sample_rate / self.hop_length)
        else:
            # ~0.12 s per character is a rough estimate for Mongolian
            chars = len(text.replace(" ", ""))
            target_len = max(50, int(chars * 0.12 * self.sample_rate / self.hop_length))

        T_total = ref_len + target_len

        # ── Build full text_ids padded to T_total ─────────────────────────────
        full_ids = _pad_text_to_len(ref_ids + target_ids, T_total, pad_id=0)
        text_ids_t = torch.tensor([full_ids], dtype=torch.long, device=device)  # [1, T_total]

        mask = torch.ones(1, T_total, dtype=torch.bool, device=device)

        # ── Sample mel via CFM ────────────────────────────────────────────────
        gen_mel = self.cfm.sample(
            text_ids=text_ids_t,
            n_mels=self.n_mels,
            target_len=target_len,
            n_steps=n_steps,
            mask=mask,
            ref_mel=ref_mel,
            ref_len=ref_len,
        )  # [1, n_mels, target_len]

        # ── Decode mel → waveform ─────────────────────────────────────────────
        waveform = self.vocoder(gen_mel).squeeze(0).cpu()
        return waveform

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "F5TTS":
        model_cfg = config.get("model", {})
        audio_cfg = config
        return cls(
            n_mels=audio_cfg.get("n_mels", 100),
            vocab_size=model_cfg.get("vocab_size", 67),
            dim=model_cfg.get("dim", 1024),
            depth=model_cfg.get("depth", 22),
            heads=model_cfg.get("heads", 16),
            ff_mult=model_cfg.get("ff_mult", 2),
            text_dim=model_cfg.get("text_dim", 512),
            conv_layers=model_cfg.get("conv_layers", 4),
            p_dropout=model_cfg.get("p_dropout", 0.1),
            vocos_dim=model_cfg.get("vocos_dim", 512),
            vocos_layers=model_cfg.get("vocos_layers", 8),
            vocos_intermediate=model_cfg.get("vocos_intermediate", 1536),
            sample_rate=audio_cfg.get("sample_rate", 24000),
            n_fft=audio_cfg.get("n_fft", 1024),
            hop_length=audio_cfg.get("hop_length", 256),
        )
