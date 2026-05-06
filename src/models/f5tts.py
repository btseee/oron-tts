"""F5TTS: top-level model combining DiT + CFM + pretrained Vocos vocoder.

Usage (inference):
    model = F5TTS.from_config(config)
    wav = model.synthesize(
        text="Сайн байна уу",
        lang="mn",
        ref_audio_path="voices/mongolian/female_young.wav",
    )

Usage (training):
    model = F5TTS.from_config(config)
    loss = model(mel, text_ids, mel_lengths)
    loss.backward()
"""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.dit import DiT
from src.models.flow import CFM
from src.utils.audio import AudioProcessor
from src.utils.text_cleaner import TextCleaner
from src.utils.tokenizer import validate_language

_logger = logging.getLogger(__name__)

# Characters present only in the Kazakh extra set; warn if seen in MN input
# because the model has only been trained on the matching language tag.
_KZ_ONLY_CHARS: frozenset[str] = frozenset("әғқңұһі")


def _stretch_text_to_len(token_ids: list[int], target_len: int) -> list[int]:
    """Linearly stretch token_ids to target_len by repetition.

    Each mel frame receives the text token at approximately its temporal
    position (token j appears at frames j*T/N through (j+1)*T/N). This
    gives every frame a meaningful text signal instead of leaving 90%+
    of positions with a filler embedding — the standard F5-TTS approach.
    """
    n = len(token_ids)
    if n == 0:
        return [-1] * target_len
    if n >= target_len:
        return token_ids[:target_len]
    return [token_ids[int(i * n / target_len)] for i in range(target_len)]


class F5TTS(nn.Module):
    """Full F5-TTS model with flow-matching training and reference-audio inference."""

    def __init__(
        self,
        n_mels: int = 100,
        vocab_size: int = 65,
        # DiT size
        dim: int = 1024,
        depth: int = 22,
        heads: int = 16,
        dim_head: int = 64,
        ff_mult: int = 4,
        text_dim: int = 512,
        conv_layers: int = 4,
        p_dropout: float = 0.1,
        # CFM
        audio_drop_prob: float = 0.3,
        cond_drop_prob: float = 0.2,
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        # Audio
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        # Memory
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length

        self._text_cleaner = TextCleaner()
        self._audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

        backbone = DiT(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            ff_mult=ff_mult,
            dropout=p_dropout,
            mel_dim=n_mels,
            vocab_size=vocab_size,
            text_dim=text_dim,
            conv_layers=conv_layers,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.cfm = CFM(
            backbone,
            audio_drop_prob=audio_drop_prob,
            cond_drop_prob=cond_drop_prob,
            frac_lengths_mask=frac_lengths_mask,
            n_mels=n_mels,
        )

    def forward(
        self,
        mel: torch.Tensor,
        text_ids: torch.Tensor,
        lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Training forward pass — returns CFM loss.

        Args:
            mel: Log-mel spectrograms [B, n_mels, T].
            text_ids: Token IDs [B, Nt].
            lens: Mel lengths [B] (long tensor), OR bool mask [B, T] for compat.
        """
        # Accept either lengths or bool mask
        if lens is not None and lens.dtype == torch.bool and lens.ndim == 2:
            lens = lens.sum(dim=-1).long()
        return self.cfm(mel, text_ids, lens=lens)

    def _get_vocos(self, device: str) -> Any:
        """Lazily load and cache the pretrained Vocos vocoder.

        Stored outside the nn.Module parameter tree so it is never trained
        and never saved in checkpoints.
        """
        from vocos import Vocos  # required: pip install vocos

        vocos: Any = self.__dict__.get("_vocos_cache")
        if vocos is None:
            vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval()
            object.__setattr__(self, "_vocos_cache", vocos)
        return vocos.to(device)

    @staticmethod
    def _warn_lang_contamination(text: str, lang: str) -> None:
        """Warn once if text contains chars the model only saw in the other lang."""
        lang = validate_language(lang)
        if lang == "mn":
            bad = {c for c in text.lower() if c in _KZ_ONLY_CHARS}
            if bad:
                _logger.warning(
                    "Mongolian input contains Kazakh-only characters %s; "
                    "the model was conditioned with [LANG_MN] and may produce "
                    "out-of-distribution audio.",
                    sorted(bad),
                )

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        lang: str = "mn",
        ref_audio_path: str | Path | None = None,
        ref_text: str | None = None,
        n_steps: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float | None = -1.0,
        speed: float = 1.0,
        target_duration_s: float | None = None,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Synthesize speech from text with optional reference audio.

        Args:
            text: Input Cyrillic text.
            lang: "mn" or "kz".
            ref_audio_path: Path to 3-10 s reference WAV for voice cloning.
            ref_text: Transcript of the reference audio clip.
            n_steps: ODE integration steps (more = slower but better quality).
            cfg_strength: Classifier-free guidance strength (0 = none).
            sway_sampling_coef: Sway sampling coefficient (None = uniform).
            speed: Speaking-rate multiplier (>1 faster, <1 slower).
                Ignored when ``target_duration_s`` is set.
            target_duration_s: Override target duration. Defaults to estimate.
            device: Inference device.

        Returns:
            Waveform tensor [T_samples] on CPU.
        """
        lang = validate_language(lang)
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if cfg_strength < 0:
            raise ValueError(f"cfg_strength must be >= 0, got {cfg_strength}")
        if speed <= 0:
            raise ValueError(f"speed must be > 0, got {speed}")
        if target_duration_s is not None and target_duration_s <= 0:
            raise ValueError(f"target_duration_s must be > 0, got {target_duration_s}")
        self.eval()
        self.to(device)

        self._warn_lang_contamination(text, lang)

        cleaner = self._text_cleaner
        audio_proc = self._audio_processor

        # ── Encode target text ────────────────────────────────────────────────
        target_ids = cleaner.text_to_sequence(text, lang=lang)

        # ── Load & encode reference audio ─────────────────────────────────────
        ref_mel: torch.Tensor | None = None
        ref_len: int = 0
        ref_ids: list[int] = []

        if ref_audio_path is not None:
            if not ref_text:
                _logger.warning(
                    "ref_audio_path was provided without ref_text; duration will fall back "
                    "to the ref-free estimate and the reference region will use filler text."
                )
            self._warn_lang_contamination(ref_text or "", lang)
            ref_wav, _ = audio_proc.load_audio(ref_audio_path)
            ref_wav = audio_proc.normalize_audio(ref_wav).to(device)
            ref_mel_raw = audio_proc.mel_spectrogram(ref_wav)  # [n_mels, T_ref]
            ref_len = ref_mel_raw.shape[-1]
            # [1, T_ref, n_mels]
            ref_mel = ref_mel_raw.unsqueeze(0).transpose(1, 2)
            if ref_text is not None:
                ref_ids = cleaner.text_to_sequence(ref_text, lang=lang)

        # ── Estimate target length ────────────────────────────────────────────
        if target_duration_s is not None:
            target_len = max(1, int(target_duration_s * self.sample_rate / self.hop_length))
        elif ref_len > 0 and ref_ids:
            # Reference-aware: scale ref-mel duration by token-count ratio.
            # This is the paper-faithful approach used by official F5-TTS.
            target_len = max(50, int(ref_len * len(target_ids) / len(ref_ids) / speed))
        else:
            # Ref-free: ~13 mel frames per char ≈ 0.139s/char at hop=256/sr=24k.
            chars = max(1, len(text.replace(" ", "")))
            target_len = max(50, int(chars * 13 / speed))

        T_total = ref_len + target_len

        # ── Build full text_ids stretched to T_total ──────────────────────────────
        # Each mel frame must have a real text token (not filler) for the model
        # to learn text-audio correspondence — the standard F5-TTS approach.
        if ref_len > 0:
            ref_stretched = _stretch_text_to_len(ref_ids, ref_len)
            target_stretched = _stretch_text_to_len(target_ids, target_len)
            full_ids = ref_stretched + target_stretched
        else:
            full_ids = _stretch_text_to_len(target_ids, T_total)
        text_ids_t = torch.tensor([full_ids], dtype=torch.long, device=device)

        # ── Build conditioning mel [1, T_total, n_mels] ──────────────────────
        if ref_mel is not None:
            cond = F.pad(ref_mel, (0, 0, 0, T_total - ref_len), value=0.0)
        else:
            cond = torch.zeros(1, T_total, self.n_mels, device=device)

        # Reference audio lengths (must always pass, even when 0,
        # otherwise CFM.sample defaults to masking the entire sequence)
        ref_lens = torch.tensor([ref_len], device=device, dtype=torch.long)
        duration = torch.tensor([T_total], device=device, dtype=torch.long)

        # ── Sample mel via CFM ────────────────────────────────────────────────
        gen_mel, _ = self.cfm.sample(
            cond=cond,
            text_ids=text_ids_t,
            duration=duration,
            lens=ref_lens,
            steps=n_steps,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
        )
        # gen_mel: [1, T_total, n_mels] → take target portion
        gen_mel = gen_mel[:, ref_len:, :].transpose(1, 2)  # [1, n_mels, target_len]

        # ── Decode mel → waveform ─────────────────────────────────────────────
        waveform = self._get_vocos(device).decode(gen_mel).squeeze(0).cpu()
        return waveform

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "F5TTS":
        model_cfg = config.get("model", {})
        audio_cfg = config
        dim = model_cfg.get("dim", 1024)
        heads = model_cfg.get("heads", 16)
        frac_lengths_mask_cfg = model_cfg.get("frac_lengths_mask", [0.7, 1.0])
        return cls(
            n_mels=audio_cfg.get("n_mels", 100),
            vocab_size=model_cfg.get("vocab_size", 65),
            dim=dim,
            depth=model_cfg.get("depth", 22),
            heads=heads,
            dim_head=dim // heads,
            ff_mult=model_cfg.get("ff_mult", 4),
            text_dim=model_cfg.get("text_dim", 512),
            conv_layers=model_cfg.get("conv_layers", 4),
            p_dropout=model_cfg.get("p_dropout", 0.1),
            audio_drop_prob=model_cfg.get("audio_drop_prob", 0.3),
            cond_drop_prob=model_cfg.get("cond_drop_prob", 0.2),
            frac_lengths_mask=tuple(frac_lengths_mask_cfg),
            sample_rate=audio_cfg.get("sample_rate", 24000),
            n_fft=audio_cfg.get("n_fft", 1024),
            hop_length=audio_cfg.get("hop_length", 256),
            gradient_checkpointing=config.get("gradient_checkpointing", False),
        )
