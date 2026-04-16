"""F5-TTS Dataset and Collator supporting reference-audio conditioning."""

import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import Audio
from torch.utils.data import Dataset

from src.utils.audio import AudioProcessor
from src.utils.text_cleaner import TextCleaner

_logger = logging.getLogger(__name__)


def _decode_audio_bytes(raw_bytes: bytes, target_sr: int) -> np.ndarray:
    """Decode raw audio bytes to float32 numpy array at target sample rate."""
    audio_array, sr = sf.read(io.BytesIO(raw_bytes))
    audio_array = audio_array.astype(np.float32)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    if sr != target_sr:
        t = torchaudio.functional.resample(
            torch.from_numpy(audio_array).unsqueeze(0),
            orig_freq=sr,
            new_freq=target_sr,
        )
        audio_array = t.squeeze(0).numpy()
    return audio_array


class TTSDataset(Dataset):
    """Dataset for F5-TTS training.

    Each sample provides:
      - mel: full log-mel spectrogram [n_mels, T]
      - text_ids: token IDs padded to T (zero-padded to mel length)
      - mask: bool mask [T], all True for real frames
      - lang: "mn" or "kz"
    """

    def __init__(
        self,
        audio_paths: list[Path] | list[str] | None = None,
        texts: list[str] | None = None,
        langs: list[str] | None = None,
        sample_rate: int = 24000,
        n_mels: int = 100,
        max_audio_len: int | None = None,
        min_audio_len: int = 2048,
        audio_arrays: list[np.ndarray] | None = None,
        audio_bytes_list: list[bytes] | None = None,
    ) -> None:
        if audio_paths is not None:
            self.audio_paths: list[Path] | None = [Path(p) for p in audio_paths]
            self.audio_arrays: list[np.ndarray] | None = None
            self.audio_bytes_list: list[bytes] | None = None
            self._len = len(audio_paths)
        elif audio_bytes_list is not None:
            self.audio_paths = None
            self.audio_arrays = None
            self.audio_bytes_list = audio_bytes_list
            self._len = len(audio_bytes_list)
        elif audio_arrays is not None:
            self.audio_paths = None
            self.audio_arrays = audio_arrays
            self.audio_bytes_list = None
            self._len = len(audio_arrays)
        else:
            raise ValueError("Must provide audio_paths, audio_arrays, or audio_bytes_list")

        if texts is None:
            raise ValueError("texts must be provided")
        assert self._len == len(texts), "Audio and text lengths must match"

        self.texts = texts
        self.langs = langs or ["mn"] * self._len
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_audio_len = max_audio_len
        self.min_audio_len = min_audio_len

        self.audio_processor = AudioProcessor(sample_rate=sample_rate, n_mels=n_mels)
        self.text_cleaner = TextCleaner()

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, Any]:
        text = self.texts[idx]
        lang = self.langs[idx]

        try:
            if self.audio_bytes_list is not None:
                audio_np = _decode_audio_bytes(self.audio_bytes_list[idx], self.sample_rate)
                audio = torch.from_numpy(audio_np).float()
            elif self.audio_arrays is not None:
                audio = torch.from_numpy(self.audio_arrays[idx]).float()
            else:
                assert self.audio_paths is not None
                audio, _ = self.audio_processor.load_audio(self.audio_paths[idx])

            audio = self.audio_processor.normalize_audio(audio)

            if self.max_audio_len and len(audio) > self.max_audio_len:
                audio = audio[: self.max_audio_len]

            if torch.isnan(audio).any() or torch.isinf(audio).any():
                raise ValueError(f"Invalid audio values at index {idx}")
            if len(audio) < self.min_audio_len:
                raise ValueError(f"Audio too short at index {idx}: {len(audio)}")

            mel = self.audio_processor.mel_spectrogram(audio)  # [n_mels, T]
        except Exception:
            _logger.warning("Sample %d failed, returning next", idx, exc_info=True)
            return self.__getitem__((idx + 1) % len(self))

        T = mel.shape[-1]

        raw_ids = self.text_cleaner.text_to_sequence(text, lang=lang)
        if len(raw_ids) > T:
            raw_ids = raw_ids[:T]
        text_ids = raw_ids + [0] * (T - len(raw_ids))

        return {
            "mel": mel,
            "text_ids": torch.LongTensor(text_ids),
            "mask": torch.ones(T, dtype=torch.bool),
            "lang": lang,
            "text": text,
        }

    @classmethod
    def from_hf_dataset(
        cls,
        hf_dataset: Any,
        audio_column: str = "audio",
        text_column: str | None = None,
        lang_column: str | None = None,
        sample_rate: int = 24000,
        n_mels: int = 100,
        default_lang: str = "mn",
        max_audio_len: int | None = None,
    ) -> "TTSDataset":
        audio_bytes_list: list[bytes] = []
        texts: list[str] = []
        langs: list[str] = []

        # decode=False keeps audio as raw bytes — small in memory, fast to pickle
        # across DataLoader workers. Each sample is decoded lazily in __getitem__.
        hf_dataset = hf_dataset.cast_column(audio_column, Audio(decode=False))

        if text_column is None:
            candidates = ["text", "sentence_norm", "sentence", "transcript", "transcription"]
            for c in candidates:
                if c in hf_dataset.column_names:
                    text_column = c
                    break
            if text_column is None:
                raise ValueError(f"No text column found. Available: {hf_dataset.column_names}")

        print(f"Using text column: {text_column}")

        for item in hf_dataset:
            audio_info = item[audio_column]
            raw_bytes: bytes | None = (
                audio_info.get("bytes") if isinstance(audio_info, dict) else None
            )
            if not raw_bytes:
                # Fallback: path-based reference — read file to bytes
                path = audio_info.get("path") if isinstance(audio_info, dict) else None
                if path and Path(path).exists():
                    raw_bytes = Path(path).read_bytes()
            if not raw_bytes:
                _logger.warning("Skipping sample: no audio bytes or path")
                continue

            audio_bytes_list.append(raw_bytes)
            texts.append(item[text_column])

            lang = default_lang
            if lang_column and lang_column in item:
                lang = item[lang_column]
            langs.append(lang)

        return cls(
            audio_bytes_list=audio_bytes_list,
            texts=texts,
            langs=langs,
            sample_rate=sample_rate,
            n_mels=n_mels,
            max_audio_len=max_audio_len,
        )


class TTSCollator:
    """Pads a batch of TTSDataset samples to uniform mel/text length."""

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        mels = [item["mel"] for item in batch]
        text_ids_list = [item["text_ids"] for item in batch]
        masks = [item["mask"] for item in batch]

        mel_lengths = torch.LongTensor([m.shape[-1] for m in mels])
        max_T = int(mel_lengths.max().item())
        n_mels = mels[0].shape[0]

        mels_padded = torch.zeros(len(batch), n_mels, max_T)
        text_ids_padded = torch.zeros(len(batch), max_T, dtype=torch.long)
        masks_padded = torch.zeros(len(batch), max_T, dtype=torch.bool)

        for i, (mel, tids, msk) in enumerate(zip(mels, text_ids_list, masks, strict=True)):
            T = mel.shape[-1]
            mels_padded[i, :, :T] = mel
            text_ids_padded[i, :T] = tids
            masks_padded[i, :T] = msk

        return {
            "mel": mels_padded,
            "text_ids": text_ids_padded,
            "mask": masks_padded,
            "mel_lengths": mel_lengths,
        }
