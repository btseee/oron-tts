"""F5-TTS Dataset and Collator supporting reference-audio conditioning."""

import io
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
        speaker_ids: list[int] | None = None,
        sample_rate: int = 24000,
        n_mels: int = 100,
        max_audio_len: int | None = None,
        min_audio_len: int = 2048,
        audio_arrays: list[np.ndarray] | None = None,
    ) -> None:
        if audio_paths is not None:
            self.audio_paths: list[Path] | None = [Path(p) for p in audio_paths]
            self.audio_arrays: list[np.ndarray] | None = None
            self._len = len(audio_paths)
        elif audio_arrays is not None:
            self.audio_paths = None
            self.audio_arrays = audio_arrays
            self._len = len(audio_arrays)
        else:
            raise ValueError("Must provide either audio_paths or audio_arrays")

        if texts is None:
            raise ValueError("texts must be provided")
        assert self._len == len(texts), "Audio and text lengths must match"

        self.texts = texts
        self.langs = langs or ["mn"] * self._len
        self.speaker_ids = speaker_ids or [0] * self._len
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_audio_len = max_audio_len
        self.min_audio_len = min_audio_len

        self.audio_processor = AudioProcessor(sample_rate=sample_rate, n_mels=n_mels)
        self.text_cleaner = TextCleaner()

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, Any]:
        try:
            text = self.texts[idx]
            lang = self.langs[idx]

            if self.audio_arrays is not None:
                audio = torch.from_numpy(self.audio_arrays[idx]).float()
            else:
                assert self.audio_paths is not None
                audio, _ = self.audio_processor.load_audio(self.audio_paths[idx])

            audio = self.audio_processor.normalize_audio(audio)

            if self.max_audio_len and len(audio) > self.max_audio_len:
                audio = audio[: self.max_audio_len]

            if torch.isnan(audio).any() or torch.isinf(audio).any():
                raise ValueError("Invalid audio values")
            if len(audio) < self.min_audio_len:
                raise ValueError(f"Audio too short: {len(audio)}")

            mel = self.audio_processor.mel_spectrogram(audio)  # [n_mels, T]
            T = mel.shape[-1]

            # Encode text and zero-pad to mel length
            raw_ids = self.text_cleaner.text_to_sequence(text, lang=lang)
            if len(raw_ids) > T:
                raw_ids = raw_ids[:T]
            text_ids = raw_ids + [0] * (T - len(raw_ids))

            return {
                "mel": mel,
                "text_ids": torch.LongTensor(text_ids),
                "mask": torch.ones(T, dtype=torch.bool),
                "lang": lang,
                "speaker_id": self.speaker_ids[idx],
                "text": text,
            }
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

    @classmethod
    def from_hf_dataset(
        cls,
        hf_dataset: Any,
        audio_column: str = "audio",
        text_column: str | None = None,
        lang_column: str | None = None,
        speaker_column: str | None = "speaker_id",
        sample_rate: int = 24000,
        n_mels: int = 100,
        default_lang: str = "mn",
    ) -> "TTSDataset":
        audio_arrays: list[np.ndarray] = []
        texts: list[str] = []
        langs: list[str] = []
        speaker_ids: list[int] = []

        # decode=False avoids the torchcodec dependency in newer datasets versions.
        # We decode the raw bytes manually using soundfile + torchaudio.
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
            raw_bytes: bytes | None = audio_info.get("bytes") if isinstance(audio_info, dict) else None
            if raw_bytes:
                audio_array, sr = sf.read(io.BytesIO(raw_bytes))
                audio_array = audio_array.astype(np.float32)
                if audio_array.ndim > 1:  # stereo → mono
                    audio_array = audio_array.mean(axis=1)
                if sr != sample_rate:
                    t = torchaudio.functional.resample(
                        torch.from_numpy(audio_array).unsqueeze(0),
                        orig_freq=sr,
                        new_freq=sample_rate,
                    )
                    audio_array = t.squeeze(0).numpy()
            elif isinstance(audio_info, dict) and audio_info.get("array") is not None:
                audio_array = np.array(audio_info["array"], dtype=np.float32)
            else:
                audio_array = np.array(audio_info, dtype=np.float32)

            audio_arrays.append(audio_array.astype(np.float32))
            texts.append(item[text_column])

            lang = default_lang
            if lang_column and lang_column in item:
                lang = item[lang_column]
            langs.append(lang)

            sid = 0
            if speaker_column and speaker_column in item:
                sid = item[speaker_column]
            speaker_ids.append(sid)

        return cls(
            audio_arrays=audio_arrays,
            texts=texts,
            langs=langs,
            speaker_ids=speaker_ids,
            sample_rate=sample_rate,
            n_mels=n_mels,
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

        for i, (mel, tids, msk) in enumerate(zip(mels, text_ids_list, masks, strict=False)):
            T = mel.shape[-1]
            mels_padded[i, :, :T] = mel
            text_ids_padded[i, :T] = tids
            masks_padded[i, :T] = msk

        speaker_ids = torch.LongTensor([item["speaker_id"] for item in batch])

        return {
            "mel": mels_padded,
            "text_ids": text_ids_padded,
            "mask": masks_padded,
            "mel_lengths": mel_lengths,
            "speaker_ids": speaker_ids,
        }
