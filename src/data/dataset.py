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
from torch.utils.data import Dataset, Sampler

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
        self.min_audio_len = min_audio_len

        self.audio_processor = AudioProcessor(sample_rate=sample_rate, n_mels=n_mels)
        self.text_cleaner = TextCleaner()
        self.durations: list[float] = []  # populated by from_hf_dataset or _compute_durations

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

            if torch.isnan(audio).any() or torch.isinf(audio).any():
                raise ValueError(f"Invalid audio values at index {idx}")
            if len(audio) < self.min_audio_len:
                raise ValueError(f"Audio too short at index {idx}: {len(audio)}")

            mel = self.audio_processor.mel_spectrogram(audio)  # [n_mels, T]
        except Exception as e:
            _logger.warning("Sample %d failed, trying next", idx, exc_info=True)
            for offset in range(1, min(11, len(self))):
                next_idx = (idx + offset) % len(self)
                try:
                    return self.__getitem__(next_idx)
                except Exception:
                    continue
            raise RuntimeError("All fallback samples failed") from e

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
    ) -> "TTSDataset":
        audio_bytes_list: list[bytes] = []
        texts: list[str] = []
        langs: list[str] = []
        durations: list[float] = []

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

            # Read duration from audio header (fast, no full decode)
            try:
                info = sf.info(io.BytesIO(raw_bytes))
                dur = info.duration
            except Exception:
                dur = 0.0

            audio_bytes_list.append(raw_bytes)
            texts.append(item[text_column])
            durations.append(dur)

            lang = default_lang
            if lang_column and lang_column in item:
                lang = item[lang_column]
            langs.append(lang)

        dataset = cls(
            audio_bytes_list=audio_bytes_list,
            texts=texts,
            langs=langs,
            sample_rate=sample_rate,
            n_mels=n_mels,
        )
        dataset.durations = durations
        return dataset


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


class DynamicBatchSampler(Sampler[list[int]]):
    """Frame-budget batch sampler (following F5-TTS upstream).

    Sorts samples by mel-frame length, then greedily packs batches so that the
    total number of frames per batch stays below *frames_threshold*.  This means
    short samples get grouped into large batches (fast) while long samples form
    small batches — potentially down to batch_size=1 — keeping peak GPU memory
    bounded.  **No samples are discarded.**
    """

    def __init__(
        self,
        durations: list[float],
        frames_threshold: int,
        max_samples: int = 0,
        sample_rate: int = 24000,
        hop_length: int = 256,
        drop_last: bool = False,
    ) -> None:
        self.frames_threshold = frames_threshold
        self.epoch = 0

        # Convert durations (seconds) → mel-frame counts
        frame_lens = [dur * sample_rate / hop_length for dur in durations]

        # Sort indices by frame length, then greedily pack batches
        sorted_pairs = sorted(enumerate(frame_lens), key=lambda x: x[1])

        batches: list[list[int]] = []
        batch: list[int] = []
        batch_frames = 0.0
        for idx, flen in sorted_pairs:
            if (batch_frames + flen <= frames_threshold) and (
                max_samples == 0 or len(batch) < max_samples
            ):
                batch.append(idx)
                batch_frames += flen
            else:
                if batch:
                    batches.append(batch)
                batch = [idx]
                batch_frames = flen

        if batch and not drop_last:
            batches.append(batch)

        self.batches = batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):  # noqa: ANN204
        g = torch.Generator()
        g.manual_seed(self.epoch)
        order = torch.randperm(len(self.batches), generator=g).tolist()
        yield from (self.batches[i] for i in order)

    def __len__(self) -> int:
        return len(self.batches)
