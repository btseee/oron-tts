"""TTS Dataset and Collator for VITS training."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Audio
from torch.utils.data import Dataset

from src.utils.audio import AudioProcessor
from src.utils.text_cleaner import TextCleaner


class TTSDataset(Dataset):
    def __init__(
        self,
        audio_paths: list[Path] | list[str] | None = None,
        texts: list[str] | None = None,
        speaker_ids: list[int] | None = None,
        sample_rate: int = 22050,
        max_audio_len: int | None = None,
        min_audio_len: int = 1024,
        audio_arrays: list[np.ndarray] | None = None,
    ) -> None:
        # Support either file paths or pre-loaded audio arrays
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
        if speaker_ids:
            assert len(speaker_ids) == len(texts)

        self.texts = texts
        self.speaker_ids = speaker_ids or [0] * len(texts)
        self.sample_rate = sample_rate
        self.max_audio_len = max_audio_len
        self.min_audio_len = min_audio_len

        self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        self.text_cleaner = TextCleaner()

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, Any]:
        text = self.texts[idx]
        speaker_id = self.speaker_ids[idx]

        if self.audio_arrays is not None:
            audio = torch.from_numpy(self.audio_arrays[idx]).float()
            audio_path = f"sample_{idx}"
        else:
            assert self.audio_paths is not None
            audio_path = str(self.audio_paths[idx])
            audio, _ = self.audio_processor.load_audio(self.audio_paths[idx])

        audio = self.audio_processor.normalize_audio(audio)

        if self.max_audio_len and len(audio) > self.max_audio_len:
            audio = audio[: self.max_audio_len]

        spec = self.audio_processor.mel_spectrogram(audio)
        text_ids = self.text_cleaner.text_to_sequence(text)

        return {
            "audio": audio,
            "spec": spec,
            "text_ids": torch.LongTensor(text_ids),
            "speaker_id": speaker_id,
            "audio_path": audio_path,
            "text": text,
        }

    @classmethod
    def from_hf_dataset(
        cls,
        hf_dataset: Any,
        audio_column: str = "audio",
        text_column: str | None = None,
        speaker_column: str | None = "speaker_id",
        sample_rate: int = 22050,
    ) -> "TTSDataset":
        audio_arrays: list[np.ndarray] = []
        texts: list[str] = []
        speaker_ids: list[int] = []

        # Cast audio column to use soundfile decoder (avoids torchcodec/FFmpeg dependency)
        hf_dataset = hf_dataset.cast_column(
            audio_column,
            Audio(sampling_rate=sample_rate, decode=True),
        )

        # Auto-detect text column if not specified
        if text_column is None:
            columns = hf_dataset.column_names
            text_candidates = ["text", "sentence", "sentence_norm", "transcript", "transcription"]
            for candidate in text_candidates:
                if candidate in columns:
                    text_column = candidate
                    break
            if text_column is None:
                raise ValueError(f"Could not find text column. Available: {columns}")

        print(f"Using text column: {text_column}")

        for item in hf_dataset:
            audio_info = item[audio_column]

            # Handle different audio data types from HF datasets
            if isinstance(audio_info, dict):
                # Old-style HF datasets with dict containing 'array'
                audio_array = audio_info["array"]
                if not isinstance(audio_array, np.ndarray):
                    audio_array = np.array(audio_array, dtype=np.float32)
            elif hasattr(audio_info, "get_all_samples"):
                # torchcodec AudioDecoder (newer datasets library)
                samples = audio_info.get_all_samples()
                audio_array = samples.data.squeeze(0).numpy()
            else:
                # Fallback: try to convert directly
                audio_array = np.array(audio_info, dtype=np.float32)

            audio_arrays.append(audio_array.astype(np.float32))

            texts.append(item[text_column])

            if speaker_column and speaker_column in item:
                speaker_ids.append(item[speaker_column])
            else:
                speaker_ids.append(0)

        return cls(
            audio_arrays=audio_arrays,
            texts=texts,
            speaker_ids=speaker_ids if speaker_column else None,
            sample_rate=sample_rate,
        )


class TTSCollator:
    def __init__(self, return_ids: bool = False) -> None:
        self.return_ids = return_ids

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        text_ids = [item["text_ids"] for item in batch]
        specs = [item["spec"] for item in batch]
        audios = [item["audio"] for item in batch]
        speaker_ids = torch.LongTensor([item["speaker_id"] for item in batch])

        text_lengths = torch.LongTensor([len(t) for t in text_ids])
        spec_lengths = torch.LongTensor([s.shape[-1] for s in specs])
        audio_lengths = torch.LongTensor([len(a) for a in audios])

        max_text_len = int(text_lengths.max().item())
        max_spec_len = int(spec_lengths.max().item())
        max_audio_len = int(audio_lengths.max().item())

        text_ids_padded = torch.zeros(len(batch), max_text_len, dtype=torch.long)
        for i, t in enumerate(text_ids):
            text_ids_padded[i, : len(t)] = t

        n_mels = specs[0].shape[0]
        specs_padded = torch.zeros(len(batch), n_mels, max_spec_len)
        for i, s in enumerate(specs):
            specs_padded[i, :, : s.shape[-1]] = s

        # Pad audio waveforms
        audios_padded = torch.zeros(len(batch), max_audio_len)
        for i, a in enumerate(audios):
            audios_padded[i, : len(a)] = a

        result: dict[str, Any] = {
            "text_ids": text_ids_padded,
            "text_lengths": text_lengths,
            "specs": specs_padded,
            "spec_lengths": spec_lengths,
            "audios": audios_padded,
            "audio_lengths": audio_lengths,
            "speaker_ids": speaker_ids,
        }

        if self.return_ids:
            result["audio_paths"] = [item["audio_path"] for item in batch]
            result["texts"] = [item["text"] for item in batch]

        return result
