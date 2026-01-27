"""Dataset handling for OronTTS with HuggingFace Hub integration.

Supports multi-speaker datasets with gender separation and
efficient streaming from HuggingFace.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import datasets
import torch
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset

from src.data.audio import AudioConfig, AudioProcessor
from src.data.cleaner import MongolianPhonemizer, MongolianTextCleaner


class Gender(Enum):
    """Speaker gender enumeration."""

    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass
class SpeakerInfo:
    """Speaker metadata."""

    id: int
    name: str
    gender: Gender = Gender.UNKNOWN
    language: str = "mn"
    accent: str | None = None


@dataclass
class Sample:
    """Single training sample."""

    audio_path: str
    text: str
    speaker_id: int
    gender: Gender = Gender.UNKNOWN
    duration_sec: float | None = None
    mel: Tensor | None = None
    phonemes: Tensor | None = None
    audio_array: Any | None = None  # For HF datasets
    sample_rate: int | None = None


@dataclass
class Batch:
    """Collated batch for training."""

    mel: Tensor  # (B, T_max, mel_dim)
    phonemes: Tensor  # (B, T_max)
    speaker_ids: Tensor  # (B,)
    mel_lengths: Tensor  # (B,)
    phoneme_lengths: Tensor  # (B,)
    mask: Tensor  # (B, T_max) - 1 for valid, 0 for padding


class OronDataset(Dataset):
    """PyTorch Dataset for Mongolian TTS training.

    Loads and processes audio-text pairs from local storage or
    HuggingFace Hub with caching support.
    """

    def __init__(
        self,
        manifest_path: str | Path | None = None,
        hf_dataset_name: str | None = None,
        hf_split: str = "train",
        audio_config: AudioConfig | None = None,
        cache_mels: bool = True,
        max_duration_sec: float = 30.0,
        min_duration_sec: float = 0.5,
        filter_by_gender: Gender | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            manifest_path: Path to local manifest JSON/CSV.
            hf_dataset_name: HuggingFace dataset identifier.
            hf_split: Dataset split to use.
            audio_config: Audio processing configuration.
            cache_mels: Cache mel-spectrograms to disk.
            max_duration_sec: Maximum audio duration.
            min_duration_sec: Minimum audio duration.
            filter_by_gender: Filter samples by speaker gender.
        """
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.hf_dataset_name = hf_dataset_name
        self.hf_split = hf_split
        self.cache_mels = cache_mels
        self.max_duration = max_duration_sec
        self.min_duration = min_duration_sec
        self.filter_gender = filter_by_gender

        self.audio_processor = AudioProcessor(audio_config)
        self.text_cleaner = MongolianTextCleaner()
        self.phonemizer = MongolianPhonemizer()

        self.samples: list[Sample] = []
        self.speakers: dict[int, SpeakerInfo] = {}
        self._phoneme_vocab: dict[str, int] = {}

        self._load_data()

    def _load_data(self) -> None:
        """Load samples from manifest or HuggingFace."""
        if self.hf_dataset_name:
            self._load_from_huggingface()
        elif self.manifest_path:
            self._load_from_manifest()
        else:
            raise ValueError("Must provide either manifest_path or hf_dataset_name")

    def _load_from_huggingface(self) -> None:
        """Load dataset from HuggingFace Hub."""
        from datasets import load_dataset

        # Use soundfile for audio decoding instead of torchcodec
        datasets.config.AUDIO_DECODE_BACKENDS = ["soundfile"]

        dataset = load_dataset(
            self.hf_dataset_name,
            split=self.hf_split,
        )

        # Cast audio to avoid automatic resampling - we'll handle it in process_array
        dataset = dataset.cast_column(
            "audio", datasets.Audio(sampling_rate=None, mono=True, decode=True)
        )

        speaker_map: dict[str, int] = {}

        for idx, item in enumerate(dataset):
            # Extract audio data - HF datasets return decoded audio as dict
            audio_data = item.get("audio", {})
            # Get the audio bytes/array directly instead of path
            audio_array = audio_data.get("array", None)
            audio_path = audio_data.get("path", f"sample_{idx}")

            # If we have the audio array, we need to save or cache it
            # For now, skip if no path and rely on in-memory processing
            if audio_array is None:
                continue

            text = item.get("text", item.get("sentence", ""))
            speaker_name = item.get("speaker", item.get("client_id", f"speaker_{idx}"))
            gender_str = item.get("gender", "unknown")
            duration = item.get("duration", None)

            # Map speaker to ID
            if speaker_name not in speaker_map:
                speaker_id = len(speaker_map)
                speaker_map[speaker_name] = speaker_id
                self.speakers[speaker_id] = SpeakerInfo(
                    id=speaker_id,
                    name=speaker_name,
                    gender=Gender(gender_str)
                    if gender_str in Gender._value2member_map_
                    else Gender.UNKNOWN,
                )
            else:
                speaker_id = speaker_map[speaker_name]

            gender = self.speakers[speaker_id].gender

            # Filter by duration
            if duration is not None and (
                duration < self.min_duration or duration > self.max_duration
            ):
                continue

            # Filter by gender
            if self.filter_gender is not None and gender != self.filter_gender:
                continue

            # Store audio array with sample
            self.samples.append(
                Sample(
                    audio_path=audio_path,
                    text=text,
                    speaker_id=speaker_id,
                    gender=gender,
                    duration_sec=duration,
                    audio_array=audio_array,  # Store the decoded audio
                    sample_rate=audio_data.get(
                        "sampling_rate", self.audio_processor.config.sample_rate
                    ),
                )
            )

    def _load_from_manifest(self) -> None:
        """Load dataset from local manifest file."""
        import json

        with open(self.manifest_path) as f:
            manifest = json.load(f)

        # Load speakers
        for speaker_data in manifest.get("speakers", []):
            speaker = SpeakerInfo(
                id=speaker_data["id"],
                name=speaker_data["name"],
                gender=Gender(speaker_data.get("gender", "unknown")),
                language=speaker_data.get("language", "mn"),
            )
            self.speakers[speaker.id] = speaker

        # Load samples
        for item in manifest.get("samples", []):
            duration = item.get("duration")

            if duration is not None and (
                duration < self.min_duration or duration > self.max_duration
            ):
                continue

            speaker_id = item.get("speaker_id", 0)
            gender = self.speakers.get(speaker_id, SpeakerInfo(0, "unknown")).gender

            if self.filter_gender is not None and gender != self.filter_gender:
                continue

            self.samples.append(
                Sample(
                    audio_path=item["audio_path"],
                    text=item["text"],
                    speaker_id=speaker_id,
                    gender=gender,
                    duration_sec=duration,
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get processed sample.

        Returns:
            Dict with 'mel', 'phonemes', 'speaker_id', 'text'.
        """
        sample = self.samples[idx]

        # Process audio - use array if available, otherwise load from path
        if sample.audio_array is not None:
            mel = self.audio_processor.process_array(
                sample.audio_array, sample.sample_rate or self.audio_cfg.sample_rate
            )
        else:
            mel = self.audio_processor.process(sample.audio_path)

        # Process text
        text_clean = self.text_cleaner(sample.text)
        phonemes_str = self.phonemizer(text_clean)
        phoneme_ids = self.phonemizer.phoneme_to_ids(phonemes_str, self._phoneme_vocab)
        phonemes = torch.tensor(phoneme_ids, dtype=torch.long)

        return {
            "mel": mel,
            "phonemes": phonemes,
            "speaker_id": sample.speaker_id,
            "text": text_clean,
            "gender": sample.gender.value,
        }

    @property
    def num_speakers(self) -> int:
        """Number of unique speakers."""
        return len(self.speakers)

    def get_speakers_by_gender(self, gender: Gender) -> list[SpeakerInfo]:
        """Get all speakers of a specific gender."""
        return [s for s in self.speakers.values() if s.gender == gender]


class OronDataCollator:
    """Collate samples into padded batches."""

    def __init__(self, pad_value: int = 0) -> None:
        self.pad_value = pad_value

    def __call__(self, samples: list[dict[str, Any]]) -> Batch:
        """Collate samples into a batch.

        Args:
            samples: List of sample dicts from OronDataset.

        Returns:
            Collated Batch object.
        """
        batch_size = len(samples)

        # Get max lengths
        max_mel_len = max(s["mel"].size(0) for s in samples)
        max_phoneme_len = max(s["phonemes"].size(0) for s in samples)
        mel_dim = samples[0]["mel"].size(-1)

        # Initialize tensors
        mel = torch.zeros(batch_size, max_mel_len, mel_dim)
        phonemes = torch.full((batch_size, max_phoneme_len), self.pad_value, dtype=torch.long)
        speaker_ids = torch.zeros(batch_size, dtype=torch.long)
        mel_lengths = torch.zeros(batch_size, dtype=torch.long)
        phoneme_lengths = torch.zeros(batch_size, dtype=torch.long)
        mask = torch.zeros(batch_size, max_mel_len)

        # Fill tensors
        for i, sample in enumerate(samples):
            mel_len = sample["mel"].size(0)
            phoneme_len = sample["phonemes"].size(0)

            mel[i, :mel_len] = sample["mel"]
            phonemes[i, :phoneme_len] = sample["phonemes"]
            speaker_ids[i] = sample["speaker_id"]
            mel_lengths[i] = mel_len
            phoneme_lengths[i] = phoneme_len
            mask[i, :mel_len] = 1.0

        return Batch(
            mel=mel,
            phonemes=phonemes,
            speaker_ids=speaker_ids,
            mel_lengths=mel_lengths,
            phoneme_lengths=phoneme_lengths,
            mask=mask,
        )


class StreamingOronDataset(IterableDataset):
    """Streaming dataset for large-scale training from HuggingFace."""

    def __init__(
        self,
        hf_dataset_name: str,
        hf_split: str = "train",
        audio_config: AudioConfig | None = None,
        shuffle_buffer_size: int = 10000,
    ) -> None:
        """Initialize streaming dataset.

        Args:
            hf_dataset_name: HuggingFace dataset identifier.
            hf_split: Dataset split.
            audio_config: Audio processing config.
            shuffle_buffer_size: Buffer size for shuffling.
        """
        self.hf_dataset_name = hf_dataset_name
        self.hf_split = hf_split
        self.shuffle_buffer_size = shuffle_buffer_size

        self.audio_processor = AudioProcessor(audio_config)
        self.text_cleaner = MongolianTextCleaner()
        self.phonemizer = MongolianPhonemizer()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        from datasets import load_dataset

        dataset = load_dataset(
            self.hf_dataset_name,
            split=self.hf_split,
            streaming=True,
            trust_remote_code=True,
        )

        # Shuffle with buffer
        dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)

        for item in dataset:
            try:
                yield self._process_item(item)
            except Exception:
                # Skip failed samples in streaming mode
                continue

    def _process_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Process a single streaming item."""
        audio_data = item.get("audio", {})
        audio_array = audio_data.get("array")
        sample_rate = audio_data.get("sampling_rate", 16000)

        # Convert to tensor
        waveform = torch.from_numpy(audio_array).float().unsqueeze(0)

        # Resample if needed
        if sample_rate != self.audio_processor.config.sample_rate:
            import torchaudio

            resampler = torchaudio.transforms.Resample(
                sample_rate, self.audio_processor.config.sample_rate
            )
            waveform = resampler(waveform)

        # Extract mel
        mel = self.audio_processor.extract_mel(waveform)

        # Process text
        text = item.get("text", item.get("sentence", ""))
        text_clean = self.text_cleaner(text)
        phonemes_str = self.phonemizer(text_clean)
        phoneme_ids = self.phonemizer.phoneme_to_ids(phonemes_str)
        phonemes = torch.tensor(phoneme_ids, dtype=torch.long)

        return {
            "mel": mel,
            "phonemes": phonemes,
            "speaker_id": 0,  # Default for streaming
            "text": text_clean,
        }
