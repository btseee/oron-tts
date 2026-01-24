"""Dataset loader for VITS2 training."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from orontts.constants import (
    HOP_LENGTH,
    MEL_FMAX,
    MEL_FMIN,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    WIN_LENGTH,
)
from orontts.exceptions import DatasetError
from orontts.preprocessing.phonemizer import text_to_phoneme_ids
from orontts.preprocessing.text import normalize_text


@dataclass
class DatasetConfig:
    """Configuration for TTS dataset."""

    data_dir: Path
    sample_rate: int = SAMPLE_RATE
    n_fft: int = N_FFT
    n_mels: int = N_MELS
    hop_length: int = HOP_LENGTH
    win_length: int = WIN_LENGTH
    mel_fmin: float = MEL_FMIN
    mel_fmax: float = MEL_FMAX
    max_audio_length: int = SAMPLE_RATE * 15  # 15 seconds
    max_text_length: int = 500
    normalize_audio: bool = True
    add_noise: bool = False
    noise_scale: float = 0.0003


class TTSDataset(Dataset):
    """PyTorch Dataset for TTS training.

    Expects data in format:
        data_dir/
            metadata.csv  (format: file_id|speaker_id|text)
            wavs/
                file_id.wav

    Attributes:
        config: Dataset configuration.
        metadata: List of (audio_path, speaker_id, text) tuples.
    """

    def __init__(
        self,
        config: DatasetConfig,
        metadata: list[tuple[Path, int, str]] | None = None,
        transform: Callable | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            config: Dataset configuration.
            metadata: Pre-loaded metadata. If None, loads from config.data_dir.
            transform: Optional audio transform.
        """
        self.config = config
        self.transform = transform
        self.metadata = metadata or self._load_metadata()

        # Initialize mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.mel_fmin,
            f_max=config.mel_fmax,
            power=1.0,
            norm="slaney",
            mel_scale="slaney",
        )

    def _load_metadata(self) -> list[tuple[Path, int, str]]:
        """Load metadata from CSV file."""
        metadata_path = self.config.data_dir / "metadata.csv"
        if not metadata_path.exists():
            raise DatasetError(f"Metadata file not found: {metadata_path}")

        metadata: list[tuple[Path, int, str]] = []
        wavs_dir = self.config.data_dir / "wavs"

        with open(metadata_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split("|")
                if len(parts) < 3:
                    continue

                file_id, speaker_id, text = parts[0], int(parts[1]), parts[2]
                audio_path = wavs_dir / f"{file_id}.wav"

                if audio_path.exists():
                    metadata.append((audio_path, speaker_id, text))

        if not metadata:
            raise DatasetError(f"No valid entries found in {metadata_path}")

        return metadata

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single training sample.

        Returns:
            Dictionary with keys:
                - audio: Raw audio tensor [1, T]
                - mel: Mel spectrogram [n_mels, T']
                - phoneme_ids: Phoneme ID tensor [L]
                - speaker_id: Speaker ID integer
                - audio_length: Number of audio samples
                - phoneme_length: Number of phonemes
        """
        audio_path, speaker_id, text = self.metadata[idx]

        # Load audio
        audio, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
            audio = resampler(audio)

        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Normalize
        if self.config.normalize_audio:
            audio = audio / (audio.abs().max() + 1e-8) * 0.95

        # Add noise for training robustness
        if self.config.add_noise:
            noise = torch.randn_like(audio) * self.config.noise_scale
            audio = audio + noise

        # Truncate if too long
        if audio.shape[1] > self.config.max_audio_length:
            audio = audio[:, : self.config.max_audio_length]

        # Apply optional transform
        if self.transform is not None:
            audio = self.transform(audio)

        # Compute mel spectrogram
        mel = self.mel_transform(audio)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        mel = mel.squeeze(0)  # [n_mels, T']

        # Process text
        normalized_text = normalize_text(text)
        phoneme_ids = text_to_phoneme_ids(normalized_text)

        # Truncate text if too long
        if len(phoneme_ids) > self.config.max_text_length:
            phoneme_ids = phoneme_ids[: self.config.max_text_length]

        return {
            "audio": audio.squeeze(0),  # [T]
            "mel": mel,  # [n_mels, T']
            "phoneme_ids": torch.LongTensor(phoneme_ids),
            "speaker_id": torch.LongTensor([speaker_id]),
            "audio_length": torch.LongTensor([audio.shape[1]]),
            "phoneme_length": torch.LongTensor([len(phoneme_ids)]),
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Collate function for batching variable-length sequences.

    Pads sequences to the maximum length in the batch.
    """
    # Get max lengths
    max_audio_len = max(item["audio"].shape[0] for item in batch)
    max_mel_len = max(item["mel"].shape[1] for item in batch)
    max_phoneme_len = max(item["phoneme_ids"].shape[0] for item in batch)

    batch_size = len(batch)
    n_mels = batch[0]["mel"].shape[0]

    # Initialize padded tensors
    audio_padded = torch.zeros(batch_size, max_audio_len)
    mel_padded = torch.zeros(batch_size, n_mels, max_mel_len)
    phoneme_ids_padded = torch.zeros(batch_size, max_phoneme_len, dtype=torch.long)

    audio_lengths = torch.zeros(batch_size, dtype=torch.long)
    mel_lengths = torch.zeros(batch_size, dtype=torch.long)
    phoneme_lengths = torch.zeros(batch_size, dtype=torch.long)
    speaker_ids = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        audio_len = item["audio"].shape[0]
        mel_len = item["mel"].shape[1]
        phoneme_len = item["phoneme_ids"].shape[0]

        audio_padded[i, :audio_len] = item["audio"]
        mel_padded[i, :, :mel_len] = item["mel"]
        phoneme_ids_padded[i, :phoneme_len] = item["phoneme_ids"]

        audio_lengths[i] = audio_len
        mel_lengths[i] = mel_len
        phoneme_lengths[i] = phoneme_len
        speaker_ids[i] = item["speaker_id"]

    return {
        "audio": audio_padded,
        "mel": mel_padded,
        "phoneme_ids": phoneme_ids_padded,
        "audio_lengths": audio_lengths,
        "mel_lengths": mel_lengths,
        "phoneme_lengths": phoneme_lengths,
        "speaker_ids": speaker_ids,
    }


class TTSDataModule(L.LightningDataModule):
    """Lightning DataModule for TTS training.

    Handles train/val/test splits and data loading.
    """

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 16,
        num_workers: int = 4,
        val_split: float = 0.1,
        test_split: float = 0.05,
        seed: int = 42,
        **dataset_kwargs: Any,
    ) -> None:
        """Initialize DataModule.

        Args:
            data_dir: Path to data directory.
            batch_size: Batch size for training.
            num_workers: Number of data loading workers.
            val_split: Fraction of data for validation.
            test_split: Fraction of data for testing.
            seed: Random seed for splitting.
            **dataset_kwargs: Additional arguments for DatasetConfig.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.dataset_kwargs = dataset_kwargs

        self.train_dataset: TTSDataset | None = None
        self.val_dataset: TTSDataset | None = None
        self.test_dataset: TTSDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage."""
        config = DatasetConfig(data_dir=self.data_dir, **self.dataset_kwargs)

        # Create full dataset to get metadata
        full_dataset = TTSDataset(config)
        metadata = full_dataset.metadata

        # Split metadata
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(len(metadata))

        n_test = int(len(metadata) * self.test_split)
        n_val = int(len(metadata) * self.val_split)
        n_train = len(metadata) - n_test - n_val

        train_indices = indices[:n_train]
        val_indices = indices[n_train : n_train + n_val]
        test_indices = indices[n_train + n_val :]

        train_meta = [metadata[i] for i in train_indices]
        val_meta = [metadata[i] for i in val_indices]
        test_meta = [metadata[i] for i in test_indices]

        if stage == "fit" or stage is None:
            self.train_dataset = TTSDataset(config, metadata=train_meta)
            self.val_dataset = TTSDataset(config, metadata=val_meta)

        if stage == "test" or stage is None:
            self.test_dataset = TTSDataset(config, metadata=test_meta)

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
