"""Audio cleaning pipeline using DeepFilterNet."""

from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Self
import sys
import types
import warnings

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from orontts.constants import SAMPLE_RATE
from orontts.exceptions import AudioProcessingError


def _patch_torchaudio_backend() -> None:
    """Patch torchaudio.backend for DeepFilterNet compatibility.
    
    DeepFilterNet 0.5.6 uses the deprecated torchaudio.backend.common.AudioMetaData
    which was removed in torchaudio 2.1+. This creates a compatibility shim.
    """
    if "torchaudio.backend" in sys.modules:
        return
    
    # Create fake backend module with AudioMetaData
    backend_module = types.ModuleType("torchaudio.backend")
    common_module = types.ModuleType("torchaudio.backend.common")
    
    # AudioMetaData is now at torchaudio.AudioMetaData
    if hasattr(torchaudio, "AudioMetaData"):
        common_module.AudioMetaData = torchaudio.AudioMetaData
    else:
        # Fallback: create a simple namedtuple-like class
        from dataclasses import dataclass as dc
        
        @dc
        class AudioMetaData:
            sample_rate: int
            num_frames: int
            num_channels: int
            bits_per_sample: int = 16
            encoding: str = "PCM_S"
        
        common_module.AudioMetaData = AudioMetaData
    
    backend_module.common = common_module
    sys.modules["torchaudio.backend"] = backend_module
    sys.modules["torchaudio.backend.common"] = common_module


# Apply patch before importing DeepFilterNet
_patch_torchaudio_backend()

# Check if DeepFilterNet is available
_DEEPFILTER_AVAILABLE = False
try:
    from df.enhance import enhance, init_df
    _DEEPFILTER_AVAILABLE = True
except ImportError:
    pass


@dataclass
class AudioCleanerConfig:
    """Configuration for audio cleaning pipeline."""

    target_sample_rate: int = SAMPLE_RATE
    normalize: bool = True
    trim_silence: bool = True
    trim_db: float = 20.0
    min_duration: float = 0.5
    max_duration: float = 15.0
    require_deepfilter: bool = False  # If True, raise error when DeepFilterNet unavailable


class AudioCleaner:
    """Audio cleaning pipeline with optional DeepFilterNet noise suppression.

    If DeepFilterNet is not installed, the cleaner will skip noise suppression
    but still apply resampling, normalization, and silence trimming.

    Attributes:
        config: Audio cleaning configuration.
        device: Torch device for processing.
    """

    def __init__(
        self,
        config: AudioCleanerConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize the audio cleaner.

        Args:
            config: Cleaning configuration. Uses defaults if None.
            device: Device for DeepFilterNet inference.
        """
        self.config = config or AudioCleanerConfig()
        self.device = torch.device(device)
        self._df_model: object | None = None
        self._df_state: object | None = None
        self._deepfilter_warned = False

    @property
    def has_deepfilter(self) -> bool:
        """Check if DeepFilterNet is available."""
        return _DEEPFILTER_AVAILABLE

    def _load_deepfilter(self) -> bool:
        """Lazy-load DeepFilterNet model.
        
        Returns:
            True if DeepFilterNet loaded successfully, False otherwise.
        """
        if not _DEEPFILTER_AVAILABLE:
            if self.config.require_deepfilter:
                raise AudioProcessingError(
                    "DeepFilterNet not installed. "
                    "Install with: pip install orontts[cleaning]"
                )
            if not self._deepfilter_warned:
                warnings.warn(
                    "DeepFilterNet not available. Noise suppression will be skipped. "
                    "Install with: pip install orontts[cleaning]",
                    UserWarning,
                    stacklevel=3,
                )
                self._deepfilter_warned = True
            return False

        if self._df_model is not None:
            return True

        self._df_model, self._df_state, _ = init_df()
        return True

    def clean(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Clean a single audio array.

        Args:
            audio: Input audio array (mono or stereo).
            sample_rate: Sample rate of input audio.

        Returns:
            Cleaned audio array at target sample rate.

        Raises:
            AudioProcessingError: If cleaning fails.
        """
        try:
            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = audio.mean(axis=-1)

            # Check if DeepFilterNet is available
            use_deepfilter = self._load_deepfilter()

            if use_deepfilter:
                # Resample to DeepFilterNet's expected rate (48kHz)
                if sample_rate != 48000:
                    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                    resampler = torchaudio.transforms.Resample(sample_rate, 48000)
                    audio_tensor = resampler(audio_tensor)
                    audio = audio_tensor.squeeze(0).numpy()

                # Apply DeepFilterNet
                audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                enhanced = enhance(self._df_model, self._df_state, audio_tensor)
                audio = enhanced.squeeze(0).numpy()

                # Resample to target rate
                if self.config.target_sample_rate != 48000:
                    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                    resampler = torchaudio.transforms.Resample(
                        48000, self.config.target_sample_rate
                    )
                    audio_tensor = resampler(audio_tensor)
                    audio = audio_tensor.squeeze(0).numpy()
            else:
                # No DeepFilterNet: just resample to target rate
                if sample_rate != self.config.target_sample_rate:
                    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                    resampler = torchaudio.transforms.Resample(
                        sample_rate, self.config.target_sample_rate
                    )
                    audio_tensor = resampler(audio_tensor)
                    audio = audio_tensor.squeeze(0).numpy()

            # Normalize amplitude
            if self.config.normalize:
                max_val = np.abs(audio).max()
                if max_val > 0:
                    audio = audio / max_val * 0.95

            # Trim silence
            if self.config.trim_silence:
                audio = self._trim_silence(audio)

            return audio

        except Exception as e:
            raise AudioProcessingError(f"Audio cleaning failed: {e}") from e

    def _trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """Trim leading and trailing silence."""
        import librosa

        _, intervals = librosa.effects.trim(
            audio, top_db=self.config.trim_db, frame_length=2048, hop_length=512
        )
        return audio[intervals[0] : intervals[1]]

    def clean_file(self, input_path: Path, output_path: Path) -> bool:
        """Clean a single audio file.

        Args:
            input_path: Path to input audio file.
            output_path: Path to save cleaned audio.

        Returns:
            True if successful and audio meets duration requirements.

        Raises:
            AudioProcessingError: If file processing fails.
        """
        try:
            audio, sr = sf.read(input_path)
            cleaned = self.clean(audio, sr)

            # Check duration constraints
            duration = len(cleaned) / self.config.target_sample_rate
            if duration < self.config.min_duration or duration > self.config.max_duration:
                return False

            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, cleaned, self.config.target_sample_rate)
            return True

        except Exception as e:
            raise AudioProcessingError(f"Failed to process {input_path}: {e}") from e

    @classmethod
    def from_config(cls, config_dict: dict) -> Self:
        """Create cleaner from configuration dictionary."""
        config = AudioCleanerConfig(**config_dict)
        return cls(config=config)


def batch_clean_audio(
    input_dir: Path,
    output_dir: Path,
    config: AudioCleanerConfig | None = None,
    extensions: tuple[str, ...] = (".wav", ".flac", ".mp3", ".ogg"),
    num_workers: int = 4,
    device: str = "cpu",
) -> dict[str, int]:
    """Batch clean audio files with parallel processing.

    Args:
        input_dir: Directory containing raw audio files.
        output_dir: Directory to save cleaned audio.
        config: Cleaning configuration.
        extensions: Audio file extensions to process.
        num_workers: Number of parallel workers.
        device: Device for DeepFilterNet.

    Returns:
        Statistics dict with counts of processed, skipped, failed files.
    """
    config = config or AudioCleanerConfig()
    cleaner = AudioCleaner(config=config, device=device)

    # Collect all audio files
    files: list[Path] = []
    for ext in extensions:
        files.extend(input_dir.rglob(f"*{ext}"))

    stats = {"processed": 0, "skipped": 0, "failed": 0}

    def process_file(input_path: Path) -> tuple[str, Path]:
        """Process single file and return status."""
        relative = input_path.relative_to(input_dir)
        output_path = output_dir / relative.with_suffix(".wav")

        try:
            if cleaner.clean_file(input_path, output_path):
                return "processed", input_path
            return "skipped", input_path
        except AudioProcessingError:
            return "failed", input_path

    # Process files with progress bar
    with tqdm(total=len(files), desc="Cleaning audio") as pbar:
        # Use sequential processing for DeepFilterNet (GPU-bound)
        for input_path in files:
            status, _ = process_file(input_path)
            stats[status] += 1
            pbar.update(1)

    return stats


def get_audio_info(file_path: Path) -> dict:
    """Get audio file metadata.

    Args:
        file_path: Path to audio file.

    Returns:
        Dict with sample_rate, duration, channels.
    """
    info = sf.info(file_path)
    return {
        "sample_rate": info.samplerate,
        "duration": info.duration,
        "channels": info.channels,
        "frames": info.frames,
    }


def iter_audio_files(
    directory: Path,
    extensions: tuple[str, ...] = (".wav", ".flac", ".mp3"),
) -> Iterator[Path]:
    """Iterate over audio files in directory."""
    for ext in extensions:
        yield from directory.rglob(f"*{ext}")
