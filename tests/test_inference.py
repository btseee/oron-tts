"""Tests for inference module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from orontts.inference.audio_output import AudioOutput


class TestAudioOutput:
    """Tests for AudioOutput class."""

    def test_creation(self) -> None:
        audio = np.random.randn(22050).astype(np.float32)
        output = AudioOutput(audio=audio, sample_rate=22050)
        assert output.duration == pytest.approx(1.0, rel=0.01)
        assert len(output) == 22050

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio = np.random.randn(22050).astype(np.float32)
            output = AudioOutput(audio=audio, sample_rate=22050)

            path = output.save(Path(tmpdir) / "test.wav")
            assert path.exists()

            # Load and verify
            import soundfile as sf

            loaded, sr = sf.read(path)
            assert sr == 22050
            assert len(loaded) == 22050

    def test_normalize(self) -> None:
        audio = np.random.randn(22050).astype(np.float32) * 0.1
        output = AudioOutput(audio=audio, sample_rate=22050)

        normalized = output.normalize(peak=0.95)
        assert np.abs(normalized.numpy).max() <= 0.95 + 1e-6

    def test_resample(self) -> None:
        audio = np.random.randn(22050).astype(np.float32)
        output = AudioOutput(audio=audio, sample_rate=22050)

        resampled = output.resample(44100)
        assert resampled.sample_rate == 44100
        assert len(resampled) == pytest.approx(44100, rel=0.1)

    def test_to_wav_bytes(self) -> None:
        audio = np.random.randn(22050).astype(np.float32)
        output = AudioOutput(audio=audio, sample_rate=22050)

        wav_bytes = output.to_wav_bytes()
        assert isinstance(wav_bytes, bytes)
        assert len(wav_bytes) > 0
        # WAV header starts with "RIFF"
        assert wav_bytes[:4] == b"RIFF"

    def test_repr(self) -> None:
        audio = np.random.randn(22050).astype(np.float32)
        output = AudioOutput(audio=audio, sample_rate=22050)

        repr_str = repr(output)
        assert "AudioOutput" in repr_str
        assert "1.00s" in repr_str or "duration=" in repr_str
