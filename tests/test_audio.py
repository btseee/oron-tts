"""Tests for audio processing."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from orontts.preprocessing.audio import AudioCleanerConfig, get_audio_info


class TestAudioCleanerConfig:
    """Tests for AudioCleanerConfig."""

    def test_default_config(self) -> None:
        config = AudioCleanerConfig()
        assert config.target_sample_rate == 22050
        assert config.normalize is True
        assert config.trim_silence is True

    def test_custom_config(self) -> None:
        config = AudioCleanerConfig(
            target_sample_rate=44100,
            min_duration=1.0,
            max_duration=10.0,
        )
        assert config.target_sample_rate == 44100
        assert config.min_duration == 1.0
        assert config.max_duration == 10.0


class TestGetAudioInfo:
    """Tests for get_audio_info function."""

    def test_get_audio_info(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test audio file
            path = Path(tmpdir) / "test.wav"
            audio = np.random.randn(22050).astype(np.float32)
            sf.write(path, audio, 22050)

            info = get_audio_info(path)
            assert info["sample_rate"] == 22050
            assert info["channels"] == 1
            assert abs(info["duration"] - 1.0) < 0.01
            assert info["frames"] == 22050
