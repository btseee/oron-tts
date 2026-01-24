"""Tests for model configuration."""

import json
import tempfile
from pathlib import Path

import pytest

from orontts.model.config import VITS2Config, load_config


class TestVITS2Config:
    """Tests for VITS2Config."""

    def test_default_config(self) -> None:
        config = VITS2Config()
        assert config.model_name == "vits2_mongolian"
        assert config.n_speakers == 1
        assert config.audio.sample_rate == 22050

    def test_light_preset(self) -> None:
        config = VITS2Config.from_preset("light")
        assert config.model_name == "vits2_mongolian_light"
        assert config.text_encoder.hidden_channels == 128
        assert config.training.batch_size == 32

    def test_hq_preset(self) -> None:
        config = VITS2Config.from_preset("hq")
        assert config.model_name == "vits2_mongolian_hq"
        assert config.text_encoder.hidden_channels == 256
        assert config.training.batch_size == 8

    def test_save_and_load(self) -> None:
        config = VITS2Config.from_preset("light")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            config.save(path)

            loaded = VITS2Config.load(path)
            assert loaded.model_name == config.model_name
            assert loaded.text_encoder.hidden_channels == config.text_encoder.hidden_channels

    def test_load_from_dict(self) -> None:
        data = {
            "model_name": "test_model",
            "n_speakers": 5,
            "audio": {"sample_rate": 44100},
        }
        config = VITS2Config.model_validate(data)
        assert config.model_name == "test_model"
        assert config.n_speakers == 5
        assert config.audio.sample_rate == 44100


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            config = VITS2Config()
            config.save(path)

            loaded = load_config(path)
            assert isinstance(loaded, VITS2Config)

    def test_load_nonexistent_raises(self) -> None:
        from orontts.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            load_config(Path("/nonexistent/path.json"))
