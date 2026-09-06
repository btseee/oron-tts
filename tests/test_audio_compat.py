"""The torchaudio shim must match torchaudio's contract, not merely return audio.

The failure this guards against is silent. F5-TTS's `preprocess_ref_audio_text`
indexes `wav[0]`, so a samples-first array does not raise -- it takes the first
*sample* as if it were a channel and synthesises from a one-sample reference.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oron_tts.audio_compat import _load_with_soundfile  # noqa: E402


def _write(tmp_path: Path, data: np.ndarray, sr: int = 24_000) -> Path:
    path = tmp_path / "clip.wav"
    sf.write(path, data, sr, subtype="FLOAT")
    return path


def test_mono_comes_back_channels_first(tmp_path):
    """(1, samples), not (samples,) and not (samples, 1)."""
    audio = np.sin(np.linspace(0, 40, 4_000)).astype("float32")
    tensor, sr = _load_with_soundfile(_write(tmp_path, audio))
    assert sr == 24_000
    assert tensor.shape == (1, 4_000), tensor.shape
    assert np.allclose(tensor[0].numpy(), audio, atol=1e-6)


def test_stereo_keeps_both_channels_and_their_order(tmp_path):
    """A transposed load would return 2 samples rather than 2 channels."""
    left = np.linspace(-0.5, 0.5, 3_000).astype("float32")
    right = np.linspace(0.5, -0.5, 3_000).astype("float32")
    tensor, _ = _write(tmp_path, np.stack([left, right], axis=1)), None
    tensor, sr = _load_with_soundfile(tmp_path / "clip.wav")
    assert tensor.shape == (2, 3_000), tensor.shape
    assert np.allclose(tensor[0].numpy(), left, atol=1e-6)
    assert np.allclose(tensor[1].numpy(), right, atol=1e-6)


def test_the_first_index_is_a_channel_not_a_sample(tmp_path):
    """The exact mistake the shim exists to avoid, stated as an assertion.

    `wav[0]` must be the whole first channel. Under a samples-first array it
    would be a length-1 vector, which every downstream stage accepts silently.
    """
    audio = np.random.default_rng(0).standard_normal(2_048).astype("float32") * 0.1
    tensor, _ = _load_with_soundfile(_write(tmp_path, audio))
    assert tensor[0].numel() == 2_048, (
        f"wav[0] has {tensor[0].numel()} samples; a reference prompt of that "
        "length synthesises silence without raising"
    )


def test_dtype_is_float32(tmp_path):
    """PCM_16 on disk must still arrive as normalised float32."""
    path = tmp_path / "clip.wav"
    sf.write(path, (np.ones(512) * 0.5).astype("float32"), 16_000, subtype="PCM_16")
    tensor, _ = _load_with_soundfile(path)
    assert tensor.dtype.is_floating_point
    assert abs(float(tensor.max()) - 0.5) < 1e-3


def test_install_is_a_noop_when_torchcodec_works(monkeypatch):
    """A working training box must keep torchaudio's own loader."""
    import oron_tts.audio_compat as compat

    monkeypatch.setattr(compat, "_INSTALLED", False)
    monkeypatch.setattr(compat, "_torchcodec_works", lambda: True)
    torchaudio = pytest.importorskip("torchaudio")
    original = torchaudio.load
    assert compat.install() is False
    assert torchaudio.load is original


def test_install_replaces_the_loader_when_torchcodec_is_broken(monkeypatch):
    import oron_tts.audio_compat as compat

    monkeypatch.setattr(compat, "_INSTALLED", False)
    monkeypatch.setattr(compat, "_torchcodec_works", lambda: False)
    torchaudio = pytest.importorskip("torchaudio")
    original = torchaudio.load
    try:
        assert compat.install() is True
        assert torchaudio.load is compat._load_with_soundfile
    finally:
        torchaudio.load = original
        compat._INSTALLED = False


def test_detection_probes_a_decode_rather_than_an_import(monkeypatch):
    """`import torchcodec` succeeds on a broken install; the libraries open lazily.

    So detection has to attempt a real decode. Simulated by making
    `torchaudio.load` raise the way a missing FFmpeg library does.
    """
    import oron_tts.audio_compat as compat

    torchaudio = pytest.importorskip("torchaudio")
    original = torchaudio.load

    def boom(*_a, **_k):
        raise OSError("Could not load this library: libtorchcodec_core4.dll")

    try:
        torchaudio.load = boom
        assert compat._torchcodec_works() is False
    finally:
        torchaudio.load = original


def test_a_file_soundfile_cannot_read_falls_back_to_ffmpeg(tmp_path, monkeypatch):
    """mp3/opus in a soundfile build without them must still load."""
    import subprocess

    import oron_tts.audio_compat as compat

    audio = np.linspace(-0.2, 0.2, 1_600).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, audio, 16_000, format="WAV", subtype="FLOAT")
    calls = {"n": 0}

    def fake_run(cmd, **kwargs):
        calls["n"] += 1
        assert "-ar" not in cmd, "must not pin the source rate"
        return subprocess.CompletedProcess(cmd, 0, stdout=buf.getvalue(), stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    tensor, sr = compat._load_with_soundfile(tmp_path / "missing.mp3")
    assert calls["n"] == 1
    assert sr == 16_000 and tensor.shape == (1, 1_600)
