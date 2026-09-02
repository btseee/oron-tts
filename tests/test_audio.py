"""The mel contract with Vocos.

`oron_tts/audio.py` was never exercised by a test, and its docstring claims the
mel "matches Vocos vocoder exactly". If it does not, generated mels decode to
noise -- a total output failure with no earlier symptom, since every other stage
would look healthy.

These tests check that claim against upstream's own implementation rather than
against a restatement of it.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

F5_SRC = Path(__file__).resolve().parents[2] / "F5-TTS" / "src"
if F5_SRC.exists() and str(F5_SRC) not in sys.path:
    sys.path.insert(0, str(F5_SRC))

upstream = pytest.importorskip(
    "f5_tts.model.modules", reason="F5-TTS checkout not beside this repo"
)

from oron_tts.audio import (  # noqa: E402
    DEFAULT_HOP_LENGTH,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_WIN_LENGTH,
    AudioProcessor,
)


def _wave(seconds=1.0, seed=0):
    torch.manual_seed(seed)
    return torch.randn(1, int(seconds * DEFAULT_SAMPLE_RATE)) * 0.1


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_our_mel_is_bit_identical_to_upstreams(seed):
    """The claim the docstring makes, checked against the code it names.

    A mismatch here is not a quality regression, it is silence or noise out of
    the vocoder.
    """
    wav = _wave(seed=seed)
    ours = AudioProcessor().mel_spectrogram(wav.squeeze(0))
    theirs = upstream.get_vocos_mel_spectrogram(wav).squeeze(0)
    assert ours.shape == theirs.shape
    assert torch.equal(ours, theirs)


def test_the_parameters_are_the_ones_upstream_defaults_to():
    """Read off `get_vocos_mel_spectrogram`'s signature, not off a comment."""
    import inspect

    defaults = {
        k: v.default
        for k, v in inspect.signature(upstream.get_vocos_mel_spectrogram).parameters.items()
    }
    assert defaults["n_fft"] == DEFAULT_N_FFT
    assert defaults["hop_length"] == DEFAULT_HOP_LENGTH
    assert defaults["win_length"] == DEFAULT_WIN_LENGTH
    assert defaults["n_mel_channels"] == DEFAULT_N_MELS
    assert defaults["target_sample_rate"] == DEFAULT_SAMPLE_RATE


def test_a_one_dimensional_waveform_is_accepted():
    wav = _wave()
    assert torch.equal(
        AudioProcessor().mel_spectrogram(wav.squeeze(0)),
        AudioProcessor().mel_spectrogram(wav),
    )


def test_the_frame_count_follows_the_hop_length():
    """center=True, so frames = 1 + samples // hop."""
    samples = DEFAULT_SAMPLE_RATE
    mel = AudioProcessor().mel_spectrogram(torch.zeros(samples))
    assert mel.shape == (DEFAULT_N_MELS, 1 + samples // DEFAULT_HOP_LENGTH)


def test_silence_is_clamped_rather_than_negative_infinity():
    """Vocos' safe_log clips at 1e-5; without it silent frames are -inf and the
    loss becomes NaN on the first batch containing one."""
    mel = AudioProcessor().mel_spectrogram(torch.zeros(DEFAULT_SAMPLE_RATE))
    assert torch.isfinite(mel).all()
    assert torch.allclose(mel, torch.full_like(mel, float(torch.log(torch.tensor(1e-5)))))


def test_normalisation_leaves_silence_alone():
    """Dividing by a near-zero peak would amplify the noise floor to full scale."""
    silent = torch.zeros(1000)
    assert torch.equal(AudioProcessor().normalize_audio(silent), silent)


def test_normalisation_reaches_full_scale_without_clipping():
    audio = torch.randn(1000) * 0.01
    out = AudioProcessor().normalize_audio(audio)
    assert out.abs().max() == pytest.approx(1.0, abs=1e-4)
    assert out.abs().max() <= 1.0
