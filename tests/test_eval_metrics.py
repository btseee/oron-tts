"""Evaluation metrics.

These decide which checkpoint ships, so the edit-distance and normalisation
behaviour is pinned. The model-backed parts are exercised separately; nothing
here loads a recogniser.
"""

import importlib.util

import numpy as np
import pytest

from oron_tts.eval.metrics import (
    HUMAN_CER_BASELINE,
    Scores,
    bandwidth_hz,
    cer,
    normalize_for_scoring,
)

SR = 16_000


def _tone(freqs, seconds=2.0, sr=SR, amp=0.2):
    t = np.arange(int(seconds * sr)) / sr
    return np.sum([amp * np.sin(2 * np.pi * f * t) for f in freqs], axis=0).astype(np.float32)


# ── CER ───────────────────────────────────────────────────────────────────────

def test_identical_text_scores_zero():
    assert cer("сайн байна уу", "сайн байна уу") == 0.0


def test_single_substitution():
    assert cer("сайн", "сайм") == pytest.approx(0.25)


def test_deletion_and_insertion():
    assert cer("сайн", "сай") == pytest.approx(0.25)
    assert cer("сайн", "сайнн") == pytest.approx(0.25)


def test_completely_different_text_scores_high():
    assert cer("сайн байна", "огт өөр") > 0.7


def test_empty_reference():
    assert cer("", "") == 0.0
    assert cer("", "текст") == 1.0


def test_empty_hypothesis_is_total_loss():
    """A model that emits silence must not look good."""
    assert cer("сайн байна", "") == 1.0


def test_cer_is_length_normalised():
    """Otherwise long sentences dominate the average."""
    assert cer("аб", "ав") == pytest.approx(cer("абабабабаб", "авабабабаб") * 5)


# ── normalisation ─────────────────────────────────────────────────────────────

def test_normalisation_strips_case_and_punctuation():
    assert normalize_for_scoring("Сайн, байна уу?") == "сайн байна уу"


def test_normalisation_preserves_mongolian_vowels():
    """ө and ү must survive; losing them would silently deflate CER."""
    assert normalize_for_scoring("Өнөөдөр үүлшинэ") == "өнөөдөр үүлшинэ"


def test_normalisation_collapses_whitespace_and_nbsp():
    assert normalize_for_scoring("сайн\xa0\xa0 байна") == "сайн байна"


def test_punctuation_difference_is_not_penalised():
    """It was never spoken, so it must not count as an error."""
    assert cer(normalize_for_scoring("Сайн байна уу?"),
               normalize_for_scoring("сайн байна уу")) == 0.0


# ── bandwidth ─────────────────────────────────────────────────────────────────

def test_bandwidth_recovers_a_known_cutoff():
    measured = bandwidth_hz(_tone([200, 900, 2400, 4800]), SR)
    assert 4300 <= measured <= 5600, measured


def test_bandwidth_separates_dull_from_bright():
    """The check that catches a voice inheriting a dull reference clip."""
    dull = bandwidth_hz(_tone([200, 900, 2400]), SR)
    bright = bandwidth_hz(_tone([200, 900, 2400, 7000]), SR)
    assert bright > dull + 2000


def test_bandwidth_of_too_short_audio_is_zero():
    assert bandwidth_hz(np.zeros(100, dtype=np.float32), SR) == 0.0


# ── baseline and reporting ────────────────────────────────────────────────────

def test_human_baseline_is_the_measured_value():
    """Measured on real human speech with human transcripts.

    A synthetic CER means nothing against zero -- the recogniser cannot do
    better than this on audio that is already correct.
    """
    assert pytest.approx(0.123, abs=0.001) == HUMAN_CER_BASELINE


def test_scores_render_only_what_was_measured():
    assert str(Scores(cer=0.2)) == "CER 0.200"
    rendered = str(Scores(cer=0.2, utmos=3.4, bandwidth_hz=7600.0))
    assert "UTMOS 3.40" in rendered and "BW 7600 Hz" in rendered
    assert "SIM-o" not in rendered


# ── checkpoint sweep ordering ─────────────────────────────────────────────────

def test_checkpoints_sort_by_update_number_not_lexically():
    """model_10000 must not come before model_2000.

    A sweep exists to show quality against training progress; out-of-order
    reporting makes the trend unreadable and the "best" line untrustworthy.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from eval_mn import sort_checkpoints

    paths = [Path(f"model_{n}.pt") for n in (10000, 2000, 500, 120000, 40000)]
    assert [p.stem for p in sort_checkpoints(paths)] == [
        "model_500", "model_2000", "model_10000", "model_40000", "model_120000",
    ]


def test_checkpoint_sort_tolerates_unnumbered_names():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from eval_mn import sort_checkpoints

    paths = [Path("model_2000.pt"), Path("model_last.pt")]
    assert sort_checkpoints(paths)[0].stem == "model_last"


# ── SIM-o ─────────────────────────────────────────────────────────────────────

# The similarity maths is torch's, even with the encoder stubbed. Skipping keeps
# the pure-text and CER tests -- the majority -- runnable without the model stack.
requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)



class _Echo:
    """Stand-in speaker encoder: embeds a waveform as its own coarse spectrum.

    Enough structure that identical audio scores 1.0 and different audio does
    not, without downloading WavLM-large.
    """

    def __call__(self, wav):
        import torch

        n = wav.shape[-1] // 8 * 8
        return torch.abs(torch.fft.rfft(wav[..., :n]))[..., :64]


def _patch_encoder(monkeypatch):
    from oron_tts.eval import metrics

    metrics._speaker_encoder.cache_clear()
    monkeypatch.setattr(metrics, "_speaker_encoder", lambda ckpt, device: _Echo())


@requires_torch
def test_sim_o_of_a_clip_with_itself_is_one(monkeypatch):
    import numpy as np

    from oron_tts.eval import sim_o

    _patch_encoder(monkeypatch)
    rng = np.random.default_rng(0)
    wav = rng.standard_normal(24000).astype("float32")
    assert sim_o(wav, wav, 24000) == pytest.approx(1.0, abs=1e-5)


@requires_torch
def test_sim_o_separates_two_different_signals(monkeypatch):
    import numpy as np

    from oron_tts.eval import sim_o

    _patch_encoder(monkeypatch)
    t = np.arange(24000) / 24000
    a = np.sin(2 * np.pi * 120 * t).astype("float32")
    b = np.sin(2 * np.pi * 240 * t).astype("float32")
    assert sim_o(a, b, 24000) < sim_o(a, a, 24000)


@requires_torch
def test_sim_o_accepts_two_different_sample_rates(monkeypatch):
    """The generated audio is 24 kHz from Vocos; the prompt is whatever the
    corpus stored. Upstream resamples each independently."""
    import numpy as np

    from oron_tts.eval import sim_o

    _patch_encoder(monkeypatch)
    gen = np.zeros(24000, dtype="float32")
    ref = np.zeros(16000, dtype="float32")
    assert isinstance(sim_o(gen, ref, 24000, 16000), float)


@requires_torch
def test_sim_o_mono_ises_a_stereo_prompt(monkeypatch):
    """soundfile returns (n, channels) for a stereo file."""
    import numpy as np

    from oron_tts.eval import sim_o

    _patch_encoder(monkeypatch)
    mono = np.random.default_rng(1).standard_normal(16000).astype("float32")
    stereo = np.stack([mono, mono], axis=1)
    assert sim_o(mono, stereo, 16000) == pytest.approx(1.0, abs=1e-5)


@requires_torch
def test_a_missing_checkpoint_is_an_error_not_a_substitute():
    """SIM-o is compared against the paper's tables, so it is only meaningful
    from the same WavLM-large model. Quietly using another would put a number
    on a different scale under the same name."""
    import numpy as np

    from oron_tts.eval import SimOUnavailable, metrics, sim_o

    metrics._speaker_encoder.cache_clear()
    with pytest.raises(SimOUnavailable) as exc:
        sim_o(np.zeros(100, "float32"), np.zeros(100, "float32"), 16000,
              checkpoint="does/not/exist.pth")
    assert "wavlm_large_finetune.pth" in str(exc.value)
