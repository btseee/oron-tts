"""Speaker similarity must produce a number, not an excuse.

The run that trained this model measured no speaker similarity at all: `sim_o`
needs a 1.3 GB checkpoint and an s3prl fetch through `torch.hub` that prompts
for trust, and on a fresh unattended machine neither was there. The voice lock
is the whole mechanism behind "the same voice every time", so the one property
the design rests on went unmeasured.
"""
from __future__ import annotations

import numpy as np
import pytest

from oron_tts.eval import SAME_SPEAKER_MIN
from oron_tts.eval.metrics import SimOUnavailable


def test_falls_back_when_sim_o_cannot_run(monkeypatch):
    """A missing WavLM checkpoint must not turn into a missing measurement."""
    import oron_tts.eval.metrics as m

    def no_wavlm(*a, **k):
        raise SimOUnavailable("checkpoint not found")

    monkeypatch.setattr(m, "sim_o", no_wavlm)
    monkeypatch.setattr(m, "speaker_similarity", lambda *a, **k: 0.73)
    value, method = m.speaker_similarity_any(np.zeros(16000, "float32"),
                                             np.zeros(16000, "float32"), 16000)
    assert value == 0.73
    assert method == "ecapa_voxceleb", "the card must be able to say which scale this is"


def test_prefers_the_paper_metric_when_available(monkeypatch):
    import oron_tts.eval.metrics as m

    monkeypatch.setattr(m, "sim_o", lambda *a, **k: 0.61)
    monkeypatch.setattr(m, "speaker_similarity",
                        lambda *a, **k: pytest.fail("fell back needlessly"))
    assert m.speaker_similarity_any(np.zeros(16000, "float32"),
                                    np.zeros(16000, "float32"), 16000) == (0.61, "sim_o")


def test_threshold_sits_between_the_measured_populations():
    """0.52 is calibrated, not chosen: real same-speaker pairs from these
    corpora ran 0.540-0.833 and different-speaker pairs 0.034-0.503."""
    assert 0.503 < SAME_SPEAKER_MIN < 0.540
