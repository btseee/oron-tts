"""Objective evaluation for Mongolian TTS.

Kept out of `oron_tts.text`, which is dependency-free by design: importing
anything here pulls torch and transformers.
"""

from oron_tts.eval.metrics import (
    HUMAN_CER_BASELINE,
    MongolianASR,
    Scores,
    SimOUnavailable,
    bandwidth_hz,
    cer,
    normalize_for_scoring,
    sim_o,
    utmos,
)

__all__ = [
    "HUMAN_CER_BASELINE",
    "MongolianASR",
    "Scores",
    "SimOUnavailable",
    "bandwidth_hz",
    "cer",
    "normalize_for_scoring",
    "sim_o",
    "utmos",
]
