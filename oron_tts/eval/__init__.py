"""Objective evaluation for Mongolian TTS.

Kept out of `oron_tts.text`, which is dependency-free by design: importing
anything here pulls torch and transformers.
"""

from oron_tts.eval.metrics import (
    ASR_MODEL,
    HUMAN_CER_BASELINE,
    MongolianASR,
    Scores,
    SimOUnavailable,
    bandwidth_hz,
    cer,
    normalize_for_scoring,
    SAME_SPEAKER_MIN,
    sim_o,
    speaker_similarity,
    speaker_similarity_any,
    utmos,
)

__all__ = [
    "ASR_MODEL",
    "HUMAN_CER_BASELINE",
    "MongolianASR",
    "Scores",
    "SimOUnavailable",
    "bandwidth_hz",
    "cer",
    "normalize_for_scoring",
    "SAME_SPEAKER_MIN",
    "sim_o",
    "speaker_similarity",
    "speaker_similarity_any",
    "utmos",
]
