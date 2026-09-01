"""Objective metrics for Mongolian TTS.

Upstream's `f5_tts/eval` supports only `zh` and `en` -- `utils_eval.py:315-318`
raises `NotImplementedError` for anything else -- so the Mongolian harness is
ours. SIM-o and UTMOS are language-agnostic and could be reused; the recogniser
cannot be.

Three things this module is careful about:

* **CER, not WER.** Mongolian is agglutinative: a single wrong suffix makes a
  whole word wrong, so WER saturates and stops discriminating.

* **The scorer has a floor.** `bayartsogt/wav2vec2-large-xlsr-mongolian` measures
  CER 0.123 median on real human speech that is correctly transcribed. Synthetic
  audio cannot beat that floor, so a raw CER means little on its own -- compare
  against `HUMAN_CER_BASELINE`, or better, against the same recogniser scoring
  the corpus's own held-out audio.

* **Both sides are normalised identically**, through the same `oron_tts.text`
  code that produced the corpus, so the metric never penalises a difference in
  written form that was never spoken.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

# Measured on 25 FLEURS test clips of real human speech with human transcripts.
# whisper-large-v3 scores 0.311 on the same clips.
HUMAN_CER_BASELINE = 0.123

ASR_MODEL = "bayartsogt/wav2vec2-large-xlsr-mongolian"
SAMPLE_RATE = 16_000


def normalize_for_scoring(text: str) -> str:
    """Casefold, strip punctuation, collapse whitespace.

    Matches `pipeline.dsp.for_comparison` in oron-cleaner, so corpus CER and
    evaluation CER are the same measurement.
    """
    text = unicodedata.normalize("NFC", text).lower().replace("\xa0", " ")
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate: edit distance over reference length.

    Implemented directly rather than via jiwer so the eval layer has no
    dependency beyond the recogniser itself.
    """
    ref, hyp = list(reference), list(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        current = [i]
        for j, hc in enumerate(hyp, 1):
            current.append(min(
                previous[j] + 1,        # deletion
                current[j - 1] + 1,     # insertion
                previous[j - 1] + (rc != hc),  # substitution
            ))
        previous = current
    return previous[-1] / len(ref)


def bandwidth_hz(audio, sr: int, drop_db: float = 40.0) -> float:
    """Highest frequency within `drop_db` of the spectral peak.

    Reported because output bandwidth follows the reference clip, and no
    Mongolian source is full-band: this is how you tell whether a voice came out
    as dull as its prompt.
    """
    import librosa
    import numpy as np

    n = min(len(audio), sr * 5)
    if n < sr // 2:
        return 0.0
    spec = np.abs(librosa.stft(audio[:n], n_fft=2048)) ** 2
    power = np.maximum(spec.mean(axis=1), 1e-20)
    db = 10.0 * np.log10(power / power.max())
    above = np.where(db > -drop_db)[0]
    if above.size == 0:
        return 0.0
    return float(np.fft.rfftfreq(2048, 1 / sr)[above[-1]])


@dataclass
class Scores:
    cer: float
    utmos: float | None = None
    sim_o: float | None = None
    bandwidth_hz: float | None = None

    def __str__(self) -> str:
        parts = [f"CER {self.cer:.3f}"]
        if self.utmos is not None:
            parts.append(f"UTMOS {self.utmos:.2f}")
        if self.sim_o is not None:
            parts.append(f"SIM-o {self.sim_o:.3f}")
        if self.bandwidth_hz is not None:
            parts.append(f"BW {self.bandwidth_hz:.0f} Hz")
        return "  ".join(parts)


class MongolianASR:
    """Transcribes Mongolian for CER scoring. Loads the model once."""

    def __init__(self, device: str = "cpu", model_name: str = ASR_MODEL) -> None:
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        self._torch = torch
        self.device = device
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModelForCTC.from_pretrained(model_name).to(device).eval()

    def transcribe(self, audio, sr: int = SAMPLE_RATE) -> str:
        if sr != SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        inputs = self._processor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).to(self.device)
        with self._torch.no_grad():
            logits = self._model(**inputs).logits
        return self._processor.batch_decode(logits.argmax(-1))[0]

    def score(self, audio, reference: str, sr: int = SAMPLE_RATE) -> float:
        return cer(
            normalize_for_scoring(reference),
            normalize_for_scoring(self.transcribe(audio, sr)),
        )


@lru_cache(maxsize=1)
def _utmos():
    """UTMOS naturalness predictor. Language-agnostic, so reused as-is."""
    import torch

    return torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)


def utmos(audio, sr: int) -> float:
    import torch

    predictor = _utmos()
    wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return float(predictor(wav, sr))
