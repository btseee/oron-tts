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

import os
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Measured on 25 FLEURS test clips of real human speech with human transcripts.
# whisper-large-v3 scores 0.311 on the same clips.
HUMAN_CER_BASELINE = 0.123

ASR_MODEL = "bayartsogt/wav2vec2-large-xlsr-mongolian"
SAMPLE_RATE = 16_000

# The paper's SIM-o uses WavLM-large fine-tuned for speaker verification. It is
# a manual download (upstream's eval README links it), so the path is
# configurable and its absence is reported rather than silently substituted.
DEFAULT_WAVLM_CKPT = "ckpts/wavlm_large_finetune.pth"


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


class SimOUnavailable(RuntimeError):
    """Raised when the speaker-verification checkpoint is not on disk.

    Deliberately an error rather than a fallback to some other model. SIM-o is
    a number a reader compares against the paper's tables (0.66 for F5-TTS on
    LibriSpeech-PC test-clean), and it is only comparable if it comes from the
    same WavLM-large verification model. A silent substitution would produce a
    number on a different scale wearing the same name.
    """


@lru_cache(maxsize=1)
def _speaker_encoder(checkpoint: str, device: str):
    """Upstream's WavLM-large ECAPA-TDNN, as used for the paper's SIM-o."""
    import torch
    from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL

    path = Path(checkpoint)
    if not path.exists():
        raise SimOUnavailable(
            f"WavLM speaker-verification checkpoint not found at {path}.\n"
            "Download wavlm_large_finetune.pth (see F5-TTS/src/f5_tts/eval/"
            "README.md, 'Download Evaluation Model Checkpoints') and pass\n"
            "    --sim-checkpoint <path>   or set ORON_WAVLM_CKPT."
        )
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state = torch.load(path, weights_only=True, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)
    return model.to(device).eval()


def sim_o(generated, reference, sr: int, ref_sr: int | None = None, *,
          checkpoint: str | None = None, device: str = "cpu") -> float:
    """Speaker similarity between generated audio and its reference prompt.

    The system's whole proposition is that voice identity transfers from a ~10 s
    reference clip. Nothing measured whether it does: `Scores.sim_o` was declared
    and rendered but never computed, so a model that produced fluent Mongolian in
    the wrong voice scored exactly as well as one that did not.

    Cosine similarity in (-1, 1) between WavLM-large ECAPA-TDNN embeddings, at
    16 kHz, matching `f5_tts.eval.utils_eval.run_sim` so the number is on the
    same scale as the paper's.
    """
    import torch
    import torch.nn.functional as F

    ckpt = checkpoint or os.environ.get("ORON_WAVLM_CKPT", DEFAULT_WAVLM_CKPT)
    model = _speaker_encoder(str(ckpt), device)

    def embed(audio, rate: int):
        wav = torch.as_tensor(audio, dtype=torch.float32)
        if wav.ndim == 2:                      # soundfile gives (n, channels)
            wav = wav.mean(dim=1)
        wav = wav.reshape(1, -1)
        # The two clips are resampled independently: the generated audio is
        # 24 kHz from Vocos and the prompt is whatever the corpus stored.
        if rate != SAMPLE_RATE:
            import torchaudio

            wav = torchaudio.transforms.Resample(rate, SAMPLE_RATE)(wav)
        with torch.no_grad():
            return model(wav.to(device))

    a = embed(generated, sr)
    b = embed(reference, ref_sr if ref_sr is not None else sr)
    return float(F.cosine_similarity(a, b)[0].item())


ECAPA_VOXCELEB = "speechbrain/spkrec-ecapa-voxceleb"

# Calibrated on this project's own corpora: 17 genuine same-speaker pairs (12
# Common Voice, 5 MBSpeech) against 30 different-speaker pairs, all real
# recordings, cosine on ECAPA-VoxCeleb embeddings.
#
#     same speaker       median 0.714   range 0.540 .. 0.833
#     different speaker  median 0.208   range 0.034 .. 0.503
#
# The two ranges do not overlap, so 0.52 separates them with margin on both
# sides. `microsoft/wavlm-base-plus-sv` was tried first and rejected: it put the
# same pairs at 0.964 against 0.886, an overlap wide enough to call two
# strangers the same person.
SAME_SPEAKER_MIN = 0.52


@lru_cache(maxsize=1)
def _ecapa_voxceleb(device: str):
    from speechbrain.inference.speaker import EncoderClassifier

    return EncoderClassifier.from_hparams(
        source=ECAPA_VOXCELEB, savedir="ckpts/ecapa", run_opts={"device": device})


def speaker_similarity(generated, reference, sr: int, ref_sr: int | None = None,
                       *, device: str = "cpu") -> float:
    """Speaker similarity that does not depend on a checkpoint nobody has.

    `sim_o` reproduces the paper's number, and should be preferred when it can
    run. It needs two things a fresh machine does not have: the 1.3 GB
    `wavlm_large_finetune.pth`, and s3prl fetched through `torch.hub`, which
    prompts for trust and so hangs or fails unattended. On the run that produced
    this model both were missing, and the consequence was not a bad number but
    no number: the voice lock is the mechanism that makes `--voice female`
    return the same person every time, and nothing measured whether it worked.

    This uses ECAPA-TDNN trained on VoxCeleb, which pip-installs and
    self-downloads. The scale is its own -- do not compare it to the paper's
    SIM-o -- so `SAME_SPEAKER_MIN` above is calibrated against real pairs from
    these corpora rather than borrowed from a leaderboard.
    """
    import torch
    import torchaudio

    encoder = _ecapa_voxceleb(device)

    def embed(audio, rate: int):
        import numpy as np

        wav = np.asarray(audio, dtype="float32")
        if wav.ndim == 2:                      # soundfile gives (n, channels)
            wav = wav.mean(axis=1)
        tensor = torch.as_tensor(wav)
        if rate != SAMPLE_RATE:
            tensor = torchaudio.transforms.Resample(rate, SAMPLE_RATE)(tensor)
        with torch.no_grad():
            return encoder.encode_batch(tensor.unsqueeze(0)).squeeze()

    a = embed(generated, sr)
    b = embed(reference, ref_sr if ref_sr is not None else sr)
    import torch.nn.functional as F
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))[0].item())


def speaker_similarity_any(generated, reference, sr: int, ref_sr: int | None = None,
                           *, checkpoint: str | None = None,
                           device: str = "cpu") -> tuple[float, str]:
    """The paper's SIM-o where possible, a measured number always.

    Returns the value and which metric produced it, because the two are on
    different scales and a card that does not say which is worse than no card.
    """
    try:
        return sim_o(generated, reference, sr, ref_sr,
                     checkpoint=checkpoint, device=device), "sim_o"
    except Exception:
        return speaker_similarity(generated, reference, sr, ref_sr,
                                  device=device), "ecapa_voxceleb"


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
