"""Build the model card from the measurements, so it cannot drift from them.

The card that shipped had seven frontmatter keys, no `model-index`, and no
`datasets`. The Hub therefore rendered no Eval Results panel and no links to the
corpora, and every measured number lived in prose or in `eval.json`, where the
Hub cannot read it.

Its `license` was also wrong in the expensive direction: `cc-by-nc-4.0`,
inherited from WorldSpeech -- a corpus that failed the 15% pass-rate gate and was
never trained on. The model saw MBSpeech (MIT), FLEURS (CC-BY-4.0) and Common
Voice (CC0), all commercial-safe. Commercial safety is why WorldSpeech was
excluded in the first place.

    python scripts/model_card.py --eval eval.json \\
        --consistency consistency.json --out README.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = "btsee/oron-tts"
DATASETS = ["btsee/mbspeech-mn", "btsee/fleurs-mn", "btsee/common-voice-26-mn"]
FINAL_STAGE = "cv"        # the last stage with a scored sweep


def best_checkpoint(stage_eval: dict) -> tuple[str, dict]:
    """The checkpoint with the lowest mean CER across genders."""
    scored = []
    for name, per_gender in stage_eval.items():
        if not isinstance(per_gender, dict):
            continue
        cers = [m["cer_median"] for m in per_gender.values()
                if isinstance(m, dict) and isinstance(m.get("cer_median"), (int, float))]
        if cers:
            scored.append((statistics.fmean(cers), name, per_gender))
    if not scored:
        raise ValueError("no scored checkpoints in this stage")
    scored.sort(key=lambda row: row[0])
    return scored[0][1], scored[0][2]


def frontmatter(evals: dict, consistency: dict, best: dict | None = None) -> dict:
    if best is None:
        _, best = best_checkpoint(evals[FINAL_STAGE])
    measured = consistency.get("measured", {})

    def metric(kind: str, value, name: str) -> dict:
        # Unrounded: this is machine-readable metadata parsed by the Hub, not
        # prose. Rounding here would let it diverge from the body's own
        # human-formatted rendering of the same measurement.
        return {"type": kind, "value": float(value), "name": name}

    results = [metric("cer", best["male"]["cer_median"], "CER, male voice"),
               metric("cer", best["female"]["cer_median"], "CER, female voice"),
               metric("utmos", best["male"]["utmos_mean"], "UTMOS, male voice"),
               metric("utmos", best["female"]["utmos_mean"], "UTMOS, female voice")]
    for key, name in (("male_demo_vs_male_prompt", "Speaker similarity, male"),
                      ("female_demo_vs_female_prompt", "Speaker similarity, female")):
        if isinstance(measured.get(key), (int, float)):
            results.append(metric("cosine_similarity", measured[key], name))

    return {
        "language": ["mn"],
        # Not cc-by-nc-4.0: WorldSpeech failed the gate and was never trained on.
        # FLEURS at CC-BY-4.0 is the most restrictive source that WAS used.
        "license": "cc-by-4.0",
        "library_name": "f5-tts",
        "pipeline_tag": "text-to-speech",
        "base_model": "SWivid/F5-TTS",
        "base_model_relation": "finetune",
        "datasets": list(DATASETS),
        "metrics": ["cer", "utmos"],
        "tags": ["text-to-speech", "tts", "mongolian", "khalkha", "cyrillic",
                 "flow-matching", "f5-tts", "dit", "vocos", "voice-cloning"],
        "model-index": [{
            "name": "oron-tts",
            "results": [{
                "task": {"type": "text-to-speech", "name": "Text-to-Speech"},
                "dataset": {"type": "btsee/common-voice-26-mn",
                            "name": "Common Voice 26 Mongolian (cleaned)",
                            "split": "withheld"},
                "metrics": results,
            }],
        }],
        # `new_version` is deliberately absent: it declares a successor repo and
        # the Hub banners every visitor to it. No successor exists.
    }


BODY = """
# OronTTS — Mongolian text to speech

Speaks Mongolian (Khalkha, Cyrillic) in two fixed voices, one male and one
female. A finetune of F5-TTS on cleaned Mongolian speech from three public
corpora.

## Listen

Male:

<audio controls src="https://huggingface.co/{repo}/resolve/main/demos/male.wav"></audio>

Female:

<audio controls src="https://huggingface.co/{repo}/resolve/main/demos/female.wav"></audio>

## Install

```bash
pip install git+https://github.com/SWivid/F5-TTS.git
pip install git+https://github.com/btseee/oron-tts.git
```

## Use

```python
import soundfile as sf
from huggingface_hub import hf_hub_download
from f5_tts.api import F5TTS
from oron_tts.text import MongolianNormalizer

VOICE = "female"          # or "male"

ckpt  = hf_hub_download("{repo}", "model.safetensors")
vocab = hf_hub_download("{repo}", "vocab.txt")
ref   = hf_hub_download("{repo}", f"voices/{{VOICE}}.wav")
rtxt  = hf_hub_download("{repo}", f"voices/{{VOICE}}.txt")

tts = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt, vocab_file=vocab, use_ema=False)
wav, sr, _ = tts.infer(
    ref_file=ref,
    ref_text=open(rtxt, encoding="utf-8").read().strip(),
    gen_text=MongolianNormalizer().normalize("Сайн байна уу. Өнөөдөр цаг агаар сайхан байна.",
                                             strict=True),
    nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-1.0, seed=0)

sf.write("out.wav", wav, sr)
```

## Two things that break it silently

**Keep `use_ema=False`.** The EMA weights synthesise fluent non-words, an order
of magnitude worse by CER than the raw tensors, while sounding like confident
speech -- so you will not hear the mistake.

**Normalise the text.** Anything outside the vocabulary is read as a space,
because unknown ids map to index 0 and index 0 is the space token. Digits, Latin
letters and punctuation all need `MongolianNormalizer`.

## Numbers

| | male | female |
| --- | --- | --- |
| CER | {cer_male:.4f} | {cer_female:.4f} |
| UTMOS | {utmos_male:.2f} | {utmos_female:.2f} |
| speaker similarity to its own prompt | {sim_male:.3f} | {sim_female:.3f} |

The two voices score {sim_cross:.3f} against each other. On this project's own
recordings, real same-speaker pairs score {same_low:.3f}–{same_high:.3f} and
different-speaker pairs {diff_low:.3f}–{diff_high:.3f}.

Per-checkpoint numbers are in `eval.json`, curves in the TensorBoard tab, the
full run in `logs/`.

## Links

* Code: [github.com/btseee/oron-tts](https://github.com/btseee/oron-tts)
* Corpus tooling: [github.com/btseee/oron-cleaner](https://github.com/btseee/oron-cleaner)
* Training data: [mbspeech-mn](https://huggingface.co/datasets/btsee/mbspeech-mn),
  [fleurs-mn](https://huggingface.co/datasets/btsee/fleurs-mn),
  [common-voice-26-mn](https://huggingface.co/datasets/btsee/common-voice-26-mn)

## Licence

CC-BY-4.0. Attribution is required because FLEURS is CC-BY-4.0. It does **not**
train on WorldSpeech, so it carries no non-commercial restriction.
"""


def render(evals: dict, consistency: dict) -> str:
    import yaml

    # Picked once here; frontmatter() takes it rather than re-selecting, so
    # the two panels can never disagree about which checkpoint is "best".
    _, best = best_checkpoint(evals[FINAL_STAGE])
    measured = consistency.get("measured", {})
    calibration = consistency.get("calibration", {})
    same_low, same_high = calibration.get("same_speaker_range", [float("nan")] * 2)
    diff_low, diff_high = calibration.get("different_speaker_range", [float("nan")] * 2)
    meta = yaml.safe_dump(frontmatter(evals, consistency, best=best), sort_keys=False,
                          allow_unicode=True, default_flow_style=False)
    body = BODY.format(
        repo=REPO,
        cer_male=best["male"]["cer_median"], cer_female=best["female"]["cer_median"],
        utmos_male=best["male"]["utmos_mean"], utmos_female=best["female"]["utmos_mean"],
        sim_male=measured.get("male_demo_vs_male_prompt", float("nan")),
        sim_female=measured.get("female_demo_vs_female_prompt", float("nan")),
        sim_cross=measured.get("male_demo_vs_female_demo", float("nan")),
        same_low=same_low, same_high=same_high,
        diff_low=diff_low, diff_high=diff_high)
    return "---\n" + meta + "---\n" + body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", required=True, type=Path)
    parser.add_argument("--consistency", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    card = render(json.loads(args.eval.read_text(encoding="utf-8")),
                  json.loads(args.consistency.read_text(encoding="utf-8")))
    args.out.write_text(card, encoding="utf-8")
    print(f"  wrote {args.out} ({len(card)} chars)")


if __name__ == "__main__":
    main()
