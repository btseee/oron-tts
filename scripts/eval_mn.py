"""Evaluate a Mongolian F5-TTS checkpoint and pick the best one.

Upstream's evaluation supports only `zh` and `en` (`utils_eval.py:315-318`
raises for anything else), so this is the Mongolian harness.

The point is checkpoint *selection*, not a single number. The paper's own
small-data evidence (Tab. 9) has a 24 h LJSpeech model peaking at 200k updates
and then degrading to twice the WER by 600k, and the previous project's run
overfitted from epoch 250 of 500. The last checkpoint is not the best one, and
training loss will not tell you which is.

    python scripts/eval_mn.py --checkpoint ckpts/.../model_20000.pt \\
        --corpus ../oron-cleaner/output/oron_mn_strict
    python scripts/eval_mn.py --sweep ckpts/oron_mn --corpus <dir>

Reported per checkpoint:

  CER      against the same recogniser that scored the corpus. Its floor on real
           human speech is ~0.12, so compare to `--baseline`, not to zero.
  UTMOS    naturalness, language-agnostic
  bandwidth of the output -- follows the reference clip, and no Mongolian source
           is full-band, so this is how you catch a dull voice
"""

import argparse
import contextlib
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_VOCAB = REPO / "data" / "oron_mn_pinyin" / "vocab.txt"

# Fixed so numbers are comparable across checkpoints. Paper defaults.
NFE_STEP = 32
CFG_STRENGTH = 2.0
SWAY_SAMPLING_COEF = -1.0


def load_test_sentences(corpus: Path, limit: int) -> list[str]:
    """Held-out text. Uses the test split, which is speaker-disjoint."""
    path = corpus / "metadata_test.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run oron-cleaner's finalize step.")
    import csv

    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f, delimiter="|"))[1:]
    return [text for _audio, text in rows][:limit]


def pick_reference(corpus: Path, gender: str) -> tuple[Path, str]:
    """Best available reference clip for a gender.

    Ranked by bandwidth first: output bandwidth follows the prompt, and the
    ≥10 kHz tail exists only in Common Voice.
    """
    manifest = corpus / "manifest.jsonl"
    if not manifest.exists():
        raise SystemExit(f"{manifest} not found.")
    best = None
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("gender_resolved") != gender:
                continue
            # Upstream clips reference audio over 12 s and wants a little
            # trailing silence; 6-10 s is the comfortable band.
            if not (6.0 <= float(r.get("duration_s") or 0) <= 10.0):
                continue
            score = (float(r.get("bandwidth_hz") or 0) / 1000.0
                     + float(r.get("dnsmos_ovr") or 0)
                     + float(r.get("align_score") or 0) * 2)
            if best is None or score > best[0]:
                best = (score, corpus / r["audio_path"], r["text"])
    if best is None:
        raise SystemExit(f"No usable {gender} reference clip in {corpus}.")
    _score, path, text = best
    return path, text


def synthesise(f5, text: str, ref_audio: Path, ref_text: str):
    """One utterance. Settings fixed so checkpoints stay comparable."""
    wav, sr, _ = f5.infer(
        ref_file=str(ref_audio),
        ref_text=ref_text,
        gen_text=text,
        nfe_step=NFE_STEP,
        cfg_strength=CFG_STRENGTH,
        sway_sampling_coef=SWAY_SAMPLING_COEF,
        remove_silence=False,
    )
    return wav, sr


def evaluate(checkpoint: Path, corpus: Path, args) -> dict:
    import numpy as np
    from f5_tts.api import F5TTS  # noqa: I001 - optional heavy dependency

    from oron_tts.eval import MongolianASR, bandwidth_hz, utmos

    sentences = load_test_sentences(corpus, args.n_sentences)
    asr = MongolianASR(device=args.device)

    results: dict[str, dict] = {}
    for gender in args.genders:
        ref_audio, ref_text = pick_reference(corpus, gender)
        f5 = F5TTS(
            model="F5TTS_v1_Base",
            ckpt_file=str(checkpoint),
            vocab_file=str(args.vocab),
            device=args.device,
            # An early finetune's EMA is still dominated by the pretrained
            # weights; upstream warns use_ema=True is harmful there.
            use_ema=args.use_ema,
        )
        cers, moses, bws = [], [], []
        for text in sentences:
            try:
                wav, sr = synthesise(f5, text, ref_audio, ref_text)
            except Exception as exc:
                print(f"  [{gender}] synthesis failed: {exc}")
                continue
            wav = np.asarray(wav, dtype="float32")
            cers.append(asr.score(wav, text, sr))
            bws.append(bandwidth_hz(wav, sr))
            if not args.no_utmos:
                # UTMOS needs torch.hub, which may be offline on a training box.
                # Its absence should not cost the CER measurement.
                with contextlib.suppress(Exception):
                    moses.append(utmos(wav, sr))
        if not cers:
            continue
        results[gender] = {
            "cer_median": statistics.median(cers),
            "cer_mean": statistics.fmean(cers),
            "bandwidth_median": statistics.median(bws),
            "utmos_mean": statistics.fmean(moses) if moses else None,
            "n": len(cers),
            "reference": str(ref_audio.name),
        }
    return results


def sort_checkpoints(paths: list[Path]) -> list[Path]:
    """Order by update number, not lexically.

    Plain sorting puts model_10000 before model_2000, so a sweep reports its
    checkpoints out of order and the "best" line becomes hard to trust.
    """
    import re

    def step(path: Path) -> tuple[int, str]:
        m = re.search(r"(\d+)", path.stem)
        return (int(m.group(1)) if m else -1, path.name)

    return sorted(paths, key=step)


def report(name: str, results: dict, baseline: float) -> None:
    print(f"\n=== {name}")
    for gender, r in results.items():
        line = (f"  {gender:<7} CER {r['cer_median']:.3f} median "
                f"({r['cer_median'] / baseline:.2f}x the human floor)  "
                f"BW {r['bandwidth_median']:.0f} Hz  n={r['n']}")
        if r["utmos_mean"] is not None:
            line += f"  UTMOS {r['utmos_mean']:.2f}"
        print(line)
        print(f"          ref: {r['reference']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, help="Single checkpoint to score")
    ap.add_argument("--sweep", type=Path, help="Directory of checkpoints to score in order")
    ap.add_argument("--corpus", type=Path, required=True, help="oron-cleaner corpus directory")
    ap.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    ap.add_argument("--genders", default="male,female")
    ap.add_argument("--n-sentences", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use-ema", action="store_true", default=True)
    ap.add_argument("--no-ema", dest="use_ema", action="store_false",
                    help="Early finetunes: EMA is still dominated by pretrained weights")
    ap.add_argument("--no-utmos", action="store_true", help="Skip UTMOS (needs torch.hub)")
    ap.add_argument("--baseline", type=float, default=None,
                    help="Human-speech CER floor; defaults to the measured 0.123")
    ap.add_argument("--out", type=Path, default=Path("eval_results.json"))
    args = ap.parse_args()

    from oron_tts.eval import HUMAN_CER_BASELINE

    baseline = args.baseline if args.baseline is not None else HUMAN_CER_BASELINE
    args.genders = [g.strip() for g in args.genders.split(",") if g.strip()]

    if args.sweep:
        checkpoints = sort_checkpoints(
            list(args.sweep.glob("model_*.pt")) + list(args.sweep.glob("model_*.safetensors"))
        )
        if not checkpoints:
            raise SystemExit(f"No checkpoints under {args.sweep}")
    elif args.checkpoint:
        checkpoints = [args.checkpoint]
    else:
        raise SystemExit("Pass --checkpoint or --sweep.")

    all_results = {}
    for ckpt in checkpoints:
        print(f"\nEvaluating {ckpt.name} …", file=sys.stderr)
        try:
            results = evaluate(ckpt, args.corpus, args)
        except Exception as exc:
            print(f"  failed: {exc}", file=sys.stderr)
            continue
        all_results[ckpt.name] = results
        report(ckpt.name, results, baseline)

    args.out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")

    if len(all_results) > 1:
        def mean_cer(r):
            vals = [v["cer_median"] for v in r.values()]
            return statistics.fmean(vals) if vals else float("inf")

        best = min(all_results, key=lambda k: mean_cer(all_results[k]))
        print(f"\nBest by mean CER across genders: {best}")
        print("Confirm by listening before shipping -- CER does not hear prosody.")


if __name__ == "__main__":
    main()
