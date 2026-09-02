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
           That recogniser is fine-tuned on Common Voice Mongolian -- the corpus
           this model trains on -- so it is contaminated in the model's favour.
           `--asr-model` takes an independent one for a second opinion.
  SIM-o    speaker similarity to the reference prompt, WavLM-large ECAPA-TDNN,
           the same model the paper uses so the number is on its scale
  UTMOS    naturalness, language-agnostic
  bandwidth of the output -- follows the reference clip, and no Mongolian source
           is full-band, so this is how you catch a dull voice
  RTF      with --rtf: wall-clock seconds per second of audio, plus latency and
           peak memory, at each of --rtf-nfe
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
# Upstream clips reference audio over 12 s; under ~6 s there is too little
# speaker evidence for a stable prompt.
MIN_REF_S = 6.0
MAX_REF_S = 10.0


def load_test_sentences(corpus: Path, limit: int, mode: str = "report") -> list[str]:
    """Sentences that occur in no training clip.

    Deliberately not `metadata_test.csv`. That file is the speaker-disjoint
    audio split, which is what a *reference prompt* needs; its text is another
    matter entirely. Common Voice mn has 28,858 clips over 6,062 distinct
    sentences, so 99.6% of test clips (1,705 of 1,712) had their text in train
    as well -- CER over them measured recall of seen text.

    `eval_sentences.txt` is the corpus's text holdout: sentences withheld from
    training so this number means intelligibility.

    `mode` halves that list by parity. Sweeping a dozen checkpoints and then
    reporting the winner's score on the same sentences is selection on the test
    set: the winner is partly the checkpoint that got lucky on those sentences,
    and its number is optimistic by however much luck was involved. `select`
    takes the even indices, `report` the odd ones, so the number that gets
    published was never used to choose anything.
    """
    path = corpus / "eval_sentences.txt"
    if path.exists():
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
        lines = [ln for ln in lines if ln]
        if mode == "select":
            lines = lines[0::2]
        elif mode == "report":
            lines = lines[1::2]
        return lines[:limit]

    legacy = corpus / "metadata_test.csv"
    if not legacy.exists():
        raise SystemExit(f"{path} not found. Run oron-cleaner's finalize step.")
    print(f"[WARN] {path.name} missing; falling back to {legacy.name}, whose text\n"
          "       is almost certainly in training too. Re-run:\n"
          "           python clean_pipeline.py --finalize-only --corpus-dir <dir>\n"
          "       The CER below measures memorisation until you do.", file=sys.stderr)
    import csv

    with open(legacy, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f, delimiter="|"))[1:]
    return [text for _audio, text in rows][:limit]


def pick_reference(corpus: Path, gender: str, split: str = "test") -> tuple[Path, str]:
    """Best reference clip for a gender, drawn from a held-out split.

    Ranked by bandwidth first: output bandwidth follows the prompt, and the
    ≥10 kHz tail exists only in Common Voice.

    The `split` restriction is the zero-shot condition. Previously this scanned
    the entire manifest, so the prompt was a *training* clip with ~90%
    probability -- which is exactly what speaker_disjoint_split exists to
    prevent. A reference the model trained on measures memorisation, not
    voice cloning.
    """
    manifest = corpus / "manifest.jsonl"
    if not manifest.exists():
        raise SystemExit(f"{manifest} not found.")
    best = None
    saw_split = False
    saw_gender = False
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "split" not in r or "gender_resolved" not in r:
                raise SystemExit(
                    "manifest.jsonl lacks 'split'/'gender_resolved'. Run:\n"
                    "    python clean_pipeline.py --finalize-only --corpus-dir <dir>"
                )
            saw_split = saw_split or r["split"] == split
            saw_gender = saw_gender or r["gender_resolved"] == gender
            if r["split"] != split or r["gender_resolved"] != gender:
                continue
            # Upstream clips reference audio over 12 s and wants a little
            # trailing silence; 6-10 s is the comfortable band.
            if not (MIN_REF_S <= float(r.get("duration_s") or 0) <= MAX_REF_S):
                continue
            score = (float(r.get("bandwidth_hz") or 0) / 1000.0
                     + float(r.get("dnsmos_ovr") or 0)
                     + float(r.get("align_score") or 0) * 2)
            if best is None or score > best[0]:
                best = (score, corpus / r["audio_path"], r["text"])
    if best is None:
        why = []
        if not saw_split:
            why.append(f"no rows in split {split!r}")
        if not saw_gender:
            why.append(f"no rows with gender_resolved={gender!r}")
        if not why:
            why.append(f"none between {MIN_REF_S:g}-{MAX_REF_S:g}s")
        raise SystemExit(f"No usable {gender} reference in {corpus}: {'; '.join(why)}.")
    _score, path, text = best
    return path, text


def synthesise(f5, text: str, ref_audio: Path, ref_text: str, seed: int,
               nfe_step: int = NFE_STEP):
    """One utterance, with the sampler's noise pinned.

    F5TTS.infer draws `seed = random.randint(...)` when none is given, so an
    unseeded harness compares checkpoints under different noise draws. The paper
    averages over three seeds (§5.1) -- in the same sentence as the CFG/sway/NFE
    constants this function already copies.
    """
    wav, sr, _ = f5.infer(
        seed=seed,
        ref_file=str(ref_audio),
        ref_text=ref_text,
        gen_text=text,
        nfe_step=nfe_step,
        cfg_strength=CFG_STRENGTH,
        sway_sampling_coef=SWAY_SAMPLING_COEF,
        remove_silence=False,
    )
    return wav, sr


def measure_rtf(f5, sentences, ref_audio: Path, ref_text: str,
                nfe_steps, device: str) -> dict:
    """Real-time factor and latency at several NFE settings.

    Nothing in this repo measured whether the model runs fast enough to use.
    The paper reports RTF 0.15 at NFE 16 on datacentre hardware; a 336M DiT
    solving an ODE is not obviously real-time anywhere else, and the setting
    that buys the speed is the same one that costs quality -- so the two have to
    be read together.

    RTF is wall-clock seconds per second of audio produced: below 1.0 is faster
    than real time. Peak memory is CUDA-only; on CPU it is reported as 0.
    """
    import time

    import torch

    out = {}
    for nfe in nfe_steps:
        # One untimed pass: the first call pays for lazy CUDA init and any
        # kernel autotuning, which is not what a served request costs.
        synthesise(f5, sentences[0], ref_audio, ref_text, 0, nfe)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        latencies, audio_s = [], 0.0
        for i, text in enumerate(sentences):
            start = time.perf_counter()
            wav, sr = synthesise(f5, text, ref_audio, ref_text, i, nfe)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)
            audio_s += len(wav) / sr

        peak = (torch.cuda.max_memory_allocated() / 2**30
                if device.startswith("cuda") else 0.0)
        out[str(nfe)] = {
            "rtf": sum(latencies) / audio_s if audio_s else float("inf"),
            "latency_median_s": statistics.median(latencies),
            "latency_p90_s": sorted(latencies)[int(0.9 * (len(latencies) - 1))],
            "peak_gib": round(peak, 2),
            "n": len(latencies),
        }
    return out


def ground_truth_topline(corpus: Path, args) -> dict:
    """Score the held-out *human* audio with the same instruments.

    The paper reports a ground-truth row in every table (WER 2.23, SIM-o 0.69,
    UTMOS 4.09 on LibriSpeech-PC test-clean) and it is the row that makes the
    others readable: it is the ceiling the metrics themselves impose, not one
    the model could exceed.

    Without it the numbers here float. CER 0.19 is either close to the ceiling
    or twice it, and nothing in the output says which. This corpus reserves the
    test audio and, until now, never scored it.
    """
    import json as _json

    import numpy as np
    import soundfile as sf

    from oron_tts.eval import MongolianASR, bandwidth_hz, sim_o, utmos

    rows: list[dict] = []
    with open(corpus / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = _json.loads(line)
                if r.get("split") == args.ref_split:
                    rows.append(r)
    if not rows:
        raise SystemExit(f"No rows in split {args.ref_split!r} to score.")

    asr = MongolianASR(device=args.device, model_name=args.asr_model)
    out: dict[str, dict] = {}
    for gender in args.genders:
        clips = [r for r in rows if r.get("gender_resolved") == gender][: args.n_sentences]
        if not clips:
            continue
        cers, moses, bws, sims = [], [], [], []
        # SIM-o against a *different* clip of the same speaker: that is the
        # ceiling for speaker similarity, since even real audio of one person
        # does not score 1.0 against itself across utterances.
        by_speaker: dict[str, list[dict]] = {}
        for r in clips:
            by_speaker.setdefault(str(r.get("client_id") or ""), []).append(r)

        for r in clips:
            wav, sr = sf.read(corpus / r["audio_path"], dtype="float32")
            wav = np.asarray(wav, dtype="float32")
            cers.append(asr.score(wav, r["text"], sr))
            bws.append(bandwidth_hz(wav, sr))
            if not args.no_utmos:
                with contextlib.suppress(Exception):
                    moses.append(utmos(wav, sr))
            if not args.no_sim:
                peers = [o for o in by_speaker[str(r.get("client_id") or "")]
                         if o["clip_id"] != r["clip_id"]]
                if peers:
                    with contextlib.suppress(Exception):
                        other, osr = sf.read(corpus / peers[0]["audio_path"], dtype="float32")
                        sims.append(sim_o(wav, np.asarray(other, dtype="float32"), sr, osr,
                                          checkpoint=args.sim_checkpoint, device=args.device))
        out[gender] = {
            "cer_median": statistics.median(cers),
            "cer_mean": statistics.fmean(cers),
            "cer_sd": statistics.stdev(cers) if len(cers) > 1 else 0.0,
            "cer_ci95": (1.96 * statistics.stdev(cers) / len(cers) ** 0.5)
            if len(cers) > 1 else float("inf"),
            "bandwidth_median": statistics.median(bws),
            "utmos_mean": statistics.fmean(moses) if moses else None,
            "utmos_n": len(moses),
            "sim_o_mean": statistics.fmean(sims) if sims else None,
            "sim_o_n": len(sims),
            "n": len(cers),
            "seeds": [],
            "reference": "(real audio)",
            "cer_values": cers,
        }
    return out


def evaluate(checkpoint: Path, corpus: Path, args) -> dict:
    import numpy as np
    import soundfile as sf
    from f5_tts.api import F5TTS  # noqa: I001 - optional heavy dependency

    from oron_tts.eval import MongolianASR, bandwidth_hz, sim_o, utmos

    sentences = load_test_sentences(corpus, args.n_sentences, args.mode)
    asr = MongolianASR(device=args.device, model_name=args.asr_model)
    sim_warned = False

    results: dict[str, dict] = {}
    for gender in args.genders:
        ref_audio, ref_text = pick_reference(corpus, gender, split=args.ref_split)
        f5 = F5TTS(
            model="F5TTS_v1_Base",
            ckpt_file=str(checkpoint),
            vocab_file=str(args.vocab),
            device=args.device,
            # An early finetune's EMA is still dominated by the pretrained
            # weights; upstream warns use_ema=True is harmful there.
            use_ema=args.use_ema,
        )
        ref_wav, ref_sr = sf.read(ref_audio, dtype="float32")
        cers, moses, bws, sims = [], [], [], []
        for text in sentences:
            # The paper averages over three random seeds (§5.1). One draw makes
            # adjacent checkpoints indistinguishable from sampler noise.
            for seed in args.seeds:
                try:
                    wav, sr = synthesise(f5, text, ref_audio, ref_text, seed)
                except Exception as exc:
                    print(f"  [{gender}] synthesis failed (seed {seed}): {exc}")
                    continue
                wav = np.asarray(wav, dtype="float32")
                cers.append(asr.score(wav, text, sr))
                bws.append(bandwidth_hz(wav, sr))
                if not args.no_sim:
                    # The whole proposition is that voice identity transfers
                    # from the prompt. Without this nothing measures whether it
                    # does; the checkpoint is a manual download, so a missing
                    # one must not cost the CER measurement.
                    try:
                        sims.append(sim_o(wav, ref_wav, sr, ref_sr,
                                          checkpoint=args.sim_checkpoint,
                                          device=args.device))
                    except Exception as exc:
                        if not sim_warned:
                            print(f"  SIM-o unavailable: {exc}")
                            sim_warned = True
                if not args.no_utmos:
                    # UTMOS needs torch.hub, which may be offline on a training
                    # box. Its absence should not cost the CER measurement.
                    with contextlib.suppress(Exception):
                        moses.append(utmos(wav, sr))
        if not cers:
            continue
        results[gender] = {
            "cer_median": statistics.median(cers),
            "cer_mean": statistics.fmean(cers),
            # Reported so a reader can tell whether two checkpoints differ at all.
            "cer_sd": statistics.stdev(cers) if len(cers) > 1 else 0.0,
            "cer_ci95": (1.96 * statistics.stdev(cers) / len(cers) ** 0.5)
            if len(cers) > 1 else float("inf"),
            "bandwidth_median": statistics.median(bws),
            "utmos_mean": statistics.fmean(moses) if moses else None,
            "sim_o_mean": statistics.fmean(sims) if sims else None,
            "sim_o_n": len(sims),
            # n for UTMOS is tracked separately: exceptions are suppressed above,
            # so it can be shorter than n and was previously displayed as if not.
            "utmos_n": len(moses),
            "n": len(cers),
            "seeds": list(args.seeds),
            "reference": str(ref_audio.name),
            # Per-utterance scores, so a CI can be recomputed from the artifact.
            "cer_values": cers,
        }
        if args.rtf and "rtf" not in results:
            results["rtf"] = measure_rtf(
                f5, sentences[: args.rtf_sentences], ref_audio, ref_text,
                args.rtf_nfe, args.device,
            )
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
        line = (f"  {gender:<7} CER {r['cer_median']:.3f} +/-{r['cer_ci95']:.3f} "
                f"({r['cer_median'] / baseline:.2f}x the human floor)  "
                f"BW {r['bandwidth_median']:.0f} Hz  n={r['n']}")
        if r["utmos_mean"] is not None:
            line += f"  UTMOS {r['utmos_mean']:.2f} (n={r['utmos_n']})"
        if r.get("sim_o_mean") is not None:
            line += f"  SIM-o {r['sim_o_mean']:.3f} (n={r['sim_o_n']})"
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
    ap.add_argument("--n-sentences", type=int, default=200,
                    help="The paper reports 1000 in-set samples; 20 cannot resolve "
                         "differences smaller than ~0.05 CER.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="Sampler seeds to average over. The paper reports the "
                         "average of three random seed generations (5.1).")
    ap.add_argument("--mode", choices=("select", "report", "all"), default=None,
                    help="select: sweep on the validation speakers and the even "
                         "half of the held-out sentences. report: score one "
                         "checkpoint on the test speakers and the odd half. "
                         "Defaults to select for --sweep, report for --checkpoint.")
    ap.add_argument("--ref-split", default=None,
                    help="Split the reference prompt is drawn from. Defaults to "
                         "the one --mode implies. Anything but a held-out split "
                         "voids the zero-shot condition.")
    ap.add_argument("--asr-model", default=None,
                    help="Recogniser for CER. The default is fine-tuned on Common "
                         "Voice Mongolian -- the same corpus this model trains on "
                         "-- so it is contaminated. Pass an independent one "
                         "(e.g. facebook/mms-1b-all) for a second opinion.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use-ema", action="store_true", default=True)
    ap.add_argument("--no-ema", dest="use_ema", action="store_false",
                    help="Early finetunes: EMA is still dominated by pretrained weights")
    ap.add_argument("--no-utmos", action="store_true", help="Skip UTMOS (needs torch.hub)")
    ap.add_argument("--no-sim", action="store_true",
                    help="Skip SIM-o speaker similarity")
    ap.add_argument("--ground-truth", action="store_true",
                    help="Also score the held-out human audio. The paper reports "
                         "this row in every table; without it a CER has no ceiling "
                         "to be read against.")
    ap.add_argument("--rtf", action="store_true",
                    help="Also measure real-time factor, latency and peak memory")
    ap.add_argument("--rtf-nfe", type=int, nargs="+", default=[8, 16, 32],
                    help="NFE settings to time. The paper reports RTF 0.15 at 16.")
    ap.add_argument("--rtf-sentences", type=int, default=20,
                    help="Utterances per NFE setting")
    ap.add_argument("--sim-checkpoint", default=None,
                    help="WavLM-large speaker-verification checkpoint "
                         "(wavlm_large_finetune.pth); or set ORON_WAVLM_CKPT")
    ap.add_argument("--baseline", type=float, default=None,
                    help="Human-speech CER floor; defaults to the measured 0.123")
    ap.add_argument("--out", type=Path, default=Path("eval_results.json"))
    args = ap.parse_args()

    from oron_tts.eval import ASR_MODEL, HUMAN_CER_BASELINE

    args.asr_model = args.asr_model or ASR_MODEL
    baseline = args.baseline if args.baseline is not None else HUMAN_CER_BASELINE
    args.genders = [g.strip() for g in args.genders.split(",") if g.strip()]

    # Choosing a checkpoint on the test set and then reporting that checkpoint's
    # test score is selection on the test set. The validation split exists for
    # the choosing -- and until now nothing read it at all: the F5-TTS trainer
    # contains no validation loop, so metadata_validation.csv was written and
    # never opened.
    if args.mode is None:
        args.mode = "select" if args.sweep else "report"
    if args.ref_split is None:
        args.ref_split = "validation" if args.mode == "select" else "test"
    if args.mode == "select" and args.ref_split == "test":
        print("[WARN] Sweeping against the test split. The winner's score will be "
              "optimistic by however much it is\n       the checkpoint that got "
              "lucky there. Use --ref-split validation.", file=sys.stderr)
    print(f"mode={args.mode}  reference split={args.ref_split}", file=sys.stderr)

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
    if args.ground_truth:
        print("\nScoring the held-out human audio …", file=sys.stderr)
        try:
            gt = ground_truth_topline(args.corpus, args)
            all_results["GROUND TRUTH"] = gt
            report("GROUND TRUTH (real audio -- the ceiling, not a checkpoint)",
                   gt, baseline)
        except Exception as exc:
            print(f"  failed: {exc}", file=sys.stderr)
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

    ranked_results = {k: v for k, v in all_results.items() if k != "GROUND TRUTH"}
    if len(ranked_results) > 1:
        def mean_cer(r):
            vals = [v["cer_median"] for k, v in r.items() if k != "rtf"]
            return statistics.fmean(vals) if vals else float("inf")

        best = min(all_results, key=lambda k: mean_cer(all_results[k]))
        print(f"\nBest by mean CER across genders: {best}")
        print("Confirm by listening before shipping -- CER does not hear prosody.")


if __name__ == "__main__":
    main()
