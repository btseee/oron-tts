"""Synthesis interface: `--voice male` / `--voice female`.

The bundled voices are reference clips, not model modes. F5-TTS takes voice
identity from the prompt, so selecting a gender means selecting which reference
wav and transcript to condition on -- and any other clip works just as well:

    oron-tts-infer --voice female --text "Сайн байна уу"
    oron-tts-infer --ref-audio my.wav --ref-text "..." --text "..."

Two details that bite in Mongolian specifically:

* **Text must be normalised first.** Digits reaching the tokenizer would have to
  be pronounced from the handful of examples that survived filtering, and any
  character outside the vocabulary silently becomes a space, because
  `list_str_to_idx` maps unknown ids to 0 and index 0 is the space token.

* **`ref_text` must be in the same script as `gen_text`.** Duration is estimated
  from the *UTF-8 byte length* ratio of the two (`utils_infer.py:503-505`), and
  Cyrillic is 2 bytes per character, so a Latin reference transcript against
  Mongolian output yields roughly twice the intended length.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from oron_tts.text import MongolianNormalizer

REPO = Path(__file__).resolve().parents[1]
DEFAULT_VOICES = REPO / "voices"
DEFAULT_VOCAB = REPO / "data" / "oron_mn_pinyin" / "vocab.txt"

# Paper defaults; also what the evaluation harness fixes, so listening matches
# the reported numbers.
NFE_STEP = 32
CFG_STRENGTH = 2.0
SWAY_SAMPLING_COEF = -1.0


@dataclass
class Voice:
    id: str
    gender: str
    audio: Path
    text: str

    @property
    def exists(self) -> bool:
        return self.audio.exists()


def load_voices(directory: Path | str = DEFAULT_VOICES) -> dict[str, Voice]:
    """Read the bundle written by scripts/select_voices.py."""
    directory = Path(directory)
    index = directory / "voices.json"
    if not index.exists():
        return {}
    entries = json.loads(index.read_text(encoding="utf-8"))
    return {
        gender: Voice(
            id=e["id"],
            gender=e.get("gender", gender),
            audio=directory / e["audio"],
            text=e["text"],
        )
        for gender, e in entries.items()
    }


def resolve_voice(
    name: str | None,
    ref_audio: Path | None,
    ref_text: str | None,
    voices_dir: Path | str = DEFAULT_VOICES,
) -> tuple[Path, str]:
    """Return the (audio, transcript) pair to condition on."""
    if ref_audio is not None:
        if not ref_text:
            raise SystemExit(
                "--ref-audio needs --ref-text. Without it the reference is "
                "transcribed by an English-first ASR, and the duration estimate "
                "compares byte lengths across two scripts."
            )
        return Path(ref_audio), ref_text

    voices = load_voices(voices_dir)
    if not voices:
        raise SystemExit(
            f"No voices in {voices_dir}. Build them with:\n"
            "  python scripts/select_voices.py --corpus <corpus> --write voices/"
        )
    if name is None:
        raise SystemExit(f"Pass --voice ({', '.join(sorted(voices))}) or --ref-audio.")
    if name not in voices:
        raise SystemExit(f"Unknown voice {name!r}. Available: {', '.join(sorted(voices))}")
    voice = voices[name]
    if not voice.exists:
        raise SystemExit(f"{voice.audio} is missing from the bundle.")
    return voice.audio, voice.text


def prepare_text(text: str, normalizer: MongolianNormalizer | None = None) -> str:
    """Normalise, and refuse anything the vocabulary cannot represent.

    Raising is the point: silently dropping a character would make the model
    speak something different from what was asked, with no indication.
    """
    normalizer = normalizer or MongolianNormalizer()
    return normalizer.normalize(text, strict=True)


def synthesize(
    text: str,
    checkpoint: Path | str,
    *,
    voice: str | None = "female",
    ref_audio: Path | None = None,
    ref_text: str | None = None,
    voices_dir: Path | str = DEFAULT_VOICES,
    vocab: Path | str = DEFAULT_VOCAB,
    speed: float = 1.0,
    nfe_step: int = NFE_STEP,
    cfg_strength: float = CFG_STRENGTH,
    device: str | None = None,
    use_ema: bool = True,
    seed: int | None = None,
):
    """Synthesize one utterance. Returns (waveform, sample_rate, seed).

    `seed` pins the ODE's initial noise. Left as None, `F5TTS.infer` draws
    `random.randint(0, sys.maxsize)`, so the same text and voice give a
    different rendering every call. The seed actually used is returned so a
    result worth keeping can be reproduced.
    """
    from f5_tts.api import F5TTS

    audio, transcript = resolve_voice(voice, ref_audio, ref_text, voices_dir)
    gen_text = prepare_text(text)
    # The reference transcript goes through the same normaliser: the duration
    # estimate is a byte-length ratio between the two, so they must be written
    # the same way.
    ref_normalized = prepare_text(transcript)

    model = F5TTS(
        model="F5TTS_v1_Base",
        ckpt_file=str(checkpoint),
        vocab_file=str(vocab),
        device=device,
        use_ema=use_ema,
    )
    wav, sr, _ = model.infer(
        ref_file=str(audio),
        ref_text=ref_normalized,
        gen_text=gen_text,
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        sway_sampling_coef=SWAY_SAMPLING_COEF,
        speed=speed,
        remove_silence=False,
        seed=seed,
    )
    # infer() stores whatever it drew when seed was None.
    return wav, sr, model.seed


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Synthesize Mongolian speech with a bundled or custom voice."
    )
    ap.add_argument("--text", required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--voice", default=None, help="Bundled voice, e.g. male | female")
    ap.add_argument("--ref-audio", type=Path, default=None, help="Custom reference clip")
    ap.add_argument("--ref-text", default=None, help="Transcript of --ref-audio")
    ap.add_argument("--voices-dir", type=Path, default=DEFAULT_VOICES)
    ap.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    ap.add_argument("--output", type=Path, default=Path("out.wav"))
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--nfe-step", type=int, default=NFE_STEP)
    ap.add_argument("--cfg-strength", type=float, default=CFG_STRENGTH)
    ap.add_argument("--seed", type=int, default=None,
                    help="Pin the sampler noise. Omitted, one is drawn at random "
                         "and printed so the run can be reproduced.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-ema", dest="use_ema", action="store_false", default=True,
                    help="Early finetunes: EMA is still dominated by pretrained weights")
    ap.add_argument("--list-voices", action="store_true")
    args = ap.parse_args()

    if args.list_voices:
        for gender, v in sorted(load_voices(args.voices_dir).items()):
            print(f"{gender:<8} {v.id}  {'ok' if v.exists else 'MISSING'}  {v.text[:60]}")
        return

    if args.voice is None and args.ref_audio is None:
        args.voice = "female"

    import soundfile as sf

    wav, sr, seed = synthesize(
        args.text,
        args.checkpoint,
        voice=args.voice,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        voices_dir=args.voices_dir,
        vocab=args.vocab,
        speed=args.speed,
        nfe_step=args.nfe_step,
        cfg_strength=args.cfg_strength,
        device=args.device,
        use_ema=args.use_ema,
        seed=args.seed,
    )
    sf.write(args.output, wav, sr)
    print(f"Wrote {args.output} ({len(wav) / sr:.2f}s at {sr} Hz, seed {seed})")


if __name__ == "__main__":
    main()
