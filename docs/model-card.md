# Model card — draft for `btsee/oron-tts`

Fill the blanks marked `<>` from the reporting run and publish this with the
weights. Everything not marked is already measured and should not change.

Four other documents in this repository say a fact belongs "in the model card";
this is that card, written before the numbers exist so it cannot be shaped
around them.

---

## What it is

Mongolian (Khalkha Cyrillic) zero-shot text-to-speech: a finetune of
[F5-TTS](https://github.com/SWivid/F5-TTS) `F5TTS_v1_Base`, 336M parameters,
flow matching with a DiT and the Vocos vocoder at 24 kHz.

Voice identity comes from a **reference clip**, not from a speaker token. The
bundled `male` and `female` voices are two such clips; any ~10 s recording works
in their place.

| | |
| --- | --- |
| language | Mongolian, Khalkha, Cyrillic script |
| base | `F5TTS_v1_Base` at update 1,250,000 |
| vocabulary | 2,550 entries — the base 2,545 plus `ө ү Ө Ү Ъ` |
| training data | `btsee/oron-mn-strict`, `<N>` h across `<S>` speakers |
| updates | `<U>`, selected by sweep rather than taken as the last |
| licence | `<see the corpus card — the merge governs it>` |

## What it sounds like, honestly

**The output is wideband, roughly 8 kHz — not full-band.** No Mongolian source
is: Common Voice's median lowpass shelf is 7.1 kHz, FLEURS and MBSpeech are
hard-capped at 7.7 kHz. That is a property of the available data, not of the
model, and no amount of training changes it. A listener should not have to
discover it.

Output bandwidth follows the *prompt*, so a dull reference clip gives a dull
voice regardless of the text.

## Measured

Scored by `scripts/eval_mn.py` on speakers and sentences withheld from training
— two separate holdouts, because a reference prompt needs an unseen speaker and
a CER target needs unseen text. Averaged over three seeds, CFG 2, sway −1,
NFE 32, with 95% confidence intervals.

| | male | female | ground truth |
| --- | --- | --- | --- |
| CER | `<>` | `<>` | `<>` |
| SIM-o | `<>` | `<>` | `<>` |
| UTMOS | `<>` | `<>` | `<>` |
| bandwidth | `<>` | `<>` | `<>` |

**The CER scorer is contaminated, and the number is optimistic because of it.**
`bayartsogt/wav2vec2-large-xlsr-mongolian` is, per its own card, fine-tuned on
Common Voice Mongolian — which is most of this model's training data. It is used
here because it is the best Mongolian recogniser available (CER 0.123 median on
correctly-transcribed human speech, against whisper-large-v3's 0.311), not
because it is neutral. Read every CER against the ground-truth row, never
against zero.

**No listening test has been run.** UTMOS is a proxy trained on English and
Japanese MOS data and has never been validated for Mongolian; treat it as a
regression detector, not a MOS. See [listening-test.md](listening-test.md) for
the protocol a naturalness claim would need.

**No baseline comparison.** See [related-work.md](related-work.md) for why the
nearest published systems (MnTTS, MnTTS2) are not directly comparable — they are
Inner Mongolian in Latin transliteration, not Khalkha Cyrillic.

## Intended use

Synthesising Mongolian speech from Mongolian text, with a reference clip you
have the right to use.

## Out of scope, and asked of anyone using this

The model clones a recognisable voice from roughly ten seconds of audio. Its
training data is crowd-sourced volunteer speech: Common Voice contributors
dedicated their recordings CC0 for speech research, and FLEURS and MBSpeech
speakers recorded for read-speech benchmarks. **None of them consented to having
their individual voice cloned.**

- **Do not** synthesise a named or identifiable person's voice without that
  person's explicit consent.
- **Do not** present synthetic audio as a real recording of anyone.
- **Do not** use it for voice-biometric spoofing, or against any system that
  authenticates people by voice.
- **Do not** use it for languages or dialects it was not trained on. It is
  Khalkha Cyrillic; Inner Mongolian in the traditional script is a different
  language variety and a different script.

**There is no watermarking.** Audio from this model cannot be detected as
synthetic by any downstream tool. If you ship a voice built from it, say so
where listeners can see it.

## Known limitations

| | |
| --- | --- |
| bandwidth | ~8 kHz, above. |
| numerals | The text normaliser **refuses** numeral case suffixes it cannot expand without guessing — measured at 3.4% of Mongolian Wikipedia sentences. Those inputs raise rather than mispronounce. See [normaliser-review.md](normaliser-review.md). |
| orthography | Mongolian is non-phonemic and this is a character-level model with no G2P. The paper supports that choice, but on English; extending it to Cyrillic is an untested claim. See [ablations.md](ablations.md). |
| male voice | Male hours were the binding constraint on the corpus. Check the per-gender numbers above before assuming the two voices are equally good. |
| case | The corpus preserves case, and upstream reports that uppercase is uttered letter by letter. Listen for spelled-out first words. |
| speakers | `<S>` speakers is a narrow slice of Mongolian. Accent and dialect coverage is untested. |

## Reproducing

```bash
python scripts/preflight.py --data <prepared-dataset>   # refuses a wrong config
accelerate launch src/f5_tts/train/train.py --config-name f5tts_mn.yaml
python scripts/eval_mn.py --checkpoint <best> --corpus <corpus> --ground-truth --rtf
```

The corpus records its own provenance — pinned model and dataset revisions,
package versions, a fingerprint of the text-normalisation source, and a content
hash — in `provenance.json`. Quote the content hash here: `<hash>`.

## Deployment

| | |
| --- | --- |
| RTF | `<>` at NFE 32, `<>` at 16, `<>` at 8 |
| latency | median `<>` s, p90 `<>` s |
| peak memory | `<>` GiB |

Measured on `<GPU>` by `scripts/eval_mn.py --rtf`. NFE trades quality for speed;
report the CER at the same setting you deploy.
