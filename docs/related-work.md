# Related work, and what this project can honestly claim

The review's sharpest structural criticism was that this work compares against
nothing — not against published Mongolian systems, not against a naive
baseline, not against its own previous checkpoint. A number with no comparison
is not evidence. This is the comparison set, written before the numbers exist
so it cannot be chosen to flatter them.

## Published Mongolian TTS

| system | data | reported | script / dialect |
| --- | --- | --- | --- |
| **MnTTS** (2022) | ~8 h, 1 female speaker | MOS > 4.0 | Inner Mongolian, **Latin transliteration** |
| **MnTTS2** (2023) | ~30 h, 3 female speakers | multi-speaker VITS / FastSpeech2 | Inner Mongolian, **Latin transliteration** |
| **EM-TTS** (2024) | low-resource | lightweight | Mongolian |
| **FullConv-TTS** (2022) | low-resource | efficiency-focused | Mongolian |
| commercial (Narakeet, SpeechGen, …) | — | shipping today | Khalkha, closed |
| **this work** | ~30 h projected, both genders | **none yet** | Khalkha **Cyrillic**, zero-shot cloning |

## What is and is not differentiated

**Genuinely different, and the reason the comparison is awkward:** MnTTS and
MnTTS2 are Inner Mongolian in Latin transliteration. This is Khalkha in
Cyrillic — a different standard variety and a different writing system, so
their numbers are not a baseline this can beat or lose to. That is a real
contribution *if measured*, and currently it is asserted.

It also means a fair comparison has to be constructed rather than quoted:

- re-run an MnTTS-class recipe (FastSpeech2 + HiFi-GAN) on **this** corpus, or
- accept that the comparison is qualitative and say so plainly in the model
  card, rather than implying a win by omission.

**Not differentiated:** commercial Khalkha voices already ship. "First Mongolian
TTS" is not a claim available here. What is available is *first
commercially-usable open Khalkha corpus and model* — and only if the licence
chain holds, which is why `resolve_licence` derives the tag from the sources
actually present.

**Also fair to note in the other direction:** MOS > 4.0 on 8 h of one clean
studio speaker is a different problem from ~30 h of crowd-sourced audio across
hundreds of speakers with a ~8 kHz ceiling. Single-speaker MOS will likely be
higher. The comparison to make is not "is our MOS higher" but "does zero-shot
cloning across many speakers work at this data scale", which is what SIM-o
answers and MnTTS does not attempt.

## The comparisons that must appear beside any result

1. **Ground truth** — `eval_mn.py --ground-truth`. The ceiling the metrics
   themselves impose.
2. **This project's own `v1-from-scratch` checkpoint** — the honest "did the
   rebuild help" comparison, and the one no one else can dispute. Note it
   overfitted from epoch 250 of 500 on ~7 h of single-speaker audio, so it is a
   low bar; say that rather than banking the margin.
3. **An external system** on the same sentences, per the caveats above.

## What would settle the open questions

See [ablations.md](ablations.md). The one a reviewer will ask first is whether a
character-level tokenizer suffices for a non-phonemic orthography — the paper
demonstrates it for English, and extending that to Mongolian Cyrillic is a new
claim.
