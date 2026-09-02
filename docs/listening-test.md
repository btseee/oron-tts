# Listening test

Objective metrics do not measure naturalness. UTMOS is a proxy trained on
English and Japanese MOS data and has never been validated for Mongolian, so
using it as the sole naturalness number is itself an unverified assumption. CER
measures whether a recogniser can read the output, not whether a person wants
to listen to it.

This is the paper's protocol (§5.1, App. C), copied rather than invented,
because a reviewer will hold this work to the method it builds on.

## What is measured

| | scale | question put to the listener |
|---|---|---|
| **CMOS** | −3 … +3, whole steps | comparative naturalness against the baseline |
| **SMOS** | 1 … 5, 0.5 steps | how similar the speaker sounds to the reference |

CMOS is comparative: the listener hears two samples of the same sentence and
rates the second relative to the first. A positive score means this system was
preferred. SMOS is absolute, and the reference prompt is played alongside.

## Panel

- **20 native Mongolian (Khalkha) speakers.** Not learners, not Inner Mongolian
  Chakhar speakers — the dialect and the script differ.
- **30 randomised rounds each**, so every listener hears every system.
- System order **randomised per round**, and the pairing anonymised: a listener
  must not be able to tell which sample is the new system.
- Report **inter-rater agreement**. A CMOS of +0.1 across 20 listeners who
  disagree with each other is not a result.

## Systems to compare

At minimum three, because a number without a comparison is not evidence:

1. **This checkpoint** — the one `eval_mn.py --mode report` scored, not a
   different one.
2. **Ground truth** — the held-out human recording of the same sentence. This
   is the ceiling, and `eval_mn.py --ground-truth` already reports its objective
   scores; the listening test should agree with them.
3. **An external baseline.** MnTTS (FastSpeech2 + HiFi-GAN, ~8 h, reported
   MOS > 4.0) is the nearest published Mongolian system. Note the caveat in the
   comparison: MnTTS and MnTTS2 are **Inner Mongolian in Latin
   transliteration**, not Khalkha Cyrillic, so they are not directly
   comparable — which is itself the differentiating claim this work makes, and
   the reason the comparison has to be stated rather than assumed.

Add this project's own `v1-from-scratch` checkpoint if it can still be loaded;
it is the honest "did the rebuild help" comparison.

## Task definition

**Cross-sentence**, as the paper defines it: the reference prompt and the target
text are different utterances. Do not let the model be prompted with the
sentence it is asked to speak.

The prompt comes from the **test** split (a speaker the model never trained on)
and the target text from `eval_sentences.txt` (sentences no training clip
contains). Those are two separate holdouts for two separate reasons — see the
oron-cleaner README.

## Sampler settings

Fixed across every system, and the same ones `eval_mn.py` uses:

```
CFG strength      2
Sway sampling     -1
NFE steps         32
seeds             averaged over three
```

If a system is also demonstrated at a lower NFE for speed, report it as a
separate row with its own RTF — not as the same system.

## Reporting

- CMOS and SMOS with **95% confidence intervals**, not bare means.
- The number of listeners, rounds, and any excluded raters, with the exclusion
  rule stated in advance.
- The full instruction text given to listeners.
- Objective metrics from `eval_mn.py` beside them, including the ground-truth
  row and the CER contamination caveat.

## What this test cannot settle

- **Orthographic coverage.** ~30 h of a non-phonemic orthography with pervasive
  vowel harmony and non-initial vowel reduction. A G2P-versus-grapheme ablation
  answers that; a listening test does not.
- **Bandwidth.** The output is wideband ~8 kHz because no Mongolian source is
  full-band. Listeners will hear it and rate it down, and no amount of training
  changes it — say so in the model card rather than discovering it in the
  scores.
- **The numeral normaliser.** Until `docs/normaliser-review.md` is filled in,
  clips containing most numeral suffixes are dropped rather than spoken. The
  listening test should include sentences with numbers *only* after that, or it
  measures a subset of the language.
