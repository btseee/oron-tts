# Ablations a reviewer will ask for

Ordered by how much of the design each one actually tests. None of these has
been run; each is written with the protocol that would settle it, so the answer
is a measurement rather than an argument.

## 1. Character-level vs G2P — the one that matters most

**The question.** Mongolian orthography is non-phonemic: non-initial short
vowels reduce or delete, vowel harmony is pervasive, and `е ё ю я` are iotated.
A character-level tokenizer must learn all of it implicitly from ~30 h.

**What the paper already establishes.** Tab. 9's conclusion is that the design
"enables stable training to learn speech-text alignment **without
grapheme-to-phoneme** with varying data amounts". So this is not an unforced
error — it is the method working as documented.

**What it does not establish.** That evidence is English (LJSpeech/LibriTTS), an
orthography with far less reduction than Mongolian Cyrillic, on a
single-speaker in-set test. Extending "char-level suffices" to a non-phonemic
Cyrillic orthography is a new claim.

**Protocol.** Same corpus, same schedule, same seeds. Arm A is the current
character vocabulary. Arm B replaces the text with a phoneme sequence and
extends the vocabulary with those symbols instead — which resets the vocabulary
argument entirely, so the comparison is only fair if both arms start from the
same checkpoint with the same number of new rows. Report CER, SIM-o and CMOS.

**Caveat that must be stated first.** Until `normaliser-review.md` is filled in,
the character stream *contains refusals* — clips with most numeral suffixes are
dropped. No conclusion about character-level modelling can be drawn from a
corpus that excludes them.

## 2. Vocabulary extension vs reuse

Does appending five rows beat mapping `ө ү Ө Ү Ъ` onto their nearest existing
letters? The extension is obviously right in principle, but it is five
randomly-initialised rows against 65 that at least exist, and the measured
evidence (§ README) is that the 65 carry little training signal either. Cheap to
run, and it directly tests the claim the whole approach rests on.

## 3. Gate strictness vs final quality

Every threshold in `pipeline/constants.py` is calibrated for *yield*, not for
downstream quality — the calibration report says what each gate keeps, and
nothing says whether keeping more makes the model worse. Train on the strict
corpus and on a deliberately looser one (drop DNSMOS by 0.4, alignment by 0.1)
and compare. This is the ablation that says whether the 24–48 h pass is buying
anything.

## 4. Case preserved vs lowercased

Upstream: *"Uppercased letters (best with form like K.F.C.) will be uttered
letter by letter"*. Every Mongolian sentence begins with a capital, so the
finetune must weaken that prior on essentially every utterance. The runbook's
smoke test catches the gross failure; this measures the cost when it is not
gross.

## 5. Male-voice data floor

Male hours are the binding acceptance criterion, and MBSpeech is one narrator
supplying a large share of them. Train with and without MBSpeech and compare
male SIM-o and CMOS. If the male voice is mostly that one narrator, the model
card should say so rather than implying a general male voice.

## What each ablation costs

Arms 2, 4 and 5 are re-runs of an already-tuned recipe. Arm 1 needs a Mongolian
G2P and a second vocabulary. Arm 3 needs a second corpus pass. Nothing here
needs new recordings, which is the one thing this project cannot buy.
