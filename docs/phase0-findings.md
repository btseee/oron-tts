# Phase 0 findings

Measurements taken 2026-09-01 before committing to a training run. Everything here
is measured, not estimated. Reproduce with `scripts/extend_vocab.py` and
`pytest tests/test_vocab_coverage.py`.

## 1. The previous run overfit from epoch 250

Parsed from `btsee/oron-tts` `tb_logs/events.out.tfevents.1778934899...` (42 MB,
448,701 events, 74,500 steps / 500 epochs on 3,846 MBSpeech clips).

| epoch | train_loss | val_loss |
|------:|-----------:|---------:|
| 1 | 8.7296 | 7.9836 |
| 10 | 0.9870 | 0.7783 |
| 50 | 0.7226 | 0.4832 |
| 100 | 0.6829 | 0.4640 |
| 200 | 0.6368 | 0.4532 |
| **250** | 0.6291 | **0.4523** ← argmin |
| 300 | 0.6011 | 0.4528 |
| 400 | 0.5804 | 0.4562 |
| 500 | 0.5718 | 0.4591 |

`val_loss` bottoms at epoch 250 and rises monotonically thereafter while
`train_loss` keeps falling — textbook overfitting. Half the run (250 epochs,
~37,000 steps) was wasted or harmful. Improvement from epoch 50 to 250 was only
6%, so the model had effectively saturated on this corpus by epoch 50.

Two caveats that make even the argmin optimistic:

- The 90/10 split is random at row level over a **single-speaker** corpus, so
  validation is fully in-distribution. It measures memorisation, not generalisation.
- No audio or mel summaries were written despite `audio_sample_interval: 10`, so
  the run has no perceptual record at all.

## 2. Mongolian survives F5-TTS tokenization intact

`convert_char_to_pinyin` was assumed to be safe for Cyrillic from code reading.
Confirmed at runtime on 300 real sentences (100 each from `btsee/fleurs-mn`,
`btsee/common-voices-24-mn`, `btsee/mbspeech_mn`):

**300/300 sentences round-trip exactly** — every Cyrillic character, space and
punctuation mark preserved as its own token.

Mechanism: Cyrillic is 2 bytes/char in UTF-8, so a Cyrillic segment matches
neither the pure-ASCII branch (`byte_len == len`) nor the pure-CJK branch
(`byte_len == 3*len`) and falls through to the verbatim `else` branch.

```
in : Сайн байна уу? Өнөөдөр үүлшинэ, 25 хэм.
out: ['С','а','й','н',' ','б','а','й','н','а',' ','у','у','?',' ',
      'Ө','н','ө','ө','д','ө','р',' ','ү','ү','л','ш','и','н','э',',',
      ' ','2','5',' ','х','э','м','.']
```

## 3. The base vocab silently corrupts 4.90% of tokens

`list_str_to_idx` maps any out-of-vocabulary character to **index 0, which is the
SPACE token** — there is no `<unk>`. Coverage gaps are therefore invisible:
training just sees spaces where letters should be.

Measured on the token stream (post-`convert_char_to_pinyin`, which is what
actually gets indexed):

| vocab | entries | distinct OOV | OOV rate |
|---|---:|---|---:|
| F5-TTS base | 2545 | `Ө ө Ү ү` + `\xa0` | **4.90%** |
| extended | 2550 | `\xa0` only | **0.01%** |

`Ө ө Ү ү` are ordinary Mongolian vowels, hence the high rate. Training on the
un-extended vocab replaces roughly one character in twenty with a space.

## 4. Corrections to earlier assumptions

Three things measured differently than expected:

- **Curly quotes are not a problem.** `convert_char_to_pinyin`'s `custom_trans`
  rewrites `“ ”` to `"` before indexing, and `"` is in the base vocab. They only
  look like OOV if you measure raw text instead of the token stream.
- **Em dash `—` U+2014 is already in the base vocab.** No mapping needed.
- **Latin should be kept, not deleted or rejected.** All 22 Latin characters
  occurring in the corpus are already in the base vocab with pretrained
  embeddings, and Latin appears in only 2.7% of rows. `remove_invalid_chars`
  currently deletes them silently, which desynchronises text from audio — the
  exact failure the CER gate exists to catch. Only `\xa0` needs normalising.

Corpus charset over the 300-sentence sample: 113 distinct characters — 64
Cyrillic, 22 Latin, 10 digits, 15 punctuation. Digits appear in 7.0% of rows and
are handled by `number_norm.py`.

`Ъ` U+042A did not occur in the sample. It is included in the extension anyway:
one extra embedding row is free, whereas rejecting a row over a rare character
is not.

## 5. Vocabulary extension is 5 tokens

Base vocab already contains 61 of the 66 Mongolian Cyrillic letters with
pretrained embeddings (lines 1628–1693). Missing: `ө ү Ө Ү Ъ`.

Case is **kept**, not folded. The base vocab has both cases of Cyrillic with
pretrained embeddings; lowercasing would discard 31 trained rows to save 3
appended ones.

```
2545 entries -> 2550 entries      (text embedding 2546 -> 2551 rows)
```

Order is load-bearing: new tokens are appended, so every pretrained index is
preserved. A regenerated "sorted unique characters" vocab would misalign all 2545.

## 6. MBSpeech is one male speaker — confirmed

MBSpeech carries no gender field. Measured median F0 (librosa pyin, voiced frames
only) against self-declared Common Voice labels as calibration:

| corpus | n | median F0 | range |
|---|---:|---:|---|
| CV24 `male_masculine` | 8 | 115.1 Hz | 103–131 |
| CV24 `female_feminine` | 6 | 239.3 Hz | 232–244 |
| **MBSpeech** | 10 | **140.7 Hz** | 112–157 |

Unambiguously male. The 45 Hz spread across 10 clips is consistent with one
speaker's prosodic range in narration, supporting the single-speaker claim.

Consequence for the male/female requirement: MBSpeech is the cleanest male audio
available (professional narration, ~6.3 h) but it is **16 kHz**, so it cannot
supply the full-band male reference clip. That must come from Common Voice.

## 7. The cleaner's CER gate is calibrated to the wrong ASR

25 FLEURS test clips, scored against human `raw_transcription`, both sides
normalised (casefold, punctuation stripped, whitespace collapsed):

| model | params | CER median | CER mean | p90 |
|---|---:|---:|---:|---:|
| `bayartsogt/wav2vec2-large-xlsr-mongolian` | 315M | **0.123** | 0.134 | 0.210 |
| whisper-large-v3 *(dataset's own values, same clips)* | 1.5B | 0.311 | 0.311 | — |
| `Mengkedalai/w2v-bert-2.0-mongolian-170h_crl` | 606M | 1.000 | 1.000 | 1.000 |

**wav2vec2-xlsr more than halves Whisper's error floor**, at a fifth of the
parameters. This matters directly: oron-cleaner gates on Whisper CER ≤ 0.35 with
a rescue to 0.50, but Whisper's *own* floor on clean, correctly-transcribed
Mongolian is 0.31. The gate has almost no headroom — it is thresholding the
ASR's error, not the transcript's. With a 0.123 floor a threshold near 0.15–0.20
starts to mean something.

The w2v-bert result is **not** a verdict on that model. It emitted empty output
on every clip, which points at a processor mismatch in this harness (w2v-bert-2.0
expects `SeamlessM4TFeatureExtractor` features, not raw Wav2Vec2 input values).
Diagnosis timed out on CPU and was not pursued, since wav2vec2-xlsr already
answers the question.

**Decision:** `bayartsogt/wav2vec2-large-xlsr-mongolian` becomes the scorer for
both the cleaner's CER signal and the Phase 4 TTS evaluation. It stays a
*relative* signal in the cleaner, per the plan — forced alignment is the primary
transcript gate.

## Open items

- **Blocked:** corpus measurement of Common Voice 25 — the Mozilla Data
  Collective endpoint needs `API_KEY`, which is not present in the environment or
  in either repo's `.env`. All CV figures used so far are measured from v20/v24
  mirrors on HuggingFace.
