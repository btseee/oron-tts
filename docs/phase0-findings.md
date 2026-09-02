# Phase 0 findings

Measurements taken 2026-09-01 before committing to a training run. Everything here
is measured, not estimated. Reproduce with `scripts/extend_vocab.py` and
`pytest tests/test_vocab_coverage.py`.

## 1. The previous run overfit from epoch 250

Parsed from `btsee/oron-tts` `tb_logs/events.out.tfevents.1778934899...` (42 MB,
448,701 events, 74,500 steps / 500 epochs on 3,846 MBSpeech clips).

| epoch | train_loss | val_loss |
| ------: | -----------: | ---------: |
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

```shell
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
| --- | ---: | --- | ---: |
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

Base vocab already contains 65 of the 70 Mongolian Cyrillic letters
(lines 1628–1693). Missing: `ө ү Ө Ү Ъ`.

Their *embeddings* are another matter, and an earlier version of this line said
"with pretrained embeddings" without checking. The paper says why those rows
exist (§5.1): "all other language characters exist in the Emilia dataset as
there are many code-switched sentences" — Emilia being Chinese/English podcast
audio. Measured on the checkpoint, Cyrillic rows (mean ‖row‖ 14.177, std 0.627)
are indistinguishable from Hangul (14.174 / 0.627) and the table mean
(14.108 / 0.624), while high-frequency ASCII lowercase sits apart at
13.576 / 0.600. Consistent with little training signal, though norm statistics
alone cannot prove it. The reused value is the acoustic prior in the DiT, not
the vocabulary.

Case is **kept**, not folded. The base vocab has both cases of Cyrillic with
pretrained embeddings; lowercasing would discard 31 trained rows to save 3
appended ones.

```shell
2545 entries -> 2550 entries      (text embedding 2546 -> 2551 rows)
```

Order is load-bearing: new tokens are appended, so every pretrained index is
preserved. A regenerated "sorted unique characters" vocab would misalign all 2545.

## 6. MBSpeech is one male speaker — confirmed

MBSpeech carries no gender field. Measured median F0 (librosa pyin, voiced frames
only) against self-declared Common Voice labels as calibration:

| corpus | n | median F0 | range |
| --- | ---: | ---: | --- |
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
| --- | ---: | ---: | ---: | ---: |
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

## 8. No Mongolian source is full-band — the model ceiling is ~8 kHz

Measured as the highest frequency whose mean band power is within 40 dB of the
spectral peak, i.e. an actual lowpass shelf. Cumulative-energy rolloff was tried
first and rejected: speech energy is dominated by sub-1 kHz formants, so it
underestimates. FLEURS and MBSpeech act as controls — both are 16 kHz native, so
a correct estimator must show them capping just under the 8 kHz Nyquist.

| corpus | container | p10 | median | p90 | max | ≥8 kHz | ≥10 kHz |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Common Voice 24 mn (n=120) | 48 kHz | 5,578 | 7,148 | 11,109 | 15,938 | 32.5% | 22.5% |
| FLEURS-mn (n=20) | 24 kHz | — | 7,664 | — | **7,723** | 0% | 0% |
| MBSpeech (n=20) | 16 kHz | — | 7,570 | — | **7,695** | 0% | 0% |

Both controls cap at ~7.7 kHz exactly as predicted, so the estimator is sound.

**Common Voice's 48 kHz container is misleading.** It is crowd-sourced phone and
laptop audio at low mp3 bitrate; the encoder has already discarded the top of the
band. Its *median* cutoff (7.1 kHz) is slightly **worse** than the 16 kHz-native
corpora, and only 1.7% of clips reach 15 kHz.

Three consequences:

- **Corrects the plan.** "Common Voice is the only large full-band source" is
  wrong. The accurate statement is narrower: CV holds the only content above
  8 kHz at all (32.5% of clips), while FLEURS and MBSpeech are hard-capped.
- **The bandwidth gate must be ≥7 kHz, not higher.** That keeps 59% of Common
  Voice and essentially all of FLEURS/MBSpeech. A gate at 10 kHz would discard
  77% of the corpus for a quality tier that barely exists.
- **A genuinely full-band reference voice is not obtainable.** Output bandwidth
  follows the reference clip, so the realistic target is **wideband (~8 kHz)**,
  not full-band. This should be stated in the model card rather than discovered
  by a listener.

Reference-voice candidates are still ample despite the compounding filters:
~96k CV clips × ~11% labelled male × 22.5% at ≥10 kHz ≈ 2,400 candidate male
clips before quality ranking, and only one good one is needed per gender.

## 9. Common Voice 24 Mongolian, measured from its own TSVs

Read from `cv-corpus-24.0-2025-12-05/mn/` via the CC0 mirror
`onlysainaa/common-voice-mn-24`. `clip_durations.tsv` gives exact durations, so
these are counted hours, not estimates.

**96,308 clips / 140.6 hours total.**

| split | clips | hours | speakers | male / female / unlabelled |
| --- | ---: | ---: | ---: | --- |
| **validated** | 33,331 | **46.9** | 511 | 7,520 / 12,574 / 13,237 |
| other *(unvalidated)* | 58,460 | 82.2 | 272 | 4,897 / 24,601 / 28,962 |
| invalidated *(voter-rejected)* | 3,164 | 9.4 | 371 | — |
| train / dev / test | 6,018 | 9.1 | — | subsets of validated |

**Use `validated` only: 46.9 h.** The plan's "~123 h" figure came from
`validated ∪ other`, which is what `Blgn94/mongolian-stt-dataset` blends and
which its own card flags as *"a deliberate volume-over-purity choice"* —
`other` is not human-confirmed and contains misreads. That is incompatible with a
strict tier. `invalidated` was actively rejected by voters and is excluded.

### The male-voice risk is smaller than projected

Within `validated`:

| | hours | speakers |
| --- | ---: | ---: |
| male_masculine | **10.7** | 114 |
| female_feminine | 17.3 | 43 |
| unlabelled / declined | — | (13,237 clips, 39.7%) |

Earlier projection from a v20 sample was ~8 h of male speech; the actual figure
is **10.7 h across 114 speakers**, and MBSpeech adds 6.3 h more. The ≥5 h male
floor in the plan's go/no-go is met with margin, from many speakers rather than
one. Note the inversion: there are 2.6× more male speakers but fewer male hours —
female contributors are individually far more prolific.

### Two gates the pipeline is not yet using

- **Speaker concentration.** The top 10 of 511 speakers hold **45.7%** of
  validated clips; the largest single contributor has 1,968. Without a
  per-speaker cap the model will collapse toward a handful of voices.
- **`down_votes`.** 4,377 validated clips (13.1%) carry at least one down-vote —
  a free, high-precision quality signal that oron-cleaner reads and discards.

### Revised budget for the strict tier

| source | raw hours |
| --- | ---: |
| Common Voice `validated` | 46.9 |
| FLEURS-mn (already filtered) | 12.3 |
| MBSpeech | 6.3 |
| **total** | **~65** |

Comfortably above the 25 h go/no-go before quality filtering.

## 10. Common Voice 26 adds nothing for Mongolian

`Common Voice Scripted Speech 26.0 - Mongolian`
(`cmqinq6zs00x8nr07elg0nyrr`, CC0-1.0, 2.87 GB, released 2026-06-17) against
v24 (2025-12-05):

| | v24 | v26 |
| --- | ---: | ---: |
| validated clips | 33,331 | 33,258 |
| **validated hours** | **46.9** | **46.9** |
| validated speakers | 511 | 520 |
| male | 10.7 h / 114 spk | 10.6 h / 114 spk |
| female | 17.3 h / 43 spk | 17.3 h / 43 spk |
| total hours | 140.6 | 130.3 |

**Mongolian gained essentially no validated speech in six months.** The corpus
has plateaued; total hours actually *fell*, because `invalidated` shrank from
9.4 h to 4.5 h as clips were re-adjudicated. Use v26 because it is the current
authoritative CC0 release, not because it adds data — and do not plan on Common
Voice growing.

After the `down_votes > 0` gate (4,400 clips, 13.2%), the usable input is
**28,858 clips / 40.3 h**, of which male is **9.1 h across 109 speakers**.

### Access notes

The API key authenticates correctly; the original blocker was purely a stale
dataset ID. Mozilla Data Collective publishes **each language of each release as
its own dataset**, so the ID is release- and language-specific, and there is no
list or search endpoint — a dead ID can only be replaced by hand from the
dataset page URL. Downloads additionally require the account to have accepted
that dataset's terms, and are capped at 30/day per organisation. The endpoint
sits behind Cloudflare, which rejects `requests`/`urllib` default User-Agents
with error 1010 before the request reaches the API.

## Open items

- `pipeline/datasets/common_voice.py` cannot be imported without pulling
  `torchcrepe`, `silero_vad`, `whisper` and `torchmetrics`, because it imports
  `audio_filter` at module level. That is why oron-cleaner's tests cannot run
  without the full ML stack, and it should be separated in Phase 2.
