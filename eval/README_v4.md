---
language:
- mn
license: cc-by-4.0
library_name: f5-tts
pipeline_tag: text-to-speech
base_model: SWivid/F5-TTS
base_model_relation: finetune
datasets:
- btsee/mbspeech-mn
- btsee/fleurs-mn
- btsee/common-voice-26-mn
metrics:
- cer
- utmos
tags:
- text-to-speech
- tts
- mongolian
- khalkha
- cyrillic
- flow-matching
- f5-tts
- dit
- vocos
- voice-cloning
model-index:
- name: oron-tts
  results:
  - task:
      type: text-to-speech
      name: Text-to-Speech
    dataset:
      type: btsee/common-voice-26-mn
      name: Common Voice 26 Mongolian (cleaned)
      split: withheld
    metrics:
    - type: cer
      value: 0.0508
      name: CER, male voice
    - type: cer
      value: 0.063
      name: CER, female voice
    - type: utmos
      value: 2.949
      name: UTMOS, male voice
    - type: utmos
      value: 2.951
      name: UTMOS, female voice
    - type: cosine_similarity
      value: 0.741
      name: Speaker similarity, male
    - type: cosine_similarity
      value: 0.7171
      name: Speaker similarity, female
---

# OronTTS — Mongolian text to speech

Speaks Mongolian (Khalkha, Cyrillic) in two fixed voices, one male and one
female. A finetune of F5-TTS on cleaned Mongolian speech from three public
corpora.

## Listen

Male:

<audio controls src="https://huggingface.co/btsee/oron-tts/resolve/main/demos/male.wav"></audio>

Female:

<audio controls src="https://huggingface.co/btsee/oron-tts/resolve/main/demos/female.wav"></audio>

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

ckpt  = hf_hub_download("btsee/oron-tts", "model.safetensors")
vocab = hf_hub_download("btsee/oron-tts", "vocab.txt")
ref   = hf_hub_download("btsee/oron-tts", f"voices/{VOICE}.wav")
rtxt  = hf_hub_download("btsee/oron-tts", f"voices/{VOICE}.txt")

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

Measured on the shipped `voices/` prompts, over held-out sentences never
used to select anything: n=201 per voice. Micro-CER and mean UTMOS, 95%
bootstrap intervals.

| | male | female |
| --- | --- | --- |
| CER | 0.0508 [0.0466–0.0552] | 0.0630 [0.0585–0.0676] |
| UTMOS | 2.95 [2.91–2.99] | 2.95 [2.92–2.98] |
| speaker similarity to own prompt | 0.741 | 0.717 |

The voices score 0.105 against each other; here real same-speaker
pairs score 0.540–0.833, different speakers
0.034–0.503.

Per-checkpoint numbers in `eval.json`, curves in TensorBoard.

## Limits

* **The output is wideband, roughly 8 kHz -- not full-band**: the corpus pipeline
  decoded to 16 kHz before measuring. Not a limit of Mongolian audio; fixed
  after this model.
* **The CER above is optimistic**: its scorer is fine-tuned on Common Voice,
  which is most of this model's training data, so read every CER against the
  ground-truth human floor, never against zero.
* **No listening test has been run** -- UTMOS is a proxy trained on English and
  Japanese MOS data, never validated for Mongolian.
* **There is no watermarking**; audio from this model cannot be detected as
  synthetic by any downstream tool.
* **Consent:** Common Voice contributors dedicated their recordings CC0 for
  speech research and did not consent to having their individual voice cloned.
* **The normaliser refuses** numeral case suffixes it cannot expand without
  guessing, so a little input is rejected rather than mispronounced.

## Links

* Code: [github.com/btseee/oron-tts](https://github.com/btseee/oron-tts)
* Full card: [docs/model-card.md](https://github.com/btseee/oron-tts/blob/main/docs/model-card.md)
  -- the longer version, with the full method and every caveat
* Corpus tooling: [github.com/btseee/oron-cleaner](https://github.com/btseee/oron-cleaner)
* Training data: [mbspeech-mn](https://huggingface.co/datasets/btsee/mbspeech-mn),
  [fleurs-mn](https://huggingface.co/datasets/btsee/fleurs-mn),
  [common-voice-26-mn](https://huggingface.co/datasets/btsee/common-voice-26-mn)

## Licence

CC-BY-4.0. Attribution is required because FLEURS is CC-BY-4.0. It does **not**
train on WorldSpeech, so it carries no non-commercial restriction.
