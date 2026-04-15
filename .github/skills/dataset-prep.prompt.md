---
mode: agent
description: Prepare TTS datasets for OronTTS — clean text, denoise audio, save WAV + metadata.json, optionally upload to Hugging Face.
---

# OronTTS Dataset Preparation

Use this skill when the user asks about preparing, processing, or cleaning audio datasets for training OronTTS.

## Key classes

| Class | File | Purpose |
|-------|------|---------|
| `TextCleaner` | `src/utils/text_cleaner.py` | Unicode NFC, punctuation map, number expansion, char filter, lowercase |
| `AudioProcessor` | `src/utils/audio.py` | Load, resample, normalize, trim silence, save WAV |
| `AudioDenoiser` | `src/data/denoiser.py` | DeepFilterNet denoising (lazy-loaded, target_sr=24000) |
| `CommonVoiceWrapper` | `src/data/hf_wrapper.py` | Loads `btsee/common-voices-24-mn` from HF |
| `MBSpeechWrapper` | `src/data/hf_wrapper.py` | Loads `btsee/mbspeech_mn` from HF |
| `TTSDataset` | `src/data/dataset.py` | Reads metadata.json, returns mel/text_ids/mask/lang |

## Audio requirements

- **Sample rate**: 24 000 Hz (all pipeline defaults are 24000)
- **Mel bins**: 100, n_fft 1024, hop_length 256
- Minimum audio length after trimming: 1024 samples (~43 ms)
- Duration gate in `clean_local_cv.py`: 0.5 s – 15.0 s

## metadata.json format

Each entry produced by `prepare.py` or `clean_local_cv.py`:

```json
{
  "audio_path": "data/processed/cv_mn/audio/cv_000042.wav",
  "text": "сайн байна уу"
}
```

`TTSDataset` accepts an optional `lang` field per entry; defaults to the `--lang` arg.

## Environment setup

```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
# DeepFilterNet denoising is optional and requires Rust/Cargo:
.venv\Scripts\pip install -e ".[denoise]"
```

## Dataset: btsee/mbspeech_mn

- Columns: `audio` (16 kHz bytes), `sentence_norm` (preferred), `sentence_orig`
- No `sentence` column — use `sentence_norm`
- `TTSDataset.from_hf_dataset` auto-detects `sentence_norm` first

## CLI commands

From HF datasets:

```bash
python scripts/prepare.py \
    --output-dir data/processed \
    --dataset common-voice \   # or mbspeech / all
    --sample-rate 24000 \
    --max-samples 5000 \
    --upload \
    --hf-repo btsee/oron-tts-dataset
```

From a local Common Voice tar.gz (no HF required):

```bash
python scripts/clean_local_cv.py \
    --input cv_mn.tar.gz \
    --output-dir data/processed/cv_mn \
    --sample-rate 24000 \
    --max-samples 5000 \
    --skip-denoise          # omit DeepFilterNet for speed
```

## Pipeline order

1. `TextCleaner.clean(text)` — normalize → filter → lowercase
2. `AudioDenoiser.denoise(audio, sr)` — DeepFilterNet (48 kHz internally, resampled to target_sr)
3. `AudioProcessor.normalize_audio(audio)` — peak normalize to ±1.0
4. `AudioProcessor.trim_silence(audio)` — librosa top_db trim
5. Write WAV with `soundfile.write` or `AudioProcessor.save_audio`
6. Append to metadata.json

## TextCleaner allowed characters

Mongolian Cyrillic + Kazakh-specific + punctuation only.
Characters outside `ALLOWED_CHARS` are silently dropped.
Numbers are expanded via `NumberNormalizer` before char filtering.
