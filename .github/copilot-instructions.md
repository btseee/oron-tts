# OronTTS Project Instructions

## General Principles
- **Language:** Python 3.12 (exactly). Strict type hints on every function.
- **Style:** Functional and modular. Avoid monolithic files.
- **Documentation:** Minimal, meaningful comments only. Self-documenting code.
- **No Coqui, no ESPnet, no Coqui-TTS.** All model code is native PyTorch.
- **No VITS.** The architecture is F5-TTS (Flow Matching + Diffusion Transformer). Do not introduce GAN, HiFi-GAN, discriminators, or duration predictors.

## Architecture: F5-TTS

OronTTS is a non-autoregressive TTS system for Mongolian (Khalkha Cyrillic) and Kazakh (Cyrillic) based on the F5-TTS paper.

### Model stack
| Component | Class | File |
|-----------|-------|------|
| Text encoder | `TextEmbedding` | `src/models/encoder.py` |
| Backbone | `DiT` (Diffusion Transformer) | `src/models/dit.py` |
| Flow matching | `CFM` (OT-Conditional Flow Matching) | `src/models/flow.py` |
| Vocoder | `VocosDecoder` (ConvNeXt + iSTFT) | `src/models/decoder.py` |
| Top-level | `F5TTS` | `src/models/f5tts.py` |
| DiT blocks | `RMSNorm, AdaLayerNorm, Attention, DiTBlock, ...` | `src/models/modules.py` |

### Training
- Flow matching loss (MSE on velocity field) is computed inline in `CFM.forward()` — no separate loss function.
- **Infilling training**: random 70–100% span masking. The unmasked portion is the conditioning signal.
- **CFG dropout**: `audio_drop_prob=0.3`, `cond_drop_prob=0.2` (drops audio/conditioning during training for classifier-free guidance).
- Optimizer: AdamW. Scheduler: LinearLR warmup. EMA maintained via `torch_ema`.
- Trainer class: `F5Trainer` (`src/training/trainer.py`).

### Inference modes
1. **Voice cloning** — pass `ref_audio_path` to `F5TTS.synthesize()`, uses reference mel as conditioning.
2. **Attribute tokens** — pass `attr_tokens` list (e.g. `["[FEMALE]", "[YOUNG]"]`), embedded as prefix.
3. **Inference params**: `cfg_strength` (classifier-free guidance, default 2.0), `sway_sampling_coef` (sway sampling, default -1.0), `n_steps` (ODE steps, default 32).

## Tokenizer

`CyrillicTokenizer` (`src/utils/tokenizer.py`) — character-level, vocab size **65**.

- 35 Mongolian Cyrillic (lower) + 7 Kazakh-specific (`әғқңұһі`) + 12 punctuation/space
- Special tokens: `<PAD>=0, <BOS>=1, <EOS>=2, <UNK>=3`
- Language tags: `[LANG_MN]=4, [LANG_KZ]=5`
- Attribute tags: `[FEMALE]=6, [MALE]=7, [YOUNG]=8, [MIDDLE]=9, [ELDERLY]=10`

Usage:
```python
tokenizer = CyrillicTokenizer()
ids = tokenizer.encode("сайн байна уу", lang="mn", attr_tokens=["[FEMALE]"])
text = tokenizer.decode(ids)
```

`TextCleaner` (`src/utils/text_cleaner.py`) wraps the tokenizer and provides `clean(text)` and `text_to_sequence(text, lang, attr_tokens)`.

## Audio

- **Sample rate:** 24 000 Hz (all defaults must be 24000).
- **Mel bins:** 100. **n_fft:** 1024. **hop_length:** 256.
- `AudioProcessor` in `src/utils/audio.py` — `mel_spectrogram()`, `normalize_audio()`, `trim_silence()`, `save_audio()`.
- `AudioDenoiser` in `src/data/denoiser.py` — wraps DeepFilterNet.

## Project Structure

```
src/
  data/
    dataset.py       # TTSDataset, TTSCollator
    denoiser.py      # AudioDenoiser (DeepFilterNet)
    hf_wrapper.py    # HFDatasetWrapper, CommonVoiceWrapper, MBSpeechWrapper
  models/
    dit.py           # DiT backbone
    decoder.py       # VocosDecoder (Vocos iSTFT vocoder)
    encoder.py       # TextEmbedding
    flow.py          # CFM (OT-CFM) with infilling + CFG
    f5tts.py         # F5TTS top-level model
    modules.py       # DiT building blocks
  training/
    trainer.py       # F5Trainer
  utils/
    audio.py         # AudioProcessor
    checkpoint.py    # CheckpointManager (single optimizer + EMA + safetensors)
    number_norm.py   # NumberNormalizer (Mongolian cardinal + ordinal)
    text_cleaner.py  # TextCleaner
    tokenizer.py     # CyrillicTokenizer
configs/
  local.yaml         # F5-TTS Small: dim=512, depth=12, heads=8
  runpod.yaml        # F5-TTS Base: dim=1024, depth=22, heads=16
scripts/
  prepare.py         # Clean + denoise + upload to HF
  train.py           # Pull from HF + train + push model
  infer.py           # Run synthesis
  clean_local_cv.py  # Process local Common Voice tar.gz without HF
  setup/
    runpod_setup.sh  # RunPod: install deps, auth wandb, smoke test
```

## Configs

Both YAML configs use the same keys. Critical keys:
- `model.dim`, `model.depth`, `model.heads`, `model.ff_mult`, `model.vocab_size` (must match tokenizer = 65)
- `sample_rate` (must be 24000), `n_mels` (must be 100) — top-level YAML keys, not nested
- `wandb_project: "oron-tts"` — Weights & Biases project name (activates logging when set)
- `batch_size`, `warmup_steps`, `num_epochs` — top-level YAML keys, not nested under `training`

## Checkpoints

`CheckpointManager` (`src/utils/checkpoint.py`):
- `save(model, optimizer, ema, step, path)` — saves `.pt` checkpoint
- `load(path, model, optimizer, ema)` — loads `.pt` checkpoint
- `load_pretrained_f5tts(path, model)` — loads official F5-TTS `.safetensors` weights with key remapping

## Formatting and Quality

- **Linter/Formatter:** Ruff + Black.
- **Imports:** isort.
- **Typing:** Strict. No `Any` unless unavoidable. Use `Final` for module-level constants.

## Workflow

1. **Local prep:** `python scripts/prepare.py --output-dir data/processed --dataset common-voice`
   - Loads from HF → cleans text → denoises audio → saves WAV + metadata.json
   - Or use `scripts/clean_local_cv.py --input cv_mn.tar.gz` for local archives.
2. **Cloud training:** `python scripts/train.py --config configs/runpod.yaml`
   - Pulls dataset from HF → trains F5-TTS → saves checkpoints → logs metrics to wandb.
3. **Inference:** `python scripts/infer.py --text "Сайн байна уу" --lang mn --output out.wav`
   - Optionally `--ref-audio ref.wav` for voice cloning.

## Environment Setup

```powershell
# Use exactly Python 3.12 — the only fully supported version
py -3.12 -m venv .venv
.venv\Scripts\pip install -e ".[dev]"

# Secrets — create .env at repo root (never commit)
# HF_TOKEN=hf_...         (HuggingFace personal access token)
# WANDB_API_KEY=...        (Weights & Biases — get from wandb.ai/settings)

# First-time wandb login (writes token to ~/.netrc)
.venv\Scripts\python -m wandb login

# Verify
.venv\Scripts\python scripts/test_pipeline.py        # synthetic only
.venv\Scripts\python scripts/test_pipeline.py --hf   # + real btsee/mbspeech_mn
```

### Dataset: btsee/mbspeech_mn

- 3,846 samples, split `train` only
- Audio column: `audio` — stored as raw bytes (16 kHz), resampled to 24 kHz at load time
- Text columns: `sentence_norm` (preferred) and `sentence_orig`
- `TTSDataset.from_hf_dataset` auto-detects `sentence_norm` via the candidates list
- `sentence` column does **not** exist — the dataset card is incorrect

## Hardware Targets

- **Local:** CPU/GPU for dataset prep and small test runs (Small config).
- **Cloud:** NVIDIA A100 or L40 on RunPod for full training (Base config).