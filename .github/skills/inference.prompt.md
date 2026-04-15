---
mode: agent
description: Run OronTTS inference — synthesise speech with voice cloning or attribute tokens, in Python or via CLI.
---

# OronTTS Inference

Use this skill when the user asks about generating speech, voice cloning, inference parameters, or running `scripts/infer.py`.

## Environment

```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
```

## Two inference modes

### Option A — Voice cloning

Conditions the DiT on a reference mel computed from a real recording.
Pass a 3–10 s WAV and its transcript.

```python
from src.models.f5tts import F5TTS

model = F5TTS.from_config(config)
model.load_state_dict(...)
model.eval()

audio = model.synthesize(
    text="Сайн байна уу",
    lang="mn",
    ref_audio_path="ref.wav",
    ref_text="Энэ бол жишээ өгүүлбэр",
    n_steps=32,
    cfg_strength=2.0,
    sway_sampling_coef=-1.0,
)
# audio: torch.Tensor [T] at 24 000 Hz
```

### Option B — Attribute tokens

No reference audio — speaker style is controlled by prefix tokens embedded into the sequence.

```python
audio = model.synthesize(
    text="Сайн байна уу",
    lang="mn",
    attr_tokens=["[FEMALE]", "[YOUNG]"],
    n_steps=32,
)
```

Available attribute tokens: `[FEMALE]`, `[MALE]`, `[YOUNG]`, `[MIDDLE]`, `[ELDERLY]`.

## CLI

```bash
# Voice cloning
python scripts/infer.py \
    --text "Сайн байна уу" \
    --lang mn \
    --ref-audio ref.wav \
    --ref-text "Энэ бол жишээ өгүүлбэр" \
    --output out.wav \
    --steps 32

# Attribute tokens
python scripts/infer.py \
    --text "Сайн байна уу" \
    --lang mn \
    --attr-tokens "[FEMALE],[YOUNG]" \
    --output out.wav

# Kazakh
python scripts/infer.py \
    --text "Сәлеметсіз бе" \
    --lang kz \
    --attr-tokens "[FEMALE]" \
    --output out_kz.wav

# Override target duration (seconds)
python scripts/infer.py \
    --text "Сайн байна уу" \
    --lang mn \
    --duration 3.5 \
    --output out.wav
```

## Key parameters

| Arg | Default | Notes |
|-----|---------|-------|
| `--steps` | 32 | Euler ODE integration steps; more = slower but smoother |
| `--duration` | `None` | Override predicted duration in seconds |
| `--lang` | `mn` | `mn` (Mongolian) or `kz` (Kazakh) |

> **Note**: `cfg_strength` (default 2.0) and `sway_sampling_coef` (default -1.0) are available in the Python API via `F5TTS.synthesize()` but not yet exposed as CLI args.

## F5TTS.synthesize() signature

```python
def synthesize(
    self,
    text: str,
    lang: str = "mn",
    attr_tokens: list[str] | None = None,
    ref_audio_path: str | Path | None = None,
    ref_text: str | None = None,
    n_steps: int = 32,
    cfg_strength: float = 2.0,
    sway_sampling_coef: float | None = -1.0,
    target_duration_s: float | None = None,
    device: str = "cuda",
) -> torch.Tensor:  # [T] at sample_rate
```

## Output audio

- Output is a 1-D `torch.Tensor` at **24 000 Hz**.
- Save with `torchaudio.save(path, audio.unsqueeze(0), 24000)` or `AudioProcessor.save_audio(path, audio)`.
