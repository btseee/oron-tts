# OronTTS

Non-autoregressive TTS for Mongolian (Khalkha Cyrillic) and Kazakh (Cyrillic) using [F5-TTS](https://arxiv.org/abs/2410.06885) — Flow Matching + Diffusion Transformer.

## Features

- **F5-TTS**: OT-CFM + DiT backbone. No GAN, no duration predictor, single MSE loss.
- **Voice cloning**: pass a 3–10 s reference WAV at inference time.
- **Attribute tokens**: `[FEMALE]`, `[MALE]`, `[YOUNG]`, `[MIDDLE]`, `[ELDERLY]`.
- **Bilingual**: Mongolian + Kazakh Cyrillic, character-level tokenizer (vocab 65).
- **Number normalisation**: Mongolian cardinal and ordinal.
- **Audio denoising**: DeepFilterNet for preprocessing non-professional recordings.

## Installation

**Windows (local dev)**:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
```

**Linux / RunPod**:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Project Structure

```
src/
  data/       # TTSDataset, AudioDenoiser, HF wrappers
  models/     # DiT, CFM, VocosDecoder, F5TTS, TextConvEmbed
  training/   # F5Trainer, cfm_loss
  utils/      # AudioProcessor, CyrillicTokenizer, TextCleaner, CheckpointManager
configs/
  local.yaml    # Small (dim=512, depth=12) — local dev
  runpod.yaml   # Base  (dim=1024, depth=22) — cloud training
scripts/
  prepare.py        # clean + denoise + upload to HF
  train.py          # train + push model
  infer.py          # synthesise speech
  clean_local_cv.py # process local Common Voice tar.gz
  setup/
    runpod_setup.sh # RunPod one-shot setup
```

## Usage

### Dataset preparation

From Hugging Face:
```bash
python scripts/prepare.py \
    --output-dir data/processed \
    --dataset all \
    --upload \
    --hf-repo btsee/oron-tts-dataset
```

From a local Common Voice tar.gz:
```bash
python scripts/clean_local_cv.py \
    --input cv_mn.tar.gz \
    --output-dir data/processed/cv_mn \
    --max-samples 5000
```

### Training

**Local** (Small config):
```bash
python scripts/train.py \
    --config configs/local.yaml \
    --dataset btsee/mbspeech_mn
```

**RunPod** (Base config):
```bash
python scripts/train.py \
    --config configs/runpod.yaml \
    --dataset btsee/mbspeech_mn \
    --push-to-hub \
    --hf-repo btsee/orontts
```

Fine-tune from a pretrained F5-TTS checkpoint:
```bash
python scripts/train.py \
    --config configs/runpod.yaml \
    --dataset btsee/mbspeech_mn \
    --pretrain-ckpt F5TTS_Base.safetensors
```

Training auto-resumes from the latest checkpoint if one exists.

### Inference

```bash
# Attribute-token synthesis
python scripts/infer.py \
    --text "Сайн байна уу" \
    --lang mn \
    --attr-tokens "[FEMALE],[YOUNG]" \
    --output out.wav

# Voice cloning
python scripts/infer.py \
    --text "Сайн байна уу" \
    --lang mn \
    --ref-audio ref.wav \
    --ref-text "Энэ бол жишээ өгүүлбэр" \
    --output out.wav

# Kazakh
python scripts/infer.py \
    --text "Сәлеметсіз бе" \
    --lang kz \
    --attr-tokens "[FEMALE]" \
    --output out_kz.wav
```

## Mongolian Numbers

| Input | Output |
|-------|--------|
| 10 | арван |
| 25 | хорин тав |
| 100 | зуун |
| 1-р | нэгдүгээр |
| 2024 | хоёр мянга хорин дөрөв |

## Environment

Create `.env` at the repo root (never commit — already in `.gitignore`):

```
HF_TOKEN=hf_...       # HuggingFace token — read + write scope
WANDB_API_KEY=...     # get from wandb.ai/settings
```

`train.py` and `prepare.py` load it automatically via `python-dotenv`.

## RunPod Training

### Pod settings

| Field | Value |
|-------|-------|
| GPU | **L40S** |
| GPU count | **1** |
| Cloud tier | **Secure Cloud** |
| Template | **RunPod PyTorch 2.4.0** |
| Container disk | **20 GB** |
| Volume disk | **50 GB** |
| Volume mount | `/workspace` |
| `HF_TOKEN` env var | your HuggingFace token (read + write) |
| `WANDB_API_KEY` env var | your W&B key from [wandb.ai/settings](https://wandb.ai/settings) |

The Base config peaks at ~13 GB VRAM; the L40S 48 GB gives a 3.5× margin. Add the env vars in the **Environment Variables** section of the pod creation form.

### Setup

Connect via **Web Terminal**, then:

```bash
tmux new-session -s setup   # keeps running if you close the browser tab

cd /workspace
git clone https://github.com/btsee/oron-tts.git
cd oron-tts
bash scripts/setup/runpod_setup.sh
```

The script installs Python 3.12, creates `.venv`, installs deps, authenticates wandb, and runs a 10-step smoke test. Close the tab at any time — re-attach with `tmux attach -t setup`.

If you skipped the env vars form, create `.env` instead (auto-loaded by `train.py`):
```bash
cat > /workspace/oron-tts/.env <<'EOF'
HF_TOKEN=hf_...
WANDB_API_KEY=...
EOF
```

### Train

```bash
source .venv/bin/activate
python scripts/train.py \
    --config configs/runpod.yaml \
    --dataset btsee/mbspeech_mn \
    --push-to-hub \
    --hf-repo YOUR_HF_USERNAME/orontts
```

Metrics stream to **wandb.ai → project oron-tts**. Checkpoints land on the 50 GB volume and survive pod restarts — re-run the same command to resume.

### Cost

| Scenario | Cost at ~$0.86/hr |
|----------|-------------------|
| Smoke test (< 5 min) | < $0.08 |
| 1 epoch × 3 846 samples | ~$0.21 |
| 500 epochs (full run) | ~$107 |

Terminate the pod (not just stop it) after training to end container disk billing.

## Configuration

Both configs use the same keys (all top-level, no nesting):

```yaml
sample_rate: 24000
n_mels: 100
n_fft: 1024
hop_length: 256

wandb_project: "oron-tts"   # remove to disable logging
wandb_run_name: null

batch_size: 16
warmup_steps: 1000
num_epochs: 500
ema_decay: 0.9999
use_tqdm: true          # set false for RunPod container logs
log_interval: 100

model:
  dim: 512        # 1024 for runpod.yaml
  depth: 12       # 22 for runpod.yaml
  heads: 8        # 16 for runpod.yaml
  vocab_size: 65
```

## Development

```bash
ruff check src/ scripts/
ruff format src/ scripts/
isort src/ scripts/
```

## License

MIT

## Citation

If you use OronTTS in your research, please cite:

```bibtex
@software{orontts2026,
  title  = {OronTTS: Mongolian and Kazakh Text-to-Speech with F5-TTS},
  author = {Badral, Battseren},
  year   = {2026},
  url    = {https://github.com/btsee/oron-tts}
}
```
