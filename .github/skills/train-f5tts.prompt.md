---
mode: agent
description: Train OronTTS F5-TTS models — configure, launch, resume, and monitor training runs locally or on RunPod.
---

# OronTTS F5-TTS Training

Use this skill when the user asks about training, fine-tuning, resuming, or configuring OronTTS model training.

## Architecture constraints

- **Single loss**: `cfm_loss(v_pred, v_target, mask)` — MSE on the velocity field only. Never add discriminator, duration, or KL losses.
- **Optimizer**: AdamW. **Scheduler**: LinearLR warmup then constant.
- **EMA**: maintained on rank-0 only via `torch_ema.ExponentialMovingAverage`.
- **No GAN**, no HiFi-GAN, no `optimizer_d`.

## Trainer API

```python
from src.training.trainer import F5Trainer

trainer = F5Trainer(
    config=config,          # dict loaded from YAML
    model=model,            # F5TTS instance (optionally DDP-wrapped by train.py)
    train_loader=loader,
    log_dir="output/logs",
    checkpoint_dir="output/checkpoints",
    rank=0,
    world_size=1,
)
trainer.train(total_epochs=config["training"]["epochs"])
```

`F5Trainer` saves checkpoints via `CheckpointManager` every `save_interval` steps and at the end of each epoch.

## Config keys (both YAML configs)

```yaml
model:
  dim: 512           # 1024 for Base
  depth: 12          # 22 for Base
  heads: 8           # 16 for Base
  vocab_size: 65     # fixed — must match CyrillicTokenizer
  ff_mult: 2
  text_dim: 256
  conv_layers: 4

audio:
  sample_rate: 24000  # never change
  n_mels: 100         # never change
  n_fft: 1024
  hop_length: 256

training:
  batch_size: 16
  learning_rate: 1.0e-4
  warmup_steps: 1000
  max_steps: 500000
  grad_accum: 1
  max_grad_norm: 1.0
  ema_decay: 0.9999
  save_interval: 5000
  log_interval: 100
  use_tqdm: true      # false for RunPod container logs
  epochs: 100
```

## CLI

Local training from metadata.json:

```bash
python scripts/train.py \
    --config configs/vits_local.yaml \
    --from-local \
    --data-dir data/processed
```

Cloud training, pulling dataset from HF:

```bash
python scripts/train.py \
    --config configs/vits_runpod.yaml \
    --dataset btsee/mbspeech_mn \
    --lang mn \
    --push-to-hub \
    --hf-repo btsee/orontts
```

Fine-tune on top of an official F5-TTS pretrained checkpoint:

```bash
python scripts/train.py \
    --config configs/vits_runpod.yaml \
    --dataset btsee/mbspeech_mn \
    --pretrain-ckpt F5TTS_Base.safetensors
```

Resume from latest checkpoint:

```bash
python scripts/train.py \
    --config configs/vits_runpod.yaml \
    --from-local \
    --data-dir data/processed \
    --resume
```

## CheckpointManager

```python
from src.utils.checkpoint import CheckpointManager

cm = CheckpointManager("output/checkpoints", model_name="f5tts", max_checkpoints=5)

# Save
cm.save(step=1000, model=model, optimizer=optimizer,
        scheduler=scheduler, ema_state=ema.state_dict(), is_best=True)

# Load
info = cm.load(model=model, optimizer=optimizer, load_best=True)
# info = {"step": int, "loss": float|None, "ema_state_dict": dict|None}

# Load official pretrained weights (safetensors or .pt)
info = cm.load_pretrained_f5tts(model, "F5TTS_Base.safetensors", strict=False)
```

## Monitoring

```bash
tensorboard --logdir output/logs
```

TensorBoard scalars: `train/loss`, `lr`.

## RunPod setup

```bash
bash scripts/setup/runpod_setup.sh
```

The script installs dependencies into the container and configures the environment for A100/L40 training.

## Environment (local)

```powershell
# Python 3.12 required
py -3.12 -m venv .venv
.venv\Scripts\pip install -e ".[dev]"

# Smoke test before launching a full run
.venv\Scripts\python scripts/test_pipeline.py
.venv\Scripts\python scripts/test_pipeline.py --hf
```
