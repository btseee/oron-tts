---
mode: agent
description: Train OronTTS F5-TTS models — configure, launch, resume, and monitor training runs locally or on RunPod.
---

# OronTTS F5-TTS Training

Use this skill when the user asks about training, fine-tuning, resuming, or configuring OronTTS model training.

## Architecture constraints

- **Loss**: Flow matching MSE on velocity field, computed inline in `CFM.forward()` — no separate loss function.
- **Infilling training**: random 70–100% span masking. Unmasked portion is the conditioning signal.
- **CFG dropout**: `audio_drop_prob=0.3`, `cond_drop_prob=0.2` during training.
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
trainer.train(num_epochs=config["num_epochs"])
```

`F5Trainer` saves checkpoints via `CheckpointManager` every `save_interval` steps and at the end of each epoch.

## Config keys (all three YAML configs)

```yaml
# == Audio (top-level, not nested) ==
sample_rate: 24000  # never change
n_mels: 100         # never change
n_fft: 1024
hop_length: 256
win_length: 1024
fmin: 0.0
fmax: 8000.0

# == Training (top-level, not nested) ==
batch_size: 16
learning_rate: 1.0e-4
betas: [0.9, 0.999]
warmup_steps: 1000
max_grad_norm: 1.0
ema_decay: 0.9999
grad_accumulation_steps: 1
log_interval: 100
save_interval: 5
use_tqdm: true      # false for RunPod container logs
num_epochs: 500
batch_size_type: frame   # "frame" for DynamicBatchSampler, "sample" for fixed
frames_threshold: 3000   # max total mel frames per batch (frame mode)
max_samples: 8           # max samples per batch cap (frame mode)
num_workers: 6
pin_memory: true

# == Performance (top-level) ==
gradient_checkpointing: true  # essential for T4/RTX 5070 Ti VRAM
compile: true                 # torch.compile on DiT backbone; set false for T4/Colab
use_tf32: true                # disable on T4
cudnn_benchmark: true         # disable on T4

# == Model (nested under model:) ==
model:
  dim: 512           # 1024 for Base
  depth: 12          # 22 for Base
  heads: 8           # 16 for Base
  vocab_size: 65     # fixed — must match CyrillicTokenizer
  ff_mult: 4
  text_dim: 256      # 512 for Base
  conv_layers: 4
  p_dropout: 0.1
  vocos_dim: 384     # 512 for Base
  vocos_layers: 6    # 8 for Base
  vocos_intermediate: 1024  # 1536 for Base
```

### Config profiles
| Config | Target GPU | dim | depth | frames_threshold | compile |
|--------|-----------|-----|-------|-----------------|---------|
| `local.yaml` | RTX 5070 Ti | 512 | 12 | 3000 | true |
| `runpod.yaml` | A100/L40 | 1024 | 22 | 38400 | true |
| `colab.yaml` | T4 (15 GB) | 512 | 12 | 12000 | false |

### torch.compile
- Compiles **only** `model.cfm.backbone` (DiT) with `dynamic=True`.
- **Never** compile the full model or CFM wrapper (Python branching in `CFM.forward()` causes graph breaks).
- Set `compile: false` on T4/Colab — inductor overhead exceeds gains for small models.

## CLI

Local training from metadata.json:

```bash
python scripts/train.py \
    --config configs/local.yaml \
    --from-local \
    --data-dir data/processed
```

Cloud training, pulling dataset from HF:

```bash
python scripts/train.py \
    --config configs/runpod.yaml \
    --dataset btsee/mbspeech_mn \
    --lang mn \
    --push-to-hub \
    --hf-repo btsee/orontts
```

Fine-tune on top of an official F5-TTS pretrained checkpoint:

```bash
python scripts/train.py \
    --config configs/runpod.yaml \
    --dataset btsee/mbspeech_mn \
    --pretrain-ckpt F5TTS_Base.safetensors
```

Resume from latest checkpoint:

```bash
python scripts/train.py \
    --config configs/runpod.yaml \
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

Console logging only (no TensorBoard integration currently).

`F5Trainer` logs per-epoch: avg_loss, val_loss, throughput (samples/s), ETA.

## RunPod setup

```bash
bash scripts/setup/runpod_setup.sh
```

The script creates `.venv` with `--system-site-packages` (inherits pre-installed PyTorch + CUDA), installs project deps, and configures the environment for A100/L40 training. Recommended image: `runpod/pytorch:1.0.3-cu1290-torch280-ubuntu2404`.

## Environment (local)

```powershell
# Python 3.12 required
py -3.12 -m venv .venv
.venv\Scripts\pip install -e ".[dev]"

# Smoke test before launching a full run
.venv\Scripts\python scripts/test_pipeline.py
.venv\Scripts\python scripts/test_pipeline.py --hf
```
