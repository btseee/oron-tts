# OronTTS RunPod Setup Guide

This guide will help you set up and run OronTTS training on RunPod.

## Quick Start

### 1. Launch RunPod Instance

**Recommended Template:**
- **Image**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **GPU**: RTX 4090 (24GB) or A100 (40GB/80GB)
- **Disk**: 50GB minimum
- **Container Disk**: 50GB recommended

### 2. Run Setup Script

After your pod starts, open a terminal and run:

```bash
cd /workspace
wget https://raw.githubusercontent.com/btseee/oron-tts/main/runpod_setup.sh
chmod +x runpod_setup.sh
./runpod_setup.sh
```

Or manually:

```bash
# Install FFmpeg
apt-get update
apt-get install -y ffmpeg libavutil-dev libavcodec-dev libavformat-dev \
    libavdevice-dev libavfilter-dev libswscale-dev libswresample-dev

# Clone repository
cd /workspace
git clone https://github.com/btseee/oron-tts.git
cd oron-tts

# Install dependencies
pip install --upgrade pip
pip install -e ".[runpod]"
```

### 3. Set HuggingFace Token

Create a `.env` file with your HuggingFace token:

```bash
echo "HF_TOKEN=hf_your_token_here" > .env
```

Or login via CLI:

```bash
huggingface-cli login --token hf_your_token_here
```

### 4. Start Training

```bash
python scripts/train.py \
    --config configs/vits_runpod.yaml \
    --from-hf \
    --dataset btsee/mbspeech_mn \
    --push-to-hub \
    --hf-repo btsee/orontts
```

## Configuration

The `configs/vits_runpod.yaml` is pre-configured for RTX 4090:

- **Batch size**: 24 (adjust based on GPU)
- **Segment size**: 64
- **FP16**: Enabled
- **Logging**: Structured logs (no tqdm)
- **Log interval**: 50 steps

For different GPUs:

| GPU          | VRAM | Batch Size | Segment Size |
|--------------|------|------------|--------------|
| RTX 3090     | 24GB | 20         | 64           |
| RTX 4090     | 24GB | 24         | 64           |
| A100 (40GB)  | 40GB | 40         | 96           |
| A100 (80GB)  | 80GB | 80         | 128          |

## Monitoring Training

### Via Container Logs

RunPod shows structured logs in the Logs panel:

```text
[2026-01-28 14:37:22] [INFO] Starting Epoch 1
[2026-01-28 14:37:24] [INFO] Step 0 | Batch 1/320 | Loss: 281.89 | Mel: 100.38 | KL: 175.99 | Dur: 0.01 | LR: 0.000200
[2026-01-28 14:37:29] [INFO] Step 10 | Batch 11/320 | Loss: 139.46 | Mel: 72.75 | KL: 63.69 | Dur: 0.01 | LR: 0.000200
```

### Via TensorBoard

```bash
# In a separate terminal
tensorboard --logdir output/logs --host 0.0.0.0 --port 6006
```

Then access via RunPod's HTTP service or port forwarding.

## Checkpoints

Checkpoints are saved to `output/checkpoints/` every N epochs (configured in yaml).

To push to HuggingFace Hub:

```bash
# Automatic during training (if --push-to-hub flag is set)
# Or manually:
python -c "
from src.utils.checkpoint import CheckpointManager
mgr = CheckpointManager('output/checkpoints')
mgr.push_to_hub('btsee/orontts', token='your_token')
"
```

## Troubleshooting

### FFmpeg Not Found

```bash
apt-get update
apt-get install -y ffmpeg libavutil-dev libavcodec-dev libavformat-dev
```

### Out of Memory

Reduce batch size or segment size in `configs/vits_runpod.yaml`:

```yaml
batch_size: 16  # Reduce from 24
segment_size: 32  # Reduce from 64
```

### Slow Data Loading

Increase workers in config:

```yaml
num_workers: 16  # Increase for multi-core CPUs
prefetch_factor: 4
```

### torchcodec Issues

If torchcodec fails even after FFmpeg installation, the code will fall back to torchaudio. You can verify:

```python
python -c "import torchcodec; print('torchcodec OK')"
```

## Multi-GPU Training

For multi-GPU pods:

```bash
python scripts/train.py \
    --config configs/vits_runpod.yaml \
    --from-hf \
    --dataset btsee/mbspeech_mn \
    --num-gpus 2  # or 4, 8
```

## Persistent Storage

To keep checkpoints across pod restarts:

1. Use RunPod Network Volumes
2. Mount to `/workspace/oron-tts/output`
3. Or regularly push to HuggingFace Hub with `--push-to-hub`

## Cost Optimization

- **Use Spot Instances**: 70% cheaper, may be interrupted
- **Auto-shutdown**: Set idle timeout in RunPod settings
- **Push checkpoints frequently**: Don't lose progress if pod stops
- **Monitor GPU utilization**: `nvidia-smi` to ensure GPU is being used

## Support

- GitHub Issues: https://github.com/btseee/oron-tts/issues
- Documentation: https://github.com/btseee/oron-tts
