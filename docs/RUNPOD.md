# OronTTS Runpod.io Deployment Guide

## Recommended Pod Configuration

### For Training (VITS2)

| Component | Recommended | Minimum | Notes |
|-----------|-------------|---------|-------|
| **GPU** | RTX A5000 (24GB) | RTX 4090 (24GB) | A100 40GB for larger batch sizes |
| **vCPU** | 8 cores | 4 cores | More helps with data loading |
| **RAM** | 32 GB | 16 GB | Dataset caching |
| **Storage** | 100 GB | 50 GB | Dataset + checkpoints |
| **Template** | RunPod Pytorch 2.4 | - | CUDA 12.4, Python 3.11 |

### Cost-Effective Options

1. **Budget Training**: RTX 4090 (24GB) - ~$0.44/hr
   - Good for: Initial experiments, small batch sizes
   - Batch size: 8-16

2. **Optimal Training**: RTX A5000 (24GB) - ~$0.39/hr
   - Good for: Full training runs
   - Batch size: 16-32

3. **Fast Training**: A100 40GB - ~$1.29/hr
   - Good for: Large batch training, quick iterations
   - Batch size: 32-64

## Quick Start on Runpod

### 1. Create Pod

1. Go to [runpod.io](https://runpod.io)
2. Click "Deploy" → "GPU Pods"
3. Select template: **RunPod Pytorch 2.4.0**
4. Choose GPU: **RTX A5000 24GB** (recommended)
5. Set volume: **100 GB** (Network Volume recommended for persistence)
6. Add environment variables:
   ```
   HF_TOKEN=your_huggingface_token
   ```
7. Deploy!

### 2. Setup Environment

SSH into your pod or use the web terminal:

```bash
# Clone the repository
cd /workspace
git clone https://github.com/orontts/oron-tts.git
cd oron-tts

# Install dependencies
pip install -e ".[dev,cleaning]"

# Verify installation
python -c "import orontts; print(orontts.__version__)"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 3. Download Dataset

```bash
# Option A: Download from HuggingFace (recommended)
python -c "
from datasets import load_dataset
ds = load_dataset('btsee/common-voices-24-mn', split='train')
ds.save_to_disk('/workspace/data/common-voices-24-mn')
print(f'Downloaded {len(ds)} samples')
"

# Option B: Use the HuggingFace dataset directly in training
# (streaming mode - no local storage needed)
```

### 4. Start Training

```bash
# Light model (faster, less VRAM)
orontts-train \
    --config configs/vits2_light.json \
    --data-dir /workspace/data/common-voices-24-mn \
    --output-dir /workspace/checkpoints \
    --max-epochs 1000

# High-quality model (slower, more VRAM)
orontts-train \
    --config configs/vits2_hq.json \
    --data-dir /workspace/data/common-voices-24-mn \
    --output-dir /workspace/checkpoints \
    --max-epochs 2000
```

### 5. Monitor Training

```bash
# TensorBoard (in another terminal)
tensorboard --logdir /workspace/checkpoints --bind_all --port 6006

# Access via: https://your-pod-id-6006.proxy.runpod.net
```

## Using Docker Locally (Testing)

### Prerequisites

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA support

### Build and Run

```bash
cd /path/to/oron-tts

# Build image
docker compose build

# Run interactive shell
docker compose run --rm orontts bash

# Or run training directly
docker compose run --rm orontts orontts-train --config configs/vits2_light.json
```

### Test Without GPU

```bash
# CPU-only testing (slow, but works)
docker build -t orontts:test .
docker run --rm -it orontts:test python -c "import orontts; print('OK')"
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `HF_TOKEN` | HuggingFace access token | Yes (for dataset download) |
| `WANDB_API_KEY` | Weights & Biases API key | No (optional logging) |
| `CUDA_VISIBLE_DEVICES` | GPU selection | No (default: all) |

## Runpod Template (Advanced)

To create a custom Runpod template:

1. Push Docker image to Docker Hub:
   ```bash
   docker build -t yourusername/orontts:latest .
   docker push yourusername/orontts:latest
   ```

2. Create template on Runpod:
   - Image: `yourusername/orontts:latest`
   - Docker Command: `/entrypoint.sh bash`
   - Expose ports: 6006, 8888

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size in config:
```json
{
  "training": {
    "batch_size": 8  // Reduce from 16/32
  }
}
```

### Slow Data Loading

Increase number of workers:
```json
{
  "training": {
    "num_workers": 4  // Increase to 8 if CPU allows
  }
}
```

### espeak-ng Not Found

```bash
apt-get update && apt-get install -y espeak-ng
```

### DeepFilterNet Issues

DeepFilterNet is optional. Training works without it if audio is already clean.
