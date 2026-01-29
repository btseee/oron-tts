#!/bin/bash
# RunPod Setup Script for OronTTS
# Tested with: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

set -e

echo "=== OronTTS RunPod Setup ==="
echo "Container: PyTorch 2.4.0 + CUDA 12.4.1 + Python 3.11"
echo ""

# Update package lists
echo "[1/6] Updating package lists..."
apt-get update -qq

# Install minimal dependencies (no FFmpeg needed - we use datasets<4.0 with soundfile)
echo "[2/6] Installing system dependencies..."
apt-get install -y -qq git libsndfile1

# Clone repository if not already cloned
if [ ! -d "/workspace/oron-tts" ]; then
    echo "[3/6] Cloning OronTTS repository..."
    cd /workspace
    git clone https://github.com/btseee/oron-tts.git
    cd oron-tts
else
    echo "[3/6] Repository already exists, pulling latest changes..."
    cd /workspace/oron-tts
    git pull
fi

# Record pre-installed PyTorch version
echo "[4/6] Checking pre-installed PyTorch..."
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "  Found PyTorch: $TORCH_VERSION"

# Install Python dependencies (preserve pre-installed torch)
echo "[5/6] Installing Python dependencies..."
pip install --upgrade pip -q

# Install datasets<4.0 first to avoid torchcodec requirement
pip install "datasets>=3.0.0,<4.0.0" -q

# Install project without upgrading torch
pip install -e . --no-deps -q
pip install numpy scipy librosa soundfile huggingface-hub deepfilternet pyyaml tqdm tensorboard einops -q

# Verify installations
echo "[6/6] Verifying installation..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torchaudio; print(f'  Torchaudio: {torchaudio.__version__}')"
python -c "import datasets; print(f'  Datasets: {datasets.__version__}')"
python -c "from src.models.vits import VITS; print('  OronTTS: OK')"

# Set HuggingFace token if provided
if [ ! -z "$HF_TOKEN" ]; then
    echo ""
    echo "Setting HuggingFace token..."
    huggingface-cli login --token $HF_TOKEN
fi

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To start training:"
echo "  nohup python scripts/train.py --config configs/vits_runpod.yaml --dataset btsee/mbspeech_mn --push-to-hub --hf-repo btsee/oron-tts > /workspace/train.log 2>&1 &"
echo ""
echo "To monitor training:"
echo "  tail -f /workspace/train.log"
echo ""
