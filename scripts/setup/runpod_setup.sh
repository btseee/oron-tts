#!/bin/bash
# RunPod Setup Script for OronTTS
# Run this script on your RunPod instance to install all dependencies

set -e

echo "=== OronTTS RunPod Setup ==="

# Update package lists
echo "Updating package lists..."
apt-get update

# Install FFmpeg and development libraries for torchcodec
echo "Installing FFmpeg..."
apt-get install -y ffmpeg libavutil-dev libavcodec-dev libavformat-dev libavdevice-dev libavfilter-dev libswscale-dev libswresample-dev

# Install git if not present
apt-get install -y git

# Clone repository if not already cloned
if [ ! -d "/workspace/oron-tts" ]; then
    echo "Cloning OronTTS repository..."
    cd /workspace
    git clone https://github.com/btseee/oron-tts.git
    cd oron-tts
else
    echo "Repository already exists, pulling latest changes..."
    cd /workspace/oron-tts
    git pull
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -e .

# Install torchcodec
echo "Installing torchcodec..."
pip install torchcodec

# Verify installations
echo "Verifying FFmpeg installation..."
ffmpeg -version | head -1

echo "Verifying Python packages..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchaudio; print(f'Torchaudio: {torchaudio.__version__}')"
python -c "import torchcodec; print(f'Torchcodec: {torchcodec.__version__}')"

# Set HuggingFace token if provided
if [ ! -z "$HF_TOKEN" ]; then
    echo "Setting HuggingFace token..."
    echo "HF_TOKEN=$HF_TOKEN" > .env
    export HF_TOKEN=$HF_TOKEN
    huggingface-cli login --token $HF_TOKEN
fi

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To start training, run:"
echo "python scripts/train.py --config configs/vits_runpod.yaml --dataset btsee/mbspeech_mn --push-to-hub --hf-repo btsee/orontts"
echo ""
echo "Or with explicit token:"
echo "python scripts/train.py --config configs/vits_runpod.yaml --dataset btsee/mbspeech_mn --push-to-hub --hf-repo btsee/orontts --hf-token \$HF_TOKEN"
echo ""
