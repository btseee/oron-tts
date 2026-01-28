#!/bin/bash
# Quick test script for Docker container

set -e

echo "=== OronTTS Docker Test ==="

# Build Docker image
echo "Building Docker image..."
docker build -t oron-tts:test .

# Run container and test
echo "Starting container and testing training..."
docker run --rm --gpus all \
    -v $(pwd)/output:/workspace/oron-tts/output \
    -v $(pwd)/data/cache:/workspace/oron-tts/data/cache \
    -e HF_TOKEN="${HF_TOKEN}" \
    oron-tts:test \
    bash -c "
        echo 'Verifying installations...'
        python -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'
        python -c 'import torchaudio; print(f\"Torchaudio: {torchaudio.__version__}\")'
        python -c 'from datasets import load_dataset; print(\"Datasets: OK\")'
        echo ''
        echo 'Starting training (2 epochs for testing)...'
        python scripts/train.py \
            --config configs/vits_local.yaml \
            --dataset btsee/mbspeech_mn \
            --num-epochs 2 || true
    "

echo ""
echo "=== Test Complete ==="
echo "Check output/logs and output/checkpoints for results"
