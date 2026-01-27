#!/bin/bash
# =============================================================================
# OronTTS RunPod Setup
# Minimal setup for training environment - no data cleaning deps
# =============================================================================

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }

# Configuration
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
PROJECT_DIR="${WORKSPACE_DIR}/oron-tts"

log_info "OronTTS RunPod Setup"

# Install system dependencies including FFmpeg dev libraries
log_info "Installing system dependencies..."
apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs libsndfile1-dev sox \
    ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswresample-dev
rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
git lfs install

# Clone repository with submodules
if [ ! -d "${PROJECT_DIR}" ]; then
    log_info "Cloning OronTTS repository..."
    cd "${WORKSPACE_DIR}"
    git clone --recurse-submodules https://github.com/btseee/oron-tts.git
else
    log_info "Updating repository and submodules..."
    cd "${PROJECT_DIR}"
    git pull
    git submodule update --init --recursive
fi

cd "${PROJECT_DIR}"

# Install F5-TTS from submodule (includes all training dependencies)
log_info "Installing F5-TTS..."
pip install --no-cache-dir --upgrade pip setuptools wheel
pip install --no-cache-dir -e third_party/F5-TTS

# Install OronTTS wrapper (minimal deps)
log_info "Installing OronTTS..."
pip install --no-cache-dir -e .

# Install Flash Attention 2 for faster training
log_info "Installing Flash Attention 2..."
pip install --no-cache-dir flash-attn --no-build-isolation || \
    log_info "Flash Attention 2 not available, using standard attention"

# HuggingFace login
if [ -n "${HF_TOKEN:-}" ]; then
    log_info "Logging in to HuggingFace..."
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential
fi

# Create directories
mkdir -p ckpts logs data

# Make training scripts executable
chmod +x "${PROJECT_DIR}/scripts/training/train_background.sh"

log_success "Setup complete!"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  OronTTS Ready for Training (RTX 4090 Optimized)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Dataset: btsee/common-voices-24-mn (HuggingFace)"
echo ""
echo "=== Quick Start (Background Training) ==="
echo ""
echo "  ./scripts/training/train_background.sh"
echo ""
echo "=== Interactive Training ==="
echo ""
echo "  python scripts/training/train.py \\"
echo "    --dataset-name btsee/common-voices-24-mn \\"
echo "    --epochs 500 \\"
echo "    --batch-size 1800"
echo ""
echo "=== Monitor Training ==="
echo ""
echo "  tail -f logs/train_*.log"
echo "  tensorboard --logdir ckpts/ --bind_all"
echo ""
echo "═══════════════════════════════════════════════════════════"
