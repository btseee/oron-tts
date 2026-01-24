#!/bin/bash
# OronTTS Runpod Setup Script
# Sets up the environment for VITS2 training on Runpod.io

set -euo pipefail

echo "======================================"
echo "OronTTS Runpod Environment Setup"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Runpod
if [[ -d "/workspace" ]]; then
    WORKSPACE="/workspace"
    log_info "Detected Runpod environment. Using /workspace"
else
    WORKSPACE="$HOME/orontts"
    log_warn "Not on Runpod. Using $WORKSPACE"
fi

cd "$WORKSPACE"

# Update system packages
log_info "Updating system packages..."
apt-get update -qq

# Install system dependencies
log_info "Installing system dependencies..."
apt-get install -y -qq \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libsndfile1 \
    ffmpeg \
    sox \
    libsox-dev \
    libsox-fmt-all \
    autoconf \
    automake \
    libtool \
    pkg-config \
    libpcaudio-dev \
    libsonic-dev

# Install espeak-ng from source (latest for Mongolian support)
log_info "Installing espeak-ng from source..."
if [[ ! -d "espeak-ng" ]]; then
    git clone --depth 1 https://github.com/espeak-ng/espeak-ng.git
fi

cd espeak-ng
./autogen.sh
./configure --prefix=/usr
make -j$(nproc)
make install
ldconfig
cd "$WORKSPACE"

# Verify espeak-ng installation
if command -v espeak-ng &> /dev/null; then
    log_info "espeak-ng installed: $(espeak-ng --version)"
else
    log_error "espeak-ng installation failed!"
    exit 1
fi

# Install Python dependencies
log_info "Installing Python dependencies..."

# Check for uv first (preferred), then pip
if command -v uv &> /dev/null; then
    log_info "Using uv for package management"
    uv pip install --upgrade pip
    if [[ -f "pyproject.toml" ]]; then
        uv pip install -e ".[dev]"
    else
        log_warn "pyproject.toml not found. Installing core dependencies..."
        uv pip install torch torchaudio lightning librosa pydantic \
            huggingface-hub datasets deepfilternet soundfile einops \
            phonemizer tqdm rich ruff pytest
    fi
else
    log_info "Using pip for package management"
    pip install --upgrade pip
    if [[ -f "pyproject.toml" ]]; then
        pip install -e ".[dev]"
    else
        log_warn "pyproject.toml not found. Installing core dependencies..."
        pip install torch torchaudio lightning librosa pydantic \
            huggingface-hub datasets deepfilternet soundfile einops \
            phonemizer tqdm rich ruff pytest
    fi
fi

# Install CUDA-specific packages if available
if command -v nvidia-smi &> /dev/null; then
    log_info "CUDA detected. GPU info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    log_warn "No CUDA detected. Training will use CPU."
fi

# Create necessary directories
log_info "Creating project directories..."
mkdir -p "$WORKSPACE"/{data,outputs,checkpoints,logs}

# Download sample data (optional)
# log_info "Downloading sample data..."
# wget -q -O "$WORKSPACE/data/sample.tar.gz" "https://example.com/sample.tar.gz"
# tar -xzf "$WORKSPACE/data/sample.tar.gz" -C "$WORKSPACE/data/"

# Set up HuggingFace authentication
if [[ -n "${HF_TOKEN:-}" ]]; then
    log_info "HuggingFace token found. Configuring authentication..."
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=True)"
else
    log_warn "HF_TOKEN not set. Set it to enable HuggingFace Hub integration."
fi

# Environment summary
echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Environment Summary:"
echo "  - Workspace: $WORKSPACE"
echo "  - Python: $(python --version)"
echo "  - PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  - CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  - espeak-ng: $(espeak-ng --version 2>&1 | head -1)"
echo ""
echo "Directories:"
echo "  - Data: $WORKSPACE/data"
echo "  - Outputs: $WORKSPACE/outputs"
echo "  - Checkpoints: $WORKSPACE/checkpoints"
echo ""
echo "Quick Start:"
echo "  1. Place your data in $WORKSPACE/data/"
echo "  2. Prepare data: orontts-prepare --input-dir data/raw --output-dir data/processed"
echo "  3. Train: orontts-train --config configs/vits2_hq.json --data-dir data/processed"
echo ""
log_info "Happy training! 🎵"
