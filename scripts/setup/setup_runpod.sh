#!/bin/bash
# =============================================================================
# OronTTS Runpod Setup
# Installs dependencies and prepares training environment
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

log_info "OronTTS Runpod Setup"

# Install system dependencies
log_info "Installing system dependencies..."
apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git git-lfs \
    libsndfile1-dev sox ffmpeg \
    autoconf automake libtool pkg-config

# Build espeak-ng from source
log_info "Building espeak-ng with Mongolian support..."
cd /tmp
git clone --depth 1 --branch 1.51.1 https://github.com/espeak-ng/espeak-ng.git
cd espeak-ng
./autogen.sh
./configure --prefix=/usr/local
make -j$(nproc)
make install
ldconfig
rm -rf /tmp/espeak-ng

# Initialize git-lfs
git lfs install

# Clone repository
if [ ! -d "${PROJECT_DIR}" ]; then
    log_info "Cloning OronTTS repository..."
    cd "${WORKSPACE_DIR}"
    git clone https://github.com/btseee/oron-tts.git
fi

cd "${PROJECT_DIR}"

# Install Python dependencies
log_info "Installing Python dependencies..."
pip install --no-cache-dir --upgrade pip setuptools wheel
pip install --no-cache-dir "packaging>=23.0,<24.0"
pip install --no-cache-dir -e ".[dev]"

# Install Flash Attention 2 (optional)
pip install --no-cache-dir flash-attn --no-build-isolation || \
    log_info "Flash Attention 2 not available, using standard attention"

# HuggingFace login
if [ -n "${HF_TOKEN:-}" ]; then
    log_info "Logging in to HuggingFace..."
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential
fi

log_success "Setup complete!"
echo ""
echo "Training command:"
echo "  accelerate launch scripts/training/train.py \\"
echo "    --config configs/config.yaml \\"
echo "    --output-dir outputs \\"
echo "    --hub-repo btsee/oron-tts \\"
echo "    --hf-dataset btsee/common-voices-24-mn"
