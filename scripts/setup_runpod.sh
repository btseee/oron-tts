#!/bin/bash
# =============================================================================
# OronTTS Runpod Environment Setup Script
# 
# Sets up all dependencies for training F5-TTS Mongolian TTS on Runpod.io
#
# Usage:
#   wget https://raw.githubusercontent.com/<repo>/main/scripts/setup_runpod.sh
#   chmod +x setup_runpod.sh
#   ./setup_runpod.sh
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# Configuration
# =============================================================================
PYTHON_VERSION="3.11"
CUDA_VERSION="12.4"
ORON_TTS_REPO="${ORON_TTS_REPO:-https://github.com/btseee/oron-tts.git}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
PROJECT_DIR="${WORKSPACE_DIR}/oron-tts"

# =============================================================================
# System Dependencies
# =============================================================================
install_system_deps() {
    log_info "Installing system dependencies..."
    
    apt-get update && apt-get install -y \
        build-essential \
        cmake \
        git \
        git-lfs \
        curl \
        wget \
        vim \
        htop \
        nvtop \
        tmux \
        libsndfile1 \
        libsndfile1-dev \
        libsox-fmt-all \
        sox \
        ffmpeg \
        portaudio19-dev \
        python3-dev \
        autoconf \
        automake \
        libtool \
        pkg-config \
        ronn \
        2>/dev/null || true
    
    log_success "System dependencies installed"
}

# =============================================================================
# espeak-ng from source (for Mongolian support)
# =============================================================================
install_espeak_ng() {
    log_info "Installing espeak-ng from source..."
    
    cd /tmp
    
    if [ -d "espeak-ng" ]; then
        rm -rf espeak-ng
    fi
    
    git clone https://github.com/espeak-ng/espeak-ng.git
    cd espeak-ng
    
    # Checkout stable version
    git checkout 1.51.1
    
    # Build and install
    ./autogen.sh
    ./configure --prefix=/usr/local
    make -j$(nproc)
    make install
    
    # Update library cache
    ldconfig
    
    # Verify installation
    if espeak-ng --version &> /dev/null; then
        log_success "espeak-ng installed: $(espeak-ng --version)"
    else
        log_error "espeak-ng installation failed"
        exit 1
    fi
    
    # Verify Mongolian support
    if espeak-ng --voices | grep -q "mn"; then
        log_success "Mongolian voice support available"
    else
        log_warning "Mongolian voice may not be fully supported"
    fi
    
    cd /tmp
    rm -rf espeak-ng
}

# =============================================================================
# Python Environment
# =============================================================================
setup_python_env() {
    log_info "Setting up Python environment..."
    
    # Create virtual environment if not exists
    if [ ! -d "${WORKSPACE_DIR}/venv" ]; then
        python${PYTHON_VERSION} -m venv ${WORKSPACE_DIR}/venv
    fi
    
    source ${WORKSPACE_DIR}/venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    log_success "Python environment ready"
}

# =============================================================================
# Install OronTTS and Dependencies
# =============================================================================
install_oron_tts() {
    log_info "Installing OronTTS..."
    
    source ${WORKSPACE_DIR}/venv/bin/activate
    
    cd ${WORKSPACE_DIR}
    
    # Clone or update repo
    if [ -d "${PROJECT_DIR}" ]; then
        log_info "Updating existing OronTTS installation..."
        cd ${PROJECT_DIR}
        git pull origin main || true
    else
        log_info "Cloning OronTTS repository..."
        git clone ${ORON_TTS_REPO} ${PROJECT_DIR} || {
            log_warning "Could not clone repo, creating empty project..."
            mkdir -p ${PROJECT_DIR}
        }
    fi
    
    cd ${PROJECT_DIR}
    
    # Install PyTorch with CUDA
    pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
    
    # Fix packaging version for deepfilternet compatibility
    pip install "packaging>=23.0,<24.0"
    
    # Install project dependencies
    if [ -f "pyproject.toml" ]; then
        pip install -e ".[dev]" 2>/dev/null || pip install -e . 2>/dev/null || {
            log_warning "Installing dependencies manually..."
            pip install \
                accelerate>=0.33.0 \
                transformers>=4.44.0 \
                datasets>=2.20.0 \
                huggingface-hub[cli]>=0.24.0 \
                einops>=0.8.0 \
                torchdiffeq>=0.2.4 \
                librosa>=0.10.2 \
                soundfile>=0.12.1 \
                deepfilternet>=0.5.6 \
                phonemizer>=3.3.0 \
                pyyaml>=6.0.1 \
                rich>=13.7.0 \
                wandb>=0.17.0 \
                python-dotenv
        }
    else
        log_warning "No pyproject.toml found, installing core dependencies..."
        pip install \
            accelerate>=0.33.0 \
            transformers>=4.44.0 \
            datasets>=2.20.0 \
            huggingface-hub[cli]>=0.24.0 \
            einops>=0.8.0 \
            torchdiffeq>=0.2.4 \
            librosa>=0.10.2 \
            soundfile>=0.12.1 \
            deepfilternet>=0.5.6 \
            phonemizer>=3.3.0 \
            pyyaml>=6.0.1 \
            rich>=13.7.0 \
            wandb>=0.17.0 \
            python-dotenv
    fi
    
    # Install Flash Attention 2
    log_info "Installing Flash Attention 2..."
    pip install flash-attn --no-build-isolation || log_warning "Flash Attention installation failed, will use fallback"
    
    log_success "OronTTS installed"
}

# =============================================================================
# HuggingFace Setup
# =============================================================================
setup_huggingface() {
    log_info "Setting up HuggingFace CLI..."
    
    source ${WORKSPACE_DIR}/venv/bin/activate
    
    # Ensure huggingface-hub is installed
    pip install --upgrade huggingface-hub[cli] > /dev/null 2>&1
    
    # Initialize git-lfs
    git lfs install
    
    # Check for HF token
    if [ -n "${HF_TOKEN:-}" ]; then
        log_info "Logging into HuggingFace Hub..."
        python -c "from huggingface_hub import login; login('${HF_TOKEN}')" 2>/dev/null || \
        huggingface-cli login --token ${HF_TOKEN} --add-to-git-credential
        log_success "HuggingFace authenticated"
    else
        log_warning "HF_TOKEN not set. Set it to enable Hub sync:"
        log_warning "  export HF_TOKEN=hf_xxx"
    fi
}

# =============================================================================
# Weights & Biases Setup
# =============================================================================
setup_wandb() {
    log_info "Setting up Weights & Biases..."
    
    source ${WORKSPACE_DIR}/venv/bin/activate
    
    # Ensure wandb is installed
    pip install --upgrade wandb > /dev/null 2>&1
    
    if [ -n "${WANDB_API_KEY:-}" ]; then
        wandb login ${WANDB_API_KEY}
        log_success "W&B authenticated"
    else
        log_warning "WANDB_API_KEY not set. Set it to enable experiment tracking:"
        log_warning "  export WANDB_API_KEY=xxx"
    fi
}

# =============================================================================
# Verify Installation
# =============================================================================
verify_installation() {
    log_info "Verifying installation..."
    
    source ${WORKSPACE_DIR}/venv/bin/activate
    
    echo ""
    echo "============================================"
    echo "Installation Verification"
    echo "============================================"
    
    # Python
    echo -n "Python: "
    cd ${PROJECT_DIR} 2>/dev/null && python --version
    
    # PyTorch
    echo -n "PyTorch: "
    python -c "import torch; print(f'{torch.__version__} (CUDA: {torch.cuda.is_available()})')"
    
    # CUDA
    echo -n "CUDA: "
    python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'N/A')"
    
    # GPU
    echo -n "GPU: "
    python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
    
    # espeak-ng
    echo -n "espeak-ng: "
    espeak-ng --version 2>/dev/null | head -1 || echo "Not installed"
    
    # Flash Attention
    echo -n "Flash Attention: "
    python -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null || echo "Not installed"
    
    # OronTTS
    echo -n "OronTTS: "
    python -c "from src.core import F5TTS; print('OK')" 2>/dev/null || echo "Import failed"
    
    echo "============================================"
    echo ""
}

# =============================================================================
# Create Helper Scripts
# =============================================================================
create_helper_scripts() {
    log_info "Creating helper scripts..."
    
    # Activation script
    cat > ${WORKSPACE_DIR}/activate.sh << 'EOF'
#!/bin/bash
source /workspace/venv/bin/activate
cd /workspace/oron-tts
echo "OronTTS environment activated!"
echo "Run: python scripts/train.py --config configs/light.yaml"
EOF
    chmod +x ${WORKSPACE_DIR}/activate.sh
    
    # Training launcher (foreground)
    cat > ${WORKSPACE_DIR}/train.sh << 'EOF'
#!/bin/bash
source /workspace/venv/bin/activate
cd /workspace/oron-tts

CONFIG="${1:-configs/light.yaml}"
HUB_REPO="${HUB_REPO:-btsee/oron-tts}"
DATASET="${DATASET:-btsee/common-voices-24-mn}"

ARGS="--config ${CONFIG} --output-dir outputs --hub-repo ${HUB_REPO} --hf-dataset ${DATASET}"

if [ "${RESUME:-false}" = "true" ]; then
    ARGS="${ARGS} --resume"
fi

accelerate launch scripts/train.py ${ARGS}
EOF
    chmod +x ${WORKSPACE_DIR}/train.sh
    
    # Background training with tmux
    cat > ${WORKSPACE_DIR}/train_bg.sh << 'EOF'
#!/bin/bash
SESSION="${1:-oron-train}"
CONFIG="${2:-configs/light.yaml}"

if tmux has-session -t ${SESSION} 2>/dev/null; then
    echo "Error: Session '${SESSION}' already running"
    echo "Attach: tmux attach -t ${SESSION}"
    echo "Kill: tmux kill-session -t ${SESSION}"
    exit 1
fi

tmux new-session -d -s ${SESSION} bash -c "
    source /workspace/venv/bin/activate
    cd /workspace/oron-tts
    export HF_TOKEN=${HF_TOKEN}
    export WANDB_API_KEY=${WANDB_API_KEY:-}
    HUB_REPO=btsee/oron-tts DATASET=btsee/common-voices-24-mn /workspace/train.sh ${CONFIG}
    echo 'Training finished. Press Enter to close.'
    read
"

echo "✓ Started: tmux attach -t ${SESSION}"
echo "✓ Detach: Ctrl+B then D"
EOF
    chmod +x ${WORKSPACE_DIR}/train_bg.sh
    
    # Background training with nohup
    cat > ${WORKSPACE_DIR}/train_nohup.sh << 'EOF'
#!/bin/bash
CONFIG="${1:-configs/light.yaml}"
LOG="${2:-/workspace/training.log}"

source /workspace/venv/bin/activate
cd /workspace/oron-tts

nohup bash -c "export HF_TOKEN=${HF_TOKEN}; export WANDB_API_KEY=${WANDB_API_KEY:-}; HUB_REPO=btsee/oron-tts DATASET=btsee/common-voices-24-mn /workspace/train.sh ${CONFIG}" > ${LOG} 2>&1 &

echo $! > /workspace/training.pid
echo "✓ PID: $(cat /workspace/training.pid)"
echo "✓ Logs: tail -f ${LOG}"
EOF
    chmod +x ${WORKSPACE_DIR}/train_nohup.sh
    
    log_success "Helper scripts created"
    log_info "  - ${WORKSPACE_DIR}/activate.sh: Activate environment"
    log_info "  - ${WORKSPACE_DIR}/train.sh: Foreground training"
    log_info "  - ${WORKSPACE_DIR}/train_bg.sh: Background (tmux)"
    log_info "  - ${WORKSPACE_DIR}/train_nohup.sh: Background (nohup)"
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo ""
    echo "============================================"
    echo "  OronTTS Runpod Setup Script"
    echo "============================================"
    echo ""
    
    cd ${WORKSPACE_DIR}
    
    install_system_deps
    install_espeak_ng
    setup_python_env
    install_oron_tts
    setup_huggingface
    setup_wandb
    create_helper_scripts
    verify_installation
    
    log_success "Setup complete!"
    echo ""
    echo "========================================"
    echo "Background Training (Recommended):"
    echo "  /workspace/train_bg.sh oron-train configs/light.yaml"
    echo ""
    echo "Attach to session:"
    echo "  tmux attach -t oron-train"
    echo "  (Detach: Ctrl+B then D)"
    echo ""
    echo "Alternative (nohup):"
    echo "  /workspace/train_nohup.sh configs/light.yaml"
    echo "  tail -f /workspace/training.log"
    echo "========================================"
    echo ""
}

main "$@"
