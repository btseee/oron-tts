#!/bin/bash
# RunPod Environment Setup for OronTTS Training
# Run this once when starting a new RunPod instance

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }

echo "======================================================================="
echo "  OronTTS RunPod Setup - Mongolian TTS Training Environment"
echo "======================================================================="
echo ""

# Check if running on GPU
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. Make sure you're running on a GPU instance."
    exit 1
fi

log_info "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Create workspace structure
log_info "Creating workspace directory structure..."
mkdir -p /workspace/output/{ckpts,logs,runs,data}
log_success "Workspace structure created"
echo ""

# Clone repository if not exists
if [ ! -d "/workspace/oron-tts" ]; then
    log_info "Cloning OronTTS repository with submodules..."
    cd /workspace
    git clone --recurse-submodules https://github.com/btseee/oron-tts.git
    cd oron-tts
    log_success "Repository cloned"
else
    log_info "Repository already exists, pulling latest changes..."
    cd /workspace/oron-tts
    git pull
    git submodule update --init --recursive
    log_success "Repository updated"
fi
echo ""

# Check Python version
log_info "Python version:"
python --version
echo ""

# Install OronTTS
log_info "Installing OronTTS (core dependencies)..."
pip install -e . --quiet
log_success "OronTTS installed"
echo ""

# Install F5-TTS from submodule
log_info "Installing F5-TTS from submodule..."
pip install -e third_party/F5-TTS --quiet
log_success "F5-TTS installed"
echo ""

# Install PyTorch (if not already installed)
log_info "Checking PyTorch installation..."
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
    log_success "PyTorch ${TORCH_VERSION} with CUDA ${CUDA_VERSION} already installed"
else
    log_info "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
    log_success "PyTorch installed"
fi
echo ""

# Verify CUDA availability
log_info "Verifying CUDA availability..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'CUDA available: {torch.cuda.get_device_name(0)}')"
log_success "CUDA verification passed"
echo ""

# Set environment variables
log_info "Setting environment variables..."
export PYTHONPATH="/workspace/oron-tts:/workspace/oron-tts/third_party/F5-TTS/src"
export HF_AUDIO_DECODER="soundfile"
export CUDA_VISIBLE_DEVICES="0"

# Add to bashrc for persistence
if ! grep -q "OronTTS Environment" ~/.bashrc; then
    cat >> ~/.bashrc << 'EOF'

# OronTTS Environment
export PYTHONPATH="/workspace/oron-tts:/workspace/oron-tts/third_party/F5-TTS/src"
export HF_AUDIO_DECODER="soundfile"
export CUDA_VISIBLE_DEVICES="0"
EOF
    log_success "Environment variables added to ~/.bashrc"
else
    log_info "Environment variables already in ~/.bashrc"
fi
echo ""

# Create .env file
log_info "Creating .env file..."
cat > /workspace/oron-tts/.env << EOF
PYTHONPATH=/workspace/oron-tts:/workspace/oron-tts/third_party/F5-TTS/src
HF_AUDIO_DECODER=soundfile
CUDA_VISIBLE_DEVICES=0
EOF
log_success ".env file created"
echo ""

# Download pretrained F5-TTS checkpoint
log_info "Checking for pretrained F5-TTS checkpoint..."
CKPT_DIR="/workspace/oron-tts/third_party/F5-TTS/ckpts"
if [ ! -f "${CKPT_DIR}/F5TTS_Base/model_1200000.safetensors" ]; then
    log_info "Downloading pretrained F5-TTS checkpoint..."
    cd /workspace/oron-tts
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='SWivid/F5-TTS',
    local_dir='third_party/F5-TTS/ckpts',
    allow_patterns=['F5TTS_Base/*']
)
"
    log_success "Pretrained checkpoint downloaded"
else
    log_success "Pretrained checkpoint already exists"
fi
echo ""

# Install additional tools (optional)
log_info "Installing additional development tools..."
pip install tensorboard wandb --quiet
log_success "Development tools installed"
echo ""

# Display summary
echo "======================================================================="
log_success "RunPod Setup Complete!"
echo ""
echo "Environment Information:"
echo "  Python:      $(python --version | cut -d' ' -f2)"
echo "  PyTorch:     $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA:        $(python -c 'import torch; print(torch.version.cuda)')"
echo "  GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""
echo "Project Location:"
echo "  Repository:  /workspace/oron-tts"
echo "  Output:      /workspace/output"
echo ""
echo "Next Steps:"
echo "  1. Prepare dataset:"
echo "     cd /workspace/oron-tts"
echo "     python scripts/data/prepare_combined_dataset.py"
echo ""
echo "  2. Start training:"
echo "     bash scripts/training/train.sh"
echo ""
echo "  3. Monitor training:"
echo "     tail -f /workspace/output/logs/train_mongolian_*.log"
echo "     watch -n 1 nvidia-smi"
echo ""
echo "  4. Launch TensorBoard:"
echo "     tensorboard --logdir /workspace/output/runs --host 0.0.0.0"
echo ""
echo "======================================================================="
