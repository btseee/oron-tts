#!/bin/bash
set -e

echo "=================================================="
echo "  OronTTS - Mongolian VITS2 TTS Training"
echo "=================================================="

# Verify CUDA
echo ""
echo "🔧 System Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  No NVIDIA GPU detected"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'  CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Verify OronTTS
echo ""
echo "📦 OronTTS:"
python -c "import orontts; print(f'  Version: {orontts.__version__}')"
python -c "from orontts.preprocessing.audio import AudioCleaner; print(f'  DeepFilterNet: {AudioCleaner().has_deepfilter}')"
python -c "from orontts.model.config import VITS2Config; print(f'  Config: OK')"

# Check for HuggingFace token
echo ""
echo "🔑 Authentication:"
if [ -n "$HF_TOKEN" ]; then
    echo "  HF_TOKEN: Set"
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)" 2>/dev/null && echo "  HuggingFace: Logged in" || echo "  HuggingFace: Login failed"
else
    echo "  HF_TOKEN: Not set (set with -e HF_TOKEN=xxx)"
fi

# Check workspace
echo ""
echo "📂 Workspace:"
echo "  Data: /workspace/data"
echo "  Checkpoints: /workspace/checkpoints"
echo "  Logs: /workspace/logs"

# Check if dataset exists
if [ -d "/workspace/data/cleaned" ]; then
    CLIP_COUNT=$(ls /workspace/data/cleaned/clips/*.wav 2>/dev/null | wc -l)
    echo "  Dataset: $CLIP_COUNT clips found"
else
    echo "  Dataset: Not found (run prepare_data.py or download from HF)"
fi

echo ""
echo "=================================================="
echo "  Ready! Run 'orontts-train --help' to start"
echo "=================================================="
echo ""

# Execute command
exec "$@"
