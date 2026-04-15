#!/usr/bin/env bash
# RunPod environment setup for OronTTS
# Recommended image: runpod/pytorch:1.0.3-cu1290-torch280-ubuntu2404
#   (Ubuntu 24.04 → system Python IS 3.12 → venv inherits pre-installed PyTorch)
# Run once after the pod starts:
#   bash scripts/setup/runpod_setup.sh
set -euo pipefail

echo "[INFO] Setting up OronTTS training environment..."

# ── Python 3.12 ────────────────────────────────────────────────────────────────
# RunPod base images (Ubuntu 22.04 & 24.04) ship Python 3.12 pre-installed
# via deadsnakes. No manual installation needed.
if ! python3.12 --version &>/dev/null; then
    echo "[ERROR] Python 3.12 not found. Use a RunPod PyTorch image (Ubuntu 24.04 recommended)."
    exit 1
fi
python3.12 --version

# ── Virtual environment ────────────────────────────────────────────────────────
# Use --system-site-packages to inherit the pre-installed PyTorch + CUDA stack.
# This avoids re-downloading torch (~2.5 GB) and ensures CUDA version match.
VENV_ARGS="--system-site-packages"
if ! python3.12 -c "import torch" 2>/dev/null; then
    echo "[WARN] System PyTorch not found for Python 3.12; creating isolated venv."
    echo "       (This will download PyTorch from PyPI — use Ubuntu 24.04 image to avoid this.)"
    VENV_ARGS=""
fi

python3.12 -m venv ${VENV_ARGS} .venv
source .venv/bin/activate

# ── Python dependencies ────────────────────────────────────────────────────────
pip install --upgrade pip --quiet
pip install -e ".[dev]"

# Verify torch + CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# ── Weights & Biases ───────────────────────────────────────────────────────────
# Set WANDB_API_KEY in RunPod → Secrets (env var) — then this auto-authenticates.
# If running interactively, run `wandb login` instead.
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb login --relogin
else
    echo "[WARN] WANDB_API_KEY not set — charts will be offline. Add it to RunPod Secrets."
fi

# ── Smoke test ─────────────────────────────────────────────────────────────────
python scripts/test_pipeline.py

echo ""
echo "Setup complete. Start training with:"
echo "  source .venv/bin/activate"
echo "  python scripts/train.py --config configs/runpod.yaml --dataset btsee/mbspeech_mn --num-gpus \$(nvidia-smi -L | wc -l)"
