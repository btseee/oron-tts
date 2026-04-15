#!/usr/bin/env bash
# RunPod environment setup for OronTTS
# Run once after the pod starts: bash scripts/setup/runpod_setup.sh
set -euo pipefail

# ── Python 3.12 ────────────────────────────────────────────────────────────────
# RunPod templates may ship Python 3.10/3.11; pyproject.toml requires >=3.12.
if ! python3.12 --version &>/dev/null 2>&1; then
    apt-get update && apt upgrade -y --no-install-recommends
    echo "[INFO] Installing Python 3.12..."
    # Remove any stale/malformed deadsnakes entry from previous attempts
    rm -f /etc/apt/sources.list.d/deadsnakes.list /etc/apt/trusted.gpg.d/deadsnakes.gpg
    apt-get update -qq
    apt-get install -y --no-install-recommends curl gnupg lsb-release
    # Add deadsnakes PPA manually — avoids add-apt-repository's broken apt_pkg dep
    curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xF23C5A6CF475977595C89F51BA6932366A755776" \
        | gpg --dearmor > /etc/apt/trusted.gpg.d/deadsnakes.gpg
    echo "deb https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/deadsnakes.list
    apt-get update -qq
    apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev
fi
python3.12 --version

# ── Virtual environment ────────────────────────────────────────────────────────
python3.12 -m venv .venv
source .venv/bin/activate

# ── Python dependencies ────────────────────────────────────────────────────────
pip install --upgrade pip --quiet
pip install -e ".[dev]"

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
