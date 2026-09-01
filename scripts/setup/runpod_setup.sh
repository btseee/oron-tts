#!/usr/bin/env bash
# RunPod setup for an F5TTS_v1_Base Mongolian finetune.
#
# Template: Runpod Pytorch 2.8.0 (image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404)
#   Ubuntu 24.04 ships Python 3.12, so the venv can inherit the preinstalled
#   PyTorch + CUDA stack instead of re-downloading ~2.5 GB.
#
# Storage: 150 GB. Each F5TTS_v1_Base checkpoint is ~5.4 GB (weights + grads +
#   AdamW state + EMA); keeping 3 plus the pretrained base plus the prepared
#   dataset does not fit the 80 GB that the previous from-scratch setup assumed.
#   Prefer a Network Volume so the corpus survives pod termination and is shared
#   with the oron-cleaner pod.
#
#   bash scripts/setup/runpod_setup.sh
set -euo pipefail

echo "[INFO] Setting up the OronTTS finetune environment..."

MIN_WORKSPACE_GB=120
if [[ -d /workspace ]]; then
    available_kb=$(df --output=avail -k /workspace | tail -n 1 | tr -d ' ')
    available_gb=$((available_kb / 1024 / 1024))
    if (( available_gb < MIN_WORKSPACE_GB )) && [[ "${ORON_ALLOW_SMALL_DISK:-0}" != "1" ]]; then
        echo "[ERROR] /workspace has ${available_gb} GB free; the finetune needs at least ${MIN_WORKSPACE_GB} GB."
        echo "        Budget: ~5.4 GB per checkpoint x 3 retained, ~1.3 GB pretrained base,"
        echo "        plus the prepared dataset and the HuggingFace cache."
        echo "        Set ORON_ALLOW_SMALL_DISK=1 for a smoke test only."
        exit 1
    fi
fi

if ! python3.12 --version &>/dev/null; then
    echo "[ERROR] Python 3.12 not found. Use a RunPod PyTorch image (Ubuntu 24.04)."
    exit 1
fi
python3.12 --version

VENV_ARGS="--system-site-packages"
if ! python3.12 -c "import torch" 2>/dev/null; then
    echo "[WARN] No system PyTorch for Python 3.12; creating an isolated venv."
    echo "       This downloads PyTorch from PyPI. Use the Ubuntu 24.04 image to avoid it."
    VENV_ARGS=""
fi

python3.12 -m venv ${VENV_ARGS} .venv
source .venv/bin/activate

# Keep caches on the persistent volume, not the ephemeral container filesystem.
# Never overwrite user-supplied secrets already in .env.
CACHE_ROOT="${RUNPOD_CACHE_ROOT:-/workspace/.cache}"
mkdir -p "${CACHE_ROOT}/huggingface" "${CACHE_ROOT}/torch"

append_env_default() {
    local key="$1" value="$2"
    if [[ ! -f .env ]] || ! grep -q "^${key}=" .env; then
        printf "%s=%s\n" "${key}" "${value}" >> .env
    fi
    export "${key}=${value}"
}

append_env_default HF_HOME "${CACHE_ROOT}/huggingface"
append_env_default TORCH_HOME "${CACHE_ROOT}/torch"

pip install --upgrade pip --quiet
pip install --no-cache-dir -e ".[dev,audio,train,eval]"

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# rjieba backs f5_tts's convert_char_to_pinyin. Mongolian passes through it
# unchanged, but the tokenizer round-trip test is skipped without it -- and that
# test is what proves no character silently becomes a space.
python -c "import rjieba" 2>/dev/null || pip install --no-cache-dir rjieba

# Rebuild the extended vocabulary from the upstream base, so the pretrained
# prefix is verified on this machine rather than trusted from the checkout.
if [[ -f ../F5-TTS/data/Emilia_ZH_EN_pinyin/vocab.txt ]]; then
    python scripts/extend_vocab.py --out data/oron_mn_pinyin/vocab.txt
else
    echo "[WARN] Upstream F5-TTS checkout not found at ../F5-TTS."
    echo "       Clone it before preparing the dataset; using the committed vocab.txt for now."
fi

# The coverage tests are the gate: an unextended vocabulary silently replaces
# 4.90% of Mongolian characters with spaces, and nothing else reports it.
pytest -q

echo ""
echo "Setup complete. Next:"
echo "  1. Prepare the corpus with oron-cleaner"
echo "  2. python scripts/extend_vocab.py --out data/oron_mn_pinyin/vocab.txt \\"
echo "         --checkpoint ckpts/F5TTS_v1_Base/model_1250000.safetensors \\"
echo "         --checkpoint-out ckpts/oron_mn/pretrained_model_1250000.safetensors"
echo "  3. accelerate launch f5_tts/train/finetune_cli.py --exp_name F5TTS_v1_Base --learning_rate 1e-5 ..."
