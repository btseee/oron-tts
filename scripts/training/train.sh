#!/bin/bash
# Complete training pipeline for OronTTS Mongolian TTS
# Runs in BACKGROUND with nohup

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[!]${NC} $1"; }

echo "======================================================================="
echo "  OronTTS Training Pipeline - Mongolian Khalkha TTS"
echo "======================================================================="
echo ""

# Step 1: Prepare combined dataset
log_info "Step 1/3: Preparing combined dataset..."
log_info "Combining mbspeech (3.8k) + Common Voice (best male/female voices)"
echo ""

python scripts/data/prepare_combined_dataset.py

if [ $? -ne 0 ]; then
    log_warn "Dataset preparation failed or already exists"
    log_info "Continuing with existing dataset..."
fi

echo ""
echo "-----------------------------------------------------------------------"
echo ""

# Step 2: Train model (IN BACKGROUND)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="/workspace/output/logs/train_mongolian_${TIMESTAMP}.log"
PID_FILE="/workspace/output/logs/train_mongolian.pid"

mkdir -p /workspace/output/logs

log_info "Step 2/3: Starting training in BACKGROUND..."
log_info "Strategy: Finetune from pretrained F5-TTS"
log_info "Output: /workspace/output/"
log_info "Log file: ${LOG_FILE}"
echo ""

nohup python scripts/training/train.py \
    --dataset /workspace/output/data/mongolian-tts-combined \
    --output-dir /workspace/output \
    --epochs 300 \
    --batch-size 2400 \
    --learning-rate 7.5e-5 \
    --save-every 2000 > "${LOG_FILE}" 2>&1 &

TRAIN_PID=$!
echo "${TRAIN_PID}" > "${PID_FILE}"

log_success "Training started in background with PID: ${TRAIN_PID}"
echo ""
echo "Monitor with:"
echo "  tail -f ${LOG_FILE}"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Stop with:"
echo "  kill ${TRAIN_PID}"
echo ""

# Wait a bit to check if training started successfully
sleep 5
if ps -p ${TRAIN_PID} > /dev/null 2>&1; then
    log_success "Training process is running"
else
    log_warn "Training process may have failed - check ${LOG_FILE}"
    exit 1
fi

# Skip step 3 since training is in background
log_info "Step 3/3: Skipping example generation (training in progress)"
log_info "Generate example after training with:"
echo "  python scripts/inference/infer.py"
echo ""

echo "======================================================================="
log_success "Training pipeline started in BACKGROUND!"
echo ""
echo "  PID:         ${TRAIN_PID}"
echo "  Log:         ${LOG_FILE}"
echo "  Checkpoints: /workspace/output/ckpts/mongolian-tts/"
echo "  TensorBoard: tensorboard --logdir /workspace/output/runs"
echo ""
echo "Monitor progress:"
echo "  tail -f ${LOG_FILE}"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "When training completes, generate example:"
echo "  python scripts/inference/infer.py"
echo ""
echo "Upload to HuggingFace:"
echo "  python scripts/utils/upload_to_hub.py --model-name btsee/oron-tts-mongolian"
echo "======================================================================="
