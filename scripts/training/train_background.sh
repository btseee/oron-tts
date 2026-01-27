#!/bin/bash
# =============================================================================
# OronTTS Background Training Script
# Runs training in background with nohup, logging to file
# =============================================================================

set -euo pipefail

# Force datasets to use soundfile instead of torchcodec
export HF_AUDIO_DECODER=soundfile

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }

# Default configuration
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
PROJECT_DIR="${PROJECT_DIR:-${WORKSPACE_DIR}/oron-tts}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
DATASET_NAME="${1:-btsee/common-voices-24-mn}"
EXTRA_ARGS="${@:2}"

# Create log directory
mkdir -p "${LOG_DIR}"

# Generate log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DATASET_SHORT=$(echo "${DATASET_NAME}" | sed 's/.*\///')
LOG_FILE="${LOG_DIR}/train_${DATASET_SHORT}_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/train_${DATASET_SHORT}.pid"

# Check if training is already running
if [ -f "${PID_FILE}" ]; then
    OLD_PID=$(cat "${PID_FILE}")
    if ps -p "${OLD_PID}" > /dev/null 2>&1; then
        log_error "Training already running with PID ${OLD_PID}"
        log_info "To stop: kill ${OLD_PID}"
        log_info "To view logs: tail -f ${LOG_FILE}"
        exit 1
    else
        log_warn "Stale PID file found, removing..."
        rm -f "${PID_FILE}"
    fi
fi

# Navigate to project directory
cd "${PROJECT_DIR}"

# Activate virtual environment if exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# RTX 4090 optimized training command (from scratch)
TRAIN_CMD="python scripts/training/train.py \
    --dataset-name ${DATASET_NAME} \
    --epochs 500 \
    --batch-size 1800 \
    --learning-rate 7.5e-5 \
    --warmup-updates 1000 \
    --save-per-updates 5000 \
    --last-per-updates 500 \
    --keep-checkpoints 5 \
    --grad-accumulation 2 \
    --max-samples 32 \
    --num-workers 4 \
    --logger tensorboard \
    ${EXTRA_ARGS}"

log_info "Starting background training..."
log_info "Dataset: ${DATASET_NAME}"
log_info "Log file: ${LOG_FILE}"
log_info "Mode: Training from scratch (RTX 4090 optimized)"
echo ""
log_info "Command: ${TRAIN_CMD}"
echo ""

# Start training in background
nohup ${TRAIN_CMD} > "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!

# Save PID
echo "${TRAIN_PID}" > "${PID_FILE}"

log_success "Training started with PID: ${TRAIN_PID}"
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  RTX 4090 Configuration:${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo "  Batch Size:      1800 frames"
echo "  Learning Rate:   7.5e-5"
echo "  Grad Accum:      2 (effective batch: 3600)"
echo "  Max Samples:     32"
echo ""
echo -e "${BLUE}Useful commands:${NC}"
echo "  View logs:       tail -f ${LOG_FILE}"
echo "  Check status:    ps -p ${TRAIN_PID}"
echo "  Stop training:   kill ${TRAIN_PID}"
echo "  GPU usage:       watch -n 1 nvidia-smi"
echo "  TensorBoard:     tensorboard --logdir ckpts/"
echo ""
echo -e "${GREEN}Training is running in background. You can safely disconnect.${NC}"
