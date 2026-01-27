#!/bin/bash
# =============================================================================
# OronTTS Background Training Script
# Runs training in background with nohup, logging to file
# =============================================================================

set -euo pipefail

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
DATASET_NAME="${1:-oron_mn}"
EXTRA_ARGS="${@:2}"

# Create log directory
mkdir -p "${LOG_DIR}"

# Generate log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${DATASET_NAME}_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/train_${DATASET_NAME}.pid"

# Check if training is already running
if [ -f "${PID_FILE}" ]; then
    OLD_PID=$(cat "${PID_FILE}")
    if ps -p "${OLD_PID}" > /dev/null 2>&1; then
        log_error "Training already running with PID ${OLD_PID}"
        log_info "To stop: kill ${OLD_PID}"
        log_info "To view logs: tail -f ${LOG_DIR}/train_${DATASET_NAME}_*.log"
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

# Build training command
TRAIN_CMD="python scripts/training/train.py \
    --dataset-name ${DATASET_NAME} \
    --finetune \
    --epochs 100 \
    --batch-size 3200 \
    --learning-rate 1e-5 \
    --warmup-updates 2000 \
    --save-per-updates 10000 \
    --last-per-updates 1000 \
    --keep-checkpoints 5 \
    --tokenizer char \
    --log-samples \
    ${EXTRA_ARGS}"

log_info "Starting background training..."
log_info "Dataset: ${DATASET_NAME}"
log_info "Log file: ${LOG_FILE}"
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
echo -e "${BLUE}Useful commands:${NC}"
echo "  View logs:      tail -f ${LOG_FILE}"
echo "  Check status:   ps -p ${TRAIN_PID}"
echo "  Stop training:  kill ${TRAIN_PID}"
echo "  GPU usage:      watch -n 1 nvidia-smi"
echo ""
echo -e "${GREEN}Training is running in background. You can safely disconnect.${NC}"
