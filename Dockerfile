# OronTTS Docker Image for Runpod.io
# Matches Runpod's pytorch template environment

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

LABEL maintainer="OronTTS Contributors"
LABEL description="Mongolian VITS2 TTS Training Environment"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV WORKSPACE=/workspace
ENV HF_HOME=/workspace/.cache/huggingface
ENV TORCH_HOME=/workspace/.cache/torch

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Audio processing
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    sox \
    libsox-dev \
    # espeak-ng for phonemization (with all language data)
    espeak-ng \
    espeak-ng-data \
    libespeak-ng-dev \
    # Build tools
    build-essential \
    cmake \
    pkg-config \
    git \
    wget \
    curl \
    # HDF5 for datasets
    libhdf5-dev \
    # Misc
    vim \
    tmux \
    htop \
    nvtop \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory
WORKDIR ${WORKSPACE}

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY orontts/ ./orontts/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e ".[dev,cleaning]" && \
    pip install \
        tensorboard \
        wandb && \
    pip uninstall -y torchcodec || true

# Verify installations
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" && \
    python -c "import orontts; print(f'OronTTS: {orontts.__version__}')" && \
    python -c "from orontts.preprocessing.audio import AudioCleaner; print(f'DeepFilterNet: {AudioCleaner().has_deepfilter}')" && \
    espeak-ng --version

# Create directories for data and checkpoints
RUN mkdir -p ${WORKSPACE}/data \
    ${WORKSPACE}/checkpoints \
    ${WORKSPACE}/logs \
    ${WORKSPACE}/.cache

# Set up entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose ports for TensorBoard and Jupyter
EXPOSE 6006 8888

# Default command
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
