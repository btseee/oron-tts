# OronTTS - F5-TTS for Mongolian Language
# Optimized for RunPod.io training environment

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

LABEL description="F5-TTS for Mongolian Khalkha Dialect - Training"
LABEL version="1.0.0"

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    git-lfs \
    libsndfile1-dev \
    sox \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

# Clone repository with submodules
RUN git clone --recurse-submodules https://github.com/btseee/oron-tts.git

WORKDIR /workspace/oron-tts

# Install F5-TTS from submodule (includes all training dependencies)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e third_party/F5-TTS

# Install OronTTS (minimal wrapper)
RUN pip install --no-cache-dir -e .

# Install Flash Attention 2 for faster training
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "Flash Attention 2 not available, using standard attention"

# Create directories
RUN mkdir -p ckpts logs data

# Environment
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface

# Default command
CMD ["/bin/bash"]
