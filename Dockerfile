# OronTTS - F5-TTS for Mongolian Language

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

LABEL description="F5-TTS for Mongolian Khalkha Dialect"
LABEL version="1.0.0"

WORKDIR /workspace/oron-tts

# Install system dependencies and FFmpeg 6 for torchcodec
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    git-lfs \
    libsndfile1-dev \
    sox \
    autoconf \
    automake \
    libtool \
    pkg-config \
    software-properties-common \
    && add-apt-repository ppa:ubuntuhandbook1/ffmpeg6 -y \
    && apt-get update \
    && apt-get install -y ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

# Build espeak-ng from source for Mongolian support
RUN cd /tmp && \
    git clone --depth 1 --branch 1.51.1 https://github.com/espeak-ng/espeak-ng.git && \
    cd espeak-ng && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf /tmp/espeak-ng

# Initialize git-lfs
RUN git lfs install

# Copy project files
COPY . /workspace/oron-tts/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir "packaging>=23.0,<24.0" && \
    pip install --no-cache-dir -e ".[dev]"

# Install Flash Attention 2 (optional)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "Flash Attention 2 not available, will use standard attention"

# Create directories
RUN mkdir -p outputs logs

# Environment
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

CMD ["/bin/bash"]
