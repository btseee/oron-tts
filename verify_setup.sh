#!/bin/bash
# Verification script for OronTTS dependencies and setup

echo "=== OronTTS Dependency Verification ==="
echo ""

# Check Python version
python_version=$(python --version 2>&1)
echo "✓ $python_version"

# Check critical packages
packages=("torch" "torchaudio" "numpy" "scipy" "librosa" "soundfile" "datasets" "huggingface_hub" "deepfilternet" "yaml" "tqdm" "tensorboard" "einops")

echo ""
echo "Checking Python packages:"
for pkg in "${packages[@]}"; do
    if python -c "import $pkg" 2>/dev/null; then
        version=$(python -c "import $pkg; print(getattr($pkg, '__version__', 'installed'))" 2>/dev/null)
        echo "✓ $pkg ($version)"
    else
        echo "✗ $pkg - MISSING"
    fi
done

# Check CUDA
echo ""
cuda_available=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [ "$cuda_available" = "True" ]; then
    cuda_version=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>/dev/null)
    echo "✓ CUDA $cuda_version available"
    echo "✓ GPU: $gpu_name"
else
    echo "✗ CUDA not available (CPU only)"
fi

echo ""
echo "=== Verification Complete ==="
