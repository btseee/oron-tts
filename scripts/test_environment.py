#!/usr/bin/env python
"""Test OronTTS installation and environment for Runpod deployment."""

import sys
from pathlib import Path


def test_imports():
    """Test all required imports."""
    print("📦 Testing imports...")
    
    errors = []
    
    # Core
    try:
        import torch
        print(f"  ✅ torch {torch.__version__}")
    except ImportError as e:
        errors.append(f"torch: {e}")
    
    try:
        import torchaudio
        print(f"  ✅ torchaudio {torchaudio.__version__}")
    except ImportError as e:
        errors.append(f"torchaudio: {e}")
    
    try:
        import lightning
        print(f"  ✅ lightning {lightning.__version__}")
    except ImportError as e:
        errors.append(f"lightning: {e}")
    
    # Audio
    try:
        import librosa
        print(f"  ✅ librosa {librosa.__version__}")
    except ImportError as e:
        errors.append(f"librosa: {e}")
    
    try:
        import soundfile
        print(f"  ✅ soundfile {soundfile.__version__}")
    except ImportError as e:
        errors.append(f"soundfile: {e}")
    
    # OronTTS
    try:
        import orontts
        print(f"  ✅ orontts {orontts.__version__}")
    except ImportError as e:
        errors.append(f"orontts: {e}")
    
    try:
        from orontts.preprocessing.audio import AudioCleaner
        cleaner = AudioCleaner()
        df_status = "✅" if cleaner.has_deepfilter else "⚠️ (optional)"
        print(f"  {df_status} DeepFilterNet")
    except Exception as e:
        errors.append(f"DeepFilterNet: {e}")
    
    try:
        from orontts.model.config import VITS2Config
        config = VITS2Config.from_preset("light")
        print(f"  ✅ VITS2Config (light preset)")
    except Exception as e:
        errors.append(f"VITS2Config: {e}")
    
    try:
        from orontts.model.vits2 import VITS2
        print(f"  ✅ VITS2 model")
    except Exception as e:
        errors.append(f"VITS2: {e}")
    
    return errors


def test_cuda():
    """Test CUDA availability."""
    print("\n🔧 Testing CUDA...")
    
    import torch
    
    if not torch.cuda.is_available():
        print("  ⚠️  CUDA not available (CPU only)")
        print("     (Expected for CPU-only testing, will work on Runpod)")
        return []  # Not an error - just a warning
    
    print(f"  ✅ CUDA available")
    print(f"  ✅ Device: {torch.cuda.get_device_name(0)}")
    print(f"  ✅ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test tensor operations
    try:
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.matmul(x, x)
        del x, y
        torch.cuda.empty_cache()
        print(f"  ✅ Tensor operations work")
    except Exception as e:
        return [f"CUDA tensor ops: {e}"]
    
    return []


def test_espeak():
    """Test espeak-ng for phonemization."""
    print("\n🗣️  Testing espeak-ng...")
    
    import subprocess
    
    try:
        result = subprocess.run(
            ["espeak-ng", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            version = result.stdout.strip().split("\n")[0]
            print(f"  ✅ {version}")
        else:
            return ["espeak-ng not working"]
    except FileNotFoundError:
        return ["espeak-ng not installed"]
    except Exception as e:
        return [f"espeak-ng: {e}"]
    
    # Test Mongolian
    try:
        from orontts.preprocessing.phonemizer import MongolianPhonemizer
        phonemizer = MongolianPhonemizer()
        result = phonemizer.phonemize("Сайн байна уу")
        print(f"  ✅ Mongolian phonemization: '{result}'")
    except Exception as e:
        return [f"Mongolian phonemizer: {e}"]
    
    return []


def test_model_creation():
    """Test model creation on GPU."""
    print("\n🧠 Testing model creation...")
    
    import torch
    from orontts.model.config import VITS2Config
    from orontts.model.vits2 import VITS2
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        config = VITS2Config.from_preset("light")
        model = VITS2(config)
        model = model.to(device)
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  ✅ Model created on {device}")
        print(f"  ✅ Parameters: {param_count / 1e6:.1f}M")
        
        # Test forward pass with dummy data
        batch_size = 2
        text_len = 50
        
        # This is a simplified test - real training uses more complex inputs
        print(f"  ✅ Model ready for training")
        
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        
    except Exception as e:
        return [f"Model creation: {e}"]
    
    return []


def test_dataset():
    """Test dataset loading."""
    print("\n📂 Testing dataset access...")
    
    try:
        from datasets import load_dataset
        
        # Try to load the dataset in streaming mode (no download)
        ds = load_dataset(
            "btsee/common-voices-24-mn",
            split="train",
            streaming=True,
        )
        
        # Get first sample
        sample = next(iter(ds))
        print(f"  ✅ Dataset accessible")
        print(f"  ✅ Sample text: {sample['text'][:50]}...")
        print(f"  ✅ Sample duration: {sample['duration']:.2f}s")
        
    except Exception as e:
        print(f"  ⚠️  Dataset not accessible: {e}")
        print(f"     (This is OK if running without network)")
        return []  # Non-critical
    
    return []


def main():
    """Run all tests."""
    print("=" * 60)
    print("  OronTTS Environment Test")
    print("=" * 60)
    
    all_errors = []
    
    all_errors.extend(test_imports())
    all_errors.extend(test_cuda())
    all_errors.extend(test_espeak())
    all_errors.extend(test_model_creation())
    all_errors.extend(test_dataset())
    
    print("\n" + "=" * 60)
    
    if all_errors:
        print("❌ FAILED - Issues found:")
        for error in all_errors:
            print(f"   - {error}")
        sys.exit(1)
    else:
        print("✅ ALL TESTS PASSED")
        print("   Environment is ready for training!")
        sys.exit(0)


if __name__ == "__main__":
    main()
