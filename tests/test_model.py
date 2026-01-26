import pytest
import torch
from src.core.model import F5TTS, F5TTSConfig

def test_model_init():
    config = F5TTSConfig(
        mel_dim=100,
        phoneme_vocab_size=256,
        dim=64,
        depth=2,
        num_heads=4
    )
    model = F5TTS(config)
    assert model is not None

def test_model_forward(mock_audio_tensor):
    config = F5TTSConfig(
        mel_dim=100,
        phoneme_vocab_size=256,
        dim=64,
        depth=2,
        num_heads=4
    )
    model = F5TTS(config)
    
    # Mock inputs
    mel = torch.randn(1, 100, 50) # B, Mel, T
    phonemes = torch.randint(0, 256, (1, 20)) # B, T_text
    
    loss_dict = model.compute_loss(
        mel=mel,
        phonemes=phonemes
    )
    
    assert "loss" in loss_dict
    assert isinstance(loss_dict["loss"], torch.Tensor)
