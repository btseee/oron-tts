import pytest
import torch

@pytest.fixture
def mock_audio_tensor():
    return torch.randn(1, 1, 24000)

@pytest.fixture
def mock_text_batch():
    return ["Сайн байна уу?", "123"]
