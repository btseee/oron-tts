from pathlib import Path

import torch

from src.models.f5tts import F5TTS
from src.utils.checkpoint import CheckpointManager, stale_remote_checkpoint_paths


def _tiny_config() -> dict[str, object]:
    return {
        "sample_rate": 24000,
        "n_fft": 1024,
        "hop_length": 256,
        "n_mels": 100,
        "model": {
            "vocab_size": 65,
            "dim": 64,
            "depth": 1,
            "heads": 2,
            "ff_mult": 2,
            "text_dim": 32,
            "conv_layers": 1,
        },
    }


def _compiled_backbone_state_dict(model: F5TTS) -> dict[str, torch.Tensor]:
    compiled_state: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if key.startswith("cfm.backbone."):
            key = key.replace("cfm.backbone.", "cfm.backbone._orig_mod.", 1)
        compiled_state[key] = value.clone()
    return compiled_state


def test_load_accepts_compiled_backbone_checkpoint(tmp_path: Path) -> None:
    source = F5TTS.from_config(_tiny_config())
    target = F5TTS.from_config(_tiny_config())
    checkpoint_path = tmp_path / "compiled.pt"
    torch.save(
        {
            "step": 12,
            "loss": 0.25,
            "model_state_dict": _compiled_backbone_state_dict(source),
        },
        checkpoint_path,
    )

    info = CheckpointManager(tmp_path).load(target, path=checkpoint_path)

    assert info["step"] == 12
    assert info["loss"] == 0.25
    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value)


def test_load_pretrained_skips_incompatible_shapes(tmp_path: Path) -> None:
    model = F5TTS.from_config(_tiny_config())
    state = {key: value.clone() for key, value in model.state_dict().items()}
    state["cfm.backbone.text_embed.text_embed.weight"] = torch.randn(10, 32)
    checkpoint_path = tmp_path / "pretrained.pt"
    torch.save({"model_state_dict": state}, checkpoint_path)

    result = CheckpointManager(tmp_path).load_pretrained_f5tts(
        model,
        checkpoint_path,
        strict=False,
    )

    assert result["skipped_keys"] == ["cfm.backbone.text_embed.text_embed.weight"]


def test_load_pretrained_prefers_ema_state_dict(tmp_path: Path) -> None:
    source = F5TTS.from_config(_tiny_config())
    target = F5TTS.from_config(_tiny_config())
    raw_state = {key: value.clone() for key, value in source.state_dict().items()}
    ema_state = {key: value.clone() for key, value in source.state_dict().items()}
    float_key = next(key for key, value in ema_state.items() if value.is_floating_point())
    ema_state[float_key] = torch.full_like(ema_state[float_key], 0.25)
    raw_state[float_key] = torch.full_like(raw_state[float_key], -0.25)
    checkpoint_path = tmp_path / "oron_best.pt"
    torch.save({"model_state_dict": raw_state, "ema_state_dict": ema_state}, checkpoint_path)

    CheckpointManager(tmp_path).load_pretrained_f5tts(target, checkpoint_path, strict=False)

    assert torch.equal(target.state_dict()[float_key], ema_state[float_key])


def test_stale_remote_checkpoint_paths_keeps_local_rotation() -> None:
    remote_paths = [
        "README.md",
        "config.json",
        "f5tts_best.pt",
        "f5tts_step_00000010.pt",
        "f5tts_step_00000020.pt",
        "f5tts_step_00000030.pt",
        "tb_logs/events.out.tfevents.test",
    ]
    local_paths = ["f5tts_step_00000020.pt", "f5tts_step_00000030.pt"]

    stale = stale_remote_checkpoint_paths(remote_paths, local_paths, "f5tts")

    assert stale == ["f5tts_step_00000010.pt"]
