"""Inference script for OronTTS F5-TTS."""

import argparse
from pathlib import Path

import torch
import torchaudio

from src.models.f5tts import F5TTS
from src.utils.checkpoint import CheckpointManager


def load_model(checkpoint_path: str, device: str) -> F5TTS:
    cm = CheckpointManager(str(Path(checkpoint_path).parent))
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = cm.load_config() or {}
    model = F5TTS.from_config(config)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    model.eval()
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="OronTTS F5-TTS Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--text", type=str, required=True, help="Cyrillic text to synthesize")
    parser.add_argument("--lang", type=str, default="mn", choices=["mn", "kz"])
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="3-10 s reference WAV for voice cloning (Option A)",
    )
    parser.add_argument("--ref-text", type=str, default=None, help="Transcript of ref-audio clip")
    parser.add_argument(
        "--attr-tokens",
        type=str,
        default=None,
        help="Comma-separated attribute tags e.g. '[FEMALE],[YOUNG]' (Option B)",
    )
    parser.add_argument("--steps", type=int, default=32, help="ODE integration steps")
    parser.add_argument("--duration", type=float, default=None, help="Target duration in seconds")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    attr_tokens = None
    if args.attr_tokens:
        attr_tokens = [t.strip() for t in args.attr_tokens.split(",")]

    print(f"Synthesising [{args.lang}]: {args.text}")
    waveform = model.synthesize(
        text=args.text,
        lang=args.lang,
        attr_tokens=attr_tokens,
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        n_steps=args.steps,
        target_duration_s=args.duration,
        device=device,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), waveform.unsqueeze(0), model.sample_rate)
    print(f"Saved: {output_path} ({len(waveform) / model.sample_rate:.2f} s)")


if __name__ == "__main__":
    main()
