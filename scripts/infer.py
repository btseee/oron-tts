"""Inference script for OronTTS F5-TTS."""

import argparse
from pathlib import Path

import soundfile as sf
import torch

from src.models.f5tts import F5TTS
from src.utils.checkpoint import CheckpointManager


def load_model(checkpoint_path: str, device: str, use_ema: bool = True) -> F5TTS:
    cm = CheckpointManager(str(Path(checkpoint_path).parent))
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = cm.load_config() or {}
    model = F5TTS.from_config(config)

    # Prefer EMA weights — they produce significantly better audio for diffusion models
    if use_ema and "ema_state_dict" in ckpt:
        state = ckpt["ema_state_dict"]
        print("Loading EMA weights (smoothed)")
    else:
        state = ckpt.get("model_state_dict", ckpt)
        if use_ema:
            print("[WARN] EMA weights not found in checkpoint, using raw training weights")
        else:
            print("Loading raw training weights (--no-ema)")

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
    parser.add_argument("--steps", type=int, default=32, help="ODE integration steps")
    parser.add_argument("--duration", type=float, default=None, help="Target duration in seconds")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speaking-rate multiplier (>1 faster, <1 slower). Ignored if --duration set.",
    )
    parser.add_argument("--no-ema", action="store_true", help="Use raw weights instead of EMA")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device, use_ema=not args.no_ema)
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(f"Synthesising [{args.lang}]: {args.text}")
    waveform = model.synthesize(
        text=args.text,
        lang=args.lang,
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        n_steps=args.steps,
        target_duration_s=args.duration,
        speed=args.speed,
        device=device,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), waveform.cpu().numpy(), model.sample_rate)
    print(f"Saved: {output_path} ({len(waveform) / model.sample_rate:.2f} s)")


if __name__ == "__main__":
    main()
