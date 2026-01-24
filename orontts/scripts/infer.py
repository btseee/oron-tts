"""Inference CLI entry point."""

import argparse
from pathlib import Path

from rich.console import Console

console = Console()


def main() -> None:
    """Main entry point for inference CLI."""
    parser = argparse.ArgumentParser(
        description="Synthesize speech with OronTTS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to checkpoint file or HuggingFace model ID",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to synthesize",
    )
    parser.add_argument(
        "--text-file",
        type=Path,
        default=None,
        help="File containing text to synthesize (one utterance per line)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output.wav"),
        help="Output audio file path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for batch synthesis",
    )
    parser.add_argument(
        "--speaker",
        type=int,
        default=0,
        help="Speaker ID for multi-speaker models",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.667,
        help="Noise scale for sampling (higher = more variation)",
    )
    parser.add_argument(
        "--length-scale",
        type=float,
        default=1.0,
        help="Duration scale (higher = slower speech)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference (cpu, cuda, cuda:0, etc.)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Output sample rate (resample if different from model)",
    )

    args = parser.parse_args()

    # Validate input
    if args.text is None and args.text_file is None:
        console.print("[red]Error:[/red] Either --text or --text-file is required")
        return

    console.print("[bold green]OronTTS Synthesis[/bold green]")

    # Import here to avoid slow startup
    from orontts.inference import Synthesizer

    try:
        # Load model
        console.print(f"Loading model: {args.model}")
        model_path = Path(args.model)

        if model_path.exists():
            synth = Synthesizer.from_checkpoint(model_path, device=args.device)
        else:
            # Assume HuggingFace model ID
            synth = Synthesizer.from_pretrained(args.model, device=args.device)

        console.print(f"Model loaded. Sample rate: {synth.sample_rate} Hz")

        # Single text synthesis
        if args.text:
            console.print(f"Synthesizing: {args.text[:50]}...")

            output = synth.synthesize(
                text=args.text,
                speaker_id=args.speaker,
                noise_scale=args.noise_scale,
                length_scale=args.length_scale,
            )

            # Resample if requested
            if args.sample_rate and args.sample_rate != output.sample_rate:
                output = output.resample(args.sample_rate)

            output.save(args.output)
            console.print(f"[green]Saved:[/green] {args.output} ({output.duration:.2f}s)")

        # Batch synthesis from file
        elif args.text_file:
            if not args.text_file.exists():
                console.print(f"[red]Error:[/red] Text file not found: {args.text_file}")
                return

            output_dir = args.output_dir or args.text_file.parent / "synthesized"
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(args.text_file, encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]

            console.print(f"Synthesizing {len(texts)} utterances...")

            from tqdm import tqdm

            for i, text in enumerate(tqdm(texts)):
                output = synth.synthesize(
                    text=text,
                    speaker_id=args.speaker,
                    noise_scale=args.noise_scale,
                    length_scale=args.length_scale,
                )

                if args.sample_rate and args.sample_rate != output.sample_rate:
                    output = output.resample(args.sample_rate)

                output_path = output_dir / f"{i:05d}.wav"
                output.save(output_path)

            console.print(f"[green]Saved {len(texts)} files to:[/green] {output_dir}")

    except Exception as e:
        console.print(f"[red]Synthesis failed:[/red] {e}")
        raise


if __name__ == "__main__":
    main()
