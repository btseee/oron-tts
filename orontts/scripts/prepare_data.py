"""Data preparation CLI entry point."""

import argparse
from pathlib import Path

from rich.console import Console

console = Console()


def main() -> None:
    """Main entry point for data preparation CLI."""
    parser = argparse.ArgumentParser(
        description="Prepare and clean audio data for OronTTS training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory containing raw audio files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for cleaned audio",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Target sample rate",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Minimum audio duration in seconds",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=15.0,
        help="Maximum audio duration in seconds",
    )
    parser.add_argument(
        "--trim-silence",
        action="store_true",
        default=True,
        help="Trim leading/trailing silence",
    )
    parser.add_argument(
        "--trim-db",
        type=float,
        default=20.0,
        help="Silence threshold in dB for trimming",
    )
    parser.add_argument(
        "--skip-cleaning",
        action="store_true",
        help="Skip DeepFilterNet noise suppression",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for DeepFilterNet (cpu or cuda)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to metadata CSV file (format: file_id|speaker_id|text)",
    )
    parser.add_argument(
        "--normalize-text",
        action="store_true",
        default=True,
        help="Normalize text in metadata",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="Push cleaned dataset to HuggingFace Hub (repo ID)",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.input_dir.exists():
        console.print(f"[red]Error:[/red] Input directory not found: {args.input_dir}")
        return

    console.print("[bold green]OronTTS Data Preparation[/bold green]")
    console.print(f"  Input: {args.input_dir}")
    console.print(f"  Output: {args.output_dir}")
    console.print()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Audio cleaning
    if not args.skip_cleaning:
        console.print("[bold]Step 1: Audio Cleaning[/bold]")

        from orontts.preprocessing.audio import AudioCleanerConfig, batch_clean_audio

        config = AudioCleanerConfig(
            target_sample_rate=args.sample_rate,
            trim_silence=args.trim_silence,
            trim_db=args.trim_db,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )

        stats = batch_clean_audio(
            input_dir=args.input_dir,
            output_dir=args.output_dir / "wavs",
            config=config,
            device=args.device,
        )

        console.print(f"  Processed: {stats['processed']}")
        console.print(f"  Skipped: {stats['skipped']}")
        console.print(f"  Failed: {stats['failed']}")
    else:
        console.print("[yellow]Skipping audio cleaning[/yellow]")
        # Copy files without cleaning
        import shutil

        wavs_dir = args.output_dir / "wavs"
        wavs_dir.mkdir(exist_ok=True)

        for ext in (".wav", ".flac", ".mp3"):
            for f in args.input_dir.rglob(f"*{ext}"):
                shutil.copy(f, wavs_dir / f.name)

    # Process metadata
    if args.metadata:
        console.print("\n[bold]Step 2: Metadata Processing[/bold]")

        from orontts.preprocessing.text import normalize_text

        output_metadata = args.output_dir / "metadata.csv"

        with open(args.metadata, encoding="utf-8") as f_in, \
             open(output_metadata, "w", encoding="utf-8") as f_out:

            for line in f_in:
                line = line.strip()
                if not line or line.startswith("#"):
                    f_out.write(line + "\n")
                    continue

                parts = line.split("|")
                if len(parts) >= 3:
                    file_id, speaker_id, text = parts[0], parts[1], parts[2]

                    # Check if audio exists
                    audio_path = args.output_dir / "wavs" / f"{file_id}.wav"
                    if not audio_path.exists():
                        continue

                    # Normalize text
                    if args.normalize_text:
                        text = normalize_text(text)

                    f_out.write(f"{file_id}|{speaker_id}|{text}\n")

        console.print(f"  Metadata saved: {output_metadata}")

    # Push to HuggingFace Hub
    if args.push_to_hub:
        console.print("\n[bold]Step 3: Pushing to HuggingFace Hub[/bold]")

        from orontts.dataset.hf_integration import push_dataset_to_hub

        url = push_dataset_to_hub(
            data_dir=args.output_dir,
            repo_id=args.push_to_hub,
        )
        console.print(f"  [green]Dataset pushed to:[/green] {url}")

    console.print("\n[bold green]Data preparation complete![/bold green]")


if __name__ == "__main__":
    main()
