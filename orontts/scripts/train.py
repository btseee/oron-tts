"""Training CLI entry point."""

import argparse
from pathlib import Path

from rich.console import Console

console = Console()


def main() -> None:
    """Main entry point for training CLI."""
    parser = argparse.ArgumentParser(
        description="Train OronTTS VITS2 model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to model configuration JSON file",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Path for outputs (checkpoints, logs)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="auto",
        help="GPU devices to use (e.g., '0', '0,1', 'auto')",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        choices=["32", "16-mixed", "bf16-mixed"],
        help="Training precision",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint path to resume training from",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="HuggingFace repo ID to push model after training",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.config.exists():
        console.print(f"[red]Error:[/red] Config file not found: {args.config}")
        return

    if not args.data_dir.exists():
        console.print(f"[red]Error:[/red] Data directory not found: {args.data_dir}")
        return

    console.print("[bold green]OronTTS Training[/bold green]")
    console.print(f"  Config: {args.config}")
    console.print(f"  Data: {args.data_dir}")
    console.print(f"  Output: {args.output_dir}")
    console.print()

    # Import here to avoid slow startup
    from orontts.training.trainer import train

    try:
        train(
            config_path=args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            max_epochs=args.epochs,
            devices=args.devices,
            precision=args.precision,
            resume_from=args.resume,
        )

        console.print("[bold green]Training complete![/bold green]")

        # Push to Hub if requested
        if args.push_to_hub:
            from orontts.dataset.hf_integration import push_model_to_hub
            from orontts.model.config import load_config

            config = load_config(args.config)
            url = push_model_to_hub(
                model_dir=args.output_dir / "checkpoints",
                repo_id=args.push_to_hub,
                config=config.model_dump(),
            )
            console.print(f"[green]Model pushed to:[/green] {url}")

    except KeyboardInterrupt:
        console.print("[yellow]Training interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Training failed:[/red] {e}")
        raise


if __name__ == "__main__":
    main()
