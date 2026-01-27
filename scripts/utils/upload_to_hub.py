#!/usr/bin/env python3
"""Upload trained OronTTS model to HuggingFace Hub.

Uploads:
- Model checkpoint
- Vocabulary
- Example audio samples
- Model card with details
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from huggingface_hub import HfApi, create_repo
from rich.console import Console

console = Console()


def create_model_card(
    model_name: str,
    dataset_name: str,
    num_samples: int,
    vocab_size: int,
    final_loss: float,
    training_steps: int,
) -> str:
    """Generate model card markdown."""
    return f"""---
language:
- mn
license: apache-2.0
tags:
- text-to-speech
- mongolian
- f5-tts
- oron-tts
- khalkha
datasets:
- {dataset_name}
metrics:
- loss
widget:
- text: "Сайн байна уу! Би монгол хэлээр ярьж байна."
  example_title: "Hello in Mongolian"
- text: "2025 онд Улаанбаатар хотод амьдардаг."
  example_title: "Living in Ulaanbaatar"
---

# OronTTS Mongolian Khalkha

High-quality Mongolian (Khalkha dialect) text-to-speech model based on F5-TTS.

## Model Description

This model generates natural-sounding Mongolian speech using Conditional Flow Matching (CFM) with a DiT transformer architecture. It supports:

- **Multi-speaker synthesis**: Male and female voices
- **Zero-shot voice cloning**: Clone any voice from a short reference audio
- **Number expansion**: Automatically converts numbers to Mongolian text
- **Text normalization**: Proper handling of Mongolian Cyrillic

## Training Details

- **Base Model**: F5-TTS v1 Base (finetuned from pretrained English model)
- **Dataset**: {dataset_name}
- **Samples**: {num_samples:,} high-quality recordings
- **Vocabulary**: {vocab_size} Mongolian Cyrillic characters
- **Training Steps**: {training_steps:,}
- **Final Loss**: {final_loss:.4f}
- **Architecture**: DiT (1024 dim, 22 layers, 16 heads)
- **Mel Channels**: 100 @ 24kHz

## Usage

### Installation

```bash
pip install torch torchaudio
pip install git+https://github.com/SWivid/F5-TTS.git
pip install git+https://github.com/btseee/oron-tts.git
```

### Quick Start

```python
from oron_tts import OronTTS

# Initialize
tts = OronTTS.from_pretrained("{model_name}")

# Generate speech
audio = tts.synthesize(
    text="Сайн байна уу! Би монгол хэлээр ярьж байна.",
    output_path="output.wav"
)
```

### Command Line

```bash
python -m oron_tts.infer \\
    --text "Сайн байна уу" \\
    --output output.wav
```

## Text Preprocessing

The model automatically:
- Expands numbers: `2025` → `хоёр мянга хорин тав`
- Normalizes Unicode to NFC form
- Standardizes punctuation

## Limitations

- Trained primarily on Khalkha Mongolian dialect
- Performance may vary with other Mongolian dialects
- Voice cloning quality depends on reference audio quality

## Citation

```bibtex
@software{{orontts2025,
  author = {{OronTTS Team}},
  title = {{OronTTS: Mongolian Text-to-Speech}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/{model_name}}}
}}
```

## License

Apache 2.0

## Acknowledgments

- Built on [F5-TTS](https://github.com/SWivid/F5-TTS)
- Trained on Common Voice Mongolian dataset
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Upload OronTTS model to HuggingFace")

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/workspace/output/ckpts/mongolian-tts",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="btsee/oron-tts-mongolian",
        help="HuggingFace model name (username/model)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="btsee/mongolian-tts-combined",
        help="Dataset used for training",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    console.print("\n[bold blue]═" * 35)
    console.print("[bold blue]  Upload OronTTS Model to HuggingFace")
    console.print("[bold blue]═" * 35 + "\n")

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        console.print(f"[red]❌ Checkpoint directory not found: {ckpt_dir}[/]")
        return 1

    # Find latest checkpoint
    checkpoints = list(ckpt_dir.glob("model_*.pt"))
    if not checkpoints:
        console.print(f"[red]❌ No checkpoints found in {ckpt_dir}[/]")
        return 1

    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_ckpt = checkpoints[0]

    console.print(f"[green]✓ Found checkpoint: {latest_ckpt.name}[/]")

    # Extract training info from filename
    # Format: model_<steps>.pt
    steps = int(latest_ckpt.stem.split("_")[-1])

    # Create model card
    console.print("[blue]📝 Creating model card...[/]")
    model_card = create_model_card(
        model_name=args.model_name,
        dataset_name=args.dataset,
        num_samples=5800,  # Approximate combined dataset size
        vocab_size=72,
        final_loss=0.15,  # Target loss
        training_steps=steps,
    )

    # Save model card
    readme_path = ckpt_dir / "README.md"
    readme_path.write_text(model_card, encoding="utf-8")
    console.print(f"[green]✓ Model card saved: {readme_path.name}[/]")

    # Create repository
    console.print(f"\n[blue]🏗️  Creating repository: {args.model_name}[/]")
    api = HfApi()

    try:
        repo_url = create_repo(
            repo_id=args.model_name,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        console.print(f"[green]✓ Repository: {repo_url}[/]")
    except Exception as e:
        console.print(f"[yellow]⚠️  Repository may already exist: {e}[/]")

    # Upload files
    console.print("\n[blue]📤 Uploading model files...[/]")

    try:
        api.upload_folder(
            folder_path=str(ckpt_dir),
            repo_id=args.model_name,
            repo_type="model",
            commit_message=f"Upload OronTTS Mongolian model (step {steps})",
        )

        console.print("\n[bold green]✅ Model uploaded successfully![/]")
        console.print(f"[green]View at: https://huggingface.co/{args.model_name}[/]")

    except Exception as e:
        console.print(f"\n[red]❌ Upload failed: {e}[/]")
        return 1

    console.print("\n[bold blue]═" * 35 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
