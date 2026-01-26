#!/usr/bin/env python3
"""Sample generation script for model evaluation.

Generates audio samples from a trained model for quality assessment.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from rich.console import Console
from rich.table import Table

from scripts.infer import Synthesizer

console = Console()


# Mongolian test sentences for evaluation
EVAL_SENTENCES = [
    # Basic greetings
    "Сайн байна уу.",
    "Баярлалаа.",
    "Уучлаарай.",
    
    # Numbers
    "Нэг, хоёр, гурав, дөрөв, тав.",
    "Энэ жил 2024 он.",
    
    # Common phrases
    "Намайг Болд гэдэг.",
    "Та хаанаас ирсэн бэ?",
    "Энэ хэдэн төгрөг вэ?",
    
    # Longer sentences
    "Монгол улс нь Төв Азид оршдог бөгөөд хойд талаараа Орос, урд талаараа Хятадтай хиллэдэг.",
    "Улаанбаатар хот бол Монгол улсын нийслэл бөгөөд хамгийн том хот юм.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evaluation samples")
    
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/samples")
    parser.add_argument("--speakers", type=str, default="0", help="Comma-separated speaker IDs")
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_dir = output_dir / f"samples_{timestamp}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    synth = Synthesizer(
        model_path=args.model,
        device=args.device,
    )
    
    speaker_ids = [int(s) for s in args.speakers.split(",")]
    
    # Results table
    table = Table(title="Generated Samples")
    table.add_column("ID", style="cyan")
    table.add_column("Speaker", style="magenta")
    table.add_column("Text", style="white", max_width=50)
    table.add_column("Duration", style="green")
    
    manifest = []
    
    for speaker_id in speaker_ids:
        speaker_dir = sample_dir / f"speaker_{speaker_id:02d}"
        speaker_dir.mkdir(exist_ok=True)
        
        for i, text in enumerate(EVAL_SENTENCES):
            audio = synth.synthesize(
                text=text,
                speaker_id=speaker_id,
                cfg_scale=args.cfg_scale,
                num_steps=args.steps,
            )
            
            filename = f"{i:03d}.wav"
            synth.save_audio(audio, speaker_dir / filename)
            
            duration = audio.size(-1) / synth.sample_rate
            
            table.add_row(
                f"{i:03d}",
                str(speaker_id),
                text[:50] + "..." if len(text) > 50 else text,
                f"{duration:.2f}s",
            )
            
            manifest.append({
                "id": i,
                "speaker_id": speaker_id,
                "text": text,
                "file": str(speaker_dir / filename),
                "duration": duration,
            })
    
    console.print(table)
    
    # Save manifest
    manifest_path = sample_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    console.print(f"\n[green]Samples saved to {sample_dir}[/]")


if __name__ == "__main__":
    main()
