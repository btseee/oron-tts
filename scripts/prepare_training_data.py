#!/usr/bin/env python
"""Prepare cleaned dataset for VITS2 training.

Converts manifest.tsv to metadata.csv format expected by TTSDataset.
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to cleaned data directory (containing manifest.tsv and clips/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for training-ready dataset",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks instead of copying audio files",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    
    manifest_path = input_dir / "manifest.tsv"
    clips_dir = input_dir / "clips"
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not clips_dir.exists():
        raise FileNotFoundError(f"Clips directory not found: {clips_dir}")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(exist_ok=True)
    
    # Map speaker IDs to integers
    speaker_to_id: dict[str, int] = {}
    next_speaker_id = 0
    
    # Read manifest and convert
    metadata_lines: list[str] = []
    
    print(f"Reading manifest from {manifest_path}...")
    with open(manifest_path, encoding="utf-8") as f:
        header = f.readline()  # Skip header
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            
            audio_path, text, speaker_hash, duration = parts[:4]
            audio_path = Path(audio_path)
            
            # Map speaker hash to integer ID
            if speaker_hash not in speaker_to_id:
                speaker_to_id[speaker_hash] = next_speaker_id
                next_speaker_id += 1
            speaker_id = speaker_to_id[speaker_hash]
            
            # Get file ID from audio path
            file_id = audio_path.stem
            
            # Create link or copy
            dest_path = wavs_dir / f"{file_id}.wav"
            if not dest_path.exists():
                if args.symlink:
                    dest_path.symlink_to(audio_path.absolute())
                else:
                    shutil.copy2(audio_path, dest_path)
            
            # Add to metadata (format: file_id|speaker_id|text)
            # Escape any pipe characters in text
            clean_text = text.replace("|", " ")
            metadata_lines.append(f"{file_id}|{speaker_id}|{clean_text}")
    
    # Write metadata.csv
    metadata_path = output_dir / "metadata.csv"
    print(f"Writing {len(metadata_lines)} entries to {metadata_path}...")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_lines))
    
    # Write speaker mapping
    speaker_map_path = output_dir / "speakers.txt"
    with open(speaker_map_path, "w", encoding="utf-8") as f:
        for speaker_hash, speaker_id in sorted(speaker_to_id.items(), key=lambda x: x[1]):
            f.write(f"{speaker_id}|{speaker_hash}\n")
    
    print(f"Done! Dataset ready at {output_dir}")
    print(f"  - {len(metadata_lines)} audio files")
    print(f"  - {len(speaker_to_id)} speakers")


if __name__ == "__main__":
    main()
