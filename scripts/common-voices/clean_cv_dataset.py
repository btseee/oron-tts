#!/usr/bin/env python
"""Clean Mozilla Common Voice Mongolian dataset.

Filters clips by vote counts and applies DeepFilterNet audio cleaning.
"""

import csv
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import soundfile as sf
from tqdm import tqdm

from orontts.preprocessing.audio import AudioCleaner, AudioCleanerConfig


# Global cleaner for reuse (initialized once per process)
_CLEANER: AudioCleaner | None = None


def get_cleaner(config_dict: dict, device: str = "cuda") -> AudioCleaner:
    """Get or create a global cleaner instance."""
    global _CLEANER
    if _CLEANER is None:
        config = AudioCleanerConfig(**config_dict)
        _CLEANER = AudioCleaner(config=config, device=device)
        # Force initialization of DeepFilterNet
        if _CLEANER.has_deepfilter:
            _CLEANER._load_deepfilter()
    return _CLEANER


def filter_validated_tsv(
    input_tsv: Path,
    min_up_votes: int = 1,
    max_down_votes: int = 0,
) -> list[dict]:
    """Filter validated.tsv based on vote criteria.
    
    Args:
        input_tsv: Path to validated.tsv file.
        min_up_votes: Minimum up_votes required (exclusive: > min_up_votes).
        max_down_votes: Maximum down_votes allowed (inclusive: <= max_down_votes).
    
    Returns:
        List of filtered row dictionaries.
    """
    filtered = []
    
    with open(input_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            up_votes = int(row.get("up_votes", 0))
            down_votes = int(row.get("down_votes", 0))
            
            # Filter: more than 0 up votes AND exactly 0 down votes
            if up_votes > min_up_votes and down_votes <= max_down_votes:
                filtered.append(row)
    
    return filtered


def clean_single_clip(
    input_path: Path,
    output_path: Path,
    cleaner: AudioCleaner,
) -> tuple[str, bool, str]:
    """Clean a single audio clip.
    
    Args:
        input_path: Input audio file path.
        output_path: Output audio file path.
        cleaner: AudioCleaner instance.
    
    Returns:
        Tuple of (filename, success, error_message)
    """
    try:
        # Load audio
        audio, sr = sf.read(input_path)
        
        # Clean
        cleaned = cleaner.clean(audio, sr)
        
        # Check duration
        duration = len(cleaned) / cleaner.config.target_sample_rate
        if duration < cleaner.config.min_duration or duration > cleaner.config.max_duration:
            return (input_path.name, False, f"Duration {duration:.2f}s out of range")
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, cleaned, cleaner.config.target_sample_rate)
        
        return (input_path.name, True, "")
    except Exception as e:
        return (input_path.name, False, str(e))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean Common Voice Mongolian dataset")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/tsee/Personal/oron-tts/data/raw/mn"),
        help="Path to extracted CV dataset (mn folder)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/tsee/Personal/oron-tts/data/cleaned"),
        help="Output directory for cleaned audio",
    )
    parser.add_argument(
        "--min-up-votes",
        type=int,
        default=0,
        help="Minimum up_votes (exclusive, default: >0)",
    )
    parser.add_argument(
        "--max-down-votes",
        type=int,
        default=0,
        help="Maximum down_votes (inclusive, default: 0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of clips to process (for testing)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for DeepFilterNet (cuda or cpu)",
    )
    
    args = parser.parse_args()
    
    # Setup paths
    validated_tsv = args.input_dir / "validated.tsv"
    clips_dir = args.input_dir / "clips"
    output_clips_dir = args.output_dir / "clips"
    output_manifest = args.output_dir / "manifest.tsv"
    
    print(f"📂 Input: {args.input_dir}")
    print(f"📂 Output: {args.output_dir}")
    
    # Filter dataset
    print("\n🔍 Filtering validated.tsv...")
    filtered_rows = filter_validated_tsv(
        validated_tsv,
        min_up_votes=args.min_up_votes,
        max_down_votes=args.max_down_votes,
    )
    print(f"   Found {len(filtered_rows)} clips with >0 up_votes and 0 down_votes")
    
    if args.limit:
        filtered_rows = filtered_rows[:args.limit]
        print(f"   Limited to {len(filtered_rows)} clips for processing")
    
    # Prepare cleaning config
    config_dict = {
        "target_sample_rate": 22050,
        "normalize": True,
        "trim_silence": True,
        "trim_db": 20.0,
        "min_duration": 0.5,
        "max_duration": 15.0,
    }
    
    # Prepare task list
    tasks = []
    for row in filtered_rows:
        clip_name = row["path"]
        input_path = clips_dir / clip_name
        # Change extension from .mp3 to .wav
        output_name = Path(clip_name).stem + ".wav"
        output_path = output_clips_dir / output_name
        
        if input_path.exists():
            tasks.append((input_path, output_path, row))
    
    print(f"   {len(tasks)} audio files found")
    
    # Initialize cleaner once (avoid multiprocessing race condition)
    print(f"\n🔧 Initializing DeepFilterNet on {args.device}...")
    cleaner = get_cleaner(config_dict, device=args.device)
    print(f"   DeepFilterNet available: {cleaner.has_deepfilter}")
    
    # Process clips sequentially (DeepFilterNet is CPU-bound anyway)
    print("\n🎵 Cleaning audio clips...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_clips_dir.mkdir(parents=True, exist_ok=True)
    
    successful = []
    failed = []
    results = []
    
    for input_path, output_path, row in tqdm(tasks, desc="Cleaning"):
        filename, success, error = clean_single_clip(input_path, output_path, cleaner)
        if success:
            successful.append(filename)
            results.append((output_path, row))
        else:
            failed.append((filename, error))
    
    print(f"\n✅ Successfully cleaned: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    if failed[:10]:
        print("\nFirst 10 failures:")
        for name, err in failed[:10]:
            print(f"   {name}: {err}")
    
    # Write output manifest
    print(f"\n📝 Writing manifest to {output_manifest}")
    with open(output_manifest, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["audio_path", "text", "speaker_id", "duration"],
            delimiter="\t",
        )
        writer.writeheader()
        
        for wav_path, row in results:
            # Get duration
            info = sf.info(wav_path)
            
            writer.writerow({
                "audio_path": str(wav_path),
                "text": row["sentence"],
                "speaker_id": row.get("client_id", "unknown")[:8],  # Truncate hash
                "duration": f"{info.duration:.3f}",
            })
    
    print("\n🎉 Done!")
    print(f"   Cleaned audio: {output_clips_dir}")
    print(f"   Manifest: {output_manifest}")


if __name__ == "__main__":
    main()
