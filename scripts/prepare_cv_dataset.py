#!/usr/bin/env python3
"""
Prepare Common Voice 24 Dataset for OronTTS.

This script:
1. Extracts the local tar.gz archive.
2. Filters high-quality data (up_votes > 0, down_votes == 0).
3. Deep cleans text using MongolianTextCleaner.
4. Validates audio files (duration, sample rate, corruption).
5. Uploads to HuggingFace Hub.
"""

import argparse
import csv
import tarfile
import shutil
import os
import re
from pathlib import Path
from typing import Dict, List, Optional
import sys

import librosa
import soundfile as sf
from datasets import Dataset, DatasetDict, Audio, Features, Value
from huggingface_hub import HfApi, login
from dotenv import load_dotenv
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.cleaner import MongolianTextCleaner

# Configuration
DATASET_NAME = "btsee/common-voices-24-mn"
DEFAULT_TAR_PATH = "/home/tsee/Downloads/cv-corpus-24.0-2025-12-05-mn.tar.gz"
SAMPLING_RATE = 24000

# Quality thresholds
MIN_AUDIO_DURATION = 1.0  # seconds
MAX_AUDIO_DURATION = 15.0  # seconds
MIN_TEXT_LENGTH = 3  # characters
MAX_TEXT_LENGTH = 500  # characters

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Common Voice Dataset")
    parser.add_argument("--tar-path", type=str, default=DEFAULT_TAR_PATH, help="Path to tar.gz")
    parser.add_argument("--extract-path", type=str, default="/tmp/cv_mn_extract", help="Extraction directory")
    parser.add_argument("--repo-id", type=str, default=DATASET_NAME, help="HF Repo ID")
    parser.add_argument("--keep-extracted", action="store_true", help="Don't delete extracted files")
    parser.add_argument("--min-upvotes", type=int, default=1, help="Minimum upvotes required")
    parser.add_argument("--max-downvotes", type=int, default=0, help="Maximum downvotes allowed")
    parser.add_argument("--validate-audio", action="store_true", default=True, help="Validate audio files")
    return parser.parse_args()


def validate_audio_file(audio_path: Path) -> Optional[Dict[str, float]]:
    """Validate audio file quality.
    
    Returns:
        Dict with duration and sample_rate if valid, None otherwise.
    """
    try:
        # Check file exists and is readable
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            return None
            
        # Load audio and check duration
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        duration = len(y) / sr
        
        # Validate duration
        if duration < MIN_AUDIO_DURATION or duration > MAX_AUDIO_DURATION:
            return None
            
        # Check for silent audio
        if librosa.feature.rms(y=y).mean() < 0.001:
            return None
            
        return {"duration": duration, "sample_rate": sr}
        
    except Exception:
        return None


def deep_clean_text(text: str, cleaner: MongolianTextCleaner) -> Optional[str]:
    """Deep clean and validate text.
    
    Returns:
        Cleaned text if valid, None otherwise.
    """
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Check length
    if len(text) < MIN_TEXT_LENGTH or len(text) > MAX_TEXT_LENGTH:
        return None
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove excessive punctuation
    text = re.sub(r'([.!?,])\1{2,}', r'\1', text)
    
    # Check for invalid characters (non-Cyrillic, non-punctuation)
    cyrillic_pattern = r'[А-Яа-яӨөҮү\s\d.,!?;:\-\'"()«»]'
    if not all(re.match(cyrillic_pattern, char) or char in ' \n\t' for char in text):
        # Contains invalid characters
        valid_text = ''.join(char for char in text if re.match(cyrillic_pattern, char) or char in ' \n\t')
        if len(valid_text) < MIN_TEXT_LENGTH:
            return None
        text = valid_text
    
    # Apply Mongolian cleaner
    try:
        cleaned = cleaner(text)
        
        # Final validation
        if len(cleaned.strip()) < MIN_TEXT_LENGTH:
            return None
            
        return cleaned.strip()
    except Exception:
        return None

def process_split(
    split_name: str,
    tsv_path: Path,
    clips_dir: Path,
    cleaner: MongolianTextCleaner,
    min_upvotes: int = 1,
    max_downvotes: int = 0,
    validate_audio: bool = True,
) -> List[Dict]:
    """Process a single split with quality filtering.
    
    Args:
        split_name: Name of the split (train/dev/test).
        tsv_path: Path to the TSV file.
        clips_dir: Directory containing audio clips.
        cleaner: Text cleaner instance.
        min_upvotes: Minimum required upvotes.
        max_downvotes: Maximum allowed downvotes.
        validate_audio: Whether to validate audio files.
    
    Returns:
        List of valid data samples.
    """
    data = []
    stats = {
        "total": 0,
        "filtered_votes": 0,
        "filtered_text": 0,
        "filtered_audio": 0,
        "valid": 0,
    }
    
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        
        for row in tqdm(reader, desc=f"Processing {split_name}"):
            stats["total"] += 1
            
            # Filter by votes
            up_votes = int(row.get("up_votes", 0))
            down_votes = int(row.get("down_votes", 0))
            
            if up_votes < min_upvotes or down_votes > max_downvotes:
                stats["filtered_votes"] += 1
                continue
            
            # Deep clean text
            sentence = row.get("sentence", "")
            cleaned_text = deep_clean_text(sentence, cleaner)
            
            if cleaned_text is None:
                stats["filtered_text"] += 1
                continue
            
            # Validate audio
            audio_path = clips_dir / row["path"]
            
            if validate_audio:
                audio_info = validate_audio_file(audio_path)
                if audio_info is None:
                    stats["filtered_audio"] += 1
                    continue
                duration = audio_info["duration"]
            else:
                if not audio_path.exists():
                    stats["filtered_audio"] += 1
                    continue
                duration = 0.0
            
            # Add valid sample
            data.append({
                "audio": str(audio_path),
                "text": cleaned_text,
                "original_text": sentence,
                "client_id": row.get("client_id", ""),
                "gender": row.get("gender", ""),
                "age": row.get("age", ""),
                "accent": row.get("accents", ""),
                "up_votes": up_votes,
                "down_votes": down_votes,
                "duration": duration,
                "split": split_name,
            })
            stats["valid"] += 1
    
    # Print statistics
    print(f"\n{split_name} Statistics:")
    print(f"  Total samples: {stats['total']}")
    print(f"  Filtered by votes: {stats['filtered_votes']}")
    print(f"  Filtered by text: {stats['filtered_text']}")
    print(f"  Filtered by audio: {stats['filtered_audio']}")
    print(f"  Valid samples: {stats['valid']}")
    print(f"  Retention rate: {stats['valid']/stats['total']*100:.2f}%\n")
    
    return data

def main():
    load_dotenv()
    args = parse_args()
    
    # Check HF Token
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN not found in .env")
        return
    login(token=token)
    
    print("=" * 60)
    print("OronTTS Common Voice Dataset Preparation")
    print("=" * 60)
    print(f"Quality Filters:")
    print(f"  Min upvotes: {args.min_upvotes}")
    print(f"  Max downvotes: {args.max_downvotes}")
    print(f"  Audio duration: {MIN_AUDIO_DURATION}s - {MAX_AUDIO_DURATION}s")
    print(f"  Text length: {MIN_TEXT_LENGTH} - {MAX_TEXT_LENGTH} chars")
    print(f"  Audio validation: {args.validate_audio}")
    print("=" * 60)
    
    # 1. Extract
    extract_path = Path(args.extract_path)
    tar_path = Path(args.tar_path)
    
    if not tar_path.exists():
        print(f"Error: Tar file not found at {tar_path}")
        return
    
    if not extract_path.exists():
        print(f"\nExtracting {tar_path.name}...")
        extract_path.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()
            for member in tqdm(members, desc="Extracting"):
                tar.extract(member, path=extract_path)
    else:
        print(f"\nUsing existing extracted files at {extract_path}")
    
    # Locate the 'mn' folder
    mn_dirs = list(extract_path.glob("**/mn"))
    if not mn_dirs:
        if (extract_path / "train.tsv").exists():
            mn_dir = extract_path
        else:
            print("Error: Could not find 'mn' directory or tsv files inside extracted archive.")
            return
    else:
        mn_dir = mn_dirs[0]
    
    print(f"Dataset root: {mn_dir}")
    clips_dir = mn_dir / "clips"
    
    if not clips_dir.exists():
        print(f"Error: Clips directory not found at {clips_dir}")
        return
    
    # 2. Initialize cleaner
    print("\nInitializing Mongolian text cleaner...")
    cleaner = MongolianTextCleaner()
    
    # 3. Process splits
    splits = ["train", "dev", "test"]
    datasets = {}
    
    features = Features({
        "audio": Audio(sampling_rate=SAMPLING_RATE),
        "text": Value("string"),
        "original_text": Value("string"),
        "client_id": Value("string"),
        "gender": Value("string"),
        "age": Value("string"),
        "accent": Value("string"),
        "up_votes": Value("int32"),
        "down_votes": Value("int32"),
        "duration": Value("float32"),
        "split": Value("string"),
    })
    
    for split in splits:
        tsv_path = mn_dir / f"{split}.tsv"
        if not tsv_path.exists():
            print(f"Warning: {split}.tsv not found, skipping.")
            continue
        
        print(f"\n{'='*60}")
        data = process_split(
            split,
            tsv_path,
            clips_dir,
            cleaner,
            args.min_upvotes,
            args.max_downvotes,
            args.validate_audio,
        )
        
        if not data:
            print(f"Warning: No valid data for {split}, skipping.")
            continue
        
        print(f"Creating dataset for {split}...")
        ds = Dataset.from_list(data, features=features)
        datasets[split] = ds
    
    if not datasets:
        print("\nError: No valid data found in any split.")
        return
    
    final_dataset = DatasetDict(datasets)
    
    # 4. Display final statistics
    print("\n" + "=" * 60)
    print("Final Dataset Summary:")
    print("=" * 60)
    total_samples = 0
    for split, ds in datasets.items():
        print(f"{split.capitalize()}: {len(ds):,} samples")
        total_samples += len(ds)
    print(f"Total: {total_samples:,} samples")
    print("=" * 60)
    
    # 5. Push to Hub
    print(f"\nPushing to {args.repo_id}...")
    final_dataset.push_to_hub(
        args.repo_id,
        private=False,
        commit_message=f"Update dataset with quality filters (upvotes>{args.min_upvotes}, downvotes<={args.max_downvotes})"
    )
    print("✓ Successfully uploaded to HuggingFace Hub!")
    
    # 6. Cleanup
    if not args.keep_extracted:
        print("\nCleaning up extracted files...")
        shutil.rmtree(extract_path)
        print("✓ Cleanup complete")
    
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
