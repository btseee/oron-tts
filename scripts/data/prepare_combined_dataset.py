#!/usr/bin/env python3
"""Prepare combined Mongolian dataset with best male/female voices.

Combines:
- btsee/mbspeech_mn (high quality, single speaker, 3846 samples)
- btsee/common-voices-24-mn (multi-speaker, 555 male + 623 female)

Strategy:
- Use ALL mbspeech for quality baseline
- Add best male and female voices from common-voices
- Filter by duration (1-15 seconds for optimal training)
- Balance male/female representation
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ["HF_AUDIO_DECODER"] = "soundfile"

from collections import defaultdict
from typing import Any

from datasets import (
    Audio,
    Dataset,  # type: ignore[import-untyped]
    concatenate_datasets,
    load_dataset,
)


def prepare_mbspeech(dataset: Any) -> Dataset:
    """Prepare mbspeech dataset."""
    print(f"Processing mbspeech: {len(dataset)} samples")

    # Rename columns to match common format
    processed = []
    for item in dataset:
        # Filter by duration (1-15 seconds)
        duration = len(item["audio"]["array"]) / item["audio"]["sampling_rate"]
        if 1.0 <= duration <= 15.0:
            processed.append(
                {
                    "audio": item["audio"],
                    "text": item["sentence_norm"],
                    "speaker_id": "mbspeech_speaker",
                    "gender": "unknown",  # mbspeech doesn't specify
                    "duration": duration,
                }
            )

    print(f"Filtered mbspeech: {len(processed)} samples (1-15 seconds)")
    return Dataset.from_list(processed).cast_column("audio", Audio(sampling_rate=24000))


def prepare_common_voices(
    dataset: Any, target_male: int = 400, target_female: int = 400
) -> Dataset:
    """Prepare common voices dataset with best male/female samples."""
    print(f"Processing common-voices: {len(dataset)} samples")

    # Group by speaker and gender
    speakers = defaultdict(list)
    for idx, item in enumerate(dataset):
        gender = item["gender"]
        if gender in ["male_masculine", "female_feminine"]:
            speaker_id = item["client_id"]
            duration = item["duration"]

            # Filter by duration and quality
            if 1.0 <= duration <= 15.0 and item["up_votes"] >= 2:
                speakers[(speaker_id, gender)].append((idx, item))

    # Select best speakers (most samples)
    male_speakers = [(k, v) for k, v in speakers.items() if k[1] == "male_masculine"]
    female_speakers = [(k, v) for k, v in speakers.items() if k[1] == "female_feminine"]

    male_speakers.sort(key=lambda x: len(x[1]), reverse=True)
    female_speakers.sort(key=lambda x: len(x[1]), reverse=True)

    print(f"Found {len(male_speakers)} male speakers, {len(female_speakers)} female speakers")

    # Collect samples
    processed = []

    # Add male samples
    male_count = 0
    for (speaker_id, _gender), samples in male_speakers:
        if male_count >= target_male:
            break
        for _idx, item in samples[: min(len(samples), target_male - male_count)]:
            processed.append(
                {
                    "audio": item["audio"],
                    "text": item["text"],
                    "speaker_id": speaker_id[:16],  # Shorten ID
                    "gender": "male",
                    "duration": item["duration"],
                }
            )
            male_count += 1

    # Add female samples
    female_count = 0
    for (speaker_id, _gender), samples in female_speakers:
        if female_count >= target_female:
            break
        for _idx, item in samples[: min(len(samples), target_female - female_count)]:
            processed.append(
                {
                    "audio": item["audio"],
                    "text": item["text"],
                    "speaker_id": speaker_id[:16],  # Shorten ID
                    "gender": "female",
                    "duration": item["duration"],
                }
            )
            female_count += 1

    print(f"Selected {male_count} male + {female_count} female samples from common-voices")
    return Dataset.from_list(processed).cast_column("audio", Audio(sampling_rate=24000))


def main() -> None:
    print("=" * 70)
    print("Preparing Combined Mongolian Dataset")
    print("=" * 70)

    # Load datasets
    print("\n📥 Loading datasets...")
    mbspeech = load_dataset("btsee/mbspeech_mn", split="train")
    common_voices = load_dataset("btsee/common-voices-24-mn", split="train")

    # Process datasets
    print("\n🔧 Processing datasets...")
    mbspeech_processed = prepare_mbspeech(mbspeech)
    cv_processed = prepare_common_voices(common_voices, target_male=600, target_female=600)

    # Combine
    print("\n🔗 Combining datasets...")
    combined = concatenate_datasets([mbspeech_processed, cv_processed])

    # Shuffle
    combined = combined.shuffle(seed=42)

    print(f"\n✅ Final dataset: {len(combined)} samples")
    print(f"   - MBSpeech: ~{len(mbspeech_processed)} samples")
    print(f"   - Common Voices: ~{len(cv_processed)} samples")

    # Save locally first
    output_dir = Path("/workspace/output/data/mongolian-tts-combined")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving to local directory: {output_dir}")
    combined.save_to_disk(str(output_dir))
    print("✅ Saved locally!")

    # Try to push to HuggingFace (optional)
    print("\n📤 Attempting to push to HuggingFace Hub...")
    try:
        combined.push_to_hub("btsee/mongolian-tts-combined", private=True)
        print("✅ Pushed to HuggingFace Hub!")
    except Exception as e:
        print(f"⚠️  Could not push to HuggingFace (skipping): {e}")
        print(f"   Dataset saved locally at: {output_dir}")

    print("\n" + "=" * 70)
    print(f"✅ Dataset prepared: {output_dir}")
    print("   Use with: --dataset /workspace/output/data/mongolian-tts-combined")
    print("=" * 70)


if __name__ == "__main__":
    main()
