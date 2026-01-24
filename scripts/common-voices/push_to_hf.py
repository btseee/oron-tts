#!/usr/bin/env python
"""Push cleaned Common Voice Mongolian dataset to Hugging Face Hub."""

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import Audio, Dataset, DatasetDict, Features, Value
from huggingface_hub import HfApi, create_repo, login

# Configuration
REPO_ID = "btsee/common-voices-24-mn"
DATA_DIR = Path("/home/tsee/Personal/oron-tts/data/cleaned")
MANIFEST_PATH = DATA_DIR / "manifest.tsv"

# Dataset card content
DATASET_CARD = """---
license: cc0-1.0
language:
- mn
task_categories:
- automatic-speech-recognition
- text-to-speech
tags:
- mongolian
- khalkha
- speech
- audio
- tts
- asr
- common-voice
- cleaned
- denoised
pretty_name: Common Voice 24.0 Mongolian (Cleaned)
size_categories:
- 10K<n<100K
---

# Common Voice 24.0 Mongolian - Cleaned Dataset

A cleaned and preprocessed version of the Mozilla Common Voice 24.0 Mongolian (Khalkha) dataset, optimized for Text-to-Speech (TTS) training.

## Dataset Description

This dataset contains high-quality, cleaned audio recordings of Mongolian speech derived from [Mozilla Common Voice](https://commonvoice.mozilla.org/) version 24.0.

### Processing Pipeline

All audio samples have been processed through the following pipeline:

1. **Quality Filtering**: Only samples with >0 upvotes and 0 downvotes were selected
2. **Noise Suppression**: [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) applied for background noise removal
3. **Resampling**: Audio resampled to 22,050 Hz (optimal for TTS)
4. **Normalization**: Peak amplitude normalized to 0.95
5. **Silence Trimming**: Leading/trailing silence removed (20dB threshold)
6. **Duration Filtering**: Only clips between 0.5-15 seconds retained

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Clips | ~28,900 |
| Total Duration | ~40 hours |
| Sample Rate | 22,050 Hz |
| Audio Format | WAV (16-bit PCM) |
| Language | Mongolian (Khalkha Cyrillic) |
| Unique Speakers | ~1,500+ |

## Dataset Structure

```python
DatasetDict({
    train: Dataset({
        features: ['audio', 'text', 'speaker_id', 'duration'],
        num_rows: ...
    })
})
```

### Features

- **audio**: Audio waveform (automatically decoded)
- **text**: Mongolian Cyrillic transcription
- **speaker_id**: Anonymized speaker identifier (first 8 chars of hash)
- **duration**: Audio duration in seconds

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("btsee/common-voices-24-mn")

# Access a sample
sample = dataset["train"][0]
print(sample["text"])  # Mongolian transcription
print(sample["audio"]["array"])  # Audio waveform
print(sample["audio"]["sampling_rate"])  # 22050
```

### For TTS Training

```python
from datasets import load_dataset

dataset = load_dataset("btsee/common-voices-24-mn", split="train")

# Filter by duration for VITS2 training
dataset = dataset.filter(lambda x: 1.0 <= x["duration"] <= 10.0)

# Group by speaker for multi-speaker TTS
speakers = dataset.unique("speaker_id")
```

## Intended Uses

- **Text-to-Speech (TTS)**: Training Mongolian speech synthesis models (VITS2, Tacotron2, etc.)
- **Automatic Speech Recognition (ASR)**: Fine-tuning Whisper or Wav2Vec2 for Mongolian
- **Voice Cloning**: Multi-speaker TTS development
- **Linguistic Research**: Mongolian phonetics and prosody studies

## Source

- **Original Dataset**: [Mozilla Common Voice 24.0](https://commonvoice.mozilla.org/mn/datasets)
- **Processing Code**: [OronTTS](https://github.com/orontts/oron-tts)

## License

This dataset inherits the [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) license from Mozilla Common Voice.

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{common_voices_24_mn_cleaned,
  title = {Common Voice 24.0 Mongolian - Cleaned Dataset},
  author = {OronTTS Contributors},
  year = {2026},
  url = {https://huggingface.co/datasets/btsee/common-voices-24-mn},
  note = {Cleaned and preprocessed for TTS training}
}

@inproceedings{commonvoice:2020,
  author = {Ardila, Rosana and others},
  title = {Common Voice: A Massively-Multilingual Speech Corpus},
  booktitle = {LREC},
  year = {2020}
}
```

## Acknowledgements

- Mozilla Foundation for the Common Voice project
- DeepFilterNet team for the noise suppression model
- All Mongolian speakers who contributed to Common Voice
"""


def load_manifest() -> list[dict]:
    """Load the manifest TSV file."""
    rows = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # Convert path to just filename for HF
            audio_path = Path(row["audio_path"])
            if audio_path.exists():
                rows.append({
                    "audio": str(audio_path),
                    "text": row["text"],
                    "speaker_id": row["speaker_id"],
                    "duration": float(row["duration"]),
                })
    return rows


def main():
    print("🚀 Pushing Common Voice 24.0 Mongolian (Cleaned) to Hugging Face")
    print(f"   Repository: {REPO_ID}")
    print(f"   Data directory: {DATA_DIR}")
    
    # Login with token from environment variable
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("\n🔑 Logging in with HF_TOKEN...")
        login(token=hf_token)
    else:
        print("\n⚠️  No HF_TOKEN found. Set it with: export HF_TOKEN='your_token'")
        print("   Get your token from: https://huggingface.co/settings/tokens")
        return
    
    # Load data
    print("\n📂 Loading manifest...")
    data = load_manifest()
    print(f"   Loaded {len(data)} samples")
    
    # Create dataset
    print("\n🔧 Creating dataset...")
    features = Features({
        "audio": Audio(sampling_rate=22050),
        "text": Value("string"),
        "speaker_id": Value("string"),
        "duration": Value("float32"),
    })
    
    dataset = Dataset.from_list(data, features=features)
    dataset_dict = DatasetDict({"train": dataset})
    
    print(f"   Dataset created: {dataset_dict}")
    
    # Create repository
    print("\n📤 Creating repository...")
    api = HfApi()
    
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="dataset",
            exist_ok=True,
        )
        print(f"   Repository created/verified: {REPO_ID}")
    except Exception as e:
        print(f"   Repository creation: {e}")
    
    # Upload README
    print("\n📝 Uploading dataset card...")
    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    
    # Push dataset
    print("\n⬆️  Pushing dataset to Hub (this may take a while)...")
    dataset_dict.push_to_hub(
        REPO_ID,
        private=False,
        max_shard_size="500MB",
    )
    
    print(f"\n✅ Dataset pushed successfully!")
    print(f"   View at: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
