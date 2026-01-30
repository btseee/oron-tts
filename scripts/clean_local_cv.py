"""Clean local Common Voice tar.gz dataset (no HF upload)."""

import argparse
import csv
import io
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from src.data.denoiser import AudioDenoiser
from src.utils.audio import AudioProcessor
from src.utils.text_cleaner import TextCleaner


def load_mp3_bytes(mp3_bytes: bytes, target_sr: int = 22050) -> torch.Tensor | None:
    """Load MP3 from bytes using ffmpeg."""
    try:
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-i", "pipe:0",
                "-f", "wav",
                "-acodec", "pcm_s16le",
                "-ar", str(target_sr),
                "-ac", "1",
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        wav_bytes, _ = process.communicate(input=mp3_bytes)
        if len(wav_bytes) < 100:
            return None
        audio, sr = sf.read(io.BytesIO(wav_bytes))
        return torch.from_numpy(audio.astype(np.float32))
    except Exception:
        return None


def extract_and_process_cv(
    tar_path: Path,
    output_dir: Path,
    sample_rate: int = 22050,
    max_samples: int | None = None,
    skip_denoise: bool = False,
) -> tuple[list[Path], list[str], list[int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    denoiser = AudioDenoiser(target_sr=sample_rate) if not skip_denoise else None
    audio_processor = AudioProcessor(sample_rate=sample_rate)
    text_cleaner = TextCleaner()

    audio_paths: list[Path] = []
    texts: list[str] = []
    speaker_ids: list[int] = []
    speaker_map: dict[str, int] = {}

    print(f"Opening {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        tsv_member = None
        clips_prefix = None

        for m in members:
            if m.name.endswith("validated.tsv") or m.name.endswith("train.tsv"):
                tsv_member = m
            if "/clips/" in m.name and m.name.endswith(".mp3"):
                clips_prefix = "/".join(m.name.split("/")[:-1]) + "/"
                break

        if tsv_member is None:
            for m in members:
                if m.name.endswith(".tsv") and "validated" not in m.name.lower():
                    tsv_member = m
                    break

        if tsv_member is None:
            raise ValueError("No TSV metadata file found in archive")

        print(f"Found metadata: {tsv_member.name}")

        tsv_file = tar.extractfile(tsv_member)
        if tsv_file is None:
            raise ValueError("Could not extract TSV file")

        reader = csv.DictReader(
            tsv_file.read().decode("utf-8").splitlines(),
            delimiter="\t",
        )
        rows = list(reader)

        if max_samples:
            rows = rows[:max_samples]

        print(f"Processing {len(rows)} samples...")
        processed = 0
        failed = 0

        for idx, row in enumerate(tqdm(rows, desc="Processing")):
            try:
                audio_filename = row.get("path", row.get("filename", ""))
                if not audio_filename:
                    continue

                text = row.get("sentence", row.get("text", ""))
                if not text or len(text.strip()) < 2:
                    continue

                client_id = row.get("client_id", "unknown")
                if client_id not in speaker_map:
                    speaker_map[client_id] = len(speaker_map)
                speaker_id = speaker_map[client_id]

                cleaned_text = text_cleaner.clean(text)
                if len(cleaned_text) < 2:
                    continue

                audio_member = None
                for prefix in [clips_prefix, "clips/", ""]:
                    if prefix is None:
                        continue
                    try:
                        test_path = prefix + audio_filename
                        audio_member = tar.getmember(test_path)
                        break
                    except KeyError:
                        for m in members:
                            if m.name.endswith(audio_filename):
                                audio_member = m
                                break
                        if audio_member:
                            break

                if audio_member is None:
                    failed += 1
                    continue

                audio_file = tar.extractfile(audio_member)
                if audio_file is None:
                    failed += 1
                    continue

                mp3_bytes = audio_file.read()
                audio = load_mp3_bytes(mp3_bytes, sample_rate)
                if audio is None:
                    failed += 1
                    continue

                if denoiser:
                    audio = denoiser.denoise(audio, sample_rate)

                audio = audio_processor.normalize_audio(audio)
                audio = audio_processor.trim_silence(audio)

                if len(audio) < 2048:
                    continue

                duration = len(audio) / sample_rate
                if duration < 0.5 or duration > 15.0:
                    continue

                output_path = audio_dir / f"cv_{idx:06d}.wav"
                sf.write(str(output_path), audio.numpy(), sample_rate)

                audio_paths.append(output_path)
                texts.append(cleaned_text)
                speaker_ids.append(speaker_id)
                processed += 1

            except Exception as e:
                failed += 1
                if failed < 10:
                    print(f"Error on sample {idx}: {e}")
                continue

    print(f"\nProcessed: {processed}, Failed: {failed}, Speakers: {len(speaker_map)}")
    return audio_paths, texts, speaker_ids


def save_metadata(
    audio_paths: list[Path],
    texts: list[str],
    speaker_ids: list[int],
    output_path: Path,
) -> None:
    import json

    metadata = []
    for audio_path, text, sid in zip(audio_paths, texts, speaker_ids, strict=False):
        metadata.append({
            "audio_path": str(audio_path),
            "text": text,
            "speaker_id": sid,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean local Common Voice dataset")
    parser.add_argument("--input", type=str, required=True, help="Path to tar.gz file")
    parser.add_argument("--output-dir", type=str, default="data/processed/common_voice")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-denoise", action="store_true", help="Skip DeepFilterNet denoising")
    args = parser.parse_args()

    tar_path = Path(args.input)
    if not tar_path.exists():
        raise FileNotFoundError(f"Input file not found: {tar_path}")

    output_dir = Path(args.output_dir)

    audio_paths, texts, speaker_ids = extract_and_process_cv(
        tar_path=tar_path,
        output_dir=output_dir,
        sample_rate=args.sample_rate,
        max_samples=args.max_samples,
        skip_denoise=args.skip_denoise,
    )

    metadata_path = output_dir / "metadata.json"
    save_metadata(audio_paths, texts, speaker_ids, metadata_path)
    print(f"\nMetadata saved to: {metadata_path}")
    print(f"Total clean samples: {len(audio_paths)}")


if __name__ == "__main__":
    main()
