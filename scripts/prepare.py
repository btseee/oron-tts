"""Dataset preparation script: Clean, denoise, and upload to Hugging Face."""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from tqdm import tqdm

from src.data.denoiser import AudioDenoiser
from src.data.hf_wrapper import CommonVoiceWrapper, HFDatasetWrapper, MBSpeechWrapper
from src.utils.audio import AudioProcessor
from src.utils.text_cleaner import TextCleaner


def process_dataset(
    hf_dataset: Dataset,
    audio_column: str,
    text_column: str,
    output_dir: Path,
    denoiser: AudioDenoiser,
    audio_processor: AudioProcessor,
    text_cleaner: TextCleaner,
    speaker_id: int = 0,
    max_samples: int | None = None,
) -> tuple[list[Path], list[str], list[int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_paths = []
    texts = []
    speaker_ids = []

    dataset_iter: Dataset = hf_dataset
    if max_samples:
        dataset_iter = hf_dataset.select(range(min(max_samples, len(hf_dataset))))

    for idx in tqdm(range(len(dataset_iter)), desc="Processing samples"):
        try:
            item = dataset_iter[idx]
            audio_info = item[audio_column]
            if isinstance(audio_info, dict):
                audio_array = torch.tensor(audio_info["array"]).float()
                sr = audio_info["sampling_rate"]
            else:
                continue

            text = item[text_column]
            if not text or len(text.strip()) < 2:
                continue

            cleaned_text = text_cleaner.clean(text)
            if len(cleaned_text) < 2:
                continue

            denoised = denoiser.denoise(audio_array, sr)
            denoised = audio_processor.normalize_audio(denoised)
            denoised = audio_processor.trim_silence(denoised)

            if len(denoised) < 1024:
                continue

            output_path = output_dir / f"sample_{idx:06d}.wav"
            audio_processor.save_audio(output_path, denoised)

            audio_paths.append(output_path)
            texts.append(cleaned_text)
            speaker_ids.append(speaker_id)

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

    return audio_paths, texts, speaker_ids


def create_metadata(
    audio_paths: list[Path],
    texts: list[str],
    speaker_ids: list[int],
    output_path: Path,
) -> None:
    metadata = []
    for audio_path, text, sid in zip(audio_paths, texts, speaker_ids):
        metadata.append({
            "audio_path": str(audio_path),
            "text": text,
            "speaker_id": sid,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TTS dataset")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--cache-dir", type=str, default="data/cache")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--hf-repo", type=str, default="btsee/oron-tts")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["common-voice", "mbspeech", "all"],
        default="all",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)

    denoiser = AudioDenoiser(target_sr=args.sample_rate)
    audio_processor = AudioProcessor(sample_rate=args.sample_rate)
    text_cleaner = TextCleaner()

    all_audio_paths: list[Path] = []
    all_texts: list[str] = []
    all_speaker_ids: list[int] = []

    if args.dataset in ["common-voice", "all"]:
        print("Loading Common Voice dataset...")
        cv_wrapper = CommonVoiceWrapper(cache_dir=cache_dir, sample_rate=args.sample_rate)
        try:
            cv_dataset_raw = cv_wrapper.load(split="train")
            if not isinstance(cv_dataset_raw, Dataset):
                raise TypeError("Expected Dataset, got DatasetDict")
            cv_output = output_dir / "common_voice"
            audio_paths, texts, speaker_ids = process_dataset(
                cv_dataset_raw,
                audio_column=cv_wrapper.get_audio_column(),
                text_column=cv_wrapper.get_text_column(),
                output_dir=cv_output,
                denoiser=denoiser,
                audio_processor=audio_processor,
                text_cleaner=text_cleaner,
                speaker_id=0,
                max_samples=args.max_samples,
            )
            all_audio_paths.extend(audio_paths)
            all_texts.extend(texts)
            all_speaker_ids.extend(speaker_ids)
            print(f"Processed {len(audio_paths)} samples from Common Voice")
        except Exception as e:
            print(f"Error loading Common Voice: {e}")

    if args.dataset in ["mbspeech", "all"]:
        print("Loading MBSpeech dataset...")
        mb_wrapper = MBSpeechWrapper(cache_dir=cache_dir, sample_rate=args.sample_rate)
        try:
            mb_dataset_raw = mb_wrapper.load(split="train")
            if not isinstance(mb_dataset_raw, Dataset):
                raise TypeError("Expected Dataset, got DatasetDict")
            mb_output = output_dir / "mbspeech"
            audio_paths, texts, speaker_ids = process_dataset(
                mb_dataset_raw,
                audio_column=mb_wrapper.get_audio_column(),
                text_column=mb_wrapper.get_text_column(),
                output_dir=mb_output,
                denoiser=denoiser,
                audio_processor=audio_processor,
                text_cleaner=text_cleaner,
                speaker_id=1,
                max_samples=args.max_samples,
            )
            all_audio_paths.extend(audio_paths)
            all_texts.extend(texts)
            all_speaker_ids.extend(speaker_ids)
            print(f"Processed {len(audio_paths)} samples from MBSpeech")
        except Exception as e:
            print(f"Error loading MBSpeech: {e}")

    metadata_path = output_dir / "metadata.json"
    create_metadata(all_audio_paths, all_texts, all_speaker_ids, metadata_path)
    print(f"Created metadata at {metadata_path}")
    print(f"Total samples: {len(all_audio_paths)}")

    if args.upload:
        print("Uploading to Hugging Face...")
        dataset = HFDatasetWrapper.create_from_files(
            audio_paths=all_audio_paths,
            texts=all_texts,
            speaker_ids=all_speaker_ids,
            sample_rate=args.sample_rate,
        )
        wrapper = HFDatasetWrapper("", cache_dir=cache_dir)
        url = wrapper.upload_processed(
            dataset=dataset,
            repo_id=args.hf_repo,
            private=True,
            token=args.hf_token,
        )
        print(f"Uploaded to: {url}")


if __name__ == "__main__":
    main()
