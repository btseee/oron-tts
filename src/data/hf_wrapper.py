"""Hugging Face Dataset wrappers for Common Voice and MBSpeech."""

from pathlib import Path
from typing import Any, Iterator

from datasets import Audio, Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi


class HFDatasetWrapper:
    def __init__(
        self,
        dataset_name: str,
        cache_dir: str | Path | None = None,
        sample_rate: int = 22050,
    ) -> None:
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.sample_rate = sample_rate
        self._dataset: Dataset | DatasetDict | None = None

    def load(
        self,
        split: str | None = None,
        streaming: bool = False,
    ) -> Dataset | DatasetDict:
        kwargs: dict[str, Any] = {
            "path": self.dataset_name,
            "streaming": streaming,
        }
        if self.cache_dir:
            kwargs["cache_dir"] = str(self.cache_dir)
        if split:
            kwargs["split"] = split

        self._dataset = load_dataset(**kwargs)

        if not streaming and self._dataset is not None and hasattr(self._dataset, "cast_column"):
            self._dataset = self._dataset.cast_column(
                "audio", Audio(sampling_rate=self.sample_rate)
            )

        if self._dataset is None:
            raise ValueError(f"Failed to load dataset: {self.dataset_name}")

        return self._dataset

    def load_common_voice(
        self,
        language: str = "mn",
        split: str = "train",
    ) -> Dataset:
        dataset = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            language,
            split=split,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            trust_remote_code=True,
        )
        dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sample_rate))
        return dataset

    def load_mbspeech(self, split: str = "train") -> Dataset:
        dataset = load_dataset(
            self.dataset_name,
            split=split,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        )
        if "audio" in dataset.column_names:
            dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sample_rate))
        return dataset

    def iterate_samples(
        self,
        dataset: Dataset,
        batch_size: int = 1,
    ) -> Iterator[dict[str, Any]]:
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            if batch_size == 1:
                yield {k: v[0] if isinstance(v, list) else v for k, v in batch.items()}
            else:
                yield batch

    def upload_processed(
        self,
        dataset: Dataset,
        repo_id: str,
        private: bool = True,
        token: str | None = None,
    ) -> str:
        dataset.push_to_hub(repo_id, private=private, token=token)
        return f"https://huggingface.co/datasets/{repo_id}"

    @staticmethod
    def create_from_files(
        audio_paths: list[Path],
        texts: list[str],
        speaker_ids: list[int] | None = None,
        sample_rate: int = 22050,
    ) -> Dataset:
        data: dict[str, list[str] | list[int]] = {
            "audio": [str(p) for p in audio_paths],
            "text": texts,
        }
        if speaker_ids:
            data["speaker_id"] = speaker_ids

        dataset = Dataset.from_dict(data)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
        return dataset


class CommonVoiceWrapper(HFDatasetWrapper):
    def __init__(
        self,
        cache_dir: str | Path | None = None,
        sample_rate: int = 22050,
    ) -> None:
        super().__init__("btsee/common-voices-24-mn", cache_dir, sample_rate)

    def load(self, split: str = "train", streaming: bool = False) -> Dataset | DatasetDict:
        return super().load(split=split, streaming=streaming)

    def get_text_column(self) -> str:
        return "sentence"

    def get_audio_column(self) -> str:
        return "audio"


class MBSpeechWrapper(HFDatasetWrapper):
    def __init__(
        self,
        cache_dir: str | Path | None = None,
        sample_rate: int = 22050,
    ) -> None:
        super().__init__("btsee/mbspeech_mn", cache_dir, sample_rate)

    def load(self, split: str = "train", streaming: bool = False) -> Dataset | DatasetDict:
        return super().load(split=split, streaming=streaming)

    def get_text_column(self) -> str:
        return "text"

    def get_audio_column(self) -> str:
        return "audio"
