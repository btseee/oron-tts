"""HuggingFace Hub integration for model and dataset management."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from huggingface_hub import (
    HfApi,
    Repository,
    create_repo,
    hf_hub_download,
    snapshot_download,
    upload_file,
    upload_folder,
)


@dataclass
class HubConfig:
    """Configuration for HuggingFace Hub operations."""

    repo_id: str
    repo_type: str = "model"  # model, dataset, space
    private: bool = True
    token: str | None = None

    def __post_init__(self) -> None:
        if self.token is None:
            self.token = os.environ.get("HF_TOKEN")


class HubManager:
    """Manager for HuggingFace Hub operations.

    Handles model uploads, checkpoint syncing, and dataset management.
    """

    def __init__(self, config: HubConfig) -> None:
        """Initialize Hub manager.

        Args:
            config: Hub configuration.
        """
        self.config = config
        self.api = HfApi(token=config.token)

        # Ensure repo exists
        self._ensure_repo()

    def _ensure_repo(self) -> None:
        """Create repository if it doesn't exist."""
        try:
            create_repo(
                repo_id=self.config.repo_id,
                repo_type=self.config.repo_type,
                private=self.config.private,
                token=self.config.token,
                exist_ok=True,
            )
        except Exception as e:
            # Repo might already exist
            pass

    def upload_checkpoint(
        self,
        checkpoint_path: str | Path,
        commit_message: str = "Update checkpoint",
        subfolder: str = "checkpoints",
    ) -> str:
        """Upload a single checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file.
            commit_message: Git commit message.
            subfolder: Folder in repo for checkpoints.

        Returns:
            URL of uploaded file.
        """
        checkpoint_path = Path(checkpoint_path)

        return upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo=f"{subfolder}/{checkpoint_path.name}",
            repo_id=self.config.repo_id,
            repo_type=self.config.repo_type,
            token=self.config.token,
            commit_message=commit_message,
        )

    def upload_training_state(
        self,
        output_dir: str | Path,
        commit_message: str = "Sync training state",
    ) -> str:
        """Upload entire training directory.

        Args:
            output_dir: Training output directory.
            commit_message: Git commit message.

        Returns:
            Repository URL.
        """
        output_dir = Path(output_dir)

        return upload_folder(
            folder_path=str(output_dir),
            repo_id=self.config.repo_id,
            repo_type=self.config.repo_type,
            token=self.config.token,
            commit_message=commit_message,
            ignore_patterns=["*.pyc", "__pycache__", ".git"],
        )

    def download_checkpoint(
        self,
        filename: str,
        local_dir: str | Path,
        subfolder: str = "checkpoints",
    ) -> Path:
        """Download a checkpoint from Hub.

        Args:
            filename: Checkpoint filename.
            local_dir: Local directory to save to.
            subfolder: Folder in repo containing checkpoint.

        Returns:
            Path to downloaded file.
        """
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        return Path(
            hf_hub_download(
                repo_id=self.config.repo_id,
                repo_type=self.config.repo_type,
                filename=f"{subfolder}/{filename}",
                local_dir=str(local_dir),
                token=self.config.token,
            )
        )

    def download_model(self, local_dir: str | Path) -> Path:
        """Download entire model repository.

        Args:
            local_dir: Local directory to save to.

        Returns:
            Path to downloaded model.
        """
        local_dir = Path(local_dir)

        return Path(
            snapshot_download(
                repo_id=self.config.repo_id,
                repo_type=self.config.repo_type,
                local_dir=str(local_dir),
                token=self.config.token,
            )
        )

    def upload_model_card(
        self,
        model_name: str,
        description: str,
        language: str = "mn",
        license: str = "apache-2.0",
        **metadata: Any,
    ) -> None:
        """Create and upload model card.

        Args:
            model_name: Model name for the card.
            description: Model description.
            language: Model language.
            license: Model license.
            **metadata: Additional metadata.
        """
        card_content = f"""---
language: {language}
license: {license}
library_name: oron-tts
pipeline_tag: text-to-speech
tags:
  - mongolian
  - tts
  - flow-matching
  - diffusion-transformer
{self._format_metadata(metadata)}
---

# {model_name}

{description}

## Model Details

- **Architecture**: F5-TTS (Diffusion Transformer + Flow Matching)
- **Language**: Mongolian (Khalkha dialect)
- **Framework**: PyTorch

## Usage

```python
from src.core import F5TTS

model = F5TTS.from_pretrained("{self.config.repo_id}")
mel = model.synthesize(phonemes, speaker_id)
```

## Training

Trained using OronTTS framework with conditional flow matching.

## Citation

```bibtex
@misc{{oron-tts,
  title={{OronTTS: Mongolian Text-to-Speech}},
  year={{2025}},
}}
```
"""
        upload_file(
            path_or_fileobj=card_content.encode(),
            path_in_repo="README.md",
            repo_id=self.config.repo_id,
            repo_type=self.config.repo_type,
            token=self.config.token,
            commit_message="Update model card",
        )

    def _format_metadata(self, metadata: dict) -> str:
        """Format additional metadata for YAML."""
        if not metadata:
            return ""
        lines = []
        for key, value in metadata.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def list_checkpoints(self, subfolder: str = "checkpoints") -> list[str]:
        """List available checkpoints in repository.

        Args:
            subfolder: Folder containing checkpoints.

        Returns:
            List of checkpoint filenames.
        """
        files = self.api.list_repo_files(
            repo_id=self.config.repo_id,
            repo_type=self.config.repo_type,
            token=self.config.token,
        )

        checkpoints = [
            f.replace(f"{subfolder}/", "")
            for f in files
            if f.startswith(subfolder) and f.endswith(".pt")
        ]

        return sorted(checkpoints)


class DatasetHubManager(HubManager):
    """Specialized manager for dataset operations."""

    def __init__(self, repo_id: str, token: str | None = None) -> None:
        """Initialize dataset manager.

        Args:
            repo_id: HuggingFace dataset repository ID.
            token: HF API token.
        """
        super().__init__(
            HubConfig(
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
        )

    def upload_samples(
        self,
        audio_dir: str | Path,
        manifest: list[dict[str, Any]],
        split: str = "train",
    ) -> None:
        """Upload audio samples with manifest.

        Args:
            audio_dir: Directory containing audio files.
            manifest: List of sample metadata dicts.
            split: Dataset split name.
        """
        audio_dir = Path(audio_dir)

        # Upload audio folder
        upload_folder(
            folder_path=str(audio_dir),
            repo_id=self.config.repo_id,
            repo_type="dataset",
            path_in_repo=f"audio/{split}",
            token=self.config.token,
            commit_message=f"Upload {split} audio",
        )

        # Upload manifest
        manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2)
        upload_file(
            path_or_fileobj=manifest_json.encode("utf-8"),
            path_in_repo=f"manifests/{split}.json",
            repo_id=self.config.repo_id,
            repo_type="dataset",
            token=self.config.token,
            commit_message=f"Upload {split} manifest",
        )
