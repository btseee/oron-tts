"""HuggingFace Hub integration for dataset and model management."""

import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
from tqdm import tqdm

from orontts.exceptions import DatasetError


def get_hf_api() -> HfApi:
    """Get HuggingFace API instance.

    Expects HF_TOKEN environment variable or cached credentials.
    """
    token = os.environ.get("HF_TOKEN")
    return HfApi(token=token)


def push_dataset_to_hub(
    data_dir: Path,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload TTS dataset",
) -> str:
    """Push a prepared dataset to HuggingFace Hub.

    Args:
        data_dir: Local directory containing the dataset.
        repo_id: Target repository ID (e.g., "username/dataset-name").
        private: Whether the repository should be private.
        commit_message: Commit message for the upload.

    Returns:
        URL of the uploaded dataset.

    Raises:
        DatasetError: If upload fails.
    """
    try:
        api = get_hf_api()

        # Create repository if it doesn't exist
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )

        # Upload the entire folder
        api.upload_folder(
            folder_path=str(data_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )

        url = f"https://huggingface.co/datasets/{repo_id}"
        return url

    except Exception as e:
        raise DatasetError(f"Failed to push dataset to Hub: {e}") from e


def load_dataset_from_hub(
    repo_id: str,
    local_dir: Path,
    revision: str | None = None,
) -> Path:
    """Download a dataset from HuggingFace Hub.

    Args:
        repo_id: Repository ID (e.g., "username/dataset-name").
        local_dir: Local directory to download to.
        revision: Git revision (branch, tag, or commit hash).

    Returns:
        Path to the downloaded dataset.

    Raises:
        DatasetError: If download fails.
    """
    try:
        from huggingface_hub import snapshot_download

        local_dir.mkdir(parents=True, exist_ok=True)

        path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            revision=revision,
        )

        return Path(path)

    except Exception as e:
        raise DatasetError(f"Failed to download dataset from Hub: {e}") from e


def push_model_to_hub(
    model_dir: Path,
    repo_id: str,
    config: dict[str, Any] | None = None,
    private: bool = False,
    commit_message: str = "Upload VITS2 model",
) -> str:
    """Push a trained model to HuggingFace Hub.

    Args:
        model_dir: Directory containing model files.
        repo_id: Target repository ID.
        config: Model configuration to include.
        private: Whether the repository should be private.
        commit_message: Commit message.

    Returns:
        URL of the uploaded model.

    Raises:
        DatasetError: If upload fails.
    """
    try:
        api = get_hf_api()

        # Create repository
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )

        # Upload config if provided
        if config is not None:
            config_path = model_dir / "config.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

        # Create model card if not exists
        readme_path = model_dir / "README.md"
        if not readme_path.exists():
            _create_model_card(readme_path, repo_id, config)

        # Upload folder
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )

        url = f"https://huggingface.co/{repo_id}"
        return url

    except Exception as e:
        raise DatasetError(f"Failed to push model to Hub: {e}") from e


def _create_model_card(
    path: Path,
    repo_id: str,
    config: dict[str, Any] | None,
) -> None:
    """Create a basic model card."""
    content = f"""---
language: mn
tags:
- tts
- vits2
- mongolian
- speech-synthesis
license: mit
---

# {repo_id.split('/')[-1]}

Mongolian (Khalkha) Text-to-Speech model using VITS2 architecture.

## Usage

```python
from inference import Synthesizer

synth = Synthesizer.from_pretrained("{repo_id}")
audio = synth.synthesize("Сайн байна уу")
audio.save("output.wav")
```

## Model Details

- **Architecture**: VITS2
- **Language**: Mongolian (Khalkha)
- **Sample Rate**: 22050 Hz

## Training

Trained using OronTTS framework.
"""

    if config:
        content += f"""
## Configuration

```json
{json.dumps(config, indent=2, ensure_ascii=False)}
```
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def download_checkpoint(
    repo_id: str,
    filename: str = "model.ckpt",
    local_dir: Path | None = None,
) -> Path:
    """Download a specific checkpoint file from Hub.

    Args:
        repo_id: Repository ID.
        filename: Name of the checkpoint file.
        local_dir: Local directory to save to.

    Returns:
        Path to the downloaded file.
    """
    from huggingface_hub import hf_hub_download

    local_dir = local_dir or Path.home() / ".cache" / "orontts"
    local_dir.mkdir(parents=True, exist_ok=True)

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
    )

    return Path(path)


def list_available_models(author: str | None = None) -> list[dict[str, Any]]:
    """List available OronTTS models on the Hub.

    Args:
        author: Filter by author/organization.

    Returns:
        List of model info dictionaries.
    """
    api = get_hf_api()

    filter_str = "vits2 mongolian"
    if author:
        filter_str = f"{author}/{filter_str}"

    models = api.list_models(
        filter=filter_str,
        sort="downloads",
        direction=-1,
    )

    return [
        {
            "id": model.modelId,
            "author": model.author,
            "downloads": model.downloads,
            "tags": model.tags,
        }
        for model in models
    ]
