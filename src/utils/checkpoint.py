"""Checkpoint management for training and inference."""

import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, hf_hub_download


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str | Path,
        model_name: str = "vits",
        max_checkpoints: int = 5,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.max_checkpoints = max_checkpoints

    def _get_checkpoint_path(self, step: int) -> Path:
        return self.checkpoint_dir / f"{self.model_name}_step_{step:08d}.pt"

    def _get_best_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / f"{self.model_name}_best.pt"

    def _get_config_path(self) -> Path:
        return self.checkpoint_dir / "config.json"

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer_g: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer,
        scheduler_g: torch.optim.lr_scheduler.LRScheduler | None = None,
        scheduler_d: torch.optim.lr_scheduler.LRScheduler | None = None,
        loss: float | None = None,
        config: dict[str, Any] | None = None,
        is_best: bool = False,
    ) -> Path:
        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict(),
            "loss": loss,
        }

        if scheduler_g is not None:
            checkpoint["scheduler_g_state_dict"] = scheduler_g.state_dict()
        if scheduler_d is not None:
            checkpoint["scheduler_d_state_dict"] = scheduler_d.state_dict()

        path = self._get_checkpoint_path(step)
        torch.save(checkpoint, path)

        if config is not None:
            with open(self._get_config_path(), "w") as f:
                json.dump(config, f, indent=2)

        if is_best:
            best_path = self._get_best_checkpoint_path()
            torch.save(checkpoint, best_path)

        self._cleanup_old_checkpoints()
        return path

    def load(
        self,
        model: torch.nn.Module,
        optimizer_g: torch.optim.Optimizer | None = None,
        optimizer_d: torch.optim.Optimizer | None = None,
        scheduler_g: torch.optim.lr_scheduler.LRScheduler | None = None,
        scheduler_d: torch.optim.lr_scheduler.LRScheduler | None = None,
        path: str | Path | None = None,
        load_best: bool = False,
        device: str = "cpu",
    ) -> dict[str, Any]:
        if path is None:
            path = self._get_best_checkpoint_path() if load_best else self._get_latest_checkpoint()

        if path is None or not Path(path).exists():
            return {"step": 0, "loss": None}

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer_g is not None and "optimizer_g_state_dict" in checkpoint:
            optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
        if optimizer_d is not None and "optimizer_d_state_dict" in checkpoint:
            optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
        if scheduler_g is not None and "scheduler_g_state_dict" in checkpoint:
            scheduler_g.load_state_dict(checkpoint["scheduler_g_state_dict"])
        if scheduler_d is not None and "scheduler_d_state_dict" in checkpoint:
            scheduler_d.load_state_dict(checkpoint["scheduler_d_state_dict"])

        return {"step": checkpoint.get("step", 0), "loss": checkpoint.get("loss")}

    def _get_latest_checkpoint(self) -> Path | None:
        checkpoints = sorted(
            self.checkpoint_dir.glob(f"{self.model_name}_step_*.pt"),
            key=lambda x: int(x.stem.split("_")[-1]),
        )
        return checkpoints[-1] if checkpoints else None

    def _cleanup_old_checkpoints(self) -> None:
        checkpoints = sorted(
            self.checkpoint_dir.glob(f"{self.model_name}_step_*.pt"),
            key=lambda x: int(x.stem.split("_")[-1]),
        )
        while len(checkpoints) > self.max_checkpoints:
            checkpoints[0].unlink()
            checkpoints.pop(0)

    def load_config(self) -> dict[str, Any] | None:
        config_path = self._get_config_path()
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return None

    def push_to_hub(
        self,
        repo_id: str,
        token: str | None = None,
        private: bool = False,
    ) -> str:
        api = HfApi()
        api.create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)

        api.upload_folder(
            folder_path=str(self.checkpoint_dir),
            repo_id=repo_id,
            token=token,
        )
        return f"https://huggingface.co/{repo_id}"

    def pull_from_hub(
        self,
        repo_id: str,
        filename: str = "vits_best.pt",
        token: str | None = None,
    ) -> Path:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
            local_dir=str(self.checkpoint_dir),
        )
        return Path(path)
