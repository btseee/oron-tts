"""F5-TTS training script for OronTTS."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from dotenv import load_dotenv
from torch.multiprocessing import spawn
from torch.utils.data import DataLoader, DistributedSampler

from src.data.dataset import TTSCollator, TTSDataset
from src.data.hf_wrapper import HFDatasetWrapper
from src.models.f5tts import F5TTS
from src.training.trainer import F5Trainer


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        if config_path.endswith(".json"):
            return json.load(f)
        import yaml

        return yaml.safe_load(f)


def setup_distributed(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed() -> None:
    dist.destroy_process_group()


def train_worker(rank: int, world_size: int, config: dict, args: argparse.Namespace) -> None:
    if world_size > 1:
        setup_distributed(rank, world_size)

    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        if config.get("cudnn_benchmark", False):
            torch.backends.cudnn.benchmark = True
        if config.get("use_tf32", False):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    sample_rate = config.get("sample_rate", 24000)
    n_mels = config.get("n_mels", 100)
    default_lang = args.lang or "mn"

    max_duration = config.get("max_duration")
    max_audio_len = int(max_duration * sample_rate) if max_duration else None

    if not args.from_local:
        if rank == 0:
            print(f"[Rank {rank}] Loading dataset from HuggingFace: {args.dataset}")
        wrapper = HFDatasetWrapper(args.dataset, cache_dir=args.cache_dir, sample_rate=sample_rate)
        hf_dataset = wrapper.load(split="train")
        train_dataset = TTSDataset.from_hf_dataset(
            hf_dataset,
            audio_column=args.audio_column,
            text_column=args.text_column,
            lang_column=args.lang_column,
            sample_rate=sample_rate,
            n_mels=n_mels,
            default_lang=default_lang,
            max_audio_len=max_audio_len,
        )
    else:
        metadata_path = Path(args.data_dir) / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        audio_paths = [Path(m["audio_path"]) for m in metadata]
        texts = [m["text"] for m in metadata]
        langs = [m.get("lang", default_lang) for m in metadata]
        train_dataset = TTSDataset(
            audio_paths=audio_paths,
            texts=texts,
            langs=langs,
            sample_rate=sample_rate,
            n_mels=n_mels,
            max_audio_len=max_audio_len,
        )

    if rank == 0:
        print(f"[Rank {rank}] Dataset size: {len(train_dataset)}")

    # Train/val split (90/10) for validation loss tracking
    val_loader = None
    val_size = int(len(train_dataset) * 0.1)
    if val_size >= 2:
        from torch.utils.data import random_split

        train_subset, val_subset = random_split(
            train_dataset,
            [len(train_dataset) - val_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        train_subset = train_dataset
        val_subset = None

    num_workers = config.get("num_workers", 4)
    use_persistent = num_workers > 0

    sampler = (
        DistributedSampler(train_subset, num_replicas=world_size, rank=rank)
        if world_size > 1
        else None
    )
    train_loader = DataLoader(
        train_subset,
        batch_size=config.get("batch_size", 16),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=TTSCollator(),
        pin_memory=config.get("pin_memory", True) and torch.cuda.is_available(),
        prefetch_factor=config.get("prefetch_factor", 2) if num_workers > 0 else None,
        persistent_workers=use_persistent,
        drop_last=True,
    )

    if val_subset is not None:
        val_loader = DataLoader(
            val_subset,
            batch_size=config.get("batch_size", 16),
            shuffle=False,
            num_workers=max(num_workers // 2, 1),
            collate_fn=TTSCollator(),
            pin_memory=config.get("pin_memory", True) and torch.cuda.is_available(),
            persistent_workers=False,
        )

    model = F5TTS.from_config(config)
    if rank == 0:
        print(f"[Rank {rank}] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # torch.compile for PyTorch 2.x speedup (skip if unavailable or disabled)
    if config.get("compile", True) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            if rank == 0:
                print("[INFO] torch.compile enabled")
        except Exception as e:
            if rank == 0:
                print(f"[WARN] torch.compile failed, using eager mode: {e}")

    # Optionally load pretrained F5-TTS checkpoint
    if args.pretrain_ckpt:
        from src.utils.checkpoint import CheckpointManager

        cm = CheckpointManager(args.checkpoint_dir)
        info = cm.load_pretrained_f5tts(model, args.pretrain_ckpt, device=device)
        if rank == 0:
            print(
                f"Loaded pretrained weights. Missing: {len(info['missing_keys'])}, Unexpected: {len(info['unexpected_keys'])}"
            )

    trainer = F5Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        rank=rank,
        world_size=world_size,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

    if args.resume:
        trainer.load_checkpoint(load_best=args.resume_best)

    num_epochs = args.num_epochs or config.get("num_epochs", 500)
    trainer.train(num_epochs=num_epochs, save_interval=config.get("save_interval", 5))

    if rank == 0 and args.push_to_hub:
        url = trainer.push_to_hub(args.hf_repo, token=args.hf_token)
        print(f"Model pushed to: {url}")

    if world_size > 1:
        cleanup_distributed()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Train OronTTS F5-TTS model")
    parser.add_argument("--config", type=str, default="configs/runpod.yaml")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument(
        "--from-local", action="store_true", help="Use local metadata.json instead of HF"
    )
    parser.add_argument("--dataset", type=str, default="btsee/mbspeech_mn")
    parser.add_argument("--audio-column", type=str, default="audio")
    parser.add_argument("--text-column", type=str, default=None)
    parser.add_argument("--lang-column", type=str, default=None)
    parser.add_argument(
        "--lang",
        type=str,
        default="mn",
        choices=["mn", "kz"],
        help="Default language if no lang column",
    )
    parser.add_argument("--cache-dir", type=str, default="output/data/cache")
    parser.add_argument("--log-dir", type=str, default="output/logs")
    parser.add_argument("--checkpoint-dir", type=str, default="output/checkpoints")
    parser.add_argument(
        "--pretrain-ckpt",
        type=str,
        default=None,
        help="Path to pretrained F5-TTS .safetensors or .pt checkpoint",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-best", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hf-repo", type=str, default="btsee/oron-tts")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.num_epochs:
        config["num_epochs"] = args.num_epochs

    world_size = args.num_gpus if torch.cuda.is_available() else 1

    if world_size > 1:
        spawn(train_worker, args=(world_size, config, args), nprocs=world_size, join=True)
    else:
        train_worker(0, 1, config, args)


if __name__ == "__main__":
    main()
