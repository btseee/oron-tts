"""VITS training script for OronTTS."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn as mp_spawn
from torch.utils.data import DataLoader, DistributedSampler

from src.data.dataset import TTSCollator, TTSDataset
from src.data.hf_wrapper import HFDatasetWrapper
from src.models.discriminator import MultiPeriodDiscriminator
from src.models.vits import VITS
from src.training.trainer import VITSTrainer
from src.utils.phonemizer import MongolianPhonemizer


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        if config_path.endswith(".json"):
            return json.load(f)
        else:
            import yaml
            return yaml.safe_load(f)


def create_model(config: dict, n_speakers: int = 2) -> tuple[VITS, MultiPeriodDiscriminator]:
    phonemizer = MongolianPhonemizer()

    model_config = config.get("model", {})
    model = VITS(
        n_vocab=phonemizer.vocab_size,
        spec_channels=config.get("n_mels", 80),
        segment_size=config.get("segment_size", 32),
        inter_channels=model_config.get("inter_channels", 192),
        hidden_channels=model_config.get("hidden_channels", 192),
        filter_channels=model_config.get("filter_channels", 768),
        n_heads=model_config.get("n_heads", 2),
        n_layers=model_config.get("n_layers", 6),
        kernel_size=model_config.get("kernel_size", 3),
        p_dropout=model_config.get("p_dropout", 0.1),
        resblock=model_config.get("resblock", "1"),
        resblock_kernel_sizes=model_config.get("resblock_kernel_sizes", [3, 7, 11]),
        resblock_dilation_sizes=model_config.get(
            "resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        ),
        upsample_rates=model_config.get("upsample_rates", [8, 8, 2, 2]),
        upsample_initial_channel=model_config.get("upsample_initial_channel", 512),
        upsample_kernel_sizes=model_config.get("upsample_kernel_sizes", [16, 16, 4, 4]),
        n_speakers=n_speakers,
        gin_channels=model_config.get("gin_channels", 256),
        use_sdp=model_config.get("use_sdp", True),
    )

    discriminator = MultiPeriodDiscriminator()

    return model, discriminator


def setup_distributed(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed() -> None:
    dist.destroy_process_group()


def train_worker(
    rank: int,
    world_size: int,
    config: dict,
    args: argparse.Namespace,
) -> None:
    if world_size > 1:
        setup_distributed(rank, world_size)

    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(rank) if torch.cuda.is_available() else None

    if args.from_hf:
        print(f"[Rank {rank}] Loading dataset from Hugging Face: {args.dataset}")
        wrapper = HFDatasetWrapper(
            args.dataset,
            cache_dir=args.cache_dir,
            sample_rate=config.get("sample_rate", 22050),
        )
        hf_dataset = wrapper.load(split="train")
        train_dataset = TTSDataset.from_hf_dataset(
            hf_dataset,
            audio_column=args.audio_column,
            text_column=args.text_column,
            speaker_column=args.speaker_column,
            sample_rate=config.get("sample_rate", 22050),
        )
    else:
        metadata_path = Path(args.data_dir) / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        audio_paths = [Path(m["audio_path"]) for m in metadata]
        texts = [m["text"] for m in metadata]
        speaker_ids = [m["speaker_id"] for m in metadata]

        train_dataset = TTSDataset(
            audio_paths=audio_paths,
            texts=texts,
            speaker_ids=speaker_ids,
            sample_rate=config.get("sample_rate", 22050),
        )

    n_speakers = len(set(train_dataset.speaker_ids))
    if rank == 0:
        print(f"[Rank {rank}] Dataset size: {len(train_dataset)}, Speakers: {n_speakers}")
        print(f"[Rank {rank}] Training Config:")
        print(f"  - Batch size: {config.get('batch_size', 16)}")
        print(f"  - Learning rate: {config.get('learning_rate', 2e-4)}")
        print(f"  - FP16: {config.get('fp16', True)}")
        print(f"  - Segment size: {config.get('segment_size', 32)}")
        print(f"  - Log interval: {config.get('log_interval', 100)}")
        print(f"  - Use tqdm: {config.get('use_tqdm', True)}")

    if world_size > 1:
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    else:
        sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 16),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config.get("num_workers", 4),
        collate_fn=TTSCollator(),
        pin_memory=True,
        drop_last=True,
    )

    model, discriminator = create_model(config, n_speakers=n_speakers)
    print(f"[Rank {rank}] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = VITSTrainer(
        config=config,
        model=model,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=None,
        device=device,
        rank=rank,
        world_size=world_size,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

    if args.resume:
        print(f"[Rank {rank}] Resuming from checkpoint...")
        trainer.load_checkpoint(load_best=args.resume_best)

    print(f"[Rank {rank}] Starting training...")
    trainer.train(
        num_epochs=config.get("num_epochs", 1000),
        save_interval=config.get("save_interval", 5),
    )

    if rank == 0 and args.push_to_hub:
        print("Pushing model to Hugging Face Hub...")
        url = trainer.push_to_hub(args.hf_repo, token=args.hf_token)
        print(f"Model pushed to: {url}")

    if world_size > 1:
        cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OronTTS VITS model")
    parser.add_argument("--config", type=str, default="configs/vits_runpod.yaml")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--from-hf", action="store_true", default=True)
    parser.add_argument("--dataset", type=str, default="btsee/mbspeech_mn")
    parser.add_argument("--audio-column", type=str, default="audio")
    parser.add_argument("--text-column", type=str, default=None, help="Auto-detect if not specified")
    parser.add_argument("--speaker-column", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="output/data/cache")
    parser.add_argument("--log-dir", type=str, default="output/logs")
    parser.add_argument("--checkpoint-dir", type=str, default="output/checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-best", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hf-repo", type=str, default="btsee/oron-tts")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    args = parser.parse_args()

    config = load_config(args.config)

    world_size = args.num_gpus
    if world_size > 1:
        mp_spawn(
            train_worker,  # type: ignore
            args=(world_size, config, args),
            nprocs=world_size,
            join=True,
        )
    else:
        train_worker(0, 1, config, args)


if __name__ == "__main__":
    main()
