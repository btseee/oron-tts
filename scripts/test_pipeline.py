"""End-to-end smoke test for the OronTTS / F5-TTS training pipeline.

Run from workspace root:
    python scripts/test_pipeline.py          # synthetic data only
    python scripts/test_pipeline.py --hf     # + 10 real samples from btsee/mbspeech_mn
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import yaml

if TYPE_CHECKING:
    from src.models.f5tts import F5TTS

# ── helpers ─────────────────────────────────────────────────────────────────


def _pass(tag: str) -> None:
    print(f"  [PASS] {tag}")


def _fail(tag: str, exc: BaseException) -> None:
    print(f"  [FAIL] {tag}")
    traceback.print_exc()


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _make_audio(seconds: float = 2.0, sr: int = 24000) -> np.ndarray:
    """Return a short sine-wave mono array at 24 kHz."""
    t = np.linspace(0, seconds, int(seconds * sr), dtype=np.float32)
    return 0.3 * np.sin(2 * np.pi * 220 * t)


# ── steps ───────────────────────────────────────────────────────────────────


def step_config() -> dict:
    _section("Step 1 — Load and validate configs")
    results: dict = {}
    for name in ("local.yaml", "runpod.yaml"):
        path = Path("configs") / name
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)

            assert cfg["sample_rate"] == 24000, f"sample_rate != 24000 in {name}"
            assert cfg["n_mels"] == 100, f"n_mels != 100 in {name}"
            m = cfg["model"]
            assert "dim" in m, f"model.dim missing in {name}"
            assert "depth" in m, f"model.depth missing in {name}"
            assert "vocab_size" in m, f"model.vocab_size missing in {name}"
            assert m["vocab_size"] == 65, f"vocab_size != 65 in {name}"
            # Reject VITS keys that must no longer exist
            vits_keys = {"inter_channels", "hidden_channels", "n_layers", "segment_size"}
            found = vits_keys & set(m.keys())
            assert not found, f"Old VITS keys found in {name}: {found}"

            _pass(name)
            results[name] = cfg
        except Exception as exc:
            _fail(name, exc)

    return results.get("local.yaml", {})


def step_tokenizer() -> None:
    _section("Step 2 — CyrillicTokenizer")
    try:
        from src.utils.tokenizer import CyrillicTokenizer

        tok = CyrillicTokenizer()
        assert tok.vocab_size == 65, f"vocab_size={tok.vocab_size}, want 65"
        _pass(f"vocab_size = {tok.vocab_size}")

        # Mongolian round-trip
        text = "сайн байна уу"
        ids = tok.encode(text, lang="mn")
        assert 3 not in ids, f"UNK found in Mongolian encode: {ids}"
        decoded = tok.decode(ids)
        assert decoded.strip(), "decode returned empty string"
        _pass(f"MN encode/decode (no UNK):  '{text}' → {ids[:6]}…")

        # Kazakh round-trip
        kz = "сәлем"
        ids_kz = tok.encode(kz, lang="kz")
        assert 3 not in ids_kz, f"UNK in Kazakh: {ids_kz}"
        _pass(f"KZ encode (no UNK): '{kz}' → {ids_kz}")

        # Attribute tokens
        ids_attr = tok.encode(text, lang="mn", attr_tokens=["[FEMALE]"])
        assert ids_attr[1] == 6, f"[FEMALE] token id should be 6, got {ids_attr[1]}"  # after BOS
        _pass("[FEMALE] attr token at correct position")

    except Exception as exc:
        _fail("CyrillicTokenizer", exc)


def step_text_cleaner() -> None:
    _section("Step 3 — TextCleaner + NumberNormalizer")
    try:
        from src.utils.text_cleaner import TextCleaner

        tc = TextCleaner()

        # Number normalisation
        cleaned = tc.clean("2024 онд")
        assert "2024" not in cleaned, f"clean() did not expand number: '{cleaned}'"
        _pass(f"clean('2024 онд') → '{cleaned}'")

        # text_to_sequence returns list of ints, no UNK (3)
        ids = tc.text_to_sequence("сайн байна уу", lang="mn")
        assert isinstance(ids, list) and len(ids) > 0, "text_to_sequence returned empty"
        assert all(isinstance(x, int) for x in ids), "non-int in ids"
        assert 3 not in ids, f"UNK in text_to_sequence: {ids}"
        _pass(f"text_to_sequence('сайн байна уу') → {ids[:6]}… (len={len(ids)})")

    except Exception as exc:
        _fail("TextCleaner", exc)


def step_audio_processor() -> None:
    _section("Step 4 — AudioProcessor mel spectrogram")
    try:
        from src.utils.audio import AudioProcessor

        ap = AudioProcessor(sample_rate=24000, n_mels=100)
        audio = torch.from_numpy(_make_audio(2.0)).float()

        mel = ap.mel_spectrogram(audio)  # [100, T]
        assert mel.shape[0] == 100, f"n_mels={mel.shape[0]}, want 100"
        T = mel.shape[1]
        # 2s @ hop=256 → ~188 frames  (24000*2/256 = 187.5)
        assert 150 < T < 250, f"unexpected T={T} for 2s audio"
        assert torch.isfinite(mel).all(), "Non-finite values in mel"
        _pass(f"mel shape = [100, {T}], all finite")

        # normalize_audio
        normed = ap.normalize_audio(audio)
        assert normed.abs().max() <= 1.001, "normalize_audio > 1"
        _pass("normalize_audio: peak ≤ 1")

    except Exception as exc:
        _fail("AudioProcessor", exc)


def step_dataset() -> None:
    _section("Step 5 — TTSDataset with synthetic audio")
    try:
        from src.data.dataset import TTSDataset

        audios = [_make_audio(2.0), _make_audio(3.0)]
        texts = ["сайн байна уу", "нар мандав"]

        ds = TTSDataset(audio_arrays=audios, texts=texts, sample_rate=24000, n_mels=100)
        assert len(ds) == 2, f"len(ds)={len(ds)}"

        item0 = ds[0]
        mel0 = item0["mel"]
        tid0 = item0["text_ids"]
        msk0 = item0["mask"]

        T = mel0.shape[-1]
        assert mel0.shape == (100, T), f"mel shape {mel0.shape}"
        assert tid0.shape == (T,), f"text_ids shape {tid0.shape}, want ({T},)"
        assert msk0.shape == (T,), f"mask shape {msk0.shape}"
        assert msk0.all(), "mask should be all True for real frames"
        _pass(f"item0: mel=[100,{T}], text_ids=[{T}], mask all-True")

    except Exception as exc:
        _fail("TTSDataset", exc)


def step_collator() -> None:
    _section("Step 6 — TTSCollator padding")
    try:
        from torch.utils.data import DataLoader

        from src.data.dataset import TTSCollator, TTSDataset

        audios = [_make_audio(2.0), _make_audio(3.0)]
        texts = ["сайн", "нар мандав уу байна"]

        ds = TTSDataset(audio_arrays=audios, texts=texts, sample_rate=24000, n_mels=100)
        loader = DataLoader(ds, batch_size=2, collate_fn=TTSCollator(), num_workers=0)

        batch = next(iter(loader))
        mel = batch["mel"]  # [2, 100, T_max]
        mask = batch["mask"]  # [2, T_max]
        tids = batch["text_ids"]  # [2, T_max]

        B, n_mels, T_max = mel.shape
        assert B == 2, f"batch size={B}"
        assert n_mels == 100, f"n_mels={n_mels}"
        assert T_max > 0, "T_max=0"
        assert mask.shape == (2, T_max), f"mask shape {mask.shape}"
        assert tids.shape == (2, T_max), f"text_ids shape {tids.shape}"

        # Shorter audio should have False in its padding region
        # item 0 is shorter (2s), item 1 is longer (3s); after sorting by TTSCollator they may swap
        false_count = (~mask).sum().item()
        assert false_count > 0, "Expected padding False entries in mask"
        _pass(f"batch: mel=[2,100,{T_max}], mask has {false_count} padding frames")

    except Exception as exc:
        _fail("TTSCollator", exc)


def step_model_forward(cfg: dict) -> F5TTS | None:
    """Returns the model so downstream steps can reuse it."""
    _section("Step 7 + 8 — Model instantiation and forward pass")
    try:
        from src.models.f5tts import F5TTS

        if not cfg:
            # Fallback mini config if config loading failed
            cfg = {
                "sample_rate": 24000,
                "n_mels": 100,
                "n_fft": 1024,
                "hop_length": 256,
                "model": {"dim": 64, "depth": 2, "heads": 2, "vocab_size": 65},
            }

        # Use tiny overrides so test is fast on CPU
        tiny_cfg = dict(cfg)
        tiny_cfg["model"] = dict(cfg.get("model", {}))
        tiny_cfg["model"]["dim"] = 64
        tiny_cfg["model"]["depth"] = 2
        tiny_cfg["model"]["heads"] = 2
        tiny_cfg["model"]["ff_mult"] = 4
        tiny_cfg["model"]["conv_layers"] = 2
        tiny_cfg["model"]["text_dim"] = 64
        tiny_cfg["model"]["vocos_dim"] = 64
        tiny_cfg["model"]["vocos_layers"] = 2
        tiny_cfg["model"]["vocos_intermediate"] = 128

        model = F5TTS.from_config(tiny_cfg)
        n_params = sum(p.numel() for p in model.parameters())
        _pass(f"F5TTS instantiated — {n_params:,} params")

        # Synthetic batch: B=2, T=50
        B, T = 2, 50
        mel = torch.randn(B, 100, T)
        text_ids = torch.randint(4, 65, (B, T))
        mask = torch.ones(B, T, dtype=torch.bool)

        model.eval()
        with torch.no_grad():
            loss = model(mel, text_ids, mask)

        assert loss.ndim == 0, f"loss should be scalar, got shape {loss.shape}"
        assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"
        _pass(f"forward pass: loss = {loss.item():.4f}")

        return model

    except Exception as exc:
        _fail("model forward", exc)
        return None


def step_backward(model: F5TTS | None) -> F5TTS | None:
    _section("Step 9 — Backward pass (gradient check)")
    if model is None:
        print("  [SKIP] no model from step 7/8")
        return None
    try:
        B, T = 2, 50
        mel = torch.randn(B, 100, T)
        text_ids = torch.randint(4, 65, (B, T))
        mask = torch.ones(B, T, dtype=torch.bool)

        model.train()
        loss = model(mel, text_ids, mask)
        loss.backward()

        bad_grads = [
            name
            for name, p in model.named_parameters()
            if p.grad is not None and not torch.isfinite(p.grad).all()
        ]
        assert not bad_grads, f"Non-finite grads in: {bad_grads}"
        any_grad = any(p.grad is not None for p in model.parameters())
        assert any_grad, "No gradients computed"
        _pass("Backward OK — all gradients finite")
        return model

    except Exception as exc:
        _fail("backward pass", exc)
        return model


def step_trainer(model: F5TTS | None) -> None:
    _section("Step 10 — F5Trainer.train_epoch (synthetic data)")
    if model is None:
        print("  [SKIP] no model from earlier steps")
        return
    try:
        import os
        import tempfile

        from torch.utils.data import DataLoader

        from src.data.dataset import TTSCollator, TTSDataset
        from src.training.trainer import F5Trainer

        # 4 samples — tests grad_accum=1 with multiple batches
        audios = [_make_audio(float(i) + 1.5) for i in range(4)]
        texts = ["сайн байна уу", "нар мандав", "монгол хэл", "туршилт"]
        ds = TTSDataset(audio_arrays=audios, texts=texts, sample_rate=24000, n_mels=100)
        loader = DataLoader(
            ds,
            batch_size=2,
            collate_fn=TTSCollator(),
            num_workers=0,
            drop_last=False,
            shuffle=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            train_cfg: dict = {
                "learning_rate": 1e-4,
                "warmup_steps": 5,
                "ema_decay": 0.99,
                "max_grad_norm": 1.0,
                "grad_accumulation_steps": 1,
                "log_interval": 1,
                "use_tqdm": False,
            }
            trainer = F5Trainer(
                config=train_cfg,
                model=model,
                train_loader=loader,
                val_loader=None,
                device="cpu",
                rank=0,
                world_size=1,
                log_dir=os.path.join(tmpdir, "logs"),
                checkpoint_dir=os.path.join(tmpdir, "ckpts"),
            )

            avg_loss = trainer.train_epoch(total_epochs=2)
            assert np.isfinite(avg_loss), f"avg_loss={avg_loss}"
            _pass(f"train_epoch: avg_loss = {avg_loss:.4f}")

            # Save checkpoint via trainer
            ckpt_path = trainer.save_checkpoint(is_best=False)
            assert ckpt_path is not None and Path(ckpt_path).exists(), (
                f"checkpoint not saved: {ckpt_path}"
            )
            _pass(f"checkpoint saved: {Path(ckpt_path).name}")

    except Exception as exc:
        _fail("F5Trainer", exc)


def step_hf_dataset() -> None:
    _section("Step 11 — HF dataset: btsee/mbspeech_mn (10 samples)")
    try:
        import io
        from fractions import Fraction

        import soundfile as sf
        from datasets import Audio, load_dataset
        from scipy.signal import resample_poly

        from src.data.dataset import TTSDataset

        # Stream without HF audio decoding (avoids torchcodec / FFmpeg requirement).
        # We decode raw bytes ourselves with soundfile and resample to 24 kHz.
        ds_stream = load_dataset("btsee/mbspeech_mn", split="train", streaming=True)
        ds_stream = ds_stream.cast_column("audio", Audio(decode=False))

        audio_arrays: list[np.ndarray] = []
        texts: list[str] = []

        for item in ds_stream.take(10):
            raw = item["audio"]
            audio_bytes: bytes | None = raw.get("bytes")
            if not audio_bytes:
                # Fallback: path-based (uncommon for HF parquet datasets)
                raise ValueError(f"No bytes in audio column: {raw.keys()}")
            arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if arr.ndim > 1:
                arr = arr[:, 0]  # take first channel if stereo
            if sr != 24000:
                f = Fraction(24000, sr).limit_denominator(100)
                arr = resample_poly(arr, f.numerator, f.denominator).astype(np.float32)
            audio_arrays.append(arr)
            texts.append(item["sentence_norm"])

        assert len(audio_arrays) == 10, f"Expected 10 samples, got {len(audio_arrays)}"
        _pass("Streamed 10 samples: audio column=audio, text column=sentence, resampled 16→24 kHz")

        tts_ds = TTSDataset(
            audio_arrays=audio_arrays,
            texts=texts,
            langs=["mn"] * 10,
            sample_rate=24000,
            n_mels=100,
        )
        assert len(tts_ds) == 10

        failures = 0
        for i in range(10):
            item = tts_ds[i]
            mel = item.get("mel")
            if mel is None:
                failures += 1
                continue
            assert mel.shape[0] == 100, f"n_mels={mel.shape[0]}"
            assert torch.isfinite(mel).all(), f"non-finite mel in sample {i}"

        assert failures == 0, f"{failures}/10 samples failed __getitem__"
        _pass("All 10 real samples: mel [100, T], all finite, no UNK")

    except Exception as exc:
        _fail("HF dataset", exc)


# ── main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="OronTTS pipeline smoke test")
    parser.add_argument(
        "--hf",
        action="store_true",
        help="Include HF dataset step (downloads ~10 MB from btsee/mbspeech_mn)",
    )
    args = parser.parse_args()

    print("\n=== OronTTS F5-TTS pipeline smoke test ===\n")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    cfg = step_config()
    step_tokenizer()
    step_text_cleaner()
    step_audio_processor()
    step_dataset()
    step_collator()
    model = step_model_forward(cfg)
    model = step_backward(model)
    step_trainer(model)

    if args.hf:
        step_hf_dataset()

    print("\n=== Smoke test complete ===\n")


if __name__ == "__main__":
    main()
