"""Extend the F5-TTS base vocabulary with the Mongolian-only Cyrillic letters.

F5TTS_v1_Base's vocabulary already covers 65 of the 70 Mongolian Cyrillic
letters (vocab.txt lines 1628-1693). Only these are missing:

    Ө U+04E8   ө U+04E9   Ү U+04AE   ү U+04AF   Ъ U+042A

Measured on 300 real sentences from fleurs-mn / common-voices-24-mn / mbspeech_mn,
those characters are 4.90% of all tokens. `list_str_to_idx` maps anything absent
from the vocab to index 0 -- which is the SPACE token -- so training on the
un-extended vocab silently replaces roughly one character in twenty with a space.

This script appends the missing entries (order-preserving: every pretrained row
keeps its index) and, given the base checkpoint, grows the text-embedding matrix
to match.

Usage:
    python scripts/extend_vocab.py --out data/oron_mn_pinyin/vocab.txt
    python scripts/extend_vocab.py --out data/oron_mn_pinyin/vocab.txt \
        --checkpoint ckpts/F5TTS_v1_Base/model_1250000.safetensors \
        --checkpoint-out ckpts/oron_mn/pretrained_model_1250000.safetensors
"""

import argparse
from pathlib import Path

# Every letter of the Mongolian Cyrillic alphabet, both cases. Anything here that
# the base vocab lacks gets appended, in this order.
MN_ALPHABET = "абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя"

EMBED_KEY = "ema_model.transformer.text_embed.text_embed.weight"


def read_vocab(path: Path) -> list[str]:
    """Read vocab.txt exactly the way f5_tts.model.utils.get_tokenizer does."""
    with open(path, encoding="utf-8") as f:
        return [line[:-1] for line in f]


def missing_tokens(vocab: list[str]) -> list[str]:
    """Mongolian letters absent from the base vocab, lower case before upper."""
    have = set(vocab)
    out = [c for c in MN_ALPHABET if c not in have]
    out += [c for c in MN_ALPHABET.upper() if c not in have]
    return out


def write_vocab(path: Path, vocab: list[str]) -> None:
    """Write one token per line, LF, trailing newline.

    The trailing newline is load-bearing: get_tokenizer does `line[:-1]`, so a
    final line without it loses its last character.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for token in vocab:
            f.write(token + "\n")


def expand_embeddings(ckpt_in: Path, ckpt_out: Path, n_new: int, vocab: list[str]) -> None:
    """Grow the text embedding by n_new rows, preserving every pretrained row.

    Upstream's expand_model_embeddings (finetune_gradio.py:974-1011) seeds new
    rows with torch.randn, std 1.0. Measured, the pretrained Cyrillic rows have
    element std 0.627 and mean row-norm 14.18, so a randn row would be about
    1.6x too long -- a real mismatch, though a smaller one than the "10-50x"
    figure this comment previously claimed before the checkpoint was measured.

    Seeding from the empirical per-dimension mean and std of the existing
    Cyrillic rows lands the new letters at row-norm 14.28 against 14.18, so they
    start in the same regime as their neighbours rather than being pulled toward
    the model by the first few gradient steps.
    """
    import torch
    from safetensors.torch import load_file, save_file

    if ckpt_in.suffix != ".safetensors":
        raise SystemExit(
            f"Expected the .safetensors base checkpoint, got {ckpt_in.name}. "
            "Upstream's .pt path leaves model_state_dict unexpanded and fails on load."
        )

    state = load_file(str(ckpt_in), device="cpu")
    if EMBED_KEY not in state:
        raise SystemExit(f"{EMBED_KEY} not found. Keys look like: {list(state)[:3]}")

    old = state[EMBED_KEY]
    n_old, dim = old.shape
    # get_tokenizer yields len(vocab) ids; DiT allocates len(vocab)+1 rows (0 = filler).
    expected = len(vocab) - n_new + 1
    if n_old != expected:
        raise SystemExit(f"Embedding has {n_old} rows, expected {expected} for this base vocab.")

    # Row i of the embedding holds vocab index i-1 (ids are shifted +1 for the filler).
    cyr_rows = [i + 1 for i, t in enumerate(vocab) if len(t) == 1 and "Ѐ" <= t <= "ӿ"]
    cyr_rows = [r for r in cyr_rows if r < n_old]
    ref = old[cyr_rows]
    mean, std = ref.mean(0), ref.std(0)
    print(f"  seeding from {len(cyr_rows)} pretrained Cyrillic rows: "
          f"|mean|={mean.abs().mean():.4f} std={std.mean():.4f} "
          f"(upstream would use randn, std 1.0)")

    new = torch.empty((n_old + n_new, dim), dtype=old.dtype)
    new[:n_old] = old
    new[n_old:] = torch.randn((n_new, dim), dtype=old.dtype) * std + mean
    state[EMBED_KEY] = new

    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(ckpt_out))
    print(f"  embedding {tuple(old.shape)} -> {tuple(new.shape)}")
    print(f"  wrote {ckpt_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extend F5-TTS vocab for Mongolian")
    parser.add_argument(
        "--base-vocab",
        type=Path,
        default=Path("../F5-TTS/data/Emilia_ZH_EN_pinyin/vocab.txt"),
        help="F5-TTS pretrained vocab.txt (2545 lines)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Extended vocab.txt to write")
    parser.add_argument("--checkpoint", type=Path, help="Base F5TTS_v1_Base .safetensors")
    parser.add_argument("--checkpoint-out", type=Path, help="Where to write the expanded checkpoint")
    parser.add_argument("--seed", type=int, default=666, help="Matches upstream's seed")
    args = parser.parse_args()

    vocab = read_vocab(args.base_vocab)
    print(f"base vocab: {len(vocab)} entries from {args.base_vocab}")
    if vocab[0] != " ":
        raise SystemExit("Index 0 of the base vocab must be a single space.")

    new_tokens = missing_tokens(vocab)
    print(f"missing Mongolian letters: {len(new_tokens)} -> "
          + " ".join(f"{c} U+{ord(c):04X}" for c in new_tokens))
    if not new_tokens:
        print("Nothing to add.")
        return

    extended = vocab + new_tokens
    write_vocab(args.out, extended)
    print(f"wrote {args.out}: {len(vocab)} -> {len(extended)} entries")

    # The prefix must survive untouched or every pretrained embedding is misaligned.
    check = read_vocab(args.out)
    assert check[: len(vocab)] == vocab, "base vocab prefix was altered"
    assert check[0] == " ", "space must stay at index 0"
    assert len(check) == len(extended), "round-trip length mismatch"
    print("verified: pretrained prefix intact, space at index 0")

    if args.checkpoint:
        if not args.checkpoint_out:
            raise SystemExit("--checkpoint requires --checkpoint-out")
        import torch

        torch.manual_seed(args.seed)
        print(f"expanding {args.checkpoint}")
        expand_embeddings(args.checkpoint, args.checkpoint_out, len(new_tokens), extended)


if __name__ == "__main__":
    main()
