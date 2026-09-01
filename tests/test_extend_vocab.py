"""Text-embedding expansion.

Every pretrained embedding row is addressed by position, so the surgery has one
absolute requirement: rows 0..2545 must come out bit-identical. A silent
misalignment here would look like a model that simply trains badly.

The checkpoint tests are skipped when the 1.35 GB base model is not present.
"""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BASE_CKPT = REPO / "ckpts" / "F5TTS_v1_Base" / "model_1250000.safetensors"
EMBED_KEY = "ema_model.transformer.text_embed.text_embed.weight"

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

pytestmark = pytest.mark.skipif(
    not BASE_CKPT.exists(), reason=f"{BASE_CKPT.name} not downloaded"
)


@pytest.fixture(scope="module")
def expanded(tmp_path_factory):
    """Run the real surgery once, into a temporary file."""
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    from extend_vocab import expand_embeddings, missing_tokens, read_vocab

    base_vocab = read_vocab(REPO / ".." / "F5-TTS" / "data" / "Emilia_ZH_EN_pinyin" / "vocab.txt")
    new_tokens = missing_tokens(base_vocab)
    out = tmp_path_factory.mktemp("ckpt") / "expanded.safetensors"
    torch.manual_seed(666)
    expand_embeddings(BASE_CKPT, out, len(new_tokens), base_vocab + new_tokens)
    return out, len(new_tokens)


@pytest.fixture(scope="module")
def tensors(expanded):
    from safetensors.torch import load_file

    out, n_new = expanded
    return load_file(str(BASE_CKPT), device="cpu"), load_file(str(out), device="cpu"), n_new


def test_embedding_grows_by_exactly_the_new_token_count(tensors):
    base, new, n_new = tensors
    assert base[EMBED_KEY].shape[0] + n_new == new[EMBED_KEY].shape[0]
    assert base[EMBED_KEY].shape[1] == new[EMBED_KEY].shape[1]


def test_pretrained_rows_are_bit_identical(tensors):
    """The whole point. Any drift here misaligns all 2545 pretrained tokens."""
    base, new, _ = tensors
    n = base[EMBED_KEY].shape[0]
    assert torch.equal(new[EMBED_KEY][:n], base[EMBED_KEY])


def test_no_other_tensor_is_touched(tensors):
    base, new, _ = tensors
    assert set(new) == set(base)
    for key in base:
        if key != EMBED_KEY:
            assert torch.equal(new[key], base[key]), key


def test_new_rows_match_the_pretrained_scale(tensors):
    """Upstream seeds with randn (std 1.0); measured rows are std ~0.63.

    A randn row is ~1.6x too long, so the first gradient steps are spent pulling
    it back toward the model rather than learning the character.
    """
    base, new, n_new = tensors
    n = base[EMBED_KEY].shape[0]
    added = new[EMBED_KEY][n:]
    pretrained_norm = base[EMBED_KEY].norm(dim=1).mean()
    added_norm = added.norm(dim=1).mean()
    assert added_norm == pytest.approx(pretrained_norm, rel=0.25)
    # And comfortably below what randn would produce.
    assert added_norm < (base[EMBED_KEY].shape[1] ** 0.5) * 0.8


def test_new_rows_are_distinct(tensors):
    """Identical rows would make the new letters indistinguishable to the model."""
    _base, new, n_new = tensors
    added = new[EMBED_KEY][-n_new:]
    for i in range(n_new):
        for j in range(i + 1, n_new):
            assert not torch.equal(added[i], added[j])


def test_pt_checkpoints_are_refused(tmp_path):
    """Upstream's .pt path leaves model_state_dict unexpanded and fails on load."""
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    from extend_vocab import expand_embeddings

    fake = tmp_path / "model.pt"
    fake.write_bytes(b"not a real checkpoint")
    with pytest.raises(SystemExit, match="safetensors"):
        expand_embeddings(fake, tmp_path / "out.pt", 5, [" "] * 2550)
