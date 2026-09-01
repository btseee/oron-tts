"""The extended vocab must cover every character the corpus can produce.

`f5_tts.model.utils.list_str_to_idx` maps any character absent from vocab.txt to
index 0 -- and index 0 is the SPACE token, not an <unk>. Coverage gaps are
therefore silent: training simply sees spaces where letters should be, and
nothing anywhere reports it.

On the base F5-TTS vocab that failure rate is 4.90% of all tokens, because
Ө ө Ү ү are missing and they are ordinary Mongolian vowels. These tests exist so
that regression cannot happen again unnoticed.
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BASE_VOCAB = REPO / ".." / "F5-TTS" / "data" / "Emilia_ZH_EN_pinyin" / "vocab.txt"
MN_VOCAB = REPO / "data" / "oron_mn_pinyin" / "vocab.txt"
FIXTURE = Path(__file__).parent / "fixtures" / "mn_text_sample.jsonl"

MN_ALPHABET = "абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя"


def _load_f5_utils():
    """Load upstream's model/utils.py directly, bypassing the f5_tts package.

    `import f5_tts.model.utils` executes the package __init__, which pulls the
    whole training stack (wandb, vocos, torchdiffeq, x_transformers). The
    function under test needs only rjieba and pypinyin, so loading the real
    source file keeps this a test of upstream's actual code without demanding a
    training environment.
    """
    import importlib.util

    src = REPO / ".." / "F5-TTS" / "src" / "f5_tts" / "model" / "utils.py"
    if not src.exists():
        pytest.skip(f"upstream F5-TTS not checked out at {src}")
    pytest.importorskip("rjieba", reason="rjieba not installed")
    pytest.importorskip("pypinyin", reason="pypinyin not installed")

    spec = importlib.util.spec_from_file_location("_f5_utils", src)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"could not load upstream utils.py: {exc}")
    return module


def read_vocab(path: Path) -> list[str]:
    """Read vocab.txt exactly the way get_tokenizer does."""
    with open(path, encoding="utf-8") as f:
        return [line[:-1] for line in f]


def fixture_texts() -> list[str]:
    with open(FIXTURE, encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f if line.strip()]


@pytest.fixture(scope="module")
def mn_vocab() -> list[str]:
    if not MN_VOCAB.exists():
        pytest.fail(f"{MN_VOCAB} missing -- run scripts/extend_vocab.py")
    return read_vocab(MN_VOCAB)


def test_space_is_index_zero(mn_vocab):
    # get_tokenizer asserts this for pinyin/char but NOT for custom, and index 0
    # doubles as the out-of-vocabulary target.
    assert mn_vocab[0] == " "


def test_pretrained_prefix_is_untouched(mn_vocab):
    # Every pretrained embedding row is addressed by position. Reordering or
    # regenerating the base entries silently misaligns all 2545 of them.
    base = read_vocab(BASE_VOCAB)
    assert mn_vocab[: len(base)] == base
    assert len(mn_vocab) == len(base) + 5


def test_no_duplicate_entries(mn_vocab):
    # A duplicate makes one embedding row permanently unreachable.
    assert len(mn_vocab) == len(set(mn_vocab))


def test_file_ends_with_newline():
    # get_tokenizer does line[:-1]; a final line without "\n" loses its last char.
    assert MN_VOCAB.read_bytes().endswith(b"\n")


def test_every_mongolian_letter_is_present(mn_vocab):
    have = set(mn_vocab)
    missing = [c for c in MN_ALPHABET + MN_ALPHABET.upper() if c not in have]
    assert not missing, "absent letters silently become spaces: " + " ".join(
        f"{c} U+{ord(c):04X}" for c in missing
    )


def test_corpus_charset_is_covered(mn_vocab):
    """The real check: nothing in actual corpus text falls through to index 0."""
    have = set(mn_vocab)
    texts = fixture_texts()
    assert texts, "fixture is empty"

    # Mirrors convert_char_to_pinyin's custom_trans, applied before indexing.
    pre = str.maketrans({";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"})
    # Normalisation oron-tts is responsible for applying before tokenization.
    pre.update(str.maketrans({"\xa0": " "}))

    uncovered = sorted(
        {c for t in texts for c in t.translate(pre) if c not in have}
    )
    assert not uncovered, "would be replaced by spaces: " + " ".join(
        f"{c!r} U+{ord(c):04X}" for c in uncovered
    )


def test_base_vocab_would_fail_this(mn_vocab):
    """Guard the guard: prove the corpus check is actually load-bearing."""
    base = set(read_vocab(BASE_VOCAB))
    texts = fixture_texts()
    n_oov = sum(1 for t in texts for c in t if c not in base and not c.isspace())
    n_total = sum(len(t) for t in texts)
    assert n_oov > 0, "if the base vocab covers everything, this test suite is vacuous"
    assert n_oov / n_total > 0.01, f"expected a material OOV rate, got {n_oov / n_total:.2%}"


def test_tokenizer_roundtrip_preserves_every_character():
    """Cyrillic must survive convert_char_to_pinyin unchanged.

    Cyrillic is 2 bytes/char, so segments match neither the pure-ASCII branch
    (len == byte_len) nor the pure-CJK branch (3*len == byte_len) and fall
    through to the verbatim else-branch. Skipped where f5_tts is not installed.
    """
    utils = _load_f5_utils()
    pre = str.maketrans({";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"})

    texts = fixture_texts()
    for text, tokens in zip(texts, utils.convert_char_to_pinyin(texts), strict=True):
        expected = [c for c in text.translate(pre) if not c.isspace()]
        actual = [t for t in tokens if not t.isspace()]
        assert actual == expected, f"tokenization altered: {text[:60]!r}"
