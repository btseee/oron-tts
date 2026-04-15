---
mode: agent
description: Work with the OronTTS CyrillicTokenizer and TextCleaner — encode/decode text, understand the vocab, add new tokens, or integrate into a custom pipeline.
---

# OronTTS Tokenizer & Text Cleaning

Use this skill when the user asks about text encoding, the tokenizer vocabulary, special tokens, language tags, or `TextCleaner`.

## CyrillicTokenizer

Character-level tokenizer. **Vocab size: 65** (fixed — must match `model.vocab_size` in YAML configs).

```python
from src.utils.tokenizer import CyrillicTokenizer

tok = CyrillicTokenizer()

# Encode with language tag and attribute prefix
ids = tok.encode("сайн байна уу", lang="mn", attr_tokens=["[FEMALE]", "[YOUNG]"])
# → [1, 6, 8, 4, <char ids...>, 2]
#    BOS FEMALE YOUNG LANG_MN  ... EOS

# Decode back to string (special tokens stripped)
text = tok.decode(ids)
# → "сайн байна уу"

print(tok.vocab_size)  # 65
```

## Token layout

| ID | Token | Description |
|----|-------|-------------|
| 0 | `<PAD>` | Padding |
| 1 | `<BOS>` | Begin of sequence |
| 2 | `<EOS>` | End of sequence |
| 3 | `<UNK>` | Unknown character |
| 4 | `[LANG_MN]` | Mongolian language tag |
| 5 | `[LANG_KZ]` | Kazakh language tag |
| 6 | `[FEMALE]` | Female speaker attribute |
| 7 | `[MALE]` | Male speaker attribute |
| 8 | `[YOUNG]` | Young speaker attribute |
| 9 | `[MIDDLE]` | Middle-aged speaker attribute |
| 10 | `[ELDERLY]` | Elderly speaker attribute |
| 11–45 | Mongolian Cyrillic lowercase (35 chars) | а–я + ө, ү |
| 46–52 | Kazakh-specific: `ә ғ қ ң ұ һ і` | |
| 53–64 | Punctuation + space | ` . , ! ? - : ; " ' ( )` |

**Encoding order**: BOS → attr_tokens → LANG_TAG → char ids → EOS

## TextCleaner

Wraps `CyrillicTokenizer` and handles text normalisation before encoding.

```python
from src.utils.text_cleaner import TextCleaner

cleaner = TextCleaner()

# Clean only (returns str)
clean = cleaner.clean("Энэ 2024 онд болсон.")
# → "энэ хоёр мянга хорин дөрөв онд болсон."

# Clean + encode (returns list[int])
ids = cleaner.text_to_sequence(
    "Энэ 2024 онд болсон.",
    lang="mn",
    attr_tokens=["[FEMALE]"],
)

print(cleaner.vocab_size)  # 67
```

Normalisation pipeline inside `clean()`:
1. Unicode NFC
2. Punctuation map (`–` → `-`, `«»` → `""`, etc.)
3. Abbreviation expansion (`км` → `километр`, etc.)
4. Number expansion via `NumberNormalizer`
5. Remove chars outside `ALLOWED_CHARS`
6. Collapse whitespace
7. Lowercase

## Adding new attribute tokens

If you extend the vocabulary, you **must**:
1. Add the new token string to `SPECIAL_TOKENS` in `tokenizer.py` (before the character vocab).
2. Increment `vocab_size` — it is `len(_VOCAB)`, computed automatically.
3. Update `model.vocab_size` in both YAML configs to match the new total.
4. Re-initialise or resize the text embedding layer in `TextConvEmbed` and `DiT`.

## Language support

| `lang=` | Script | Extra chars |
|---------|--------|-------------|
| `mn` | Mongolian Khalkha Cyrillic | standard Cyrillic + ө ү |
| `kz` | Kazakh Cyrillic | + ә ғ қ ң ұ һ і |

Pass `lang` to both `CyrillicTokenizer.encode()` and `TextCleaner.text_to_sequence()`.
