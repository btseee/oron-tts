# Numeral suffixes: the table a native speaker has to fill

`oron_tts/text/numbers.py` refuses to expand most numeral+suffix combinations.
This document is the reason and the remedy. Filling in the table below is the
single highest-value item in the project, and the only one that cannot be done
from the code.

## Why it refuses

The normaliser used to concatenate a written case suffix onto the numeral's
citation form. On ordinary Mongolian input that produces non-words:

| input | produced | |
|---|---|---|
| `20-иос` | `хорьиос` | ь is not dropped |
| `50-иас` | `тавьиас` | " |
| `20-ийг` | `хорьийг` | " |
| `100-д` | `зууд` | hidden -н lost |
| `1-нд` | `нэгнд` | no epenthetic vowel |
| `3/4` | `дөрөвдүгээрийн гурав` | an ordinal, not a part |
| `5-ын хувь` | `таван хувь` | the genitive deleted outright |
| `Жин нь 5 г.` | `Жин нь таван оны` | "г." shadowed the gram unit |

**This matters more here than in most projects.** The corpus text, the reference
CER is scored against, and the training target are deliberately one string. So a
wrong expansion is published, learned, *and* invisible to the CER gate — the
gate compares the audio to the corrupted string, so it agrees with itself.

Three generation rules were tried and each produced a non-word somewhere:

| rule | fixes | breaks |
|---|---|---|
| concatenate onto the citation form | — | `20-иос` → `хорьиос` |
| drop stem-final ь, then concatenate | `20-иос`, `50-иас`, `20-ийг`, `5-ын` | `20-аас` → `хораас` |
| use the attributive as the stem | `100-д` → `зуунд` | `3-ийн` → `гуравийн` |

The morphology is not recoverable from the citation/attributive pairs already in
the file. The stem interacts with the written suffix orthographically (`хори` +
`ийг` contracts to `хорийг`), the tens behave unlike the ones (`гуч` → `гучаас`,
not `гучиас`), and stems with an unstable vowel reduce (`гурав` → `гурваас`).

The existing `ABLATIVE` table is the proof that a table is the right answer: it
is hand-written verified data and it is correct wherever it applies. Note that
it is also **internally inconsistent** — `гурав → гурваас` and `дөрөв → дөрвөөс`
drop the unstable vowel, while `арав → араваас` keeps it, though `арав/арван`
alternates identically. One of those three is wrong; a speaker should say which.

So the normaliser raises `NumeralSuffixError` on anything not tabulated, and
`normalize(strict=True)` — the corpus path — turns that into a dropped clip.
**Measured cost: 3 of 56 real Mongolian sentences (5.4%)** in
`tests/fixtures/mn_text_sample.jsonl`. All three previously produced non-words.

## What is already verified, so you can skip it

The **ordinals are correct** and need no review. An earlier draft of this
document's parent review listed `convert_ordinal(100)` → `зуудугаар` as a
non-word; Wiktionary lists exactly that form, and `мянгадугаар` and
`хорьдугаар` too. The claim was wrong and is withdrawn.

| n | produced | Wiktionary |
|---|---|---|
| 20 | `хорьдугаар` | ✓ |
| 100 | `зуудугаар` | ✓ |
| 1000 | `мянгадугаар` | ✓ |

The **ablative table is correct** wherever it applies — it is hand-written
verified data and is what `SUFFIXED_FORMS` is seeded from. One entry is worth a
second look: `гурав → гурваас` and `дөрөв → дөрвөөс` drop the unstable
vowel, while `арав → араваас` keeps it, though `арав/арван` alternates
identically. Is `араваас` or `арваас` right?

What background reading did establish, and what it did not: the **fleeting-n**
(hidden-n) surfaces before the genitive, dative and ablative markers, which is
the right general shape. It does not tell you which allomorph each of the 24
numeral stems takes, and that is exactly the gap below.

## What to fill in

`SUFFIXED_FORMS` in `oron_tts/text/numbers.py`, keyed by numeral stem, then by
the suffix **as written after the hyphen**:

```python
SUFFIXED_FORMS["хорь"]["ийг"] = "хорийг"   # 20-ийг
SUFFIXED_FORMS["зуу"]["д"]    = "зуунд"    # 100-д
```

The 24 stems are the keys of `ABLATIVE`: тэг нэг хоёр гурав дөрөв тав зургаа
долоо найм ес арав хорь гуч дөч тавь жар дал ная ер зуу мянга сая тэрбум наяд.

The suffix families that matter, by frequency in the corpus:

| family | written as | example |
|---|---|---|
| genitive | `-ны -ний -ын -ийн` | `2024-ны`, `5-ын`, `3-ийн` |
| dative-locative | `-д -т -нд` | `100-д`, `15-нд`, `37-д` |
| accusative | `-ыг -ийг` | `20-ийг` |
| ablative | `-аас -ээс -оос -өөс -иас -иос` | **already done** |
| instrumental | `-аар -ээр -оор -өөр` | `5-аар` |
| comitative | `-тай -тэй -той` | `10-тай` |

Genitive and dative-locative alone cover the three refusals measured above.

Two more answers are needed outside the table:

1. **Fractions.** `_fraction` handles only `1/2 → хагас`. What is the correct
   form of `3/4`, `2/3`, `1/5`? If it is the genitive of the cardinal
   denominator plus the numerator, the genitive rows above supply it and
   `_fraction` can be rebuilt on them.
2. **`нэг сая` vs `сая`.** `convert(1000000)` returns `сая`. Should a leading
   `нэг` be supplied for 1 000 000, 1 000 000 000, and for 100 and 1000?

## How to check the answers

```bash
pytest tests/test_text_normalization.py -q
python - <<'PY'
from oron_tts.text import MongolianNormalizer
n = MongolianNormalizer()
for s in ["2024-ны", "15-нд", "1-ний", "3-ийн", "20-ийг", "100-д", "5-ын хувь"]:
    print(s, "->", n.normalize(s, strict=False))
PY
```

Every line that prints instead of raising is one more construction the corpus
keeps. Add a case to `test_a_tabulated_suffix_expands` for each, and remove it
from `test_an_untabulated_suffix_is_refused_not_guessed`.

**Do not fill these in by pattern-matching the examples above.** That is exactly
how the three broken rules were produced.

## Sources consulted

- [Trying to understand the "fleeting-n" in Mongolian](https://thelanguagecloset.com/2021/08/28/trying-to-understand-the-fleeting-n-in-mongolian/)
  — confirms the -н surfaces before genitive, dative and ablative markers.
- [Wiktionary: хорь](https://en.wiktionary.org/wiki/%D1%85%D0%BE%D1%80%D1%8C),
  [зуу](https://en.wiktionary.org/wiki/%D0%B7%D1%83%D1%83),
  [мянга](https://en.wiktionary.org/wiki/%D0%BC%D1%8F%D0%BD%D0%B3%D0%B0)
  — cardinal, attributive and ordinal forms. **No case declension tables**, which
  is why this document exists.
- [Монгол хэлний зөв бичих дүрмийн журамласан толь](https://toli.gov.mn/r)
  — the state orthographic dictionary. Documents the rules; does not give
  declined forms per word.
- [Mongolian numerals](https://en.wikipedia.org/wiki/Mongolian_numerals),
  [Numeral Morphology in Mongolian](https://lisatravis2012.wordpress.com/2016/11/21/numeral-morphology-in-mongolian/)
  — numerals are morphosyntactically distinct from ordinary nominals, which is
  why a general noun-declension rule does not transfer to them.
