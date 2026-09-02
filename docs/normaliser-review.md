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
**Measured cost: 3.33%.** 18,839 sentences from 400 Mongolian Wikipedia articles
(`wikimedia/wikipedia`, `20231101.mn`), of which 6,688 contain digits: 18,211
normalise, 628 are refused, 0 fail any other way. Every one of the 628 previously
produced a non-word. An earlier figure of 5.4% came from a 56-sentence fixture
and was too small to trust.

## What the normalization spec settled

The specification supplied forms this document was blocked on, and they are in
`numbers.py` verbatim:

| | |
|---|---|
| `1/2` | `хоёрны нэг` — not `хагас`, which is what the code used to emit |
| `3/4` | `дөрөвний гурав` |
| `5-аар` | `таваар` (distributive) |
| `5-уул` | `тавуул` (collective) |
| `5.1.2` | `тавын нэгийн хоёр` — genitives for 1, 3, 4, 5, 10 |

Note there are **two** genitive tables, confirmed as context-dependent rather
than a typo: `FRACTION_GENITIVE` has the n-stem form (`дөрөвний`) and
`REFERENCE_GENITIVE` the reduced one (`дөрвийн`).

## Two of the three questions are closed

1. **~~The decimal place word.~~** Closed. A decimal reads as a *fraction*:
   the whole part, the place as a genitive, then the digits, in that order and
   with no `бүхэл`. Section 4's `бүхэл` is superseded, and the induced
   place-word rule is gone rather than patched.

   | | |
   |---|---|
   | `3.14` | `гурав зууны арван дөрөв` |
   | `0.05` | `тэг зууны тав` |
   | `12.5%` | `арван хоёр арваны таван хувь` |

   Note the place word is `арваны` (`арван` + `ы`), matching `зууны`, not the
   spec's `аравны`.

2. **~~`1:2` versus `3:1`.~~** Closed, and simpler than the spec suggested:
   **a ratio reads as a fraction.** `1:2` is `хоёрны нэг` — the same string as
   `1/2` — and `3:1` is `нэгний гурав`. There is no sports-score reading, so
   nothing needs to tell them apart. `FRACTION_GENITIVE` gained `1: нэгний`.

3. **The `хорь` ablative — still open, and three sources could not settle it.**

   The evidence points both ways. Noun ablatives do surface the fleeting-n --
   `ямаа` → `ямаанаас`, `амьтан` → `амьтнаас` -- which is the pattern behind
   the spec's `хорьноос`. But this file's own table has `гурав` → `гурваас`,
   not `*гурванаас`, so numerals plainly do not follow the noun rule, and
   `хориос` fits that instead. Neither `mongoltoli.mn` nor `toli.gov.mn` lists
   a declined form for `хорь`, and Wiktionary has no case table for it.

   The table's `хориос` was chosen and stands. The question below is what a
   speaker needs to answer. Section 13 gives `2020-2024` as `... хорьноос ...`,
   while the verified `ABLATIVE` table has `хориос`. The table was chosen, so
   the spec line is treated as a slip — but the fleeting-n rule leans the other
   way, and if the spec is right the whole table needs the same review.

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

### Ordered by how much each row buys

Counted over the 628 real refusals, so this is the order to work in rather than
the alphabet. Ten suffixes cover 81%; **`-нд` alone covers 36%**.

| written suffix | refusals | cumulative | one example to answer |
|---|---:|---:|---|
| `-нд` | 220 | 36% | `нэг` + `-нд` = ? |
| `-аад` | 87 | 51% | `ная` + `-аад` = ? |
| `-ны` | 64 | 61% | `зургаа` + `-ны` = ? |
| `-н` | 26 | 65% | `нэг` + `-н` = ? |
| `-ний` | 21 | 69% | `нэг` + `-ний` = ? |
| `-д` | 21 | 72% | `зургаа` + `-д` = ? |
| `-ээд` | 16 | 75% | `ер` + `-ээд` = ? |
| `-с` | 13 | 77% | `зургаа` + `-с` = ? |
| `-т` | 13 | 79% | `хоёр` + `-т` = ? |
| `-тын` | 10 | 81% | `мянга` + `-тын` = ? |

`-аад` / `-ээд` / `-өөд` is the **approximative** ("about eighty"), not a case:
`80-аад` is read `наяад`. Wiktionary lists this form for some numerals
(`зуугаад`, `мянгаад`), so those rows may be fillable from a dictionary rather
than from memory.

There are **146 distinct (stem, suffix) pairs** in the whole sample, and **86 of
them cover 90%** of the refusals. Filling the top ten suffixes for the stems that
actually occur with them is a far smaller job than the full 24 × 7 grid, and it
is most of the benefit.

The families, for completeness:

| family | written as | share of refusals |
|---|---|---|
| dative-locative | `-д -т -нд` | **44% — start here** |
| approximative | `-аад -ээд -өөд` | 19%, partly in Wiktionary |
| genitive | `-ны -ний -ын -ийн` | 15% |
| accusative | `-ыг -ийг` | small in text, common in speech |
| ablative | `-аас -ээс -оос -өөс -иас -иос` | **already done** |
| instrumental | `-аар -ээр -оор -өөр` | small |
| comitative | `-тай -тэй -той` | small |

Two more answers are needed outside the table:

1. **Fractions.** `_fraction` handles only `1/2 → хагас`. What is the correct
   form of `3/4`, `2/3`, `1/5`? If it is the genitive of the cardinal
   denominator plus the numerator, the genitive rows above supply it and
   `_fraction` can be rebuilt on them.
2. **`нэг сая` vs `сая`.** `convert(1000000)` returns `сая`. Should a leading
   `нэг` be supplied for 1 000 000, 1 000 000 000, and for 100 and 1000?

## Re-measuring as you go

```bash
python scripts/measure_refusals.py              # Mongolian Wikipedia, ~19k sentences
python scripts/measure_refusals.py --corpus <dir>
```

Prints the refusal rate and regenerates the priority table above, so the rate
falls visibly as rows land and the ordering re-sorts itself. It samples
Wikipedia rather than the corpus for the headline number on purpose: the corpus
has already been through the audio gates, so measuring on it would report the
rate among clips that survived everything else, not the rate in the language.

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
