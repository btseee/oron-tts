# Lexicons

Lookup tables the normaliser cannot derive. Each is a two-column TSV:
`source<TAB>spoken`, `#` starts a comment, blank lines ignored.

**These are frames, seeded from the normalization spec. They are meant to be
extended by someone who speaks Mongolian.** An entry that is not listed is
*not* guessed at — see each file's header for what happens instead. That is the
same discipline as `SUFFIXED_FORMS`: in this project the corpus text, the CER
reference and the training target are one string, so a wrong expansion is
published, scored against itself, and learned.

| file | what it holds | absent entry falls back to |
| --- | --- | --- |
| `abbreviations.tsv` | УИХ, МУИС, д-р, проф. | spelling out, letter by letter |
| `latin_letters.tsv` | Latin letter names (`a` → эй) | the letter is left as-is |
| `cyrillic_letters.tsv` | Cyrillic letter names (`б` → бэ) | the letter is left as-is |
| `foreign_words.tsv` | Scholz, Google, Frankfurt, project | the word is left in Latin |
| `emoji.tsv` | ❤️ → улаан зүрх | the emoji is dropped |
| `units.tsv` | км, м², kW | the unit is left as-is |
| `reference_words.tsv` | book names that make `3:16` a verse, not a time | it is read as a time |

## Checking your additions

```bash
python scripts/check_lexicons.py     # format, duplicates, vocabulary coverage
pytest tests/test_lexicons.py -q
```

The vocabulary check matters: a spoken form containing a character absent from
`data/oron_mn_pinyin/vocab.txt` becomes a **space** at training time, silently.
