"""How much text the numeral normaliser currently refuses, and what would fix it.

The normaliser raises on numeral suffixes it cannot expand without guessing (see
docs/normaliser-review.md). That is the right trade -- a dropped clip costs less
than a published non-word -- but the size of the trade should be a measurement,
not an impression, and it should shrink visibly as `SUFFIXED_FORMS` is filled in.

Run it after adding rows to see the rate fall and the priority list reorder:

    python scripts/measure_refusals.py                       # Mongolian Wikipedia
    python scripts/measure_refusals.py --corpus <dir>        # a built corpus
    python scripts/measure_refusals.py --articles 1000       # tighter estimate

Sampling Wikipedia rather than the corpus is deliberate for the headline number:
the corpus has already been through the audio gates, so measuring on it would
report the refusal rate among clips that survived everything else, not the rate
in the language.
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from oron_tts.text import MongolianNormalizer  # noqa: E402
from oron_tts.text.numbers import NumeralSuffixError  # noqa: E402

MN_RE = re.compile(r"[а-яөүёА-ЯӨҮЁ]")
CAUSE_RE = re.compile(r"for '([^']+)' \+ -(\S+)")
MIN_LEN, MAX_LEN = 20, 300


def from_wikipedia(articles: int) -> list[str]:
    from datasets import load_dataset

    stream = load_dataset(
        "wikimedia/wikipedia", "20231101.mn", split="train", streaming=True
    )
    out: list[str] = []
    for i, row in enumerate(stream):
        if i >= articles:
            break
        for line in row["text"].split("\n"):
            for sentence in re.split(r"(?<=[.!?])\s+", line.strip()):
                sentence = sentence.strip()
                if MIN_LEN <= len(sentence) <= MAX_LEN and MN_RE.search(sentence):
                    out.append(sentence)
    return out


def from_corpus(corpus: Path) -> list[str]:
    path = corpus / "manifest.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found.")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=None,
                    help="Measure a built corpus instead of Wikipedia")
    ap.add_argument("--articles", type=int, default=400,
                    help="Wikipedia articles to sample (default 400, ~19k sentences)")
    ap.add_argument("--top", type=int, default=10, help="Suffixes to list")
    args = ap.parse_args()

    sentences = from_corpus(args.corpus) if args.corpus else from_wikipedia(args.articles)
    if not sentences:
        raise SystemExit("No sentences to measure.")

    normalizer = MongolianNormalizer()
    by_suffix: collections.Counter[str] = collections.Counter()
    by_pair: collections.Counter[tuple[str, str]] = collections.Counter()
    example: dict[str, str] = {}
    ok = fractions = other = 0

    for sentence in sentences:
        try:
            normalizer.normalize(sentence, strict=False)
            ok += 1
        except NumeralSuffixError as exc:
            message = str(exc)
            if "fraction" in message:
                fractions += 1
                continue
            match = CAUSE_RE.search(message)
            if match:
                stem, suffix = match.group(1), match.group(2)
                by_suffix[suffix] += 1
                by_pair[(stem, suffix)] += 1
                example.setdefault(suffix, stem)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            other += 1
            example.setdefault(f"!{type(exc).__name__}", sentence[:60])

    total = len(sentences)
    refused = sum(by_suffix.values()) + fractions
    print(f"sentences            {total:,}")
    print(f"  normalised         {ok:,}  ({100 * ok / total:.2f}%)")
    print(f"  refused            {refused:,}  ({100 * refused / total:.2f}%)")
    print(f"    numeral suffix   {sum(by_suffix.values()):,}")
    print(f"    fraction         {fractions:,}")
    if other:
        print(f"  OTHER failures     {other:,}  <- not expected; investigate")

    if not by_suffix:
        print("\nNothing refused. SUFFIXED_FORMS covers this sample.")
        return

    print("\nby written suffix -- fill these in this order:")
    print(f"  {'suffix':<10}{'refusals':>9}{'cumulative':>12}   example to answer")
    running = 0
    for suffix, count in by_suffix.most_common(args.top):
        running += count
        print(f"  -{suffix:<9}{count:>9}{100 * running / sum(by_suffix.values()):>11.0f}%"
              f"   {example[suffix]} + -{suffix} = ?")

    pairs = by_pair.most_common()
    needed = next(
        (i + 1 for i in range(len(pairs))
         if sum(c for _, c in pairs[: i + 1]) >= 0.9 * sum(by_pair.values())),
        len(pairs),
    )
    print(f"\n{len(by_pair)} distinct (stem, suffix) pairs; "
          f"{needed} of them cover 90% of the refusals.")


if __name__ == "__main__":
    main()
