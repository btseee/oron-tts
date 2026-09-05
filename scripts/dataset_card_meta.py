"""Add the descriptive metadata a dataset card needs, without rewriting it.

The four published corpora carry accurate, generated cards -- clips, hours,
splits, gates, every column typed from the shipped data. What they lack is the
metadata the Hub filters and searches on, and any link to the model they exist to
train. Both are additive, so this edits in place rather than regenerating: those
cards cost a GPU pass to produce and their numbers are correct.

    python scripts/dataset_card_meta.py --repo btsee/fleurs-mn --used-by btsee/oron-tts
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Every corpus here is one language, read or spoken by many people, transcribed
# by whoever produced the upstream release rather than by this project.
DEFAULTS = {
    "annotations_creators": ["found"],
    "language_creators": ["crowdsourced"],
    "multilinguality": "monolingual",
    "source_datasets": ["original"],
    "task_ids": ["text-to-speech"],
}

USED_BY_HEADING = "## Used by"


def split_card(card: str) -> tuple[dict, str]:
    """Frontmatter as a dict, and the body after it."""
    import yaml

    if not card.startswith("---\n"):
        return {}, card
    end = card.index("\n---\n", 3)
    return yaml.safe_load(card[4:end]) or {}, card[end + 5:]


def enrich(card: str, *, used_by: str | None, note: str | None) -> str:
    """Add missing keys, a link to the model, and an optional note. Idempotent."""
    import yaml

    meta, body = split_card(card)
    for key, value in DEFAULTS.items():
        meta.setdefault(key, value)

    if used_by and USED_BY_HEADING not in body:
        body = body.rstrip() + (
            f"\n\n{USED_BY_HEADING}\n\n"
            f"[{used_by}](https://huggingface.co/{used_by}) — Mongolian "
            f"text-to-speech, trained on this corpus.\n")
    if note and note not in body:
        body = body.rstrip() + f"\n\n{note}\n"

    return "---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True,
                                    default_flow_style=False) + "---\n" + body


def main() -> None:
    import os

    from huggingface_hub import HfApi, hf_hub_download

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--used-by", default=None)
    parser.add_argument("--note", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    path = Path(hf_hub_download(args.repo, "README.md", repo_type="dataset",
                                token=token, force_download=True))
    updated = enrich(path.read_text(encoding="utf-8"),
                     used_by=args.used_by, note=args.note)
    if args.dry_run:
        print(updated[:updated.index("\n---\n", 3) + 5])
        return
    path.write_text(updated, encoding="utf-8")
    api.upload_file(path_or_fileobj=str(path), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="dataset",
                    commit_message="Add descriptive metadata and link the model")
    print(f"  updated {args.repo}")


if __name__ == "__main__":
    main()
