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
    # No `task_ids`. On the Hub a task_id is a *subtask* of a task_category, and
    # the validated list has none for text-to-speech -- its audio entries are
    # keyword-spotting, speaker-identification and the audio classification
    # tasks, none of which describe this. `task_ids: [text-to-speech]` was
    # published on all four cards and the Hub warned on every one of them.
    # `task_categories: [text-to-speech]`, which the cards already carry, is the
    # correct and sufficient declaration.
}

# Keys published in error, removed on the next run. `enrich` only ever adds, so
# a bad key would otherwise outlive the fix on every card already carrying it.
INVALID_KEYS = ("task_ids",)

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
    for key in INVALID_KEYS:
        meta.pop(key, None)
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


def missing_from(card: str, *, used_by: str | None, note: str | None) -> list[str]:
    """Everything this script should have added that the given card does not carry.

    Read against the *server's* copy after the upload. This project has twice
    published something an upload call reported as successful and that was not
    there, so "the call returned" is not evidence that the card changed.
    """
    meta, body = split_card(card)
    missing = [key for key in DEFAULTS if key not in meta]
    if used_by and (USED_BY_HEADING not in body
                    or f"https://huggingface.co/{used_by}" not in body):
        missing.append(USED_BY_HEADING)
    if note and note not in body:
        missing.append("the note")
    return missing


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

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set")
    api = HfApi(token=token)

    def from_server() -> tuple[Path, str]:
        path = Path(hf_hub_download(args.repo, "README.md", repo_type="dataset",
                                    token=token, force_download=True))
        return path, path.read_text(encoding="utf-8")

    path, current = from_server()
    updated = enrich(current, used_by=args.used_by, note=args.note)
    if args.dry_run:
        print(updated[:updated.index("\n---\n", 3) + 5])
        return
    if updated == current:
        # enrich is idempotent, so a second run has nothing to add. Uploading
        # anyway spends a commit on the dataset's history saying nothing.
        print(f"  {args.repo} already carries all of it; nothing uploaded")
        return
    path.write_text(updated, encoding="utf-8")
    api.upload_file(path_or_fileobj=str(path), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="dataset",
                    commit_message="Add descriptive metadata and link the model")
    missing = missing_from(from_server()[1], used_by=args.used_by, note=args.note)
    if missing:
        raise SystemExit(f"{args.repo} VERIFY FAILED, not on the server: {missing}")
    print(f"  updated {args.repo}, verified from the server")


if __name__ == "__main__":
    main()
