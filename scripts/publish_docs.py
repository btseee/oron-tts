"""Push the regenerated card and TensorBoard tree, then read the server back.

`verify` exists because this project has published an artifact that the upload
call reported as successful and that was not there -- and separately, a dataset
whose upload succeeded and whose content was 19 clips. The server's view is the
only view that counts.

The TensorBoard replacement uploads the new tree before removing anything that
used to be there. Deleting first and uploading second means an empty or
partial `--tensorboard` directory can leave the server with nothing at all
while the upload call still returns normally -- that has happened here too.

    python scripts/publish_docs.py --card README.md --tensorboard tensorboard
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO = "btsee/oron-tts"


def verify(api, repo: str, expected: list[str]) -> list[str]:
    """Paths that are not on the server, in the order they were expected."""
    present = {s.rfilename for s in api.model_info(repo, files_metadata=False).siblings}
    return [path for path in expected if path not in present]


def local_files(tensorboard_dir: Path) -> list[str]:
    """Repo-relative paths for every file under `tensorboard_dir`, sorted.

    Refuses rather than returning an empty list. An empty result here is
    exactly the condition that let a previous run delete the remote
    TensorBoard tree, upload nothing in its place, and still report success --
    `upload_folder` raises on nothing when there is nothing to commit.
    """
    if not tensorboard_dir.is_dir():
        raise SystemExit(f"--tensorboard {tensorboard_dir} is not a directory")
    files = ["tensorboard/" + p.relative_to(tensorboard_dir).as_posix()
             for p in sorted(tensorboard_dir.rglob("*")) if p.is_file()]
    if not files:
        raise SystemExit(f"--tensorboard {tensorboard_dir} contains no files")
    return files


def stale_paths(present: list[str], expected: list[str]) -> list[str]:
    """Paths under `tensorboard/` on the server that the new upload does not cover.

    Scoped to `tensorboard/` so a stray README or model weight sibling is never
    a delete candidate -- this function only ever cleans up the tree it just
    replaced.
    """
    expected_set = set(expected)
    return [path for path in present
            if path.startswith("tensorboard/") and path not in expected_set]


def publish_tensorboard(api, repo: str, tensorboard_dir: Path) -> list[str]:
    """Replace the remote TensorBoard tree without ever leaving it empty.

    Upload first, delete stale paths second: a failure between the two steps
    leaves the server holding either the old tree or the new one, never
    neither. The stale-path listing is read *before* the upload so it reflects
    what needs cleaning up, not what was just added.
    """
    files = local_files(tensorboard_dir)  # refuses before any network call
    present = {s.rfilename for s in api.model_info(repo, files_metadata=False).siblings}
    api.upload_folder(folder_path=str(tensorboard_dir), path_in_repo="tensorboard",
                      repo_id=repo, commit_message="Publish per-stage TensorBoard runs")
    # Sorted rather than left in set order: a failed run should delete the
    # same paths in the same order every time, so its log is reproducible.
    for path in stale_paths(sorted(present), files):
        api.delete_file(path_in_repo=path, repo_id=repo,
                        commit_message="Remove stale TensorBoard file")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--card", type=Path, help="README.md to upload")
    parser.add_argument("--tensorboard", type=Path, help="tensorboard/ directory to upload")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be uploaded and, more to the point, "
                             "what would be deleted. Changes nothing.")
    args = parser.parse_args()

    # A run with neither flag touches nothing and would still print "verified
    # 0 files" -- indistinguishable from a real publish unless it refuses here.
    if not args.card and not args.tensorboard:
        raise SystemExit("nothing to publish: pass --card and/or --tensorboard")

    # Deferred: CI does not install huggingface_hub, and check_ci_imports.py
    # enforces that statically by walking module-scope imports only.
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    expected: list[str] = []

    if args.dry_run:
        # Reads the server and writes nothing. The delete list is the reason
        # this flag exists and it cannot be derived without knowing what is
        # already there, so the read is not optional -- the sibling
        # dataset_card_meta.py --dry-run downloads the card for the same reason.
        files = local_files(args.tensorboard) if args.tensorboard else []
        present = sorted(s.rfilename for s in
                         api.model_info(args.repo, files_metadata=False).siblings)
        for path in (["README.md"] if args.card else []) + files:
            print(f"  + {path}")
        for path in (stale_paths(present, files) if args.tensorboard else []):
            print(f"  - {path}")
        print(f"  dry run: nothing was uploaded or deleted on {args.repo}")
        return

    if args.card:
        api.upload_file(path_or_fileobj=str(args.card), path_in_repo="README.md",
                        repo_id=args.repo, commit_message="Rewrite the card around usage")
        expected.append("README.md")

    if args.tensorboard:
        expected += publish_tensorboard(api, args.repo, args.tensorboard)

    missing = verify(api, args.repo, expected)
    if missing:
        raise SystemExit(f"publish VERIFY FAILED, missing: {missing}")
    print(f"  verified {len(expected)} files on {args.repo}")


if __name__ == "__main__":
    main()
