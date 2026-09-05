"""Push the regenerated card and TensorBoard tree, then read the server back.

`verify` exists because this project has published an artifact that the upload
call reported as successful and that was not there -- and separately, a dataset
whose upload succeeded and whose content was 19 clips. The server's view is the
only view that counts.

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


def main() -> None:
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--card", type=Path, help="README.md to upload")
    parser.add_argument("--tensorboard", type=Path, help="tensorboard/ directory to upload")
    parser.add_argument("--repo", default=REPO)
    args = parser.parse_args()

    api = HfApi(token=os.environ["HF_TOKEN"])
    expected: list[str] = []

    if args.card:
        api.upload_file(path_or_fileobj=str(args.card), path_in_repo="README.md",
                        repo_id=args.repo, commit_message="Rewrite the card around usage")
        expected.append("README.md")

    if args.tensorboard:
        # The old flat files are replaced wholesale: a stale events file beside
        # the new tree would show up as an unnamed extra run.
        api.delete_folder(path_in_repo="tensorboard", repo_id=args.repo,
                          commit_message="Replace the flat TensorBoard files")
        api.upload_folder(folder_path=str(args.tensorboard), path_in_repo="tensorboard",
                          repo_id=args.repo, commit_message="Publish per-stage TensorBoard runs")
        expected += ["tensorboard/" + p.relative_to(args.tensorboard).as_posix()
                     for p in sorted(args.tensorboard.rglob("*")) if p.is_file()]

    missing = verify(api, args.repo, expected)
    if missing:
        raise SystemExit(f"publish VERIFY FAILED, missing: {missing}")
    print(f"  verified {len(expected)} files on {args.repo}")


if __name__ == "__main__":
    main()
