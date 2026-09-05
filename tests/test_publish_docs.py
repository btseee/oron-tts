"""A publish that reports success and lands nothing has happened here before."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import publish_docs  # noqa: E402


class FakeSibling:
    def __init__(self, name): self.rfilename = name


class FakeInfo:
    def __init__(self, names): self.siblings = [FakeSibling(n) for n in names]


class FakeApi:
    def __init__(self, names): self._names = names
    def model_info(self, repo, **kw): return FakeInfo(self._names)


def test_missing_files_are_reported():
    api = FakeApi(["README.md", "model.safetensors"])
    missing = publish_docs.verify(api, "btsee/oron-tts",
                                  ["README.md", "tensorboard/cv/events.out.tfevents.1"])
    assert missing == ["tensorboard/cv/events.out.tfevents.1"]


def test_a_complete_upload_reports_nothing_missing():
    api = FakeApi(["README.md", "tensorboard/cv/events.out.tfevents.1"])
    assert publish_docs.verify(api, "btsee/oron-tts",
                               ["README.md", "tensorboard/cv/events.out.tfevents.1"]) == []
