"""A publish that reports success and lands nothing has happened here before."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import publish_docs  # noqa: E402


class FakeSibling:
    def __init__(self, name): self.rfilename = name


class FakeInfo:
    def __init__(self, names): self.siblings = [FakeSibling(n) for n in names]


class FakeApi:
    """Records every call so a test can assert on what was, or was not, done."""

    def __init__(self, names):
        self._names = names
        self.deleted: list[str] = []
        self.uploaded_folders: list[str] = []

    def model_info(self, repo, **kw):
        return FakeInfo(self._names)

    def upload_folder(self, **kw):
        self.uploaded_folders.append(kw["folder_path"])

    def delete_file(self, **kw):
        self.deleted.append(kw["path_in_repo"])

    def delete_folder(self, **kw):
        raise AssertionError("delete_folder must not be called: it can empty the tree")


def test_missing_files_are_reported():
    api = FakeApi(["README.md", "model.safetensors"])
    missing = publish_docs.verify(api, "btsee/oron-tts",
                                  ["README.md", "tensorboard/cv/events.out.tfevents.1"])
    assert missing == ["tensorboard/cv/events.out.tfevents.1"]


def test_a_complete_upload_reports_nothing_missing():
    api = FakeApi(["README.md", "tensorboard/cv/events.out.tfevents.1"])
    assert publish_docs.verify(api, "btsee/oron-tts",
                               ["README.md", "tensorboard/cv/events.out.tfevents.1"]) == []


def test_empty_tensorboard_dir_refuses_before_any_delete_or_upload(tmp_path):
    # The regression test for the critical finding: delete_folder-then-upload
    # let an empty directory silently wipe the remote tree. If that ordering
    # comes back, local_files stops refusing early and this starts failing.
    empty = tmp_path / "tensorboard"
    empty.mkdir()
    api = FakeApi(["README.md", "tensorboard/old/events.out.tfevents.1"])

    with pytest.raises(SystemExit, match="contains no files"):
        publish_docs.publish_tensorboard(api, "btsee/oron-tts", empty)

    assert api.deleted == []
    assert api.uploaded_folders == []


def test_nonexistent_tensorboard_dir_refuses_the_same_way(tmp_path):
    missing = tmp_path / "does-not-exist"
    api = FakeApi(["README.md"])

    with pytest.raises(SystemExit, match="is not a directory"):
        publish_docs.publish_tensorboard(api, "btsee/oron-tts", missing)

    assert api.deleted == []
    assert api.uploaded_folders == []


def test_stale_paths_are_the_old_flat_files_only():
    present = ["README.md", "model.safetensors",
               "tensorboard/events.out.tfevents.cv",
               "tensorboard/events.out.tfevents.mn",
               "tensorboard/events.out.tfevents.mbspeech",
               "tensorboard/events.out.tfevents.female"]
    expected = ["tensorboard/stage1-cv/events.out.tfevents.1",
                "tensorboard/stage2-mn/events.out.tfevents.1"]

    stale = publish_docs.stale_paths(present, expected)

    assert stale == ["tensorboard/events.out.tfevents.cv",
                      "tensorboard/events.out.tfevents.mn",
                      "tensorboard/events.out.tfevents.mbspeech",
                      "tensorboard/events.out.tfevents.female"]
    assert "README.md" not in stale
    assert "model.safetensors" not in stale
    for path in expected:
        assert path not in stale


def test_publish_tensorboard_uploads_before_deleting_stale_paths(tmp_path):
    tb = tmp_path / "tensorboard"
    (tb / "stage1-cv").mkdir(parents=True)
    (tb / "stage1-cv" / "events.out.tfevents.1").write_text("x")

    api = FakeApi(["README.md", "tensorboard/events.out.tfevents.cv"])

    files = publish_docs.publish_tensorboard(api, "btsee/oron-tts", tb)

    assert files == ["tensorboard/stage1-cv/events.out.tfevents.1"]
    assert api.uploaded_folders == [str(tb)]
    assert api.deleted == ["tensorboard/events.out.tfevents.cv"]


def test_no_flags_refuses(monkeypatch):
    # Neither --card nor --tensorboard: expected would stay empty and the
    # script would print "verified 0 files" for a run that touched nothing.
    monkeypatch.setattr(sys, "argv", ["publish_docs.py"])
    with pytest.raises(SystemExit, match="nothing to publish"):
        publish_docs.main()
