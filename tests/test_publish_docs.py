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
    """Records every call so a test can assert on what was, or was not, done.

    `calls` is one ordered log across all methods, so a test can check
    *relative* order between e.g. an upload and a later delete -- the
    per-method lists below only ever show final membership, which can't
    distinguish "uploaded then deleted" from "deleted then uploaded".
    """

    def __init__(self, names):
        self._names = names
        self.calls: list[tuple] = []
        self.deleted: list[str] = []
        self.uploaded_folders: list[str] = []
        self.uploaded_files: list[str] = []

    def model_info(self, repo, **kw):
        self.calls.append(("model_info", repo))
        return FakeInfo(self._names)

    def upload_folder(self, **kw):
        self.calls.append(("upload_folder", kw["folder_path"]))
        self.uploaded_folders.append(kw["folder_path"])

    def delete_file(self, **kw):
        self.calls.append(("delete_file", kw["path_in_repo"]))
        self.deleted.append(kw["path_in_repo"])

    def upload_file(self, **kw):
        self.calls.append(("upload_file", kw["path_in_repo"]))
        self.uploaded_files.append(kw["path_in_repo"])

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

    # Membership alone (above) passes even if delete ran before upload --
    # only a position comparison on the shared call log actually pins the
    # order down.
    upload_index = api.calls.index(("upload_folder", str(tb)))
    for path in api.deleted:
        delete_index = api.calls.index(("delete_file", path))
        assert upload_index < delete_index, (
            f"upload_folder must precede delete_file({path!r}): {api.calls}"
        )


def test_no_flags_refuses(monkeypatch):
    # Neither --card nor --tensorboard: expected would stay empty and the
    # script would print "verified 0 files" for a run that touched nothing.
    monkeypatch.setattr(sys, "argv", ["publish_docs.py"])
    with pytest.raises(SystemExit, match="nothing to publish"):
        publish_docs.main()


# ── the dry run ───────────────────────────────────────────────────────────────

def run_main(monkeypatch, api, *argv):
    """Drive main() against the fake. huggingface_hub is imported inside the
    function, so a stand-in module makes a network call impossible."""
    import types

    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(HfApi=lambda token=None: api))
    monkeypatch.setattr(sys, "argv", ["publish_docs.py", *argv])
    publish_docs.main()


def test_a_dry_run_shows_the_deletes_and_touches_nothing(monkeypatch, tmp_path, capsys):
    """This is the script that calls delete_file in a loop, and it was the only
    one of the pair with no way to see the list first."""
    tb = tmp_path / "tensorboard"
    (tb / "cv").mkdir(parents=True)
    (tb / "cv" / "events.out.tfevents.1").write_text("x")
    api = FakeApi(["README.md", "tensorboard/events.out.tfevents.old"])

    run_main(monkeypatch, api, "--tensorboard", str(tb), "--card", "README.md", "--dry-run")

    out = capsys.readouterr().out
    assert "+ README.md" in out
    assert "+ tensorboard/cv/events.out.tfevents.1" in out
    assert "- tensorboard/events.out.tfevents.old" in out
    assert api.deleted == []
    assert api.uploaded_folders == []
    assert api.uploaded_files == []


def test_a_dry_run_refuses_an_empty_tree_the_same_way(monkeypatch, tmp_path):
    """The dry run must not report a publish that the real run would refuse."""
    empty = tmp_path / "tensorboard"
    empty.mkdir()
    api = FakeApi(["README.md"])

    with pytest.raises(SystemExit, match="contains no files"):
        run_main(monkeypatch, api, "--tensorboard", str(empty), "--dry-run")


def test_a_card_only_dry_run_proposes_no_deletes(monkeypatch, tmp_path, capsys):
    """Without --tensorboard nothing replaces the remote tree, so nothing in it
    is stale -- listing it as a delete candidate would be a lie about the run."""
    api = FakeApi(["README.md", "tensorboard/events.out.tfevents.old"])

    run_main(monkeypatch, api, "--card", "README.md", "--dry-run")

    out = capsys.readouterr().out
    assert "+ README.md" in out
    assert " - " not in out
    assert api.deleted == []
