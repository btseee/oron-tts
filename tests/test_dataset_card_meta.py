"""Dataset cards keep their long form; they were missing the metadata keys the
Hub filters and searches on, and nothing linked them to the model they exist to
train.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import dataset_card_meta as dcm  # noqa: E402

CARD = """---
language:
- mn
license: cc0-1.0
pretty_name: Common Voice 26.0 Mongolian (cleaned)
task_categories:
- text-to-speech
---

# Common Voice 26.0 Mongolian (cleaned)

A quality-filtered corpus.
"""


def test_the_existing_metadata_and_body_survive():
    """This edits a card that took a GPU pass to produce; it must not rewrite it."""
    out = dcm.enrich(CARD, used_by="btsee/oron-tts", note=None)
    meta, body = dcm.split_card(out)
    assert meta["license"] == "cc0-1.0"
    assert meta["pretty_name"] == "Common Voice 26.0 Mongolian (cleaned)"
    assert "A quality-filtered corpus." in body


def test_the_missing_descriptive_keys_are_added():
    meta, _ = dcm.split_card(dcm.enrich(CARD, used_by=None, note=None))
    for key in ("annotations_creators", "language_creators", "multilinguality",
                "source_datasets", "task_ids"):
        assert key in meta, f"missing {key}"
    assert meta["multilinguality"] == "monolingual"


def test_the_model_is_linked_from_the_dataset():
    out = dcm.enrich(CARD, used_by="btsee/oron-tts", note=None)
    assert "https://huggingface.co/btsee/oron-tts" in out
    assert "## Used by" in out


def test_a_note_can_be_carried():
    """WorldSpeech is CC-BY-NC and the model does not use it. Without saying so,
    a reader seeing an NC dataset beside a CC-BY model assumes a mistake."""
    out = dcm.enrich(CARD, used_by=None, note="The model does not train on this corpus.")
    assert "The model does not train on this corpus." in out


def test_running_it_twice_changes_nothing():
    once = dcm.enrich(CARD, used_by="btsee/oron-tts", note=None)
    assert dcm.enrich(once, used_by="btsee/oron-tts", note=None) == once


# ── the upload is not evidence ────────────────────────────────────────────────

class FakeHub:
    """Stands in for huggingface_hub, which main() imports inside the function.

    `remote` is the server's copy. `upload_file` writes to it only when
    `lands` is true, so a test can reproduce the failure this project has hit
    twice: an upload call that returns normally and changes nothing.
    """

    def __init__(self, tmp_path, card, *, lands=True):
        self.remote = card
        self.lands = lands
        self.path = tmp_path / "README.md"
        self.downloads = 0
        self.uploads: list[str] = []

    # huggingface_hub surface
    def HfApi(self, token=None):  # noqa: N802 - mirrors the real name
        return self

    def hf_hub_download(self, repo_id, filename, **kw):
        self.downloads += 1
        self.path.write_text(self.remote, encoding="utf-8")
        return str(self.path)

    def upload_file(self, *, path_or_fileobj, **kw):
        body = Path(path_or_fileobj).read_text(encoding="utf-8")
        self.uploads.append(body)
        if self.lands:
            self.remote = body


def run_main(monkeypatch, hub, *argv):
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(sys, "argv", ["dataset_card_meta.py", "--repo", "btsee/cv-mn", *argv])
    dcm.main()


def test_the_server_copy_is_read_back_after_the_upload(monkeypatch, tmp_path, capsys):
    hub = FakeHub(tmp_path, CARD)
    run_main(monkeypatch, hub, "--used-by", "btsee/oron-tts")

    assert len(hub.uploads) == 1
    assert hub.downloads == 2, "the card must be re-read from the server, not trusted"
    assert "verified from the server" in capsys.readouterr().out


def test_an_upload_that_lands_nothing_is_caught(monkeypatch, tmp_path):
    """The whole reason publish_docs.py has a verify(): an upload call that
    reports success and leaves the server unchanged has happened here twice."""
    hub = FakeHub(tmp_path, CARD, lands=False)

    with pytest.raises(SystemExit) as excinfo:
        run_main(monkeypatch, hub, "--used-by", "btsee/oron-tts")

    message = str(excinfo.value)
    assert "VERIFY FAILED" in message
    assert "annotations_creators" in message, "it must name what is missing"
    assert dcm.USED_BY_HEADING in message


def test_a_card_that_is_already_current_is_not_uploaded_again(monkeypatch, tmp_path, capsys):
    """enrich is idempotent, so a re-run has nothing to add; uploading anyway
    spends a commit on the dataset's history saying nothing."""
    hub = FakeHub(tmp_path, dcm.enrich(CARD, used_by="btsee/oron-tts", note=None))

    run_main(monkeypatch, hub, "--used-by", "btsee/oron-tts")

    assert hub.uploads == []
    assert "nothing uploaded" in capsys.readouterr().out


def test_a_dry_run_uploads_nothing(monkeypatch, tmp_path):
    hub = FakeHub(tmp_path, CARD)
    run_main(monkeypatch, hub, "--used-by", "btsee/oron-tts", "--dry-run")
    assert hub.uploads == []


NOTE = "The model does not train on this corpus."


def test_missing_from_names_every_absent_piece():
    assert dcm.missing_from(CARD, used_by="btsee/oron-tts", note=NOTE) == [
        *dcm.DEFAULTS, dcm.USED_BY_HEADING, "the note"]
    complete = dcm.enrich(CARD, used_by="btsee/oron-tts", note=NOTE)
    assert dcm.missing_from(complete, used_by="btsee/oron-tts", note=NOTE) == []


def test_a_missing_token_says_which_variable(monkeypatch, tmp_path):
    """A bare KeyError names the dict key, not the fix."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHub(tmp_path, CARD))
    monkeypatch.setattr(sys, "argv", ["dataset_card_meta.py", "--repo", "btsee/cv-mn"])

    with pytest.raises(SystemExit, match="HF_TOKEN is not set"):
        dcm.main()
