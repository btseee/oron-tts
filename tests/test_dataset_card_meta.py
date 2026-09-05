"""Dataset cards keep their long form; they were missing the metadata keys the
Hub filters and searches on, and nothing linked them to the model they exist to
train.
"""

import sys
from pathlib import Path

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
