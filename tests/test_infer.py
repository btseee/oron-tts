"""Voice resolution and text preparation.

Everything here runs without a model. The synthesis call itself is a thin
wrapper over f5_tts.api; what is worth pinning is the logic around it, because
each of these failures is silent at runtime.
"""

import json

import pytest

from oron_tts.infer import Voice, load_voices, prepare_text, resolve_voice
from oron_tts.text import VocabError


@pytest.fixture
def voices_dir(tmp_path):
    for name in ("mn_male_01", "mn_female_01"):
        (tmp_path / f"{name}.wav").write_bytes(b"RIFF")
    (tmp_path / "voices.json").write_text(json.dumps({
        "male": {"id": "mn_male_01", "gender": "male", "audio": "mn_male_01.wav",
                 "text": "Энэ бол эрэгтэй хоолой"},
        "female": {"id": "mn_female_01", "gender": "female", "audio": "mn_female_01.wav",
                   "text": "Энэ бол эмэгтэй хоолой"},
    }, ensure_ascii=False), encoding="utf-8")
    return tmp_path


# ── the bundle ────────────────────────────────────────────────────────────────

def test_both_genders_load(voices_dir):
    voices = load_voices(voices_dir)
    assert set(voices) == {"male", "female"}
    assert all(v.exists for v in voices.values())


def test_missing_bundle_is_empty_not_an_error(voices_dir, tmp_path):
    assert load_voices(tmp_path / "nope") == {}


def test_voice_carries_its_transcript(voices_dir):
    """A wrong ref_text poisons every generation: duration is estimated from
    the ref text/audio ratio, so the transcript ships with the clip."""
    assert load_voices(voices_dir)["male"].text == "Энэ бол эрэгтэй хоолой"


# ── resolution ────────────────────────────────────────────────────────────────

def test_named_voice_resolves(voices_dir):
    audio, text = resolve_voice("female", None, None, voices_dir)
    assert audio.name == "mn_female_01.wav"
    assert "эмэгтэй" in text


def test_custom_reference_overrides_the_bundle(voices_dir, tmp_path):
    custom = tmp_path / "mine.wav"
    custom.write_bytes(b"RIFF")
    audio, text = resolve_voice(None, custom, "миний бичлэг", voices_dir)
    assert audio == custom and text == "миний бичлэг"


def test_custom_reference_without_transcript_is_refused(voices_dir, tmp_path):
    """Without ref_text, upstream transcribes with an English-first ASR and the
    duration estimate compares byte lengths across two scripts."""
    custom = tmp_path / "mine.wav"
    custom.write_bytes(b"RIFF")
    with pytest.raises(SystemExit, match="ref-text"):
        resolve_voice(None, custom, None, voices_dir)


def test_unknown_voice_lists_what_is_available(voices_dir):
    with pytest.raises(SystemExit, match="female"):
        resolve_voice("robot", None, None, voices_dir)


def test_no_voice_and_no_reference_is_refused(voices_dir):
    with pytest.raises(SystemExit, match="--voice"):
        resolve_voice(None, None, None, voices_dir)


def test_absent_bundle_points_at_the_builder(tmp_path):
    with pytest.raises(SystemExit, match="select_voices"):
        resolve_voice("male", None, None, tmp_path)


def test_bundle_entry_with_missing_audio_is_reported(voices_dir):
    (voices_dir / "mn_male_01.wav").unlink()
    with pytest.raises(SystemExit, match="missing"):
        resolve_voice("male", None, None, voices_dir)


# ── text preparation ──────────────────────────────────────────────────────────

def test_text_is_normalised_before_synthesis():
    """Digits must never reach the tokenizer."""
    out = prepare_text("2024 онд 25 хувь")
    assert out == "хоёр мянга хорин дөрвөн онд хорин таван хувь"
    assert not any(c.isdigit() for c in out)


def test_unrepresentable_text_raises_rather_than_being_edited():
    """Silently dropping a character makes the model speak something other than
    what was asked, with no indication that it happened."""
    with pytest.raises(VocabError):
        prepare_text("сайн 你好 байна")


def test_mongolian_specific_vowels_survive():
    assert "ө" in prepare_text("Өнөөдөр")
    assert "ү" in prepare_text("үүлшинэ")


def test_voice_dataclass_reports_a_missing_file(tmp_path):
    assert not Voice("x", "male", tmp_path / "nope.wav", "текст").exists


# ── refusals reach the user as a sentence, not a traceback ───────────────────

def test_a_refusable_numeral_exits_with_an_explanation(monkeypatch, capsys):
    """The normaliser refuses text it cannot expand without guessing. That is
    right for the corpus, which drops the clip -- but a person asking for
    speech was getting a traceback from four frames down on an ordinary
    Mongolian sentence."""
    import sys

    from oron_tts import infer

    monkeypatch.setattr(sys, "argv",
                        ["oron-tts-infer", "--text", "100-д уулзая",
                         "--checkpoint", "nonexistent.pt"])
    with pytest.raises(SystemExit) as exc:
        infer.main()
    message = str(exc.value)
    assert "Cannot say that yet" in message
    assert "100-д" in message
    assert "normaliser-review" in message


def test_unrepresentable_text_exits_with_an_explanation(monkeypatch):
    """A character absent from the vocabulary would be read as a space."""
    import sys

    from oron_tts import infer

    monkeypatch.setattr(sys, "argv",
                        ["oron-tts-infer", "--text", "сайн \u0457 байна",
                         "--checkpoint", "nonexistent.pt"])
    with pytest.raises(SystemExit) as exc:
        infer.main()
    assert "silently" in str(exc.value)


def test_ordinary_text_gets_past_the_check(monkeypatch):
    """The guard must not stand between normal input and synthesis: this should
    fail later, on the missing checkpoint, not earlier on the text."""
    import sys

    from oron_tts import infer

    monkeypatch.setattr(sys, "argv",
                        ["oron-tts-infer", "--text", "Сайн байна уу",
                         "--checkpoint", "nonexistent.pt"])
    with pytest.raises(BaseException) as exc:
        infer.main()
    assert "Cannot say that yet" not in str(exc.value)
