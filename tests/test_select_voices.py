"""Reference-voice selection.

The shipped voices ARE these clips -- F5-TTS has no gender conditioning -- so
this ranking is the whole male/female deliverable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from select_voices import MAX_REF_S, MIN_REF_S, candidates, score  # noqa: E402


def clip(cid, gender="male", dur=8.0, bw=7500.0, dnsmos=3.2, align=0.85, spk="s1"):
    return {"clip_id": cid, "gender_resolved": gender, "duration_s": dur,
            "bandwidth_hz": bw, "dnsmos_ovr": dnsmos, "align_score": align,
            "snr_db": 20.0, "client_id": spk, "text": "текст",
            "audio_path": f"wavs/{cid}.wav"}


def test_bandwidth_dominates_the_ranking():
    """Output bandwidth follows the prompt, and the >=10 kHz tail is rare."""
    bright = clip("bright", bw=11000.0, dnsmos=3.0, align=0.80)
    clean = clip("clean", bw=6500.0, dnsmos=4.5, align=0.95)
    assert score(bright) > score(clean)


def test_clips_outside_the_reference_window_are_excluded():
    """Upstream clips anything over 12 s; under ~6 s there is too little
    speaker evidence for a stable prompt."""
    pool = [clip("short", dur=MIN_REF_S - 1), clip("long", dur=MAX_REF_S + 5),
            clip("good", dur=8.0)]
    picked = candidates(pool, "male", top=5)
    assert [r["clip_id"] for r in picked] == ["good"]


def test_one_candidate_per_speaker_by_default():
    """Otherwise the shortlist is one prolific contributor several times over."""
    pool = [clip(f"c{i}", spk="same", bw=11000.0) for i in range(5)]
    assert len(candidates(pool, "male", top=5)) == 1


def test_candidates_span_speakers():
    pool = [clip(f"c{i}", spk=f"s{i}") for i in range(4)]
    picked = candidates(pool, "male", top=3)
    assert len({r["client_id"] for r in picked}) == 3


def test_only_the_requested_gender_is_returned():
    pool = [clip("m", gender="male"), clip("f", gender="female", spk="s2")]
    assert [r["clip_id"] for r in candidates(pool, "female", top=5)] == ["f"]


def test_unresolved_gender_is_never_offered():
    """A clip whose gender could not be determined must not become 'the male voice'."""
    assert candidates([clip("u", gender="")], "male", top=5) == []


def test_empty_pool_returns_nothing_rather_than_failing():
    assert candidates([], "male", top=5) == []


def test_higher_alignment_wins_when_audio_is_comparable():
    """A wrong ref_text poisons every generation: duration is estimated from
    the ref text/audio ratio."""
    good = clip("good", align=0.95, spk="a")
    poor = clip("poor", align=0.70, spk="b")
    assert score(good) > score(poor)
