"""The card is the only page most people will read, and it was prose.

Two separate failures are fixed here. The frontmatter carried seven keys and no
`model-index`, so the Hub rendered no Eval Results panel and no links to the
corpora -- every measured number lived where the Hub cannot see it. And the
licence said `cc-by-nc-4.0`, inherited from WorldSpeech, a corpus that failed the
quality gate and was never trained on; that gave away the commercial safety the
corpus selection existed to protect.
"""

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import model_card  # noqa: E402

EVALS = {
    "fleurs": {
        "model_8000.pt": {"male": {"cer_median": 0.0882, "utmos_mean": 2.23},
                          "female": {"cer_median": 0.1034, "utmos_mean": 2.69}},
    },
    "cv": {
        "model_2000.pt": {"male": {"cer_median": 0.0789, "utmos_mean": 2.56},
                          "female": {"cer_median": 0.0950, "utmos_mean": 2.14}},
        "model_12000.pt": {"male": {"cer_median": 0.06329113924050633,
                                    "utmos_mean": 2.483718866109848},
                           "female": {"cer_median": 0.1015, "utmos_mean": 2.1107}},
    },
}

CONSISTENCY = {"measured": {"male_demo_vs_male_prompt": 0.7251,
                            "female_demo_vs_female_prompt": 0.8082,
                            "male_demo_vs_female_demo": 0.1029},
               "calibration": {"same_speaker_range": [0.540, 0.833],
                               "different_speaker_range": [0.034, 0.503],
                               "same_speaker_threshold": 0.52}}


def parse_frontmatter(card: str) -> dict:
    assert card.startswith("---\n"), "a card without frontmatter renders no metadata"
    end = card.index("\n---\n", 3)
    return yaml.safe_load(card[4:end])


def test_the_best_checkpoint_is_the_lowest_mean_cer_not_the_last():
    name, _ = model_card.best_checkpoint(EVALS["cv"])
    assert name == "model_12000.pt"


def test_the_licence_matches_what_the_model_trained_on():
    """WorldSpeech is CC-BY-NC and failed the gate; it was never trained on."""
    meta = model_card.frontmatter(EVALS, CONSISTENCY)
    assert meta["license"] == "cc-by-4.0"


def test_new_version_is_absent():
    """It declares a successor repo and the Hub banners visitors to it. There is
    no successor, so setting it would send every reader to a dead end."""
    assert "new_version" not in model_card.frontmatter(EVALS, CONSISTENCY)


def test_every_field_the_hub_renders_is_present():
    meta = model_card.frontmatter(EVALS, CONSISTENCY)
    for key in ("language", "license", "library_name", "pipeline_tag", "base_model",
                "base_model_relation", "datasets", "metrics", "tags", "model-index"):
        assert key in meta, f"missing {key}"
    assert meta["datasets"] == ["btsee/mbspeech-mn", "btsee/fleurs-mn",
                               "btsee/common-voice-26-mn"]


def test_eval_results_are_read_from_the_measurements_not_typed_in():
    """A hand-copied number drifts from the run that produced it."""
    meta = model_card.frontmatter(EVALS, CONSISTENCY)
    metrics = {m["name"]: m["value"] for m in meta["model-index"][0]["results"][0]["metrics"]}
    assert metrics["CER, male voice"] == pytest.approx(0.06329113924050633)
    assert metrics["CER, female voice"] == pytest.approx(0.1015)
    assert metrics["Speaker similarity, male"] == pytest.approx(0.7251)


def test_frontmatter_values_are_full_precision_not_rounded():
    """Rounding here is how the Eval Results panel and the body's Numbers
    table end up showing two different figures for one measurement."""
    meta = model_card.frontmatter(EVALS, CONSISTENCY)
    metrics = {m["name"]: m["value"] for m in meta["model-index"][0]["results"][0]["metrics"]}
    assert metrics["CER, male voice"] == 0.06329113924050633


def test_frontmatter_and_the_body_table_agree_on_the_same_measurement():
    card = model_card.render(EVALS, CONSISTENCY)
    meta = parse_frontmatter(card)
    metrics = {m["name"]: m["value"] for m in meta["model-index"][0]["results"][0]["metrics"]}
    body = card[card.index("\n---\n", 3) + 5:]
    assert f"{metrics['CER, male voice']:.4f}" in body


def test_the_calibration_sentence_is_read_from_consistency_json():
    """The four numbers in this sentence are already in consistency.json's
    calibration block; typing them by hand would let the card drift from it."""
    body = model_card.render(EVALS, CONSISTENCY)
    calibration = CONSISTENCY["calibration"]
    same_low, same_high = calibration["same_speaker_range"]
    diff_low, diff_high = calibration["different_speaker_range"]
    assert f"{same_low:.3f}" in body
    assert f"{same_high:.3f}" in body
    assert f"{diff_low:.3f}" in body
    assert f"{diff_high:.3f}" in body


def test_the_calibration_sentence_changes_when_the_fixture_does():
    other = {**CONSISTENCY,
             "calibration": {"same_speaker_range": [0.601, 0.777],
                             "different_speaker_range": [0.011, 0.222],
                             "same_speaker_threshold": 0.5}}
    default_body = model_card.render(EVALS, CONSISTENCY)
    changed_body = model_card.render(EVALS, other)
    assert default_body != changed_body
    assert "0.601" in changed_body
    assert "0.601" not in default_body


def test_the_body_leads_with_usage_and_stays_short():
    card = model_card.render(EVALS, CONSISTENCY)
    body = card[card.index("\n---\n", 3) + 5:]
    assert "<audio" in body, "the demos should be playable on the page"
    assert "pip install" in body
    assert "github.com/btseee/oron-tts" in body
    assert body.index("pip install") < body.index("use_ema"), \
        "installation comes before caveats; a reader wants to run it first"
    # Raised 3000 -> 4000 when `## Limits` landed (six disclosures, ~1,065
    # characters), then 4000 -> 4200 when the numbers gained bootstrap intervals
    # and a line saying what they were measured on. Both times the bound moved
    # for measurement, never for prose: an interval is what makes a number
    # interpretable, and "n=360 on the shipped prompts" is what stops a reader
    # assuming the figures describe the configuration they are running. The
    # bound still exists because "short" is the requirement that keeps this card
    # readable -- anything that is not a measurement belongs in
    # docs/model-card.md, which the Links section points at so this one need not
    # grow.
    assert len(body) < 4200, "the body is instructions, not a paper"


def test_the_published_card_carries_every_disclosure_the_long_card_makes():
    """docs/model-card.md is not the card that ships; this one is.

    The two diverged once, and the published copy was the one missing the
    consent and contamination caveats -- the two a reader cannot reconstruct
    from the audio. Each assertion matches a distinctive phrase rather than a
    common word, so dropping a disclosure fails here instead of passing on an
    incidental collision.
    """
    body = model_card.render(EVALS, CONSISTENCY)
    for claim, marker in [
        ("the ~8 kHz bandwidth ceiling", "roughly 8 kHz -- not full-band"),
        ("the CER scorer's contamination", "scorer is fine-tuned on Common Voice"),
        ("that CER must be read against the human floor",
         "never against zero"),
        ("the absence of a listening test", "No listening test has been run"),
        ("UTMOS being an unvalidated proxy", "never validated for Mongolian"),
        ("the absence of watermarking", "There is no watermarking"),
        ("the consent basis", "did not consent to having their individual voice cloned"),
        ("the normaliser's numeral refusals",
         "refuses** numeral case suffixes it cannot expand"),
    ]:
        assert marker.lower() in body.lower(), f"the published card omits {claim}"


def test_the_published_card_points_at_the_long_form_one():
    """Six one-line disclosures are a summary. The reader who wants the method,
    the per-source bandwidth figures or the out-of-scope list needs somewhere
    to go, or the short card reads as the whole story."""
    body = model_card.render(EVALS, CONSISTENCY)
    assert "docs/model-card.md" in body
    assert "github.com/btseee/oron-tts/blob/main/docs/model-card.md" in body


def test_the_two_silent_failures_are_stated():
    """Both produce confident, plausible, wrong audio -- so they are usage
    instructions, not background."""
    body = model_card.render(EVALS, CONSISTENCY)
    assert "use_ema=False" in body
    assert "normalize" in body or "normalise" in body


def test_the_ema_warning_has_no_hand_typed_cer_numbers():
    """Those figures were real once, but from a run no artifact this script
    reads can reproduce; the warning must make its point without them."""
    body = model_card.render(EVALS, CONSISTENCY)
    assert "0.921" not in body
    assert "0.026" not in body


def test_the_opening_sentence_has_no_hand_typed_duration():
    """No artifact this script reads carries a corpus-duration field."""
    body = model_card.render(EVALS, CONSISTENCY)
    assert "25 hours" not in body


def test_best_checkpoint_raises_value_error_not_system_exit():
    """A caller handling 'no scored checkpoints' shouldn't have to catch
    SystemExit; only main() should ever exit the process."""
    with pytest.raises(ValueError):
        model_card.best_checkpoint({"not_a_checkpoint": "no per-gender scores here"})


def test_a_malformed_calibration_range_leaves_a_blank_rather_than_raising():
    """The default covered a missing key only; a key present with any other
    length raised ValueError from the unpack, halfway through rendering a card
    whose other numbers were already correct. Blanks are this card's convention
    for what was not measured, so an unusable range becomes one."""
    assert model_card.calibration_range({}, "same_speaker_range") != (0, 0)
    for bad in ([0.5], [0.1, 0.2, 0.3], "0.5-0.8", None, [None, None]):
        low, high = model_card.calibration_range({"same_speaker_range": bad},
                                                 "same_speaker_range")
        assert low != low and high != high, f"{bad!r} should give NaNs, got {low}, {high}"

    broken = dict(CONSISTENCY, calibration={"same_speaker_range": [0.5],
                                            "different_speaker_range": [0.034, 0.503],
                                            "same_speaker_threshold": 0.52})
    card = model_card.render(EVALS, broken)          # must not raise
    assert "nan" in card


def test_a_well_formed_range_still_reaches_the_card():
    assert model_card.calibration_range(CONSISTENCY["calibration"],
                                        "same_speaker_range") == (0.540, 0.833)


def test_the_shipped_prompt_measurement_supersedes_eval_json():
    """`--measured` must drive the card, not merely be accepted.

    eval.json was produced with prompts `pick_reference` chose from the
    manifest; the card has to report the `voices/` clips a caller actually gets.
    Measured on those, the female voice leads on both metrics -- the opposite of
    what eval.json says -- so a card that silently kept the old numbers would
    tell every reader the wrong voice is the better one.
    """
    measured = {
        "shipped-cv-12000": {
            "male": {"cer_micro": 0.0708, "cer_lo": 0.0659, "cer_hi": 0.0753,
                     "utmos": 2.2632, "utmos_lo": 2.2371, "utmos_hi": 2.2897,
                     "n": 360},
            "female": {"cer_micro": 0.0591, "cer_lo": 0.0551, "cer_hi": 0.0634,
                       "utmos": 2.4237, "utmos_lo": 2.3981, "utmos_hi": 2.4489,
                       "n": 360},
        }
    }
    card = model_card.render(EVALS, CONSISTENCY, measured)
    body = card[card.index("\n---\n", 3) + 5:]

    assert "0.0708" in body and "2.42" in body, "the measured numbers must reach the body"
    assert "n=360" in body, "a reader cannot weigh a number without its n"
    assert "[0.0659–0.0753]" in body, "intervals must be rendered, not dropped"
    assert "voices/" in body, "the body must say which prompts these describe"

    # The frontmatter feeds the Hub's Eval Results panel; it must not keep
    # showing eval.json's numbers while the body shows these.
    import yaml
    meta = yaml.safe_load(card.split("---")[1])
    metrics = meta["model-index"][0]["results"][0]["metrics"]
    by_name = {m["name"]: m["value"] for m in metrics}
    assert by_name["UTMOS, female voice"] == 2.4237
    assert by_name["UTMOS, male voice"] == 2.2632
    assert by_name["UTMOS, female voice"] > by_name["UTMOS, male voice"],         "the panel must agree with the body about which voice scores higher"


def test_a_measurement_without_intervals_still_renders():
    """Older results carry no CI; the card must degrade, not crash."""
    measured = {
        "x": {"male": {"cer_micro": 0.07, "utmos": 2.3, "n": 60},
              "female": {"cer_micro": 0.06, "utmos": 2.4, "n": 60}},
    }
    body = model_card.render(EVALS, CONSISTENCY, measured)
    assert "0.0700" in body and "n=60" in body


def test_the_card_reports_n_per_voice_not_one_voices_n_for_both():
    """`n=201 per voice` was published while female was n=360.

    The note took the male count and labelled it "per voice". Different n means
    the two voices were scored on different material, which the side-by-side
    table otherwise invites a reader to ignore.
    """
    measured = {"x": {
        "male":   {"cer_micro": 0.05, "utmos": 2.9, "n": 201},
        "female": {"cer_micro": 0.06, "utmos": 2.9, "n": 360}}}
    body = model_card.render(EVALS, CONSISTENCY, measured)
    assert "n=201 per voice" not in body, "one voice's n must not stand for both"
    assert "male n=201" in body and "female n=360" in body
    assert "different sentence sets" in body.lower(), \
        "a reader comparing the two columns must be told they are not comparable"
    assert "not against each other" in body.lower(), \
        "the note must say what not to do, not merely that the sets differ"
    # The length bound applies to this path too. It was only ever checked
    # against the equal-n note, so rendering the real numbers produced a 4,269
    # character body and the card was published past its own guard.
    text = body[body.index("\n---\n", 3) + 5:]
    assert len(text) < 4200, f"unequal-n body is {len(text)} chars"


def test_equal_n_still_reads_as_one_number():
    """When both voices really were scored on the same material, say so simply."""
    measured = {"x": {
        "male":   {"cer_micro": 0.05, "utmos": 2.9, "n": 360},
        "female": {"cer_micro": 0.06, "utmos": 2.9, "n": 360}}}
    body = model_card.render(EVALS, CONSISTENCY, measured)
    assert "n=360 per voice" in body
    assert "different sentence sets" not in body
