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
                            "male_demo_vs_female_demo": 0.1029}}


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


def test_the_body_leads_with_usage_and_stays_short():
    card = model_card.render(EVALS, CONSISTENCY)
    body = card[card.index("\n---\n", 3) + 5:]
    assert "<audio" in body, "the demos should be playable on the page"
    assert "pip install" in body
    assert "github.com/btseee/oron-tts" in body
    assert body.index("pip install") < body.index("use_ema"), \
        "installation comes before caveats; a reader wants to run it first"
    assert len(body) < 3000, "the body is instructions, not a paper"


def test_the_two_silent_failures_are_stated():
    """Both produce confident, plausible, wrong audio -- so they are usage
    instructions, not background."""
    body = model_card.render(EVALS, CONSISTENCY)
    assert "use_ema=False" in body
    assert "normalize" in body or "normalise" in body
