"""The TensorBoard tab has to show the run, not two anonymous scalars.

Four faults are fixed here, and the first is the one that made the tab
unreadable: every tfevents file sat flat in one directory, and TensorBoard
derives runs from *subdirectories*, so nothing could be selected or overlaid.
The second is that every CER and UTMOS number the sweeps produced went to
eval.json and never reached TensorBoard, which is the main thing the tab is
for.

Output is read back with EventAccumulator rather than trusted from the writer,
because that is the reader a person actually uses.
"""

import sys
from pathlib import Path

import pytest

# CI installs neither torch nor tensorboard: oron_tts.text is pure stdlib, and a
# ~2 GB torch install would make CI slow enough to be switched off (see
# scripts/check_ci_imports.py and .github/workflows/test.yml). Skip cleanly
# instead of failing when they are absent.
torch = pytest.importorskip("torch")
pytest.importorskip("tensorboard")

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import tb_report  # noqa: E402

CV_EVAL = {
    "model_2000.pt": {
        "male": {"cer_median": 0.0789, "utmos_mean": 2.56, "bandwidth_median": 6700.0},
        "female": {"cer_median": 0.0950, "utmos_mean": 2.14, "bandwidth_median": 7500.0},
    },
    "model_12000.pt": {
        "male": {"cer_median": 0.06329113924050633, "utmos_mean": 2.483718866109848,
                 "bandwidth_median": 6761.71875},
        "female": {"cer_median": 0.1015, "utmos_mean": 2.1107, "bandwidth_median": 7535.0},
    },
    "model_last.pt": {
        "male": {"cer_median": 0.0701, "utmos_mean": 2.50, "bandwidth_median": 6700.0},
        "female": {"cer_median": 0.1158, "utmos_mean": 2.16, "bandwidth_median": 7500.0},
    },
}


def read_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    acc = EventAccumulator(str(run_dir))
    acc.Reload()
    return {tag: [(e.step, e.value) for e in acc.Scalars(tag)]
            for tag in acc.Tags()["scalars"]}


def test_checkpoint_update_reads_the_step_from_the_name():
    assert tb_report.checkpoint_update("model_12000.pt") == 12000
    assert tb_report.checkpoint_update("model_2000.pt") == 2000


def test_model_last_has_no_update_so_it_cannot_be_plotted():
    """`model_last.pt` carries no update number. Plotting it at a guessed step
    would put a point on the curve that did not happen there."""
    assert tb_report.checkpoint_update("model_last.pt") is None
    assert tb_report.checkpoint_update("pretrained_model_1250000.safetensors") is None


def test_eval_series_puts_every_metric_on_the_update_axis():
    series = tb_report.eval_series(CV_EVAL)
    assert series["eval/cer_male"] == [(2000, 0.0789), (12000, 0.06329113924050633)]
    assert series["eval/cer_female"] == [(2000, 0.0950), (12000, 0.1015)]
    assert series["eval/utmos_male"] == [(2000, 2.56), (12000, 2.483718866109848)]
    assert series["eval/bandwidth_female"] == [(2000, 7500.0), (12000, 7535.0)]


def test_cer_mean_is_the_number_the_sweep_selected_on():
    """best_checkpoint ranks on the mean across genders, so the chart must show
    the quantity the decision was actually made with."""
    series = tb_report.eval_series(CV_EVAL)
    assert series["eval/cer_mean"][1] == (12000, pytest.approx((0.06329113924050633 + 0.1015) / 2))


def test_a_stage_becomes_its_own_run_directory(tmp_path):
    out = tmp_path / "tensorboard"
    run = tb_report.write_stage_run(
        out, "cv", tb_report.eval_series(CV_EVAL), events=None,
        hparams={"stage": "cv", "updates": 16150, "corpus_hours": 25.2},
        metrics={"final/cer_mean": 0.0824})
    assert run == out / "cv"
    assert run.is_dir(), "TensorBoard names runs after subdirectories; a flat file has no name"
    scalars = read_scalars(run)
    assert scalars["eval/cer_male"][-1] == (12000, pytest.approx(0.06329113924050633))


def test_the_training_curve_is_carried_into_the_stage_run(tmp_path):
    """The loss/lr events already exist and must survive verbatim; the point of
    the report is to put them somewhere selectable, not to regenerate them."""
    source = tmp_path / "src"
    source.mkdir()
    from torch.utils.tensorboard import SummaryWriter
    w = SummaryWriter(log_dir=str(source))
    for step in (1, 2, 3):
        w.add_scalar("loss", 0.5 + step / 10, step)
    w.close()
    events = next(source.glob("events.out.tfevents.*"))

    out = tmp_path / "tensorboard"
    run = tb_report.write_stage_run(out, "cv", {}, events=events, hparams={}, metrics={})
    scalars = read_scalars(run)
    assert [s for s, _ in scalars["loss"]] == [1, 2, 3]


def test_metrics_are_written_even_without_hparams(tmp_path):
    """The production command never passes --stages, so hparams is {} on every
    real call. add_hparams only requires its first argument to be a dict, not
    a non-empty one, so gating the write on `hparams and metrics` silently
    dropped final/cer_mean and final/best_update -- the numbers checkpoint
    selection actually ranks on -- in the one invocation that matters."""
    out = tmp_path / "tensorboard"
    run = tb_report.write_stage_run(
        out, "cv", {}, events=None, hparams={},
        metrics={"final/cer_mean": 0.0824, "final/best_update": 12000.0})
    acc = EventAccumulator(str(run))
    acc.Reload()
    hparam_metrics = {tag: acc.Scalars(tag)[-1].value for tag in acc.Tags()["scalars"]}
    assert hparam_metrics["final/cer_mean"] == pytest.approx(0.0824)
    assert hparam_metrics["final/best_update"] == pytest.approx(12000.0)


def test_copied_training_curve_and_new_eval_series_both_survive(tmp_path):
    """write_stage_run copies an existing events file into the run directory
    and then opens a SummaryWriter on that same directory -- main() does this
    for every stage with both training events and eval data. Steps are
    non-monotonic between the two sets (loss at 1..3, eval at 2000/12000)
    because that is the case that would trigger a TensorBoard purge if its
    out-of-order heuristics changed; a purge here would silently erase the
    copied curve instead of merely failing loudly."""
    source = tmp_path / "src"
    source.mkdir()
    from torch.utils.tensorboard import SummaryWriter
    w = SummaryWriter(log_dir=str(source))
    for step in (1, 2, 3):
        w.add_scalar("loss", 0.5 + step / 10, step)
    w.close()
    events = next(source.glob("events.out.tfevents.*"))

    out = tmp_path / "tensorboard"
    series = tb_report.eval_series(CV_EVAL)
    run = tb_report.write_stage_run(out, "cv", series, events=events, hparams={}, metrics={})
    scalars = read_scalars(run)
    assert [s for s, _ in scalars["loss"]] == [1, 2, 3]
    assert scalars["eval/cer_male"] == [(2000, pytest.approx(0.0789)),
                                        (12000, pytest.approx(0.06329113924050633))]


def test_an_empty_events_file_is_recognised(tmp_path):
    """One published file holds no scalars at all -- the aborted first fleurs
    attempt, which died before its first update. Copying it in would add a run
    that draws nothing."""
    from torch.utils.tensorboard import SummaryWriter
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    SummaryWriter(log_dir=str(empty_dir)).close()
    empty = next(empty_dir.glob("events.out.tfevents.*"))
    assert tb_report.is_empty_events(empty) is True

    full_dir = tmp_path / "full"
    full_dir.mkdir()
    w = SummaryWriter(log_dir=str(full_dir))
    w.add_scalar("loss", 0.5, 1)
    w.close()
    assert tb_report.is_empty_events(next(full_dir.glob("events.out.tfevents.*"))) is False


CONSISTENCY = {
    "metric": "ecapa_voxceleb",
    "calibration": {"same_speaker_threshold": 0.52,
                    "same_speaker_range": [0.540, 0.833],
                    "different_speaker_range": [0.034, 0.503]},
    "measured": {"male_demo_vs_male_prompt": 0.7251,
                 "female_demo_vs_female_prompt": 0.8082,
                 "male_demo_vs_female_demo": 0.1029},
}


def _tone(seconds=1.0, sr=24000, freq=220.0):
    import numpy as np
    t = np.arange(int(seconds * sr)) / sr
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype("float32")


def test_the_demos_are_listenable_in_tensorboard(tmp_path):
    """The two demos are the artifact a reader most wants, and they were not in
    the tab at all."""
    out = tmp_path / "tensorboard"
    run = tb_report.write_summary_run(
        out, {"demo_male": (_tone(), 24000), "demo_female": (_tone(freq=330.0), 24000)},
        CONSISTENCY, ["fleurs", "cv", "voicelock"])
    acc = EventAccumulator(str(run), size_guidance={"audio": 10, "images": 10})
    acc.Reload()
    assert set(acc.Tags()["audio"]) == {"audio/demo_male", "audio/demo_female"}


def test_the_spectrograms_are_rendered(tmp_path):
    out = tmp_path / "tensorboard"
    run = tb_report.write_summary_run(out, {"demo_male": (_tone(), 24000)}, CONSISTENCY, ["cv"])
    acc = EventAccumulator(str(run), size_guidance={"images": 10})
    acc.Reload()
    assert "mel/demo_male" in acc.Tags()["images"]


def test_speaker_similarity_reaches_the_charts(tmp_path):
    """The voice lock is the mechanism behind "the same voice every time" and
    nothing measured it during the run."""
    out = tmp_path / "tensorboard"
    run = tb_report.write_summary_run(out, {}, CONSISTENCY, ["cv"])
    scalars = read_scalars(run)
    assert scalars["similarity/male_demo_vs_male_prompt"][0][1] == pytest.approx(0.7251)
    assert scalars["similarity/male_demo_vs_female_demo"][0][1] == pytest.approx(0.1029)


def test_mel_image_is_a_renderable_array():
    import numpy as np
    image = tb_report.mel_image(_tone(), 24000)
    assert image.dtype == np.uint8
    assert image.ndim == 3 and image.shape[0] == 3, "add_image wants CHW"


# ── the two deliverables that reached nothing ─────────────────────────────────

STAGE_META = {
    "corpora": ["MBSpeech", "FLEURS", "Common Voice 26"],
    "clips": 41230,
    "hours": 25.2,
    "speakers": 187,
    "male_hours": 11.4,
    "female_hours": 13.8,
    "learning_rate": 1e-05,
}


def read_text_tags(run_dir: Path) -> dict[str, str]:
    """add_text lands as a tensor tag with a "/text_summary" suffix; the suffix
    is TensorBoard's, so it is stripped back to the tag that was written."""
    acc = EventAccumulator(str(run_dir), size_guidance={"tensors": 10})
    acc.Reload()
    return {tag.removesuffix("/text_summary"):
            acc.Tensors(tag)[-1].tensor_proto.string_val[0].decode("utf-8")
            for tag in acc.Tags()["tensors"]}


def test_the_stage_says_what_it_trained_on_and_what_won(tmp_path):
    """The spec asks for an add_text naming the corpora, the chosen checkpoint
    and how it was chosen. Nothing in the published tab said any of it."""
    out = tmp_path / "tensorboard"
    run = tb_report.write_stage_run(out, "cv", tb_report.eval_series(CV_EVAL), events=None,
                                    hparams=STAGE_META, metrics={})
    text = read_text_tags(run)["stage/summary"]
    assert "Common Voice 26" in text and "FLEURS" in text
    assert "model_12000.pt" in text, "the winning checkpoint must be named"
    assert "chosen by CER" in text
    assert "0.0824" in text, "the mean CER it won on must be stated"
    assert "2 evaluated checkpoints" in text


def test_a_stage_with_no_eval_data_says_the_choice_was_a_fallback(tmp_path):
    """voicelock is the case: loss and lr and nothing else, and no indication
    anywhere that its checkpoint was taken by fallback because its sweep
    produced nothing scoreable. A reader cannot tell that from the charts."""
    out = tmp_path / "tensorboard"
    run = tb_report.write_stage_run(out, "voicelock", {}, events=None,
                                    hparams={"corpora": ["the two chosen speakers"]},
                                    metrics={})
    text = read_text_tags(run)["stage/summary"]
    assert "fallback" in text.lower()
    assert "chosen by CER" not in text
    assert "the two chosen speakers" in text


def test_a_stage_with_no_metadata_says_so_rather_than_inventing_corpora(tmp_path):
    """--stages is optional. Naming no corpus is honest; naming a guessed one
    would be a number this report made up."""
    out = tmp_path / "tensorboard"
    run = tb_report.write_stage_run(out, "fleurs", {}, events=None, hparams={}, metrics={})
    assert "not recorded" in read_text_tags(run)["stage/summary"]


def test_corpus_scalars_come_from_the_stage_metadata(tmp_path):
    """--stages was decorative: the spec asked for corpus/* scalars and the
    flag carried the numbers no further than hparams."""
    out = tmp_path / "tensorboard"
    run = tb_report.write_stage_run(out, "cv", {}, events=None,
                                    hparams=STAGE_META, metrics={})
    scalars = read_scalars(run)
    assert scalars["corpus/clips"] == [(0, pytest.approx(41230.0))]
    assert scalars["corpus/hours"] == [(0, pytest.approx(25.2))]
    assert scalars["corpus/speakers"] == [(0, pytest.approx(187.0))]
    assert scalars["corpus/male_hours"] == [(0, pytest.approx(11.4))]
    assert scalars["corpus/female_hours"] == [(0, pytest.approx(13.8))]


def test_the_schedule_is_not_dressed_up_as_a_corpus_measurement():
    """learning_rate is in the same JSON but describes the schedule, not the
    corpus; under corpus/ it would read as a property of the data."""
    assert "corpus/learning_rate" not in tb_report.corpus_scalars(STAGE_META)
    assert tb_report.corpus_scalars({}) == {}


def test_no_stage_metadata_means_no_corpus_scalars(tmp_path):
    out = tmp_path / "tensorboard"
    run = tb_report.write_stage_run(out, "cv", {}, events=None, hparams={}, metrics={})
    assert not [tag for tag in read_scalars(run) if tag.startswith("corpus/")]


def test_the_calibration_sentence_is_omitted_when_nothing_calibrated_it(tmp_path):
    """--consistency is optional, and without it the summary read "Same-speaker
    pairs of real recordings scored None ... so None separates them" -- which
    reads as a measurement rather than as a gap."""
    out = tmp_path / "tensorboard"
    run = tb_report.write_summary_run(out, {}, {}, ["cv"])
    text = read_text_tags(run)["summary/speaker_similarity"]
    assert "None" not in text
    assert "separates them" not in text


def test_a_partial_calibration_is_not_half_reported(tmp_path):
    """Two of the three numbers make no sentence: the threshold is what the
    ranges are read against."""
    out = tmp_path / "tensorboard"
    run = tb_report.write_summary_run(
        out, {}, {"metric": "ecapa_voxceleb",
                  "calibration": {"same_speaker_range": [0.54, 0.83]},
                  "measured": {"male_demo_vs_male_prompt": 0.7251}}, ["cv"])
    text = read_text_tags(run)["summary/speaker_similarity"]
    assert "separates them" not in text
    assert "ecapa_voxceleb" in text
    assert "male_demo_vs_male_prompt" in text, "the measured table must survive"


def test_the_calibration_sentence_is_written_when_the_data_is_there(tmp_path):
    out = tmp_path / "tensorboard"
    run = tb_report.write_summary_run(out, {}, CONSISTENCY, ["cv"])
    text = read_text_tags(run)["summary/speaker_similarity"]
    assert "separates them" in text
    assert "0.52" in text


def test_corpora_given_as_a_string_is_not_split_into_characters():
    """`", ".join` over a plain string walks it one character at a time, which
    rendered "mbspeech + fleurs" as "m, b, s, p, e, e, c, h" in the published
    tree. A string is the natural way to write this field."""
    text = tb_report.stage_summary("cv", {}, {"corpora": "mbspeech + fleurs"})
    assert "**Trained on:** mbspeech + fleurs" in text
    assert "m, b, s" not in text


def test_corpora_given_as_a_list_still_reads_as_a_list():
    text = tb_report.stage_summary("cv", {}, {"corpora": ["mbspeech", "fleurs"]})
    assert "**Trained on:** mbspeech, fleurs" in text
