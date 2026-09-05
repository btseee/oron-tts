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

import json
import sys
from pathlib import Path

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

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
