"""Upstream logs two scalars. This adds the rest, and refuses to pretend.

The failure mode this guards against is not a bad patch but a silent one: if an
anchor moves in a new upstream release and the patch quietly matches nothing, the
next run looks instrumented and is not -- which is how four stages shipped with
`loss` and `lr` and nothing else.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import patch_trainer_logging as ptl  # noqa: E402

TRAINER = '''
                if self.logger == "tensorboard" and self.accelerator.is_main_process:
                    self.writer.add_scalar("loss", loss.item(), global_update)
                    self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)
'''

CLIP = '''
                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
'''

SAMPLES = '''
                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_gen.wav", gen_audio, target_sample_rate
                        )
'''

WHOLE = CLIP + TRAINER + SAMPLES


def test_every_patch_applies_to_the_real_upstream_shape():
    patched, applied = ptl.apply_patches(WHOLE)
    assert set(applied) == {name for name, _, _ in ptl.PATCHES}
    assert patched != WHOLE


def test_the_grad_norm_reaches_tensorboard():
    patched, _ = ptl.apply_patches(WHOLE)
    assert "train/grad_norm" in patched


def test_the_sample_audio_reaches_tensorboard():
    """The trainer already synthesises this audio and writes it to disk, where
    it dies with the pod."""
    patched, _ = ptl.apply_patches(WHOLE)
    assert "add_audio" in patched
    assert "add_image" in patched


def test_a_moved_anchor_raises_rather_than_no_opping():
    """The whole point. A patch that matches nothing must not report success."""
    with pytest.raises(SystemExit) as excinfo:
        ptl.apply_patches(TRAINER)          # missing the clip and sample anchors
    assert "grad_norm" in str(excinfo.value) or "sample" in str(excinfo.value)


def test_applying_twice_is_refused():
    """Running it again on an already patched file would double the inserts."""
    patched, _ = ptl.apply_patches(WHOLE)
    with pytest.raises(SystemExit):
        ptl.apply_patches(patched)


def test_the_mel_image_is_normalised_before_it_is_logged():
    """F5-TTS mel is `log(clamp(mel, min=1e-5))`, so its values run about -11.5
    to +2.5. `add_image` treats float input as 0-1 and does
    `(tensor * 255).clip(0, 255)`, so the raw tensor renders as a ~98% black
    rectangle -- an image that reads as a spectrogram of silence rather than as
    a bug, which is exactly the kind of plausible-looking nothing this script
    exists to stop shipping."""
    patched, _ = ptl.apply_patches(WHOLE)
    assert "gen_mel_spec[0].detach().cpu().unsqueeze(0)" not in patched, \
        "the raw log-magnitude tensor must not reach add_image"
    logged = patched[patched.index("'sample/mel_generated'"):]
    assert "((mel - mel.min()) / span)" in logged


def test_the_normalisation_actually_lands_in_zero_to_one():
    """Asserting the source *mentions* min and max would pass for a formula
    that divides by the wrong thing. This runs the injected arithmetic on a
    real F5-TTS-shaped mel and checks where the values end up."""
    import numpy as np

    class FakeTensor:
        """Only the tensor methods the injected line uses. It runs inside
        upstream's loop, so it may use nothing else."""

        def __init__(self, a): self.a = np.asarray(a, dtype="float64")
        def detach(self): return self
        def cpu(self): return self
        def float(self): return self
        def min(self): return FakeTensor(self.a.min())
        def max(self): return FakeTensor(self.a.max())
        def unsqueeze(self, _dim): return FakeTensor(self.a[None])
        def clamp(self, min): return FakeTensor(np.maximum(self.a, min))
        def __sub__(self, other): return FakeTensor(self.a - other.a)
        def __truediv__(self, other): return FakeTensor(self.a / other.a)
        def __getitem__(self, i): return FakeTensor(self.a[i])

    # The real range: torch.log(torch.clamp(mel, min=1e-5)) over a quiet clip.
    rng = np.random.default_rng(0)
    gen_mel_spec = FakeTensor(rng.uniform(-11.5, 2.5, size=(1, 100, 200)))

    body = [line.strip('" ') for line in ptl._SAMPLE_PATCH.split("\n")]
    source = "\n".join(line for line in body
                       if line.startswith(("mel =", "span =")))
    scope = {"gen_mel_spec": gen_mel_spec}
    exec(compile(source, "<injected>", "exec"), scope)          # noqa: S102
    image = ((scope["mel"] - scope["mel"].min()) / scope["span"]).unsqueeze(0)

    assert image.a.min() == pytest.approx(0.0)
    assert image.a.max() == pytest.approx(1.0)
    # What add_image does to float input. Before the fix this was ~98% zero.
    rendered = np.clip(image.a * 255, 0, 255)
    assert (rendered > 8).mean() > 0.9, "the spectrogram is still mostly black"
