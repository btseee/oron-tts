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
