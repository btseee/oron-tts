"""Teach upstream's trainer to log more than two scalars.

`Trainer.train` writes exactly `loss` and `lr`. It also synthesises sample audio
every `save_per_updates`, and writes it to `ckpts/<name>/samples/*.wav` -- on
disk, never in TensorBoard, and gone when the pod is deleted.

F5-TTS is refetched fresh on every run, so this is a patch script rather than a
fork: a fork of a training loop is how `load_checkpoint` surprised this project
once already. Each patch is anchored on an exact source fragment and **asserts it
applied**. A patch that silently matches nothing is worse than no patch, because
the run then looks instrumented and is not -- which is how four stages shipped
with two scalars.

    python scripts/patch_trainer_logging.py --trainer /workspace/F5-TTS/src/f5_tts/model/trainer.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "# oron-tts: richer tensorboard logging"

_CLIP_ANCHOR = (
    "                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)")
_CLIP_PATCH = (
    "                        grad_norm = self.accelerator.clip_grad_norm_(\n"
    "                            self.model.parameters(), self.max_grad_norm)\n"
    f"                        {MARKER}\n"
    "                        self._oron_grad_norm = float(grad_norm) if grad_norm is not None else None")

_SCALAR_ANCHOR = (
    '                    self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)')
_SCALAR_PATCH = (
    '                    self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)\n'
    f"                    {MARKER}\n"
    "                    if getattr(self, '_oron_grad_norm', None) is not None:\n"
    '                        self.writer.add_scalar("train/grad_norm", self._oron_grad_norm, global_update)')

_SAMPLE_ANCHOR = (
    "                        torchaudio.save(\n"
    '                            f"{log_samples_path}/update_{global_update}_gen.wav", gen_audio, target_sample_rate\n'
    "                        )")
# The mel is `log(clamp(mel, min=1e-5))`, so its values run about -11.5 to +2.5.
# `add_image` treats float input as 0-1 and does `(tensor * 255).clip(0, 255)`,
# so the raw tensor renders as a ~98% black rectangle with the rest saturated --
# an image that reads as a spectrogram of silence rather than as a bug. Min-max
# to 0-1 first; add_image supplies the 255. Tensor methods only: this runs inside
# upstream's loop, where the only names in scope are upstream's.
_SAMPLE_PATCH = (
    _SAMPLE_ANCHOR + "\n"
    f"                        {MARKER}\n"
    "                        if self.logger == 'tensorboard' and self.writer is not None:\n"
    "                            self.writer.add_audio('sample/generated', gen_audio,\n"
    "                                                  global_update, sample_rate=target_sample_rate)\n"
    "                            mel = gen_mel_spec[0].detach().cpu().float()\n"
    "                            span = (mel.max() - mel.min()).clamp(min=1e-5)\n"
    "                            self.writer.add_image('sample/mel_generated',\n"
    "                                                  ((mel - mel.min()) / span).unsqueeze(0),\n"
    "                                                  global_update)")

PATCHES = [
    ("grad_norm capture", _CLIP_ANCHOR, _CLIP_PATCH),
    ("grad_norm scalar", _SCALAR_ANCHOR, _SCALAR_PATCH),
    ("sample audio and mel", _SAMPLE_ANCHOR, _SAMPLE_PATCH),
]


def apply_patches(source: str) -> tuple[str, list[str]]:
    """Apply every patch, or refuse. Never partially, never silently."""
    if MARKER in source:
        raise SystemExit("trainer.py is already patched; applying again would "
                         "duplicate every insert")
    missing = [name for name, anchor, _ in PATCHES if source.count(anchor) != 1]
    if missing:
        raise SystemExit(
            "these anchors are absent or ambiguous in this trainer.py: "
            + ", ".join(missing)
            + ". Upstream has moved; re-derive them rather than shipping a run "
              "that looks instrumented and is not.")
    applied = []
    for name, anchor, replacement in PATCHES:
        source = source.replace(anchor, replacement, 1)
        applied.append(name)
    return source, applied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trainer", required=True, type=Path)
    args = parser.parse_args()
    original = args.trainer.read_text(encoding="utf-8")
    patched, applied = apply_patches(original)
    args.trainer.write_text(patched, encoding="utf-8")
    print(f"  patched {args.trainer}: {', '.join(applied)}")


if __name__ == "__main__":
    main()
