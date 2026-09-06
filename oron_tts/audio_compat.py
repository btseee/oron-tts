"""Make `torchaudio.load` work where torchcodec cannot.

torchaudio 2.9 removed its own decoding backends in favour of torchcodec, which
does not decode anything itself either -- it dlopens FFmpeg's *shared libraries*.
A machine with the ffmpeg binary but no matching `avcodec`/`avformat`/`avutil`
DLLs therefore fails at every `torchaudio.load`, with an error naming a missing
`libtorchcodec_core*.dll` rather than the actual cause. Windows is the common
case: the usual FFmpeg distributions are static builds, and PyAV's bundled
libraries are renamed with a content hash so they cannot satisfy a plain
`avcodec-62.dll` lookup either.

oron-cleaner hit the same wall from the other side and solved it by shelling out
to the ffmpeg binary. Here the fix is smaller, because everything this project
loads is a WAV file that soundfile reads directly.

`install()` is a no-op when torchcodec works, so it costs nothing on a training
box and rescues a laptop. Call it before importing anything that loads audio.
"""

from __future__ import annotations

import io
import subprocess

import numpy as np

_INSTALLED = False


def _torchcodec_works() -> bool:
    """Can torchaudio actually decode, or only import?

    The import succeeds on a broken install -- the FFmpeg libraries are opened
    lazily, on the first real load -- so this has to attempt a decode. A tiny
    in-memory WAV is enough and costs a millisecond.
    """
    try:
        import soundfile as sf
        import torch  # noqa: F401
        import torchaudio
    except Exception:
        return False

    buf = io.BytesIO()
    sf.write(buf, np.zeros(64, dtype="float32"), 16_000, format="WAV", subtype="FLOAT")
    buf.seek(0)
    try:
        torchaudio.load(buf)
        return True
    except Exception:
        return False


def _load_with_soundfile(uri, *args, **kwargs):
    """`torchaudio.load`'s contract, served by soundfile with an ffmpeg fallback.

    Returns `(Tensor[channels, samples], sample_rate)` in float32, which is what
    torchaudio guarantees and what every caller here assumes. Channels-first is
    not incidental: F5-TTS's `preprocess_ref_audio_text` indexes `wav[0]`, so a
    samples-first array would silently take the first *sample* as a channel.
    """
    import soundfile as sf
    import torch

    try:
        data, sr = sf.read(uri, dtype="float32", always_2d=True)
    except Exception:
        # soundfile refuses mp3/opus in some builds; the binary handles those.
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(uri),
             "-f", "wav", "-ac", "1", "pipe:1"],
            capture_output=True, check=True)
        data, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32", always_2d=True)
    return torch.from_numpy(np.ascontiguousarray(data.T)), int(sr)


def install(*, force: bool = False) -> bool:
    """Replace `torchaudio.load` when torchcodec cannot decode. Returns whether it did."""
    global _INSTALLED
    if _INSTALLED:
        return True
    if not force and _torchcodec_works():
        return False
    import torchaudio

    torchaudio.load = _load_with_soundfile
    _INSTALLED = True
    return True
