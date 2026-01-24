"""CLI scripts for OronTTS."""

from orontts.scripts.train import main as train_main
from orontts.scripts.infer import main as infer_main
from orontts.scripts.prepare_data import main as prepare_main

__all__ = [
    "train_main",
    "infer_main",
    "prepare_main",
]
