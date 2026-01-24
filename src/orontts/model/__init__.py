"""VITS2 model architecture module."""

from orontts.model.config import VITS2Config, load_config
from orontts.model.vits2 import VITS2
from orontts.model.modules import (
    TextEncoder,
    PosteriorEncoder,
    Generator,
    DiscriminatorP,
    DiscriminatorS,
    MultiPeriodDiscriminator,
)

__all__ = [
    "VITS2Config",
    "load_config",
    "VITS2",
    "TextEncoder",
    "PosteriorEncoder",
    "Generator",
    "DiscriminatorP",
    "DiscriminatorS",
    "MultiPeriodDiscriminator",
]
