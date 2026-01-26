"""Configuration loading utilities."""

from pathlib import Path
from typing import Any

import yaml


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML configuration file by name."""
    config_path = Path(__file__).parent / f"{name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config '{name}' not found at {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


__all__ = ["load_config"]
