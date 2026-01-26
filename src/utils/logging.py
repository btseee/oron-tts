"""Logging utilities with Rich formatting."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Global console for consistent output
console = Console()


def setup_logger(
    name: str = "oron",
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    log_file: str | Path | None = None,
    rich_tracebacks: bool = True,
) -> logging.Logger:
    """Set up a logger with Rich console handler.

    Args:
        name: Logger name.
        level: Logging level.
        log_file: Optional file path for persistent logs.
        rich_tracebacks: Enable Rich tracebacks.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Rich console handler
    console_handler = RichHandler(
        console=console,
        show_path=False,
        rich_tracebacks=rich_tracebacks,
        tracebacks_show_locals=True,
    )
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    return logger


def get_progress_bar(**kwargs) -> Progress:
    """Create a Rich progress bar for training.

    Returns:
        Configured Progress instance.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        **kwargs,
    )


class TrainingLogger:
    """Structured logging for training runs."""

    def __init__(
        self,
        name: str = "oron.train",
        log_dir: str | Path | None = None,
    ) -> None:
        """Initialize training logger.

        Args:
            name: Logger name.
            log_dir: Directory for log files.
        """
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set up file logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"train_{timestamp}.log"
        self.logger = setup_logger(name, log_file=log_file)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int,
        prefix: str = "train",
    ) -> None:
        """Log training metrics.

        Args:
            metrics: Dictionary of metric name -> value.
            step: Training step.
            prefix: Metric prefix (train/val/test).
        """
        # Format metrics for console
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self.logger.info(f"[Step {step}] {prefix} | {metrics_str}")

    def log_hyperparameters(self, config: dict) -> None:
        """Log hyperparameters at start of training."""
        self.logger.info("Hyperparameters:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")

    def log_model_summary(self, num_params: int, model_name: str = "F5-TTS") -> None:
        """Log model parameter count."""
        params_m = num_params / 1e6
        self.logger.info(f"{model_name} parameters: {params_m:.2f}M")

    def finish(self) -> None:
        """Finalize logging session."""
        pass
