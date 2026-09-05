"""Logging setup built on rich.

Library code never calls ``print``; it uses ``get_logger(__name__)``. Only the
CLI and the tests are allowed to write to stdout directly (via ``console``).
"""

from __future__ import annotations

import logging
import os
from typing import Final

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

_THEME: Final = Theme(
    {
        "logging.level.debug": "dim cyan",
        "logging.level.info": "green",
        "logging.level.warning": "yellow",
        "logging.level.error": "bold red",
        "stage": "bold cyan",
        "clip": "magenta",
        "cost": "bold yellow",
    }
)

#: Shared rich console. CLI output (tables, prompts) goes through this.
console: Final = Console(theme=_THEME, highlight=False)

_configured = False


def setup_logging(level: str | int | None = None) -> None:
    """Install the rich log handler on the root logger (idempotent).

    Args:
        level: Log level name or number. Defaults to ``$YTEDIT_LOG_LEVEL`` or
            ``INFO``.
    """
    global _configured
    if _configured:
        if level is not None:
            logging.getLogger().setLevel(level)
        return

    resolved = level if level is not None else os.environ.get("YTEDIT_LOG_LEVEL", "INFO")
    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_path=False,
        show_time=True,
        omit_repeated_times=False,
        markup=True,
        log_time_format="[%H:%M:%S]",
    )
    logging.basicConfig(
        level=resolved,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
        force=True,
    )
    # Third-party noise.
    for noisy in ("httpx", "httpcore", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger.

    Args:
        name: Usually ``__name__``.

    Returns:
        A ``logging.Logger`` with the rich handler installed.
    """
    setup_logging()
    return logging.getLogger(name)
