"""
Package-wide logging utilities using Rich for pretty console output.

Usage:
    from gef.utils.logging import get_logger
    logger = get_logger(__name__)

This configures a single Rich handler for the root logger on first use
and returns a namespaced child logger thereafter.
"""
from __future__ import annotations

import logging
from typing import Optional

try:
    from rich.logging import RichHandler
except Exception:  # rich is provided via typer[all]; fallback gracefully
    RichHandler = None  # type: ignore


_CONFIGURED = False


def _configure_root(level: int = logging.INFO, rich_tracebacks: bool = False) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    handlers: list[logging.Handler] = []

    if RichHandler is not None:
        handlers.append(
            RichHandler(
                rich_tracebacks=rich_tracebacks,
                markup=True,
                show_time=True,
                show_path=False,
                log_time_format="%Y-%m-%d %H:%M:%S",
            )
        )
        fmt = "%(message)s"  # Rich renders level/time
    else:
        # Plain logging fallback
        handlers.append(logging.StreamHandler())
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(level=level, handlers=handlers, format=fmt)

    # Make third-party noisy loggers quieter by default
    for noisy in ("matplotlib", "numba", "numexpr"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    _CONFIGURED = True


def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Return a package-consistent logger.

    - Configures root logging once with Rich handler.
    - Returns a namespaced logger with the requested level.
    """
    _configure_root(level=level)
    logger = logging.getLogger(name if name else "gef")
    logger.setLevel(level)
    return logger
