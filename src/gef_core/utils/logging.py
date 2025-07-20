"""
Logging configuration and utilities for the GEF package.

Exports:
    - setup_logger: Main logger setup function (console and optional file).
    - setup_analysis_logger: Analysis logger setup (file in output dir).
"""

import logging
import os
import sys

__all__ = [
    "setup_logger",
    "setup_analysis_logger",
]

def setup_logger(
    name: str,
    level: int = None,
    log_file: str = None,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    quiet: bool = False
) -> logging.Logger:
    """
    Set up and return a configured logger instance.
    Args:
        name (str): Logger name (usually __name__).
        level (int, optional): Logging level. If None, tries LOG_LEVEL env var, else INFO.
        log_file (str, optional): File path for log output. If None, file logging is disabled.
        fmt (str): Log message format.
        datefmt (str): Date/time format for log entries.
        quiet (bool): If True, suppress console logging (file logging only).
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Level selection (handles str like "DEBUG" or int)
    if level is None:
        env_level = os.environ.get("LOG_LEVEL", "INFO")
        if isinstance(env_level, str):
            level = getattr(logging, env_level.upper(), logging.INFO)
        else:
            try:
                level = int(env_level)
            except Exception:
                level = logging.INFO
    logger.setLevel(level)
    logger.propagate = False  # Prevent double logging from parent loggers

    # Only add handlers if none exist (prevents duplicate handlers)
    if not logger.handlers:
        if not quiet:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
            logger.addHandler(console_handler)
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
            logger.addHandler(file_handler)
            logger.info(f"Logging initialized. File: {log_file}")

    return logger

def setup_analysis_logger(
    analysis_name: str,
    level: int = None,
    quiet: bool = False
) -> logging.Logger:
    """
    Set up a logger for a specific analysis, including file logging.
    Args:
        analysis_name (str): Name of the analysis (used for log filename).
        level (int, optional): Logging level. If None, uses default or LOG_LEVEL env var.
        quiet (bool): If True, suppress console logging.
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Import here to avoid circular import (if config imports logging)
    from . import config

    log_dir = config.get_output_dir(analysis_name)
    log_file = os.path.join(log_dir, f"{analysis_name}.log")
    return setup_logger(
        f"gef.analysis.{analysis_name}",
        level=level,
        log_file=log_file,
        quiet=quiet
    )
