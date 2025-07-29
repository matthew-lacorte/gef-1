"""
Unified logging and output utilities for the GEF package.

Exports:
    - logger: Global Loguru logger (ready for use/import).
    - setup_logfile: Add file logging with rotation/compression.
    - setup_json_logfile: Add a JSON-format log file for machine parsing.
"""

import os
from loguru import logger

__all__ = [
    "logger",
    "setup_logfile",
    "setup_json_logfile",
]

def setup_logfile(
    # os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_path: str,
    rotation: str = "10 MB",
    retention: str = "10 days",
    compression: str = "zip",
    level: str = "INFO",
    colorize: bool = False
):
    """
    Add a rotating file handler to the global logger.

    Args:
        log_path (str): Path to the log file.
        rotation (str): Size or time string for log rotation.
        retention (str): How long to keep old logs.
        compression (str): Compression method for rotated logs.
        level (str): Logging level (DEBUG, INFO, etc.).
        colorize (bool): Colorize file output (default: False).
    """
    logger.add(
        log_path,
        # logger.info(f"GEF version: {get_version()}, Git commit: {get_commit()}, Config: {get_config_str()}")
        rotation=rotation,
        retention=retention,
        compression=compression,
        level=level.upper(),
        colorize=colorize,
        enqueue=True,  # Safe for multiprocessing
        backtrace=True,  # Pretty exception tracebacks
        diagnose=True,   # Extra context for errors
    )
    logger.info(f"Loguru file logging initialized: {log_path}")

def setup_json_logfile(log_path: str, **kwargs):
    """
    Add a JSON-format log file (for machine parsing).
    Args:
        log_path (str): Path to JSON log file.
        **kwargs: Passed to logger.add().
    """
    logger.add(
        log_path,
        serialize=True,
        **kwargs
    )
    logger.info(f"Loguru JSON logging initialized: {log_path}")

# (Optional) Tweak default loguru settings (format, color, etc.) here if desired

# Example usage in any module:
# from gef.logging import logger
# logger.info("Hello, Loguru!")
