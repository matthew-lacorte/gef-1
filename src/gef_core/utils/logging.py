# src/gef/logging.py
"""
Logging configuration for the GEF package.
"""

import logging
import os
import sys

def setup_logger(name, level=logging.INFO, log_file=None):
    """
    Set up a logger with consistent formatting.
    
    Args:
        name (str): Logger name (usually __name__ of the calling module)
        level (int): Logging level (default: logging.INFO)
        log_file (str, optional): Path to log file if file logging is desired
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Only add handlers if none exist (prevents duplicate handlers)
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler (if requested)
        if log_file:
            # Ensure directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
    
    return logger

def setup_analysis_logger(analysis_name, level=logging.INFO):
    """
    Set up a logger specifically for an analysis with file output.
    
    Args:
        analysis_name (str): Name of the analysis (used for log filename)
        level (int): Logging level (default: logging.INFO)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Import here to avoid circular import
    from . import config
    
    # Create log file in the analysis directory
    log_dir = config.get_output_dir(analysis_name)
    log_file = os.path.join(log_dir, f"{analysis_name}.log")
    
    return setup_logger(f"gef.analysis.{analysis_name}", level, log_file)