"""
Logging utilities for DocumentReader application.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from ..config import LOGGING_CONFIG, DEFAULT_LOG_DIR


class UnicodeFormatter(logging.Formatter):
    """Custom formatter that handles Unicode characters properly."""
    
    def format(self, record):
        """Format log record with proper Unicode handling."""
        try:
            return super().format(record)
        except UnicodeEncodeError:
            # Handle Unicode encoding errors
            record.msg = str(record.msg).encode('utf-8', errors='replace').decode('utf-8')
            return super().format(record)


def setup_logger(name: str, log_file: Optional[str] = None, level: str = 'INFO') -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = UnicodeFormatter(
        fmt=LOGGING_CONFIG['format'],
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            DEFAULT_LOG_DIR / log_file,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


# Create main application logger
main_logger = setup_logger(
    'DocumentReader',
    LOGGING_CONFIG['filename'],
    LOGGING_CONFIG['level']
) 