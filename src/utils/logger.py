"""
Logging Configuration
=====================

Centralized logging setup for the entire application.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from loguru import logger
import os

# Remove default handler
logger.remove()

# Get log level from environment
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Create logs directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Console handler with color
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=LOG_LEVEL,
)

# File handler for all logs
logger.add(
    LOG_DIR / "hiv_analytics_{time:YYYY-MM-DD}.log",
    rotation="00:00",  # Rotate at midnight
    retention="30 days",  # Keep logs for 30 days
    compression="zip",  # Compress old logs
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
)

# Error-only file handler
logger.add(
    LOG_DIR / "errors_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="90 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="ERROR",
)


def get_logger(name: Optional[str] = None) -> logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (typically __name__ of the module)
    
    Returns:
        Configured logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


# Intercept standard logging
class InterceptHandler(logging.Handler):
    """
    Intercept standard logging messages and redirect to loguru.
    """
    
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where logged message originated
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


# Replace standard logging
logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

