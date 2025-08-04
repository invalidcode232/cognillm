import logging
import sys
import os
from datetime import datetime
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        # Get the original formatted message
        original_format = super().format(record)

        # Add color to the log level
        level_name = record.levelname
        if level_name in self.COLORS:
            colored_level = (
                f"{self.COLORS[level_name]}{level_name}{self.COLORS['RESET']}"
            )
            # Replace the level name in the formatted message
            colored_format = original_format.replace(level_name, colored_level, 1)
            return colored_format

        return original_format


def setup_logger(name: str = "CogniLLM", log_dir: str = None) -> logging.Logger:
    """
    Setup a comprehensive logger with colored console output and file logging.

    Args:
        name (str): Logger name
        log_dir (str): Directory to save log files. If None, uses the directory where this function is called from.

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)

    # Clear any existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    # Set logging level
    logger.setLevel(logging.DEBUG)

    # Set log directory to logs subfolder if not specified
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), "logs")

    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    colored_formatter = ColoredFormatter(
        fmt="%(asctime)s | %(name)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(colored_formatter)

    # File handler for all logs
    log_filename = f"cognillm_{datetime.now().strftime('%Y%m%d')}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    file_handler = logging.FileHandler(log_filepath, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Error file handler for errors and critical logs only
    error_log_filename = f"cognillm_errors_{datetime.now().strftime('%Y%m%d')}.log"
    error_log_filepath = os.path.join(log_dir, error_log_filename)

    error_file_handler = logging.FileHandler(
        error_log_filepath, mode="a", encoding="utf-8"
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(detailed_formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_file_handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger
