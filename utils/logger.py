import logging
import os
from datetime import datetime

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        logger.addHandler(handler)

        # Also print to console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def write_progress_report(phase, status, details, filename="reports/progress_report.md"):
    """Appends to a markdown progress report."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filename, "a") as f:
        f.write(f"## {phase}\n")
        f.write(f"**Status**: {status}\n")
        f.write(f"**Time**: {timestamp}\n\n")
        f.write(f"{details}\n\n")
        f.write("---\n\n")
