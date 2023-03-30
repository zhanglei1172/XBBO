from __future__ import annotations

import logging
import sys

from pathlib import Path

_logger_init = False


def init_logger(file: Path | None = None) -> None:
    """
    Initialize the logger. Log to stdout by default.
    """
    global _logger_init
    if _logger_init:
        return

    logger = logging.getLogger('XBBO')
    logger.setLevel(level=logging.INFO)
    add_handler(logger)
    if file is not None:
        add_handler(logger, file)

    _logger_init = True


def add_handler(logger: logging.Logger, file: Path | None = None) -> logging.Handler:
    """
    Add a logging handler.
    If ``file`` is specified, log to file.
    Otherwise, add a handler to stdout.
    """
    fmt = '[%(asctime)s] %(levelname)s (%(threadName)s:%(name)s) %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'

    formatter = logging.Formatter(fmt, datefmt)
    if file is None:
        # Log to stdout.
        handler = logging.StreamHandler(sys.stdout)
    else:
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(file, mode='w')  # overwrite.
    handler.setLevel(level=logging.DEBUG)  # Print all the logs.
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return handler


def get_logger():
    return logging.getLogger('XBBO')
