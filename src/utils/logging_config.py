"""Logging configuration for the project."""

from __future__ import annotations

import logging
import sys
from functools import lru_cache

from config.settings import get_settings

_LOGGER_NAME = "cine-match-ai"


@lru_cache(maxsize=1)
def configure_logging() -> logging.Logger:
    """Set up a single process-wide logger with a console handler."""
    settings = get_settings()
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(settings.log_level.upper())
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            )
        )
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the project logger (optionally namespaced)."""
    base = configure_logging()
    if name:
        return base.getChild(name)
    return base
