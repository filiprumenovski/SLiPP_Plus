"""Shared logging configuration for CLI entry points."""

from __future__ import annotations

import logging
import os


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging for SLiPP++ commands."""
    resolved_level = os.environ.get("SLIPP_LOG_LEVEL", level).upper()
    logging.basicConfig(
        level=getattr(logging, resolved_level, logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
