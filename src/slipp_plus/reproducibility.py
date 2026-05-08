"""Seed initialization helpers for reproducible command-line runs."""

from __future__ import annotations

import os
import random

import numpy as np


def initialize_master_seed(seed: int) -> None:
    """Initialize process-level pseudo-random seeds."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
