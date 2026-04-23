from __future__ import annotations

from pathlib import Path

import pytest

from slipp_plus.config import Settings, load_settings


@pytest.fixture(scope="session")
def settings() -> Settings:
    return load_settings(Path("configs/day1.yaml"))
