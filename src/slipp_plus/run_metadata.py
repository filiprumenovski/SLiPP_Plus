"""Run metadata sidecars for persisted model artifacts."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from .__version__ import __version__
from .config import Settings


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _package_version(module_name: str) -> str:
    try:
        return str(getattr(import_module(module_name), "__version__", "unknown"))
    except Exception:
        return "unknown"


def artifact_metadata_sidecar_path(artifact_path: Path) -> Path:
    """Return the run-metadata sidecar path for a persisted artifact."""

    return artifact_path.with_suffix(f"{artifact_path.suffix}.metadata.json")


def build_run_metadata(
    settings: Settings,
    *,
    seed: int | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build reproducibility metadata for a persisted model artifact."""

    metadata: dict[str, Any] = {
        "slipp_plus_version": __version__,
        "sklearn_version": _package_version("sklearn"),
        "xgboost_version": _package_version("xgboost"),
        "lightgbm_version": _package_version("lightgbm"),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "config_path": str(settings.config_path) if settings.config_path else None,
        "config_sha256": settings.config_sha256,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
    }
    if extra:
        metadata.update(extra)
    return metadata


def write_run_metadata_sidecar(
    artifact_path: Path,
    settings: Settings,
    *,
    seed: int | None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write reproducibility metadata beside a persisted model artifact."""

    sidecar_path = artifact_metadata_sidecar_path(artifact_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(
        json.dumps(build_run_metadata(settings, seed=seed, extra=extra), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return sidecar_path
