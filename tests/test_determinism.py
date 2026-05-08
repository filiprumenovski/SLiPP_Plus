from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from slipp_plus.config import Settings
from slipp_plus.ingest import run_ingest


def _settings_for_tmp_ingest(settings: Settings, root: Path) -> Settings:
    return settings.model_copy(
        update={
            "paths": settings.paths.model_copy(
                update={
                    "processed_dir": root / "processed",
                    "reports_dir": root / "reports",
                }
            )
        },
        deep=True,
    )


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_ingest_full_pockets_parquet_is_byte_deterministic(
    settings: Settings,
    tmp_path: Path,
) -> None:
    first = run_ingest(_settings_for_tmp_ingest(settings, tmp_path / "run_a"))
    second = run_ingest(_settings_for_tmp_ingest(settings, tmp_path / "run_b"))

    assert _sha256_file(first["full_pockets"]) == _sha256_file(second["full_pockets"])
