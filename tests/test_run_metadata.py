from __future__ import annotations

import json
from pathlib import Path

from slipp_plus.config import load_settings
from slipp_plus.run_metadata import (
    artifact_metadata_sidecar_path,
    build_run_metadata,
    write_run_metadata_sidecar,
)


def test_build_run_metadata_records_reproducibility_fields() -> None:
    settings = load_settings(Path("configs/day1.yaml"))

    metadata = build_run_metadata(
        settings,
        seed=17,
        extra={"artifact_type": "unit_test_bundle"},
    )

    assert metadata["artifact_type"] == "unit_test_bundle"
    assert metadata["slipp_plus_version"]
    assert metadata["sklearn_version"]
    assert metadata["xgboost_version"]
    assert metadata["lightgbm_version"]
    assert metadata["numpy_version"]
    assert metadata["python_version"]
    assert metadata["config_path"] == "configs/day1.yaml"
    assert metadata["config_sha256"] == settings.config_sha256
    assert metadata["git_commit"]
    assert metadata["timestamp_utc"].endswith("+00:00")
    assert metadata["seed"] == 17


def test_write_run_metadata_sidecar_uses_joblib_adjacent_name(tmp_path: Path) -> None:
    settings = load_settings(Path("configs/day1.yaml"))
    artifact = tmp_path / "rf_multiclass.joblib"
    artifact.write_bytes(b"placeholder")

    sidecar = write_run_metadata_sidecar(
        artifact,
        settings,
        seed=42,
        extra={"artifact_type": "flat_multiclass_model", "model_key": "rf"},
    )

    assert sidecar == artifact_metadata_sidecar_path(artifact)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "flat_multiclass_model"
    assert payload["model_key"] == "rf"
    assert payload["seed"] == 42
