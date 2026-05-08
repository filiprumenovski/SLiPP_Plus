from __future__ import annotations

import joblib
import pytest

from slipp_plus.artifact_schema import (
    artifact_schema_sidecar_path,
    build_feature_schema_metadata,
    compute_feature_schema_hash,
    read_artifact_schema_sidecar,
    validate_feature_schema_metadata,
    write_artifact_schema_sidecar,
)
from slipp_plus.constants import CLASS_10
from slipp_plus.plm_ste_holdout import _load_model_bundles


def test_feature_schema_hash_is_order_sensitive() -> None:
    assert compute_feature_schema_hash(["f0", "f1", "f2"]) != compute_feature_schema_hash(
        ["f1", "f0", "f2"]
    )


def test_prediction_schema_sidecar_round_trip(tmp_path) -> None:
    artifact_path = tmp_path / "test_predictions.parquet"
    metadata = {
        "artifact_type": "test_predictions",
        "class_order": CLASS_10,
        "models": ["rf", "xgb", "lgbm"],
        **build_feature_schema_metadata(
            feature_set="v49",
            feature_columns=["f0", "f1", "f2"],
        ),
    }

    sidecar_path = write_artifact_schema_sidecar(artifact_path, metadata)

    assert sidecar_path == artifact_schema_sidecar_path(artifact_path)
    loaded = read_artifact_schema_sidecar(artifact_path)
    assert loaded == metadata
    validated = validate_feature_schema_metadata(
        loaded,
        expected_feature_columns=["f0", "f1", "f2"],
        expected_feature_set="v49",
        artifact_label="prediction artifact test_predictions.parquet",
    )
    assert validated["feature_schema_hash"] == metadata["feature_schema_hash"]


def test_load_model_bundles_rejects_feature_schema_drift(tmp_path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    payload = {
        "model": None,
        "class_order": CLASS_10,
        **build_feature_schema_metadata(
            feature_set="v49",
            feature_columns=["f0", "f1", "f2"],
        ),
    }
    for key in ["rf", "xgb", "lgbm"]:
        joblib.dump(payload, models_dir / f"{key}_multiclass.joblib")

    with pytest.raises(ValueError, match="feature schema mismatch"):
        _load_model_bundles(
            models_dir,
            expected_feature_columns=["f1", "f0", "f2"],
            expected_feature_set="v49",
        )
