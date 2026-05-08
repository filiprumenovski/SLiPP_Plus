"""Helpers for pinning persisted artifacts to an ordered feature schema."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

FEATURE_SCHEMA_VERSION = 1


def _normalized_feature_columns(feature_columns: Iterable[str]) -> list[str]:
    return [str(column) for column in feature_columns]


def compute_feature_schema_hash(feature_columns: Iterable[str]) -> str:
    """Compute the stable hash for an ordered feature schema.

    Parameters
    ----------
    feature_columns:
        Ordered feature-column names used by a model or prediction artifact.

    Returns
    -------
    str
        SHA-256 hex digest of the JSON-normalized ordered column list.
    """
    columns = _normalized_feature_columns(feature_columns)
    payload = json.dumps(columns, separators=(",", ":"), ensure_ascii=True)
    return sha256(payload.encode("utf-8")).hexdigest()


def build_feature_schema_metadata(
    *,
    feature_set: str,
    feature_columns: Iterable[str],
) -> dict[str, Any]:
    """Build metadata that pins an artifact to a feature schema.

    Parameters
    ----------
    feature_set:
        Configured feature-set name, such as ``v14`` or ``v_sterol``.
    feature_columns:
        Ordered feature-column names used to create the artifact.

    Returns
    -------
    dict[str, Any]
        JSON-serializable metadata containing the feature set, ordered columns,
        schema version, and order-sensitive schema hash.
    """
    columns = _normalized_feature_columns(feature_columns)
    return {
        "feature_set": feature_set,
        "feature_columns": columns,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_hash": compute_feature_schema_hash(columns),
    }


def artifact_schema_sidecar_path(artifact_path: Path) -> Path:
    """Return the schema-sidecar path for a persisted artifact.

    Parameters
    ----------
    artifact_path:
        Main artifact path, usually a parquet or joblib file.

    Returns
    -------
    pathlib.Path
        Sidecar path using ``<artifact suffix>.schema.json``.
    """
    return artifact_path.with_suffix(f"{artifact_path.suffix}.schema.json")


def write_artifact_schema_sidecar(
    artifact_path: Path,
    metadata: Mapping[str, Any],
) -> Path:
    """Write schema metadata beside an artifact.

    Parameters
    ----------
    artifact_path:
        Main artifact path whose sidecar should be written.
    metadata:
        JSON-serializable schema metadata.

    Returns
    -------
    pathlib.Path
        Path of the written sidecar JSON file.
    """
    sidecar_path = artifact_schema_sidecar_path(artifact_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(
        json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sidecar_path


def read_artifact_schema_sidecar(artifact_path: Path) -> dict[str, Any] | None:
    """Read schema metadata beside an artifact if present.

    Parameters
    ----------
    artifact_path:
        Main artifact path whose sidecar should be read.

    Returns
    -------
    dict[str, Any] | None
        Parsed sidecar metadata, or ``None`` when no sidecar exists.

    Raises
    ------
    ValueError
        If the sidecar exists but does not contain a JSON object.
    """
    sidecar_path = artifact_schema_sidecar_path(artifact_path)
    if not sidecar_path.exists():
        return None
    with sidecar_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"artifact schema sidecar must contain a JSON object: {sidecar_path}")
    return payload


def _first_schema_difference(expected: list[str], actual: list[str]) -> str:
    if len(expected) != len(actual):
        return f"expected {len(expected)} columns, got {len(actual)}"
    for index, (expected_name, actual_name) in enumerate(zip(expected, actual, strict=True)):
        if expected_name != actual_name:
            return (
                f"first difference at position {index}: "
                f"expected {expected_name!r}, got {actual_name!r}"
            )
    return "column order differs"


def validate_feature_schema_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_feature_columns: Iterable[str] | None = None,
    expected_feature_set: str | None = None,
    artifact_label: str = "artifact",
) -> dict[str, Any]:
    """Validate persisted feature-schema metadata.

    Parameters
    ----------
    metadata:
        Metadata loaded from a model bundle or sidecar JSON.
    expected_feature_columns:
        Optional ordered feature columns expected by the caller.
    expected_feature_set:
        Optional feature-set name expected by the caller.
    artifact_label:
        Human-readable label used in error messages.

    Returns
    -------
    dict[str, Any]
        Normalized schema metadata with a recomputed feature-schema hash.

    Raises
    ------
    KeyError
        If ``feature_columns`` metadata is absent.
    ValueError
        If the stored hash is inconsistent, the feature set differs from the
        expected feature set, or ordered feature columns differ from the
        expected schema.
    """
    if "feature_columns" not in metadata:
        raise KeyError(f"{artifact_label} is missing feature_columns metadata")

    feature_columns = _normalized_feature_columns(metadata["feature_columns"])
    computed_hash = compute_feature_schema_hash(feature_columns)
    stored_hash = metadata.get("feature_schema_hash")
    if stored_hash is not None and stored_hash != computed_hash:
        raise ValueError(
            f"{artifact_label} has inconsistent feature schema metadata: "
            f"stored_hash={stored_hash[:12]} computed_hash={computed_hash[:12]}"
        )

    feature_set = metadata.get("feature_set")
    if expected_feature_set is not None and feature_set is not None and feature_set != expected_feature_set:
        raise ValueError(
            f"{artifact_label} feature_set mismatch: "
            f"expected {expected_feature_set!r}, got {feature_set!r}"
        )

    if expected_feature_columns is not None:
        expected_columns = _normalized_feature_columns(expected_feature_columns)
        expected_hash = compute_feature_schema_hash(expected_columns)
        if computed_hash != expected_hash:
            diff = _first_schema_difference(expected_columns, feature_columns)
            raise ValueError(
                f"{artifact_label} feature schema mismatch: "
                f"expected_hash={expected_hash[:12]} artifact_hash={computed_hash[:12]}; "
                f"{diff}"
            )

    return {
        "feature_set": feature_set,
        "feature_columns": feature_columns,
        "feature_schema_version": metadata.get(
            "feature_schema_version",
            FEATURE_SCHEMA_VERSION,
        ),
        "feature_schema_hash": computed_hash,
    }
