"""Smoke tests for incoming legacy CAVER cache JSONs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from slipp_plus.feature_builders.tunnel_features import TUNNEL_FEATURES_15, TUNNEL_MISSINGNESS_3

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = REPO_ROOT / "processed" / "v_tunnel" / "structure_json"


@pytest.mark.skipif(not CACHE_ROOT.exists(), reason="legacy CAVER cache JSONs not available")
def test_incoming_tunnel_cache_json_schema_smoke() -> None:
    files = sorted(CACHE_ROOT.glob("*.json"))
    assert files, "expected at least one incoming cache json"

    required_top = {"label", "rows", "warnings"}
    total_rows = 0
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert required_top.issubset(payload), path.name
        assert isinstance(payload["label"], str), path.name
        assert isinstance(payload["rows"], list), path.name
        assert isinstance(payload["warnings"], list), path.name
        total_rows += len(payload["rows"])

        for row in payload["rows"][:5]:
            for column in TUNNEL_FEATURES_15:
                assert column in row, f"{path.name}: missing {column}"
                assert math.isfinite(float(row[column])), f"{path.name}: non-finite {column}"
            present_indicators = [column for column in TUNNEL_MISSINGNESS_3 if column in row]
            if present_indicators:
                assert set(present_indicators) == set(TUNNEL_MISSINGNESS_3), path.name
                for column in TUNNEL_MISSINGNESS_3:
                    assert math.isfinite(float(row[column])), f"{path.name}: non-finite {column}"

    assert total_rows > 0


@pytest.mark.skipif(not CACHE_ROOT.exists(), reason="legacy CAVER cache JSONs not available")
def test_incoming_tunnel_cache_labels_match_filename_prefix() -> None:
    files = sorted(CACHE_ROOT.glob("*.json"))
    assert files, "expected at least one incoming cache json"

    for path in files[:25]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        label = payload["label"]
        class_code = path.name.split("__", 1)[0]
        assert label.startswith(f"{class_code}/"), path.name
