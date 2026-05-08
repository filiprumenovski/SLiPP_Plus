"""Tests for CAVER-derived ``v_tunnel`` features."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from slipp_plus.constants import FEATURE_SETS
from slipp_plus.tunnel_features import (
    TUNNEL_FEATURES_15,
    TUNNEL_FEATURES_18,
    TunnelBuildThresholds,
    _cache_fingerprint,
    _enforce_quality_gates,
    _extract_from_analysis,
    _preflight_validate_task_inputs,
    _process_structure,
    _process_structure_cached,
    _quality_metrics,
    _run_tasks_and_write,
    _safe_defaults,
    _select_structure_batch,
    extract_pocket_tunnel_features,
)


def test_tunnel_registry_shape() -> None:
    assert len(TUNNEL_FEATURES_15) == 15
    assert len(TUNNEL_FEATURES_18) == 18
    assert FEATURE_SETS["v_tunnel"] == FEATURE_SETS["v_sterol"] + TUNNEL_FEATURES_18


def test_parse_tunnels_csv_smoke(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    (analysis / "tunnels.csv").write_text(
        "\n".join(
            [
                "Tunnel cluster,Starting point,Length,Bottleneck radius,Curvature,Throughput,Avg R",
                "1,0,24.5,1.2,1.6,0.91,2.4",
                "2,0,12.0,0.8,1.2,0.12,1.4",
                "3,1,5.0,0.5,1.1,0.03,0.9",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (analysis / "residues.csv").write_text(
        "\n".join(
            [
                "Tunnel cluster,Residue",
                "1,LEU 42 A",
                "1,PHE 88 A",
                "2,ASP 10 A",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (analysis / "tunnel_profiles.csv").write_text(
        "\n".join(
            [
                "Tunnel cluster,Distance from origin,X,Y,Z,R",
                "1,0.0,0,0,0,2.0",
                "1,2.0,1,0,0,2.0",
                "2,0.0,0.1,0,0,1.2",
                "2,2.0,1.1,0,0,1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    features = _extract_from_analysis(analysis, [7, 8], {7: 10.0, 8: 2.0})

    assert set(features[7]) == set(TUNNEL_FEATURES_18)
    assert features[7]["tunnel_count"] == 2
    assert features[7]["tunnel_primary_length"] == 24.5
    assert features[7]["tunnel_primary_bottleneck_radius"] == 1.2
    assert features[7]["tunnel_primary_charge"] == 0.0
    assert features[7]["tunnel_primary_aromatic_fraction"] == 0.5
    assert features[7]["tunnel_extends_beyond_pocket"] == 1
    assert features[7]["tunnel_pocket_context_present"] == 1
    assert features[7]["tunnel_caver_profile_present"] == 1
    assert features[7]["tunnel_has_tunnel"] == 1
    assert features[8]["tunnel_count"] == 1


def test_feature_dict_completeness_and_finiteness() -> None:
    defaults = _safe_defaults()
    assert set(defaults) == set(TUNNEL_FEATURES_18)
    for value in defaults.values():
        assert math.isfinite(float(value))


def test_extract_from_analysis_marks_no_tunnel_without_collapsing_to_failure(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    (analysis / "tunnels.csv").write_text(
        "Tunnel cluster,Starting point,Length,Bottleneck radius,Curvature,Throughput,Avg R\n",
        encoding="utf-8",
    )

    features = _extract_from_analysis(analysis, [7], {7: 10.0})

    assert features[7]["tunnel_count"] == 0
    assert features[7]["tunnel_pocket_context_present"] == 1
    assert features[7]["tunnel_caver_profile_present"] == 1
    assert features[7]["tunnel_has_tunnel"] == 0


def test_safe_defaults_on_caver_failure(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    pdb = tmp_path / "protein.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "END\n",
        encoding="utf-8",
    )
    jar = tmp_path / "caver.jar"
    jar.write_text("", encoding="utf-8")

    def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(args=args[0], returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    features = extract_pocket_tunnel_features(
        protein_pdb_path=pdb,
        pocket_centroid=np.array([0.0, 0.0, 0.0]),
        pocket_axial_length=15.0,
        caver_jar=jar,
    )
    assert features == _safe_defaults(pocket_context_present=1)


def test_process_structure_marks_missing_pocket_context(tmp_path: Path) -> None:
    structure_dir = tmp_path / "CLR" / "pdb1ABC_out"
    (structure_dir / "pockets").mkdir(parents=True)
    protein_pdb = tmp_path / "CLR" / "pdb1ABC.pdb"
    protein_pdb.parent.mkdir(parents=True, exist_ok=True)
    protein_pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "END\n",
        encoding="utf-8",
    )
    task = {
        "label": "CLR/pdb1ABC.pdb",
        "rows": [{"pdb_ligand": "CLR/pdb1ABC.pdb", "matched_pocket_number": 1}],
        "structure_dir": str(structure_dir),
        "protein_pdb": str(protein_pdb),
        "caver_jar": str(tmp_path / "missing.jar"),
        "settings": {
            "probe_radius": 0.9,
            "shell_radius": 3.0,
            "shell_depth": 4.0,
            "clustering_threshold": 3.5,
            "timeout_s": 45,
            "max_structure_timeout_s": 300,
            "use_multi_start": False,
            "java_heap": "768m",
        },
    }

    result = _process_structure(task)

    row = result["rows"][0]
    assert row["tunnel_pocket_context_present"] == 0
    assert row["tunnel_caver_profile_present"] == 0
    assert row["tunnel_has_tunnel"] == 0


def test_process_structure_cached_reads_existing_json(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    row = {"pdb_ligand": "CLR/pdb1ABC.pdb", "matched_pocket_number": 1}
    task = {
        "label": "CLR/pdb1ABC.pdb",
        "rows": [row],
        "structure_dir": str(tmp_path / "missing_out"),
        "protein_pdb": str(tmp_path / "missing.pdb"),
        "caver_jar": str(tmp_path / "missing.jar"),
        "settings": {
            "probe_radius": 0.9,
            "shell_radius": 3.0,
            "shell_depth": 4.0,
            "clustering_threshold": 3.5,
            "timeout_s": 45,
            "max_structure_timeout_s": 300,
            "use_multi_start": False,
            "java_heap": "768m",
        },
        "cache_dir": str(cache_dir),
    }
    payload = {
        "cache_version": 4,
        "caver_config_version": 2,
        "settings_fingerprint": _cache_fingerprint(task),
        "label": "CLR/pdb1ABC.pdb",
        "rows": [{**row, **_safe_defaults()}],
        "warnings": [],
    }
    (cache_dir / "CLR__pdb1ABC.pdb.json").write_text(json.dumps(payload), encoding="utf-8")

    result = _process_structure_cached(task)

    assert result["cached"] is True
    assert result["rows"] == payload["rows"]


def test_process_structure_cached_ignores_stale_json(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    task = {
        "label": "CLR/pdb1ABC.pdb",
        "rows": [{"pdb_ligand": "CLR/pdb1ABC.pdb", "matched_pocket_number": 1}],
        "structure_dir": str(tmp_path / "missing_out"),
        "protein_pdb": str(tmp_path / "missing.pdb"),
        "caver_jar": str(tmp_path / "missing.jar"),
        "settings": {
            "probe_radius": 0.9,
            "shell_radius": 3.0,
            "shell_depth": 4.0,
            "clustering_threshold": 3.5,
            "timeout_s": 45,
            "max_structure_timeout_s": 300,
            "use_multi_start": False,
            "java_heap": "768m",
        },
        "cache_dir": str(cache_dir),
    }
    stale_payload = {
        "cache_version": 3,
        "caver_config_version": 2,
        "settings_fingerprint": _cache_fingerprint(task),
        "label": "CLR/pdb1ABC.pdb",
        "rows": [],
        "warnings": [],
    }
    (cache_dir / "CLR__pdb1ABC.pdb.json").write_text(
        json.dumps(stale_payload), encoding="utf-8"
    )

    result = _process_structure_cached(task)

    assert result["cached"] is False
    assert result["rows"][0]["tunnel_pocket_context_present"] == 0


def test_preflight_rejects_missing_structure_inputs() -> None:
    tasks = [
        {
            "label": "A",
            "protein_pdb": "missing/a.pdb",
            "structure_dir": "missing/a_out",
        },
        {
            "label": "B",
            "protein_pdb": "missing/b.pdb",
            "structure_dir": "missing/b_out",
        },
    ]
    thresholds = TunnelBuildThresholds(
        max_missing_structure_frac=0.1,
        min_context_present_frac=0.0,
        min_profile_present_frac=0.0,
    )
    with pytest.raises(ValueError, match="preflight failed"):
        _preflight_validate_task_inputs(tasks, thresholds=thresholds)


def test_select_structure_batch_filters_tasks_and_base_rows() -> None:
    base = pd.DataFrame(
        [
            {"pdb_ligand": "A/one.pdb", "matched_pocket_number": 1},
            {"pdb_ligand": "A/one.pdb", "matched_pocket_number": 2},
            {"pdb_ligand": "B/two.pdb", "matched_pocket_number": 1},
            {"pdb_ligand": "C/three.pdb", "matched_pocket_number": 1},
        ]
    )
    tasks = [
        {"label": "A/one.pdb"},
        {"label": "B/two.pdb"},
        {"label": "C/three.pdb"},
    ]

    selected_tasks, selected_base, batch = _select_structure_batch(
        tasks,
        base,
        key_column="pdb_ligand",
        batch_index=0,
        batch_size=2,
    )

    assert [task["label"] for task in selected_tasks] == ["A/one.pdb", "B/two.pdb"]
    assert selected_base["pdb_ligand"].tolist() == ["A/one.pdb", "A/one.pdb", "B/two.pdb"]
    assert batch is not None
    assert batch.selected_structures == 2
    assert batch.total_structures == 3


def test_select_structure_batch_requires_pair() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        _select_structure_batch(
            [{"label": "A"}],
            pd.DataFrame([{"pdb_ligand": "A"}]),
            key_column="pdb_ligand",
            batch_index=0,
            batch_size=None,
        )


def test_quality_gate_fails_when_profile_missing() -> None:
    frame = pd.DataFrame(
        [
            {"tunnel_pocket_context_present": 1, "tunnel_caver_profile_present": 0, "tunnel_has_tunnel": 0},
            {"tunnel_pocket_context_present": 1, "tunnel_caver_profile_present": 0, "tunnel_has_tunnel": 0},
        ]
    )
    quality = _quality_metrics(frame)
    thresholds = TunnelBuildThresholds(
        max_missing_structure_frac=1.0,
        min_context_present_frac=0.9,
        min_profile_present_frac=0.5,
    )
    with pytest.raises(ValueError, match="quality gate failed"):
        _enforce_quality_gates(quality, thresholds=thresholds)


def test_quality_gate_allows_zero_tunnel_when_context_and_profile_present() -> None:
    frame = pd.DataFrame(
        [
            {"tunnel_pocket_context_present": 1, "tunnel_caver_profile_present": 1, "tunnel_has_tunnel": 0},
            {"tunnel_pocket_context_present": 1, "tunnel_caver_profile_present": 1, "tunnel_has_tunnel": 0},
        ]
    )
    quality = _quality_metrics(frame)
    thresholds = TunnelBuildThresholds(
        max_missing_structure_frac=1.0,
        min_context_present_frac=0.9,
        min_profile_present_frac=0.9,
    )
    _enforce_quality_gates(quality, thresholds=thresholds)


def test_manifest_requires_analysis_output_root(tmp_path: Path) -> None:
    base = pd.DataFrame(
        [
            {
                "pdb_ligand": "CLR/pdb1ABC.pdb",
                "matched_pocket_number": 1,
            }
        ]
    )
    task = {
        "label": "CLR/pdb1ABC.pdb",
        "rows": base.to_dict(orient="records"),
        "structure_dir": str(tmp_path / "missing_out"),
        "protein_pdb": str(tmp_path / "missing.pdb"),
        "caver_jar": str(tmp_path / "missing.jar"),
        "settings": {
            "probe_radius": 0.9,
            "shell_radius": 3.0,
            "shell_depth": 4.0,
            "clustering_threshold": 3.5,
            "timeout_s": 45,
            "max_structure_timeout_s": 300,
            "use_multi_start": False,
            "java_heap": "768m",
        },
    }
    with pytest.raises(ValueError, match="analysis_manifest_path requires analysis_output_root"):
        _run_tasks_and_write(
            [task],
            base=base,
            output_path=tmp_path / "out.parquet",
            workers=1,
            reports_dir=tmp_path / "reports",
            cache_dir=None,
            thresholds=TunnelBuildThresholds(
                max_missing_structure_frac=1.0,
                min_context_present_frac=0.0,
                min_profile_present_frac=0.0,
            ),
            analysis_manifest_path=tmp_path / "manifest.csv",
            analysis_output_root=None,
        )
