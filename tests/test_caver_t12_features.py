from __future__ import annotations

from pathlib import Path

import pandas as pd

from slipp_plus.caver_t12_features import build_training_v_caver_t12_parquet
from slipp_plus.constants import CAVER_T12_FEATURES_17, FEATURE_SETS


def test_caver_t12_registry_shape() -> None:
    assert FEATURE_SETS["v_caver_t12"][-17:] == CAVER_T12_FEATURES_17
    assert len(FEATURE_SETS["v_caver_t12"]) == len(FEATURE_SETS["v_sterol"]) + 17


def test_build_training_v_caver_t12_parquet_smoke(tmp_path: Path) -> None:
    base = pd.DataFrame(
        [
            {
                "pdb_ligand": "CLR/pdb1ABC.pdb",
                "matched_pocket_number": 1,
                "class_10": "CLR",
                "pock_vol": 1.0,
            },
            {
                "pdb_ligand": "CLR/pdb1ABC.pdb",
                "matched_pocket_number": 2,
                "class_10": "CLR",
                "pock_vol": 2.0,
            },
        ]
    )
    base_parquet = tmp_path / "base.parquet"
    base.to_parquet(base_parquet, index=False)

    analysis = tmp_path / "analysis" / "CLR__pdb1ABC"
    analysis.mkdir(parents=True)
    (analysis / "tunnels.csv").write_text(
        "\n".join(
            [
                "Tunnel cluster,Starting point,Length,Bottleneck radius,Curvature,Throughput,Avg R",
                "1,0,12.0,2.1,1.0,0.9,2.4",
                "2,1,5.0,1.2,1.0,0.1,1.5",
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
                "1,6.0,6,0,0,2.3",
                "1,12.0,12,0,0,2.6",
                "2,0.0,0,0,0,1.2",
                "2,5.0,0,5,0,1.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (analysis / "residues.csv").write_text(
        "Tunnel cluster,Residue\n1,LEU 1 A\n2,ASP 2 A\n",
        encoding="utf-8",
    )

    manifest = pd.DataFrame(
        [
            {
                "pdb_ligand": "CLR/pdb1ABC.pdb",
                "matched_pocket_number": 1,
                "starting_point_index": 0,
                "pocket_axial_length": 6.0,
                "pocket_axis_x": 1.0,
                "pocket_axis_y": 0.0,
                "pocket_axis_z": 0.0,
                "analysis_subdir": "CLR__pdb1ABC",
            },
            {
                "pdb_ligand": "CLR/pdb1ABC.pdb",
                "matched_pocket_number": 2,
                "starting_point_index": 1,
                "pocket_axial_length": 3.0,
                "pocket_axis_x": 0.0,
                "pocket_axis_y": 1.0,
                "pocket_axis_z": 0.0,
                "analysis_subdir": "CLR__pdb1ABC",
            },
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    output_path = tmp_path / "out.parquet"
    summary = build_training_v_caver_t12_parquet(
        base_parquet=base_parquet,
        manifest_path=manifest_path,
        output_path=output_path,
        analysis_root=tmp_path / "analysis",
        reports_dir=tmp_path / "reports",
    )

    assert summary["rows"] == 2
    enriched = pd.read_parquet(output_path)
    assert len(enriched) == 2
    assert "caver_primary_length" in enriched.columns
    assert enriched.loc[0, "caver_primary_length"] == 12.0
    assert enriched.loc[1, "caver_primary_length"] == 5.0
