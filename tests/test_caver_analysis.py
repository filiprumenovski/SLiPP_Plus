from __future__ import annotations

import math
from pathlib import Path

from slipp_plus.caver_analysis import (
    CAVER_T12_FEATURES_17,
    CaverPocketContext,
    cast_caver_t12_features,
    derive_caver_t12_features_by_pocket,
    parse_caver_tunnels,
    safe_caver_t12_defaults,
)


def test_caver_t12_defaults_complete_and_finite() -> None:
    defaults = safe_caver_t12_defaults()
    assert set(defaults) == set(CAVER_T12_FEATURES_17)
    for value in defaults.values():
        assert math.isfinite(float(value))


def test_parse_and_derive_caver_t12_features(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    (analysis / "tunnel_characteristics.csv").write_text(
        "\n".join(
            [
                "Tunnel cluster,Starting point,Length,Bottleneck radius,Curvature,Throughput,Avg R",
                "1,0,20.0,2.0,1.1,0.8,2.5",
                "2,0,15.0,1.5,1.2,0.4,2.0",
                "3,1,8.0,1.0,1.0,0.2,1.2",
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
                "1,5.0,5,0,0,2.2",
                "1,10.0,10,1,0,2.7",
                "1,20.0,20,1,0,3.0",
                "2,0.0,0,0,0,1.8",
                "2,7.5,6,0,0,1.5",
                "2,15.0,12,0,0,2.1",
                "3,0.0,0,0,0,1.0",
                "3,4.0,0,4,0,1.3",
                "3,8.0,0,8,0,1.4",
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
                "3,TYR 7 A",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    tunnels = parse_caver_tunnels(analysis)
    assert len(tunnels) == 3

    feature_map = derive_caver_t12_features_by_pocket(
        analysis,
        [
            CaverPocketContext(7, 0, 10.0, (1.0, 0.0, 0.0)),
            CaverPocketContext(8, 1, 4.0, (0.0, 1.0, 0.0)),
        ],
    )

    assert feature_map[7]["caver_tunnel_count"] == 2
    assert feature_map[7]["caver_primary_length"] == 20.0
    assert feature_map[7]["caver_median_bottleneck_radius"] == 1.75
    assert feature_map[7]["caver_primary_radius_max"] == 3.0
    assert feature_map[7]["caver_primary_length_over_axial"] == 2.0
    assert feature_map[8]["caver_tunnel_count"] == 1
    assert feature_map[8]["caver_primary_alignment_angle_deg"] == 0.0

    casted = cast_caver_t12_features(feature_map[7])
    assert set(casted) == set(CAVER_T12_FEATURES_17)


def test_one_based_starting_points_are_matched_correctly(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    (analysis / "tunnels.csv").write_text(
        "\n".join(
            [
                "Tunnel cluster,Starting point,Length,Bottleneck radius,Curvature,Throughput,Avg R",
                "1,1,12.0,2.0,1.0,0.9,2.1",
                "2,2,7.0,1.3,1.0,0.5,1.4",
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
                "1,12.0,12,0,0,2.1",
                "2,0.0,0,0,0,1.3",
                "2,7.0,0,7,0,1.4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    feature_map = derive_caver_t12_features_by_pocket(
        analysis,
        [
            CaverPocketContext(1, 0, 6.0, (1.0, 0.0, 0.0)),
            CaverPocketContext(2, 1, 3.5, (0.0, 1.0, 0.0)),
        ],
    )

    assert feature_map[1]["caver_primary_length"] == 12.0
    assert feature_map[2]["caver_primary_length"] == 7.0


def test_missing_axis_uses_neutral_alignment_angle(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    (analysis / "tunnels.csv").write_text(
        "\n".join(
            [
                "Tunnel cluster,Starting point,Length,Bottleneck radius,Curvature,Throughput,Avg R",
                "1,0,9.0,2.0,1.0,0.9,2.1",
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
                "1,9.0,9,0,0,2.1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    feature_map = derive_caver_t12_features_by_pocket(
        analysis,
        [CaverPocketContext(1, 0, 3.0, None)],
    )

    assert feature_map[1]["caver_primary_alignment_angle_deg"] == 90.0
