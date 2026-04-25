"""Tests for cheap alpha-sphere graph tunnel features."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from slipp_plus.constants import FEATURE_SETS
from slipp_plus.graph_tunnel_features import (
    GRAPH_TUNNEL_FEATURES_3,
    AlphaSphere,
    graph_tunnel_features_for_pocket,
    load_alpha_spheres,
)


def _sphere(x: float, radius: float, pocket_number: int = 1) -> AlphaSphere:
    return AlphaSphere(
        coord=np.asarray([x, 0.0, 0.0], dtype=float),
        radius=radius,
        pocket_number=pocket_number,
        atom_name="C",
    )


def test_graph_tunnel_registry_shape() -> None:
    assert GRAPH_TUNNEL_FEATURES_3 == [
        "tunnel_length",
        "tunnel_bottleneck_radius",
        "tunnel_length_over_axial_length",
    ]
    assert FEATURE_SETS["v_graph_tunnel"] == FEATURE_SETS["v_sterol"] + GRAPH_TUNNEL_FEATURES_3


def test_load_alpha_spheres_from_fpocket_pqr(tmp_path: Path) -> None:
    pqr = tmp_path / "pocket1_vert.pqr"
    pqr.write_text(
        "\n".join(
            [
                "HEADER example",
                "ATOM      1    O STP     7      15.115  13.645  30.390    0.00     3.63",
                "ATOM      2    C STP     7      19.815  -2.757  22.179    0.00     3.59",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    spheres = load_alpha_spheres(pqr)

    assert len(spheres) == 2
    assert spheres[0].pocket_number == 7
    assert spheres[0].atom_name == "O"
    np.testing.assert_allclose(spheres[0].coord, np.array([15.115, 13.645, 30.390]))
    assert spheres[0].radius == 3.63


def test_graph_tunnel_dijkstra_path_and_bottleneck() -> None:
    pocket = [
        _sphere(0.0, 2.0),
        _sphere(3.0, 2.0),
        _sphere(6.0, 0.9),
    ]
    outside = [
        _sphere(9.0, 2.0, pocket_number=2),
        _sphere(12.0, 2.2, pocket_number=2),
    ]

    features = graph_tunnel_features_for_pocket(
        pocket,
        pocket + outside,
        throat_radius=10.0,
        touch_eps=0.2,
    )

    assert math.isclose(features["tunnel_length"], 9.0)
    assert math.isclose(features["tunnel_bottleneck_radius"], 0.9)
    assert features["tunnel_length_over_axial_length"] > 1.0


def test_graph_tunnel_real_fpocket_smoke() -> None:
    root = Path("data/structures/source_pdbs/PLM/pdb7A77_out")
    if not root.exists():
        return
    pocket = load_alpha_spheres(root / "pockets" / "pocket1_vert.pqr", default_pocket_number=1)
    all_spheres = load_alpha_spheres(root / "pdb7A77_pockets.pqr")

    features = graph_tunnel_features_for_pocket(pocket, all_spheres)

    assert set(features) == set(GRAPH_TUNNEL_FEATURES_3)
    assert all(math.isfinite(float(value)) for value in features.values())
    assert features["tunnel_length"] >= 0.0
    assert features["tunnel_bottleneck_radius"] >= 0.0
