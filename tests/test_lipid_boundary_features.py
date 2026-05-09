"""Tests for the v_lipid_boundary feature extractor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from slipp_plus.constants import FEATURE_SETS, LIPID_BOUNDARY_FEATURES_22
from slipp_plus.feature_builders.lipid_boundary_features import (
    _collect_vert_coords,
    build_training_v_lipid_boundary_parquet,
    features_from_coordinates_and_chains,
)


class _Atom:
    element = "C"

    def __init__(self, coord: tuple[float, float, float]) -> None:
        self._coord = np.asarray(coord, dtype=float)

    def get_coord(self) -> np.ndarray:
        return self._coord


class _Residue:
    def __init__(self, resname: str, resid: int, coord: tuple[float, float, float]) -> None:
        self._resname = resname
        self._id = (" ", resid, " ")
        self._atoms = [_Atom(coord)]

    def __iter__(self):
        return iter(self._atoms)

    def get_resname(self) -> str:
        return self._resname

    def get_id(self) -> tuple[str, int, str]:
        return self._id


def _chain(residues: list[_Residue], sequence: str | None = None) -> dict[str, object]:
    return {
        "chain_id": "A",
        "sequence": sequence or "".join("X" for _ in residues),
        "residues": residues,
    }


def _reflected_chain(chain: dict[str, object]) -> dict[str, object]:
    residues = []
    for residue in chain["residues"]:  # type: ignore[union-attr]
        atom = next(iter(residue))
        x, y, z = atom.get_coord()
        residues.append(_Residue(residue.get_resname(), residue.get_id()[1], (-x, y, z)))
    return {"chain_id": chain["chain_id"], "sequence": chain["sequence"], "residues": residues}


def _vert_coords() -> np.ndarray:
    return np.asarray(
        [
            [-5.0, 0.0, 0.0],
            [-3.0, 0.5, 0.0],
            [-1.0, 0.2, 0.4],
            [1.0, -0.2, 0.2],
            [3.0, 0.5, -0.1],
            [5.0, 0.0, 0.0],
        ],
        dtype=float,
    )


def test_feature_registry_shape() -> None:
    assert FEATURE_SETS["v_lipid_boundary"][-22:] == LIPID_BOUNDARY_FEATURES_22
    assert len(FEATURE_SETS["v_lipid_boundary"]) == len(FEATURE_SETS["v_sterol"]) + 22


def test_axial_features_are_stable_under_axis_reflection() -> None:
    residues = [
        _Residue("SER", 1, (-5.0, 0.0, 0.0)),
        _Residue("LYS", 2, (-4.5, 0.0, 0.0)),
        _Residue("LEU", 3, (0.0, 0.4, 0.0)),
        _Residue("GLY", 4, (0.5, 0.0, 0.0)),
        _Residue("VAL", 5, (5.0, 0.0, 0.0)),
    ]
    chain = _chain(residues, sequence="SKLGV")

    original = features_from_coordinates_and_chains(_vert_coords(), [chain])
    reflected = features_from_coordinates_and_chains(
        _vert_coords() * np.asarray([-1.0, 1.0, 1.0]),
        [_reflected_chain(chain)],
    )

    assert set(original) == set(LIPID_BOUNDARY_FEATURES_22)
    for column in LIPID_BOUNDARY_FEATURES_22:
        assert original[column] == pytest.approx(reflected[column], rel=1e-8, abs=1e-8)


def test_p_loop_and_anchor_counts_on_synthetic_chain() -> None:
    sequence = "GAAAAGKS"
    residues = [
        _Residue("GLY", 1, (-5.0, 0.0, 0.0)),
        _Residue("ALA", 2, (-4.0, 0.0, 0.0)),
        _Residue("ALA", 3, (-3.0, 0.0, 0.0)),
        _Residue("ALA", 4, (-2.0, 0.0, 0.0)),
        _Residue("ALA", 5, (-1.0, 0.0, 0.0)),
        _Residue("GLY", 6, (-0.5, 0.0, 0.0)),
        _Residue("LYS", 7, (-4.8, 0.0, 0.0)),
        _Residue("SER", 8, (-5.1, 0.0, 0.0)),
        _Residue("LEU", 9, (5.0, 0.0, 0.0)),
    ]
    features = features_from_coordinates_and_chains(_vert_coords(), [_chain(residues, sequence)])

    assert features["lb_p_loop_like_motif_count"] >= 1.0
    assert features["lb_polar_end_donor_count"] >= 1.0
    assert features["lb_polar_end_acceptor_count"] >= 1.0
    assert features["lb_phosphate_anchor_score"] > 0.0
    assert all(np.isfinite(float(features[column])) for column in LIPID_BOUNDARY_FEATURES_22)


def _atom_line(serial: int, resname: str, resid: int, x: float, y: float, z: float) -> str:
    return (
        f"ATOM  {serial:5d}  CA  {resname:>3} A{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n"
    )


def _vert_line(serial: int, x: float, y: float, z: float) -> str:
    return f"ATOM  {serial:5d}  AP  STP A{serial:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  0.00  1.50\n"


def test_build_training_v_lipid_boundary_parquet_smoke(tmp_path: Path) -> None:
    source_root = tmp_path / "source_pdbs"
    structure_dir = source_root / "COA" / "pdb1ABC_out"
    pockets_dir = structure_dir / "pockets"
    pockets_dir.mkdir(parents=True)
    protein = source_root / "COA" / "pdb1ABC.pdb"
    protein.write_text(
        "".join(
            [
                _atom_line(1, "GLY", 1, -5.0, 0.0, 0.0),
                _atom_line(2, "ALA", 2, -4.0, 0.0, 0.0),
                _atom_line(3, "ALA", 3, -3.0, 0.0, 0.0),
                _atom_line(4, "ALA", 4, -2.0, 0.0, 0.0),
                _atom_line(5, "ALA", 5, -1.0, 0.0, 0.0),
                _atom_line(6, "GLY", 6, -0.5, 0.0, 0.0),
                _atom_line(7, "LYS", 7, -4.8, 0.0, 0.0),
                _atom_line(8, "SER", 8, -5.1, 0.0, 0.0),
                _atom_line(9, "LEU", 9, 5.0, 0.0, 0.0),
                "END\n",
            ]
        ),
        encoding="utf-8",
    )
    (pockets_dir / "pocket1_vert.pqr").write_text(
        "".join(_vert_line(i + 1, *coord) for i, coord in enumerate(_vert_coords())),
        encoding="utf-8",
    )

    row = {
        "pdb_ligand": "COA/pdb1ABC.pdb",
        "class_10": "PP",
        "class_binary": 0,
        "matched_pocket_number": 1,
    }
    row.update({column: 0.0 for column in FEATURE_SETS["v_sterol"]})
    base = pd.DataFrame([row])
    base_path = tmp_path / "base.parquet"
    base.to_parquet(base_path, index=False)

    output_path = tmp_path / "out.parquet"
    summary = build_training_v_lipid_boundary_parquet(
        base_parquet=base_path,
        source_pdbs_root=source_root,
        output_path=output_path,
        structural_join_parquet=None,
        reports_dir=tmp_path / "reports",
        workers=1,
    )

    assert summary["rows"] == 1
    enriched = pd.read_parquet(output_path)
    assert len(enriched) == 1
    assert enriched.loc[0, "class_10"] == "PP"
    assert _collect_vert_coords(pockets_dir / "pocket1_vert.pqr").shape == (6, 3)
    for column in LIPID_BOUNDARY_FEATURES_22:
        assert column in enriched.columns
        assert np.isfinite(float(enriched.loc[0, column]))
