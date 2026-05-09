from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from Bio.PDB import PDBParser

from slipp_plus.feature_builders.aromatic_aliphatic import (
    ALIPHATIC_RESIDUES,
    AROMATIC_RESIDUES,
    _classify_residue_name,
    _closest_heavy_atom_distance,
    _compute_centroid,
    _extract_features_with_stats,
    _shell_index,
    _symmetric_jeffreys_log_ratio,
    extract_features,
)


def _write_pqr(path: Path, coordinates: list[tuple[float, float, float]]) -> None:
    lines = []
    for serial, (x, y, z) in enumerate(coordinates, start=1):
        lines.append(
            f"ATOM  {serial:5d}  C   APH A{serial:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  0.00  1.50\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _write_pdb(path: Path, residues: list[tuple[str, int, float, float, float, str]]) -> None:
    lines = []
    for serial, (resname, resid, x, y, z, atom_name) in enumerate(residues, start=1):
        lines.append(
            f"ATOM  {serial:5d} {atom_name:^4s}{resname:>4s} A{resid:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{20.00:6.2f}           C\n"
        )
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")


def _write_residue_atoms(
    path: Path,
    residue_atoms: list[tuple[str, int, list[tuple[str, str, float, float, float]]]],
) -> None:
    lines = []
    serial = 1
    for resname, resid, atoms in residue_atoms:
        for atom_name, element, x, y, z in atoms:
            lines.append(
                f"ATOM  {serial:5d} {atom_name:^4s}{resname:>4s} A{resid:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{20.00:6.2f}          {element:>2s}\n"
            )
            serial += 1
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")


def _make_structure(
    base_dir: Path,
    pdb_id: str,
    pocket_index: int,
    residues: list[tuple[str, int, float, float, float, str]],
    coordinates: list[tuple[float, float, float]],
) -> None:
    pockets_dir = base_dir / f"{pdb_id}_out" / "pockets"
    pockets_dir.mkdir(parents=True, exist_ok=True)
    _write_pdb(pockets_dir / f"pocket{pocket_index}_atm.pdb", residues)
    _write_pqr(pockets_dir / f"pocket{pocket_index}_vert.pqr", coordinates)


def test_centroid_computation(tmp_path: Path) -> None:
    pqr_path = tmp_path / "pocket0_vert.pqr"
    _write_pqr(pqr_path, [(0.0, 0.0, 0.0), (2.0, 4.0, 6.0), (4.0, 8.0, 12.0)])

    centroid = _compute_centroid(pqr_path)

    np.testing.assert_allclose(centroid, np.array([2.0, 4.0, 6.0]))


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        (0.0, 0),
        (2.999, 0),
        (3.0, 1),
        (5.999, 1),
        (6.0, 2),
        (8.999, 2),
        (9.0, 3),
        (12.0, 3),
        (12.001, None),
    ],
)
def test_shell_binning(distance: float, expected: int | None) -> None:
    assert _shell_index(distance) == expected


def test_closest_atom_distance_binning(tmp_path: Path) -> None:
    pockets_dir = tmp_path / "ABCD_out" / "pockets"
    pockets_dir.mkdir(parents=True)
    _write_pqr(pockets_dir / "pocket0_vert.pqr", [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
    _write_residue_atoms(
        pockets_dir / "pocket0_atm.pdb",
        [
            ("PHE", 1, [("CB", "C", 2.5, 0.0, 0.0)]),
            ("LEU", 2, [("CB", "C", 4.5, 0.0, 0.0)]),
            ("TYR", 3, [("CZ", "C", 8.5, 0.0, 0.0)]),
            ("MET", 4, [("SD", "S", 11.5, 0.0, 0.0)]),
        ],
    )

    frame = extract_features(tmp_path)

    assert frame["aromatic_count_shell1"].item() == 1
    assert frame["aliphatic_count_shell2"].item() == 1
    assert frame["aromatic_count_shell3"].item() == 1
    assert frame["aliphatic_count_shell4"].item() == 1


def test_aromatic_classification() -> None:
    for residue in ("PHE", "TYR", "TRP", "HIS"):
        assert residue in AROMATIC_RESIDUES
        assert _classify_residue_name(residue) == "aromatic"

    for residue in ("LEU", "ILE"):
        assert _classify_residue_name(residue) != "aromatic"


def test_aliphatic_classification() -> None:
    for residue in ("LEU", "ILE", "VAL", "ALA", "MET", "PRO", "CYS"):
        assert residue in ALIPHATIC_RESIDUES
        assert _classify_residue_name(residue) == "aliphatic"

    assert _classify_residue_name("PHE") != "aliphatic"


def test_ratio_smoothing(tmp_path: Path) -> None:
    _make_structure(
        tmp_path,
        pdb_id="ABCD",
        pocket_index=0,
        residues=[("PHE", 1, 1.0, 0.0, 0.0, "CA")],
        coordinates=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
    )
    _make_structure(
        tmp_path,
        pdb_id="ABCD",
        pocket_index=1,
        residues=[("LEU", 1, 1.0, 0.0, 0.0, "CA")],
        coordinates=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
    )

    frame = extract_features(tmp_path).sort("pocket_id")

    assert frame.height == 2
    assert frame["aromatic_count_shell1"].to_list() == [1, 0]
    assert frame["aliphatic_count_shell1"].to_list() == [0, 1]
    expected = math.log(3.0)
    ratios = frame["aromatic_aliphatic_ratio_shell1"].to_list()
    assert ratios[0] == pytest.approx(expected)
    assert ratios[1] == pytest.approx(-expected)


def test_symmetric_jeffreys_log_ratio_is_antisymmetric() -> None:
    expected = math.log((4.0 + 0.5) / (1.0 + 0.5))

    assert _symmetric_jeffreys_log_ratio(4, 1) == pytest.approx(expected)
    assert _symmetric_jeffreys_log_ratio(1, 4) == pytest.approx(-expected)
    assert _symmetric_jeffreys_log_ratio(3, 3) == pytest.approx(0.0)


def test_closest_atom_preferred_over_ca(tmp_path: Path) -> None:
    pockets_dir = tmp_path / "ABCD_out" / "pockets"
    pockets_dir.mkdir(parents=True)
    _write_pqr(pockets_dir / "pocket0_vert.pqr", [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
    _write_residue_atoms(
        pockets_dir / "pocket0_atm.pdb",
        [
            ("PHE", 1, [("CA", "C", 10.0, 0.0, 0.0), ("CB", "C", 4.0, 0.0, 0.0)]),
        ],
    )

    frame = extract_features(tmp_path)

    assert frame["aromatic_count_shell2"].item() == 1
    assert frame["aromatic_count_shell4"].item() == 0

    residue = next(
        PDBParser(QUIET=True)
        .get_structure("ABCD_0", pockets_dir / "pocket0_atm.pdb")
        .get_residues()
    )
    direct_distance = _closest_heavy_atom_distance(residue, np.array([0.0, 0.0, 0.0]))
    assert direct_distance == pytest.approx(4.0)


def test_extract_features_end_to_end(tmp_path: Path) -> None:
    _make_structure(
        tmp_path,
        pdb_id="ABCD",
        pocket_index=0,
        residues=[
            ("PHE", 1, 1.0, 0.0, 0.0, "CA"),
            ("LEU", 2, 4.0, 0.0, 0.0, "CA"),
            ("TYR", 3, 8.0, 0.0, 0.0, "CA"),
            ("MET", 4, 10.0, 0.0, 0.0, "CA"),
            ("GLY", 5, 1.5, 0.0, 0.0, "CA"),
        ],
        coordinates=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
    )

    frame = extract_features(tmp_path).sort("pocket_id")

    expected = pl.DataFrame(
        {
            "pocket_id": ["ABCD_0"],
            "aromatic_count_shell1": [1],
            "aromatic_count_shell2": [0],
            "aromatic_count_shell3": [1],
            "aromatic_count_shell4": [0],
            "aliphatic_count_shell1": [0],
            "aliphatic_count_shell2": [1],
            "aliphatic_count_shell3": [0],
            "aliphatic_count_shell4": [1],
            "aromatic_aliphatic_ratio_shell1": [math.log(3.0)],
            "aromatic_aliphatic_ratio_shell2": [-math.log(3.0)],
            "aromatic_aliphatic_ratio_shell3": [math.log(3.0)],
            "aromatic_aliphatic_ratio_shell4": [-math.log(3.0)],
        }
    )

    for column in frame.columns:
        actual = frame[column].to_list()
        target = expected[column].to_list()
        if (
            column.endswith("ratio_shell1")
            or column.endswith("ratio_shell2")
            or column.endswith("ratio_shell3")
            or column.endswith("ratio_shell4")
        ):
            assert actual == pytest.approx(target)
        else:
            assert actual == target


def test_extract_features_skips_unmatched_and_malformed(tmp_path: Path) -> None:
    pockets_dir = tmp_path / "WXYZ_out" / "pockets"
    pockets_dir.mkdir(parents=True)
    _write_pqr(pockets_dir / "pocket0_vert.pqr", [(0.0, 0.0, 0.0)])
    _write_pdb(pockets_dir / "pocket1_atm.pdb", [("PHE", 1, 1.0, 0.0, 0.0, "CA")])
    _write_pqr(pockets_dir / "pocket2_vert.pqr", [(0.0, 0.0, 0.0)])
    (pockets_dir / "pocket2_atm.pdb").write_text("not a pdb\n", encoding="utf-8")

    result = _extract_features_with_stats(tmp_path)

    assert result.extracted_pockets == 0
    assert any("missing atm file for pocket 0" in warning for warning in result.warnings)
    assert any("missing vert file for pocket 1" in warning for warning in result.warnings)
    assert any("failed to parse atm pdb" in warning for warning in result.warnings)


def test_residue_without_heavy_atoms_is_skipped(tmp_path: Path) -> None:
    pockets_dir = tmp_path / "ABCD_out" / "pockets"
    pockets_dir.mkdir(parents=True)
    _write_pqr(pockets_dir / "pocket0_vert.pqr", [(0.0, 0.0, 0.0)])
    _write_residue_atoms(
        pockets_dir / "pocket0_atm.pdb",
        [("PHE", 1, [("H1", "H", 1.0, 0.0, 0.0), ("H2", "H", 2.0, 0.0, 0.0)])],
    )

    result = _extract_features_with_stats(tmp_path)

    assert result.extracted_pockets == 1
    assert any("has no heavy atoms; skipped" in warning for warning in result.warnings)
    frame = extract_features(tmp_path)
    assert frame["aromatic_count_shell1"].item() == 0


def test_fixture_end_to_end() -> None:
    fixture_dir = Path("fixtures/fpocket_fixture")
    if not fixture_dir.exists():
        pytest.skip("fixtures/fpocket_fixture is not available in this checkout")

    frame = extract_features(fixture_dir)
    expected_rows = len(list(fixture_dir.glob("*_out/pockets/pocket*_vert.pqr")))

    assert frame.height == expected_rows
    aq3 = frame.filter(pl.col("pocket_id").str.starts_with("8AQ3_"))
    assert aq3.height > 0
    assert aq3["aromatic_count_shell1"].sum() > 0 or aq3["aromatic_count_shell2"].sum() > 0
