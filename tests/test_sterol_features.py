"""Tests for the ``v_sterol`` sterol-targeted feature extractor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from slipp_plus.aromatic_aliphatic import _compute_centroid, _symmetric_jeffreys_log_ratio
from slipp_plus.constants import AA20, FEATURE_SETS
from slipp_plus.sterol_features import (
    CHEMISTRY_GROUP_ORDER,
    CHEMISTRY_GROUPS,
    POCKET_GEOMETRY_COLS,
    STEROL_CHEMISTRY_SHELL_COLS,
    STEROL_FEATURES_38,
    _collect_vert_coords,
    _pocket_pca_features,
    _polar_hydrophobic_ratios,
    _protein_ca_stats,
    build_training_v_sterol_parquet,
    extract_pocket_sterol_features,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PDBS = REPO_ROOT / "data" / "structures" / "source_pdbs"
BASE_PARQUET = REPO_ROOT / "processed" / "v49" / "full_pockets.parquet"
V_STEROL_PARQUET = REPO_ROOT / "processed" / "v_sterol" / "full_pockets.parquet"


# ---------------------------------------------------------------------------
# Group shape invariants
# ---------------------------------------------------------------------------
def test_chemistry_groups_cover_19_of_20_amino_acids() -> None:
    covered = set().union(*CHEMISTRY_GROUPS.values())
    expected_minus_gly = set(AA20) - {"GLY"}
    assert covered == expected_minus_gly
    assert "GLY" not in covered


def test_chemistry_groups_are_disjoint() -> None:
    seen: set[str] = set()
    for residues in CHEMISTRY_GROUPS.values():
        overlap = seen & residues
        assert not overlap, f"residues in two groups: {overlap}"
        seen |= residues


def test_group_order_matches_registry() -> None:
    assert set(CHEMISTRY_GROUP_ORDER) == set(CHEMISTRY_GROUPS.keys())
    assert len(STEROL_CHEMISTRY_SHELL_COLS) == 7 * 4 + 4
    assert len(POCKET_GEOMETRY_COLS) == 6
    assert len(STEROL_FEATURES_38) == 38
    assert len(FEATURE_SETS["v_sterol"]) == 17 + 20 + 12 + 32 + 6


# ---------------------------------------------------------------------------
# Single-pocket extraction against a real fpocket output
# ---------------------------------------------------------------------------
DIVERSE_STRUCTURE = ("CLR", "pdb1LRI", 1)


@pytest.mark.skipif(
    not (SOURCE_PDBS / DIVERSE_STRUCTURE[0] / f"{DIVERSE_STRUCTURE[1]}_out").exists(),
    reason="diverse_subset fixture not available",
)
def test_diverse_subset_extraction_matches_shared_centroid() -> None:
    class_code, stem, pocket_index = DIVERSE_STRUCTURE
    structure_dir = SOURCE_PDBS / class_code / f"{stem}_out"
    atm = structure_dir / "pockets" / f"pocket{pocket_index}_atm.pdb"
    vert = structure_dir / "pockets" / f"pocket{pocket_index}_vert.pqr"
    protein = SOURCE_PDBS / class_code / f"{stem}.pdb"

    features = extract_pocket_sterol_features(atm, vert, protein)
    assert set(features.keys()) == set(STEROL_FEATURES_38)

    shared_centroid = _compute_centroid(vert)
    vert_coords = _collect_vert_coords(vert)
    pca_direct = _pocket_pca_features(vert_coords)
    for key in ("pocket_lam1", "pocket_lam2", "pocket_lam3"):
        assert features[key] == pytest.approx(pca_direct[key], rel=1e-9, abs=1e-12)

    # ``_compute_centroid`` and vert coords mean align to within floating point:
    np.testing.assert_allclose(shared_centroid, vert_coords.mean(axis=0), rtol=1e-9)


@pytest.mark.skipif(
    not (SOURCE_PDBS / DIVERSE_STRUCTURE[0] / f"{DIVERSE_STRUCTURE[1]}_out").exists(),
    reason="diverse_subset fixture not available",
)
def test_sanity_ranges_on_diverse_structure() -> None:
    class_code, stem, pocket_index = DIVERSE_STRUCTURE
    structure_dir = SOURCE_PDBS / class_code / f"{stem}_out"
    atm = structure_dir / "pockets" / f"pocket{pocket_index}_atm.pdb"
    vert = structure_dir / "pockets" / f"pocket{pocket_index}_vert.pqr"
    protein = SOURCE_PDBS / class_code / f"{stem}.pdb"

    features = extract_pocket_sterol_features(atm, vert, protein)

    # Eigenvalue ordering guarantees elongation >= 1 and planarity >= 1.
    assert features["pocket_elongation"] >= 1.0 - 1e-9
    assert features["pocket_planarity"] >= 1.0 - 1e-9
    assert features["pocket_lam1"] >= features["pocket_lam2"] - 1e-9
    assert features["pocket_lam2"] >= features["pocket_lam3"] - 1e-9
    # Burial in [0, ~1.5] by construction (1.0 = at bounding radius).
    assert 0.0 <= features["pocket_burial"] <= 2.0
    # Counts must be non-negative ints.
    for column in STEROL_CHEMISTRY_SHELL_COLS:
        if column.endswith(("1", "2", "3", "4")) and "count" in column:
            assert int(features[column]) >= 0


@pytest.mark.skipif(
    not (SOURCE_PDBS / DIVERSE_STRUCTURE[0] / f"{DIVERSE_STRUCTURE[1]}.pdb").exists(),
    reason="source_pdbs fixture not available",
)
def test_protein_ca_stats_returns_bounded_values() -> None:
    class_code, stem, _ = DIVERSE_STRUCTURE
    protein = SOURCE_PDBS / class_code / f"{stem}.pdb"
    centroid, max_spread = _protein_ca_stats(protein)
    assert centroid.shape == (3,)
    assert max_spread > 0


def test_polar_hydrophobic_ratios_are_symmetric() -> None:
    counts = {
        "aromatic_pi_count_shell1": 0,
        "aromatic_polar_count_shell1": 1,
        "bulky_hydrophobic_count_shell1": 0,
        "small_special_count_shell1": 0,
        "polar_neutral_count_shell1": 0,
        "cationic_count_shell1": 0,
        "anionic_count_shell1": 0,
        "aromatic_pi_count_shell2": 1,
        "aromatic_polar_count_shell2": 0,
        "bulky_hydrophobic_count_shell2": 0,
        "small_special_count_shell2": 0,
        "polar_neutral_count_shell2": 0,
        "cationic_count_shell2": 0,
        "anionic_count_shell2": 0,
        "aromatic_pi_count_shell3": 1,
        "aromatic_polar_count_shell3": 1,
        "bulky_hydrophobic_count_shell3": 0,
        "small_special_count_shell3": 0,
        "polar_neutral_count_shell3": 0,
        "cationic_count_shell3": 0,
        "anionic_count_shell3": 0,
        "aromatic_pi_count_shell4": 0,
        "aromatic_polar_count_shell4": 0,
        "bulky_hydrophobic_count_shell4": 0,
        "small_special_count_shell4": 0,
        "polar_neutral_count_shell4": 0,
        "cationic_count_shell4": 0,
        "anionic_count_shell4": 0,
    }

    ratios = _polar_hydrophobic_ratios(counts)

    expected = _symmetric_jeffreys_log_ratio(1, 0)
    assert ratios["polar_hydrophobic_ratio_shell1"] == pytest.approx(expected)
    assert ratios["polar_hydrophobic_ratio_shell2"] == pytest.approx(-expected)
    assert ratios["polar_hydrophobic_ratio_shell3"] == pytest.approx(0.0)
    assert ratios["polar_hydrophobic_ratio_shell4"] == pytest.approx(0.0)


def test_polar_hydrophobic_ratios_match_shared_symmetric_helper() -> None:
    counts = {
        f"{group}_count_shell{shell}": 0
        for group in CHEMISTRY_GROUP_ORDER
        for shell in (1, 2, 3, 4)
    }
    counts["aromatic_polar_count_shell1"] = 2
    counts["polar_neutral_count_shell1"] = 1
    counts["bulky_hydrophobic_count_shell1"] = 4
    counts["aromatic_pi_count_shell1"] = 1

    ratios = _polar_hydrophobic_ratios(counts)

    assert ratios["polar_hydrophobic_ratio_shell1"] == pytest.approx(
        _symmetric_jeffreys_log_ratio(3, 5)
    )


# ---------------------------------------------------------------------------
# Full-pipeline schema check
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not V_STEROL_PARQUET.exists(), reason="v_sterol parquet not built")
def test_v_sterol_parquet_schema() -> None:
    frame = pd.read_parquet(V_STEROL_PARQUET)
    for column in FEATURE_SETS["v_sterol"]:
        assert column in frame.columns, f"missing column {column}"
    # counts non-negative, ratios finite, elongation >= 1, burial finite.
    for column in STEROL_CHEMISTRY_SHELL_COLS:
        if "count" in column:
            assert (frame[column] >= 0).all()
        else:
            assert np.isfinite(frame[column]).all()
    assert (frame["pocket_elongation"] >= 1.0 - 1e-9).all()
    assert (frame["pocket_planarity"] >= 1.0 - 1e-9).all()
    assert np.isfinite(frame["pocket_burial"]).all()


# ---------------------------------------------------------------------------
# Small subset build-and-match check
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not BASE_PARQUET.exists() or not SOURCE_PDBS.exists(),
    reason="v49 base parquet / source_pdbs not available",
)
def test_training_build_on_small_subset(tmp_path: Path) -> None:
    full = pd.read_parquet(BASE_PARQUET)
    target = "CLR/pdb1LRI.pdb"
    if target not in full["pdb_ligand"].unique():
        pytest.skip(f"{target} absent from base parquet")
    subset = full[full["pdb_ligand"] == target].head(3).copy()
    subset_path = tmp_path / "small_base.parquet"
    subset.to_parquet(subset_path, index=False)

    out_path = tmp_path / "small_sterol.parquet"
    summary = build_training_v_sterol_parquet(
        base_parquet=subset_path,
        source_pdbs_root=SOURCE_PDBS,
        output_path=out_path,
        workers=1,
    )
    assert summary["rows"] == len(subset)
    enriched = pd.read_parquet(out_path)
    for column in STEROL_FEATURES_38:
        assert column in enriched.columns
    assert len(enriched) == len(subset)
