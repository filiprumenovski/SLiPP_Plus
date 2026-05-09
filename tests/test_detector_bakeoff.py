from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from slipp_plus.experiments.detector_bakeoff import (
    SCORE_SCHEMA,
    compute_hit_metrics,
    extract_fpocket_predictions,
    extract_ligand_atoms,
    score_structure,
    summarize,
)
from slipp_plus.feature_builders.aromatic_aliphatic import _compute_centroid

REPO_ROOT = Path(__file__).resolve().parents[1]
ADN_1BX4_PDB = REPO_ROOT / "data" / "structures" / "source_pdbs" / "ADN" / "pdb1BX4.pdb"
ADN_1BX4_OUT = REPO_ROOT / "data" / "structures" / "source_pdbs" / "ADN" / "pdb1BX4_out"


def test_compute_hit_metrics_synthetic() -> None:
    pred_centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    ligand_copy_a = np.array([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float)
    ligand_copy_b = np.array([[20.0, 0.0, 0.0]], dtype=float)

    dcc, dca = compute_hit_metrics(pred_centers, [ligand_copy_a, ligand_copy_b])

    np.testing.assert_allclose(dcc, np.array([3.0, 7.0, 2.0]))
    np.testing.assert_allclose(dca, np.array([2.0, 6.0, 1.0]))


def test_extract_ligand_atoms_adn() -> None:
    copies = extract_ligand_atoms(ADN_1BX4_PDB, "ADN")
    assert len(copies) >= 1
    for copy in copies:
        assert copy.ndim == 2
        assert copy.shape[1] == 3
        assert copy.shape[0] > 5


def test_extract_ligand_atoms_missing_raises(tmp_path: Path) -> None:
    empty_pdb = tmp_path / "empty.pdb"
    empty_pdb.write_text("END\n", encoding="utf-8")
    with pytest.raises(ValueError):
        extract_ligand_atoms(empty_pdb, "ADN")


def test_extract_fpocket_predictions_1BX4() -> None:
    frame = extract_fpocket_predictions(ADN_1BX4_OUT)
    assert frame.height >= 1
    assert frame.columns == ["pocket_rank", "center_x", "center_y", "center_z", "score"]
    first_row = frame.row(0, named=True)
    assert first_row["pocket_rank"] == 1

    direct = _compute_centroid(ADN_1BX4_OUT / "pockets" / "pocket1_vert.pqr")
    np.testing.assert_allclose(
        np.array([first_row["center_x"], first_row["center_y"], first_row["center_z"]]),
        direct,
        atol=1e-6,
    )


def test_score_structure_fpocket_only_1BX4() -> None:
    scores = score_structure(
        pdb_path=ADN_1BX4_PDB,
        ligand_code="ADN",
        fpocket_dir=ADN_1BX4_OUT,
        p2rank_csv=None,
    )
    assert scores.height > 0
    detectors = set(scores["detector"].unique().to_list())
    assert detectors == {"fpocket"}
    assert scores.schema == SCORE_SCHEMA

    top_row = scores.sort("pocket_rank").row(0, named=True)
    assert top_row["pocket_rank"] == 1
    assert top_row["dcc"] < 10.0
    assert top_row["dca"] < 5.0
    assert float(scores["dcc"].min()) == pytest.approx(top_row["dcc"])


def test_summarize_shape() -> None:
    df = pl.DataFrame(
        {
            "structure_id": ["s1", "s1", "s1", "s2", "s2", "s2"],
            "ligand_class": ["ADN", "ADN", "ADN", "ADN", "ADN", "ADN"],
            "detector": ["fpocket"] * 6,
            "pocket_rank": [1, 2, 3, 1, 2, 3],
            "center_x": [0.0] * 6,
            "center_y": [0.0] * 6,
            "center_z": [0.0] * 6,
            "score": [1.0, 0.5, 0.1, 2.0, 1.0, 0.1],
            "dcc": [1.0, 5.0, 10.0, 8.0, 9.0, 10.0],
            "dca": [0.5, 4.5, 9.0, 7.0, 8.5, 9.0],
            "hit_dcc_4A": [True, False, False, False, False, False],
            "hit_dca_4A": [True, False, False, False, False, False],
        }
    ).cast(SCORE_SCHEMA)

    summary = summarize(df)
    expected_columns = {
        "detector",
        "ligand_class",
        "n_structures",
        "top1_dcc",
        "top3_dcc",
        "top5_dcc",
        "top1_dca",
        "top3_dca",
        "top5_dca",
        "mean_rank_first_dcc_hit",
        "n_no_hit",
    }
    assert set(summary.columns) == expected_columns
    assert summary.height == 2
    assert set(summary["ligand_class"].to_list()) == {"ADN", "ALL"}

    adn_row = summary.filter(pl.col("ligand_class") == "ADN").row(0, named=True)
    assert adn_row["n_structures"] == 2
    assert adn_row["top1_dcc"] == pytest.approx(0.5)
    assert adn_row["top3_dcc"] == pytest.approx(0.5)
    assert adn_row["n_no_hit"] == 1
    assert adn_row["mean_rank_first_dcc_hit"] == pytest.approx(1.0)


def test_summarize_writes_markdown(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "structure_id": ["s1"],
            "ligand_class": ["ADN"],
            "detector": ["fpocket"],
            "pocket_rank": [1],
            "center_x": [0.0],
            "center_y": [0.0],
            "center_z": [0.0],
            "score": [1.0],
            "dcc": [1.0],
            "dca": [0.5],
            "hit_dcc_4A": [True],
            "hit_dca_4A": [True],
        }
    ).cast(SCORE_SCHEMA)

    out_md = tmp_path / "summary.md"
    summarize(df, out_markdown=out_md)
    assert out_md.exists()
    text = out_md.read_text(encoding="utf-8")
    assert "detector" in text
    assert "ADN" in text
    assert "ALL" in text
