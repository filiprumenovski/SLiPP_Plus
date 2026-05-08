from __future__ import annotations

from pathlib import Path

import pandas as pd

from slipp_plus.constants import FEATURE_SETS
from slipp_plus.v_sterol_ablation import build_v_sterol_ablation_from_v_sterol


def _training_frame() -> pd.DataFrame:
    row = {column: 1.0 for column in FEATURE_SETS["v_sterol"]}
    row.update(
        {
            "pdb_ligand": "CLR/pdb1ABC.pdb",
            "class_10": "CLR",
            "class_binary": 1,
        }
    )
    return pd.DataFrame([row])


def _holdout_frame() -> pd.DataFrame:
    row = {column: 1.0 for column in FEATURE_SETS["v_sterol"]}
    row.update(
        {
            "structure_id": "1ABC",
            "ligand": "CLR",
            "class_binary": 1,
        }
    )
    return pd.DataFrame([row])


def test_build_v_sterol_derived_ablation(tmp_path: Path) -> None:
    v_sterol_dir = tmp_path / "v_sterol"
    out_dir = tmp_path / "v_sterol_derived"
    v_sterol_dir.mkdir()
    _training_frame().to_parquet(v_sterol_dir / "full_pockets.parquet", index=False)
    _holdout_frame().to_parquet(v_sterol_dir / "apo_pdb_holdout.parquet", index=False)
    _holdout_frame().to_parquet(v_sterol_dir / "alphafold_holdout.parquet", index=False)

    result = build_v_sterol_ablation_from_v_sterol(
        v_sterol_dir=v_sterol_dir,
        output_dir=out_dir,
        feature_set="v_sterol+derived",
    )

    assert result["holdout_supported"] is True
    full = pd.read_parquet(out_dir / "full_pockets.parquet")
    apo = pd.read_parquet(out_dir / "apo_pdb_holdout.parquet")
    assert set(FEATURE_SETS["v_sterol+derived"]).issubset(full.columns)
    assert set(FEATURE_SETS["v_sterol+derived"]).issubset(apo.columns)
    assert len(full) == 1
    assert len(apo) == 1
