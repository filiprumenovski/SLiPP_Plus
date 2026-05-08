"""Build ablation parquets from v_sterol artifacts.

This module turns feature-family hypotheses into explicit artifact sets:

- `v_sterol+derived`: holdout-safe 26 derived features on top of `v_sterol`
- `v_sterol+vdw22`: raw vdw22 surfaces added to the training parquet only
- `v_sterol+vdw22+derived`: full training/test variant with both families

The supporting-file holdouts do not ship raw vdw22 surfaces, so variants that
include `EXTRA_VDW22` are intentionally test-split-first. For those variants we
still copy holdout parquets through so the directory shape stays consistent, but
we report that holdout validation is not supported.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import DERIVED_FEATURES_26, EXTRA_VDW22, FEATURE_SETS
from .derived_features import compute_derived_features
from .ingest import _read_training_csv
from .schemas import validate_holdout, validate_training

_VDW22_FEATURE_SETS = frozenset({"v_sterol+vdw22", "v_sterol+vdw22+derived"})
_DERIVED_FEATURE_SETS = frozenset({"v_sterol+derived", "v_sterol+vdw22+derived", "v_sterol_v2"})


def _add_training_vdw22(base: pd.DataFrame, training_csv: Path) -> pd.DataFrame:
    raw = _read_training_csv(training_csv)[["pdb_ligand", *EXTRA_VDW22]].copy()
    enriched = base.merge(raw, on="pdb_ligand", how="left", validate="many_to_one")
    missing = [column for column in EXTRA_VDW22 if column not in enriched.columns]
    if missing:
        raise ValueError(f"failed to join raw vdw22 columns: {missing}")
    if enriched[list(EXTRA_VDW22)].isna().any().any():
        raise ValueError("training vdw22 join introduced NaN values")
    return enriched


def _materialize_frame(
    base: pd.DataFrame,
    *,
    feature_set: str,
    dataset_label: str,
    training_csv: Path | None,
) -> pd.DataFrame:
    frame = base.copy()
    if feature_set in _VDW22_FEATURE_SETS:
        if dataset_label == "training":
            if training_csv is None:
                raise ValueError(
                    f"{feature_set} requires training_csv to recover raw vdw22 surfaces"
                )
            frame = _add_training_vdw22(frame, training_csv)
        else:
            # Holdouts do not contain raw vdw22 surfaces in the published xlsx.
            pass
    if feature_set in _DERIVED_FEATURE_SETS:
        frame = compute_derived_features(frame)
    return frame


def build_v_sterol_ablation_from_v_sterol(
    v_sterol_dir: Path,
    output_dir: Path,
    *,
    feature_set: str,
    training_csv: Path | None = None,
) -> dict[str, object]:
    """Materialize an ablation parquet set from `processed/v_sterol` artifacts."""

    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature_set {feature_set!r}")
    if feature_set not in (_VDW22_FEATURE_SETS | _DERIVED_FEATURE_SETS):
        raise ValueError(f"feature_set {feature_set!r} is not a supported v_sterol ablation")

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = FEATURE_SETS[feature_set]

    full = _materialize_frame(
        pd.read_parquet(v_sterol_dir / "full_pockets.parquet"),
        feature_set=feature_set,
        dataset_label="training",
        training_csv=training_csv,
    )
    apo = _materialize_frame(
        pd.read_parquet(v_sterol_dir / "apo_pdb_holdout.parquet"),
        feature_set=feature_set,
        dataset_label="apo-PDB holdout",
        training_csv=training_csv,
    )
    af = _materialize_frame(
        pd.read_parquet(v_sterol_dir / "alphafold_holdout.parquet"),
        feature_set=feature_set,
        dataset_label="AlphaFold holdout",
        training_csv=training_csv,
    )

    validate_training(full, feature_columns)
    holdout_supported = feature_set not in _VDW22_FEATURE_SETS
    if holdout_supported:
        validate_holdout(apo, feature_columns)
        validate_holdout(af, feature_columns)

    full_path = output_dir / "full_pockets.parquet"
    apo_path = output_dir / "apo_pdb_holdout.parquet"
    af_path = output_dir / "alphafold_holdout.parquet"
    full.to_parquet(full_path, index=False)
    apo.to_parquet(apo_path, index=False)
    af.to_parquet(af_path, index=False)

    return {
        "feature_set": feature_set,
        "full_rows": len(full),
        "apo_rows": len(apo),
        "af_rows": len(af),
        "holdout_supported": holdout_supported,
        "full_output": full_path,
        "apo_output": apo_path,
        "af_output": af_path,
        "added_columns": [
            column for column in feature_columns if column not in FEATURE_SETS["v_sterol"]
        ],
        "derived_columns": list(DERIVED_FEATURES_26)
        if feature_set in _DERIVED_FEATURE_SETS
        else [],
        "vdw22_columns": list(EXTRA_VDW22) if feature_set in _VDW22_FEATURE_SETS else [],
    }
