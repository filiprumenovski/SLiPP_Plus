"""Pandera schemas enforcing PROMPT.md Rule 1 at ingest time.

Schemas here are *dataframe-shape* guarantees; per-class count assertions that
depend on runtime configuration live in ``ingest.assert_rule_1``.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from .constants import CLASS_10, SELECTED_17


def _numeric_feature_columns(columns: list[str]) -> dict[str, pa.Column]:
    return {c: pa.Column(float, nullable=False, coerce=True) for c in columns}


def training_schema(feature_columns: list[str]) -> pa.DataFrameSchema:
    """Schema for processed/train_pockets.parquet and test_pockets.parquet."""
    cols: dict[str, pa.Column] = _numeric_feature_columns(feature_columns)
    cols["class_10"] = pa.Column(
        str, checks=pa.Check.isin(CLASS_10), nullable=False
    )
    cols["class_binary"] = pa.Column(
        int, checks=pa.Check.isin({0, 1}), nullable=False
    )
    cols["pdb_ligand"] = pa.Column(str, nullable=False)  # e.g. "ADN/pdb1BX4.pdb"
    return pa.DataFrameSchema(cols, strict="filter", coerce=True)


def holdout_schema(feature_columns: list[str]) -> pa.DataFrameSchema:
    """Schema for apo_pdb_holdout.parquet and alphafold_holdout.parquet.

    Holdouts carry the same 17 descriptors but a much shorter row list; we do
    not require balanced per-class counts here. ``class_binary`` is derived
    from the ligand annotation in the supporting-file xlsx.
    """
    cols: dict[str, pa.Column] = _numeric_feature_columns(feature_columns)
    cols["class_binary"] = pa.Column(
        int, checks=pa.Check.isin({0, 1}), nullable=False
    )
    cols["structure_id"] = pa.Column(str, nullable=False)
    cols["ligand"] = pa.Column(str, nullable=True)
    return pa.DataFrameSchema(cols, strict="filter", coerce=True)


def validate_training(
    df: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    return training_schema(feature_columns).validate(df, lazy=True)


def validate_holdout(
    df: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    return holdout_schema(feature_columns).validate(df, lazy=True)


# Re-export for convenience.
__all__ = [
    "SELECTED_17",
    "CLASS_10",
    "training_schema",
    "holdout_schema",
    "validate_training",
    "validate_holdout",
]
