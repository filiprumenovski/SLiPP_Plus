"""Pandera schemas enforcing PROMPT.md Rule 1 at ingest time.

Schemas here are *dataframe-shape* guarantees; per-class count assertions that
depend on runtime configuration live in ``ingest.assert_rule_1``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import pandas as pd

from .constants import CLASS_10, SELECTED_17


def _load_pandera() -> Any:
    try:
        return import_module("pandera.pandas")
    except ImportError:  # pandera<0.20 exposed the pandas API at top level.
        return import_module("pandera")


pa = _load_pandera()


def _numeric_feature_columns(columns: list[str]) -> dict[str, Any]:
    return {c: pa.Column(float, nullable=False, coerce=True) for c in columns}


def training_schema(feature_columns: list[str]) -> Any:
    """Build the Pandera schema for processed training pockets.

    Parameters
    ----------
    feature_columns:
        Exact numeric feature columns required by the active feature set.

    Returns
    -------
    pandera.DataFrameSchema
        Schema that keeps only expected columns, coerces feature values to
        numeric types, and requires ``class_10``, ``class_binary``, and
        ``pdb_ligand`` labels.
    """
    cols: dict[str, Any] = _numeric_feature_columns(feature_columns)
    cols["class_10"] = pa.Column(str, checks=pa.Check.isin(CLASS_10), nullable=False)
    cols["class_binary"] = pa.Column(int, checks=pa.Check.isin({0, 1}), nullable=False)
    cols["pdb_ligand"] = pa.Column(str, nullable=False)  # e.g. "ADN/pdb1BX4.pdb"
    return pa.DataFrameSchema(cols, strict="filter", coerce=True)


def holdout_schema(feature_columns: list[str]) -> Any:
    """Build the Pandera schema for processed holdout pockets.

    Holdouts carry the same descriptor family as the active feature set but a
    much shorter row list; balanced per-class counts are not required here.
    ``class_binary`` is derived from the ligand annotation in the supporting
    workbook.

    Parameters
    ----------
    feature_columns:
        Exact numeric feature columns required by the active feature set.

    Returns
    -------
    pandera.DataFrameSchema
        Schema that keeps only expected columns, coerces feature values to
        numeric types, and requires ``class_binary``, ``structure_id``, and
        ``ligand`` metadata.
    """
    cols: dict[str, Any] = _numeric_feature_columns(feature_columns)
    cols["class_binary"] = pa.Column(int, checks=pa.Check.isin({0, 1}), nullable=False)
    cols["structure_id"] = pa.Column(str, nullable=False)
    cols["ligand"] = pa.Column(str, nullable=True)
    return pa.DataFrameSchema(cols, strict="filter", coerce=True)


def validate_training(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Validate a processed training frame against the active feature schema.

    Parameters
    ----------
    df:
        Candidate processed training frame.
    feature_columns:
        Exact numeric feature columns required by the active feature set.

    Returns
    -------
    pandas.DataFrame
        Validated and coerced training frame.

    Raises
    ------
    pandera.errors.SchemaErrors
        If required columns are missing, labels are invalid, feature values are
        null, or feature values cannot be coerced to numeric types.
    """
    return training_schema(feature_columns).validate(df, lazy=True)


def validate_holdout(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Validate a processed holdout frame against the active feature schema.

    Parameters
    ----------
    df:
        Candidate processed holdout frame.
    feature_columns:
        Exact numeric feature columns required by the active feature set.

    Returns
    -------
    pandas.DataFrame
        Validated and coerced holdout frame.

    Raises
    ------
    pandera.errors.SchemaErrors
        If required columns are missing, binary labels are invalid, feature
        values are null, or feature values cannot be coerced to numeric types.
    """
    return holdout_schema(feature_columns).validate(df, lazy=True)


# Re-export for convenience.
__all__ = [
    "CLASS_10",
    "SELECTED_17",
    "holdout_schema",
    "training_schema",
    "validate_holdout",
    "validate_training",
]
