"""Feature-set materialization: given a settings object, return X, y slices."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings


def feature_matrix(df: pd.DataFrame, settings: Settings) -> np.ndarray:
    """Extract the configured numeric feature matrix from a frame.

    Parameters
    ----------
    df
        Pocket feature table.
    settings
        Loaded configuration whose ``feature_set`` determines the required
        columns.

    Returns
    -------
    np.ndarray
        Two-dimensional feature matrix in configured column order.

    Raises
    ------
    KeyError
        If one or more configured feature columns are absent.
    """

    cols = settings.feature_columns()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"feature columns missing from frame: {missing}. "
            f"feature_set={settings.feature_set}"
        )
    return df[cols].to_numpy(dtype=np.float64)


def class10_labels(df: pd.DataFrame) -> np.ndarray:
    """Return the raw ``class_10`` string labels."""
    return df["class_10"].to_numpy()


def binary_labels(df: pd.DataFrame) -> np.ndarray:
    """Return binary lipid-vs-rest labels as integer values.

    Parameters
    ----------
    df
        Pocket table containing ``class_binary``.

    Returns
    -------
    np.ndarray
        ``int8`` binary labels aligned to ``df`` rows.
    """

    return df["class_binary"].to_numpy(dtype=np.int8)
