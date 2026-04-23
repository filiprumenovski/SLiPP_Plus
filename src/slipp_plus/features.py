"""Feature-set materialization: given a settings object, return X, y slices."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings


def feature_matrix(df: pd.DataFrame, settings: Settings) -> np.ndarray:
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
    return df["class_binary"].to_numpy(dtype=np.int8)
