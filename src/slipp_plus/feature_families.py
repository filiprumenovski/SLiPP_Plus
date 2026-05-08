"""Feature-family registry and split-local normalization for composite backbones."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import (
    AA20,
    AA_SIGNAL_12,
    AA_SIGNAL_16,
    AROMATIC_ALIPHATIC_12,
    CAVER_T12_FEATURES_17,
    DERIVED_FEATURES_26,
    EXTRA_VDW22,
    GRAPH_TUNNEL_FEATURES_3,
    LIPID_BOUNDARY_FEATURES_22,
    POCKET_GEOMETRY_COLS,
    SELECTED_17,
    SHELL_SIGNAL_6,
    STEROL_CHEMISTRY_SHELL_COLS,
    TUNNEL_CHEM_5,
    TUNNEL_FEATURES_18,
    TUNNEL_GEOM_9,
    TUNNEL_SHAPE_AVAIL_6,
    TUNNEL_SHAPE_SIGNAL_3,
)


@dataclass(frozen=True)
class FeatureFamilySpec:
    name: str
    columns: tuple[str, ...]
    optional: bool = False


FEATURE_FAMILY_REGISTRY: dict[str, FeatureFamilySpec] = {
    "paper17": FeatureFamilySpec("paper17", tuple(SELECTED_17)),
    "aa20": FeatureFamilySpec("aa20", tuple(AA20)),
    "aa_signal12": FeatureFamilySpec("aa_signal12", tuple(AA_SIGNAL_12)),
    "aa_signal16": FeatureFamilySpec("aa_signal16", tuple(AA_SIGNAL_16)),
    "shell12": FeatureFamilySpec("shell12", tuple(AROMATIC_ALIPHATIC_12)),
    "shell_signal6": FeatureFamilySpec("shell_signal6", tuple(SHELL_SIGNAL_6)),
    "sterol_chemistry": FeatureFamilySpec(
        "sterol_chemistry",
        tuple(STEROL_CHEMISTRY_SHELL_COLS),
    ),
    "pocket_geometry": FeatureFamilySpec("pocket_geometry", tuple(POCKET_GEOMETRY_COLS)),
    "boundary22": FeatureFamilySpec(
        "boundary22",
        tuple(LIPID_BOUNDARY_FEATURES_22),
        optional=True,
    ),
    "graph_tunnel": FeatureFamilySpec(
        "graph_tunnel",
        tuple(GRAPH_TUNNEL_FEATURES_3),
        optional=True,
    ),
    "tunnel": FeatureFamilySpec("tunnel", tuple(TUNNEL_FEATURES_18), optional=True),
    "tunnel_shape": FeatureFamilySpec(
        "tunnel_shape",
        tuple(TUNNEL_SHAPE_AVAIL_6),
        optional=True,
    ),
    "tunnel_shape_signal3": FeatureFamilySpec(
        "tunnel_shape_signal3",
        tuple(TUNNEL_SHAPE_SIGNAL_3),
        optional=True,
    ),
    "tunnel_chem": FeatureFamilySpec("tunnel_chem", tuple(TUNNEL_CHEM_5), optional=True),
    "tunnel_geom": FeatureFamilySpec("tunnel_geom", tuple(TUNNEL_GEOM_9), optional=True),
    "caver_t12": FeatureFamilySpec("caver_t12", tuple(CAVER_T12_FEATURES_17), optional=True),
    "derived": FeatureFamilySpec("derived", tuple(DERIVED_FEATURES_26), optional=True),
    "vdw22": FeatureFamilySpec("vdw22", tuple(EXTRA_VDW22), optional=True),
}


@dataclass(frozen=True)
class FamilyScaler:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale


def resolve_family_specs(names: tuple[str, ...] | list[str]) -> list[FeatureFamilySpec]:
    if not names:
        names = ("paper17", "aa20", "shell12", "sterol_chemistry", "pocket_geometry")
    specs: list[FeatureFamilySpec] = []
    for name in names:
        if name not in FEATURE_FAMILY_REGISTRY:
            raise KeyError(f"unknown feature family: {name}")
        specs.append(FEATURE_FAMILY_REGISTRY[name])
    return specs


def fit_family_scalers(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    specs: list[FeatureFamilySpec],
) -> dict[str, FamilyScaler]:
    scalers: dict[str, FamilyScaler] = {}
    for spec in specs:
        missing = [column for column in spec.columns if column not in df.columns]
        if missing and not spec.optional:
            raise KeyError(f"required feature family {spec.name!r} missing columns: {missing}")
        values = _family_values(df, spec)
        train_values = values[train_idx]
        mean = train_values.mean(axis=0)
        scale = train_values.std(axis=0)
        scale[scale < 1e-6] = 1.0
        scalers[spec.name] = FamilyScaler(mean=mean.astype(np.float32), scale=scale.astype(np.float32))
    return scalers


def materialize_family_arrays(
    df: pd.DataFrame,
    specs: list[FeatureFamilySpec],
    scalers: Mapping[str, FamilyScaler],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    masks = np.zeros((len(df), len(specs)), dtype=np.float32)
    for family_index, spec in enumerate(specs):
        values = _family_values(df, spec)
        if all(column in df.columns for column in spec.columns):
            masks[:, family_index] = 1.0
        arrays[spec.name] = scalers[spec.name].transform(values).astype(np.float32)
    return arrays, masks


def _family_values(df: pd.DataFrame, spec: FeatureFamilySpec) -> np.ndarray:
    if all(column in df.columns for column in spec.columns):
        return df[list(spec.columns)].to_numpy(dtype=np.float32)
    if spec.optional:
        return np.zeros((len(df), len(spec.columns)), dtype=np.float32)
    missing = [column for column in spec.columns if column not in df.columns]
    raise KeyError(f"required feature family {spec.name!r} missing columns: {missing}")
