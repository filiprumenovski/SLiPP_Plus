"""Configuration loader backed by pydantic.

Reads ``configs/day1.yaml`` (or any passed path) and returns a typed settings
object used throughout the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from .constants import CLASS_10, FEATURE_SETS

FeatureSet = Literal["v14", "v14+v22", "v14+aa", "v14+v22+aa"]
ModelKey = Literal["rf", "xgb", "lgbm"]


class Paths(BaseModel):
    training_csv: Path
    supporting_file_2_xlsx: Path
    supporting_file_3_xlsx: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path


class GroundTruthSplit(BaseModel):
    f1: float
    auroc: float
    accuracy: float | None = None
    precision: float | None = None
    sensitivity: float | None = None


class GroundTruth(BaseModel):
    test: GroundTruthSplit
    apo_pdb: GroundTruthSplit
    alphafold: GroundTruthSplit


class Validation(BaseModel):
    training_total_exact: int
    per_class_exact: dict[str, int]

    @field_validator("per_class_exact")
    @classmethod
    def keys_match_class10(cls, v: dict[str, int]) -> dict[str, int]:
        missing = set(CLASS_10) - set(v.keys())
        extra = set(v.keys()) - set(CLASS_10)
        if missing or extra:
            raise ValueError(
                f"validation.per_class_exact keys must be exactly CLASS_10. "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )
        return v


class Settings(BaseModel):
    seed_base: int = 42
    n_iterations: int = 25
    test_fraction: float = 0.10
    feature_set: FeatureSet = "v14"
    models: list[ModelKey] = Field(default_factory=lambda: ["rf", "xgb", "lgbm"])
    paths: Paths
    ground_truth: GroundTruth
    validation: Validation

    @field_validator("feature_set")
    @classmethod
    def feature_set_known(cls, v: str) -> str:
        if v not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {sorted(FEATURE_SETS)}")
        return v

    def feature_columns(self) -> list[str]:
        return FEATURE_SETS[self.feature_set]


def load_settings(path: str | Path) -> Settings:
    """Load settings from a YAML file and validate."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open() as f:
        raw = yaml.safe_load(f)
    return Settings.model_validate(raw)
