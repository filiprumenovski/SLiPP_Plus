"""Configuration loader backed by pydantic.

Reads ``configs/day1.yaml`` (or any passed path) and returns a typed settings
object used throughout the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from .constants import CLASS_10, FEATURE_SETS, HIERARCHICAL_BUNDLE_NAME

CANONICAL_GROUPED_SPLIT_STRATEGY = "grouped"
GROUPED_SPLIT_STRATEGY_ALIASES = frozenset(
    {
        "grouped_mmseqs30",
        "grouped_uniprot_clustered",
        "grouped_uniprot",
    }
)
SUPPORTED_SPLIT_STRATEGIES = frozenset(
    {"stratified_shuffle", CANONICAL_GROUPED_SPLIT_STRATEGY}
).union(GROUPED_SPLIT_STRATEGY_ALIASES)

FeatureSet = Literal[
    "v14",
    "v14+v22",
    "v14+aa",
    "v14+v22+aa",
    "v49",
    "v61",
    "v_sterol",
    "v_sterol+vdw22",
    "v_sterol+derived",
    "v_sterol+vdw22+derived",
    "v_sterol_v2",
    "v_sterol_lean",
    "v_plm_ste",
    "v_lipid_boundary",
    "v_tunnel",
    "v_graph_tunnel",
    "v_caver_t12",
]
ModelKey = Literal["rf", "xgb", "lgbm"]
SplitStrategy = Literal["stratified_shuffle", "grouped"]
PipelineMode = Literal["flat", "hierarchical"]
Stage1Source = Literal["ensemble", "trained"]
NonlipidSource = Literal["dedicated_head", "flat_fallback"]
SpecialistGateMode = Literal["heuristic", "utility"]
LipidFamilyMode = Literal["softmax", "binary_ovr"]


def normalize_split_strategy(value: str) -> SplitStrategy:
    if value in GROUPED_SPLIT_STRATEGY_ALIASES:
        return CANONICAL_GROUPED_SPLIT_STRATEGY
    if value in SUPPORTED_SPLIT_STRATEGIES:
        return value  # type: ignore[return-value]
    raise ValueError(
        "split_strategy must be one of "
        f"{sorted(SUPPORTED_SPLIT_STRATEGIES)}"
    )


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


class BoundaryHeadSettings(BaseModel):
    name: str
    positive_label: str
    negative_labels: tuple[str, ...]
    margin: float = Field(default=0.20, ge=0.0)
    max_rank: int | None = Field(default=None, ge=2, le=len(CLASS_10))
    feature_set: FeatureSet | None = None

    @field_validator("positive_label")
    @classmethod
    def positive_label_known(cls, value: str) -> str:
        if value not in CLASS_10:
            raise ValueError(f"unknown positive_label {value!r}; expected one of {CLASS_10}")
        return value

    @field_validator("negative_labels")
    @classmethod
    def negative_labels_known(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("negative_labels cannot be empty")
        unknown = sorted(set(value) - set(CLASS_10))
        if unknown:
            raise ValueError(f"unknown negative_labels: {unknown}")
        if len(set(value)) != len(value):
            raise ValueError("negative_labels cannot contain duplicates")
        return value

    @model_validator(mode="after")
    def positive_not_negative(self) -> BoundaryHeadSettings:
        if self.positive_label in self.negative_labels:
            raise ValueError("positive_label cannot also appear in negative_labels")
        return self

    def to_boundary_rule(self):
        from .boundary_head import BoundaryRule

        return BoundaryRule(
            name=self.name,
            positive_label=self.positive_label,
            negative_labels=self.negative_labels,
            margin=self.margin,
            max_rank=self.max_rank,
        )


class HierarchicalSettings(BaseModel):
    stage1_source: Stage1Source = "ensemble"
    ste_threshold: float = Field(default=0.40, ge=0.0, le=1.0)
    workers: int = Field(default=8, ge=1)
    nonlipid_source: NonlipidSource = "dedicated_head"
    persist_stage_predictions: bool = True
    bundle_name: str = HIERARCHICAL_BUNDLE_NAME
    specialist_gate: SpecialistGateMode = "heuristic"
    utility_threshold_default: float = Field(default=0.50, ge=0.0, le=1.0)
    utility_threshold_top1_plm: float | None = Field(default=None, ge=0.0, le=1.0)
    boundary_heads: list[BoundaryHeadSettings] = Field(default_factory=list)
    lipid_family_mode: LipidFamilyMode = "softmax"
    lipid_family_feature_set: FeatureSet | None = None
    specialist_feature_set: FeatureSet | None = None
    nonlipid_feature_set: FeatureSet | None = None


class Settings(BaseModel):
    seed_base: int = 42
    n_iterations: int = 25
    test_fraction: float = 0.10
    feature_set: FeatureSet = "v14"
    pipeline_mode: PipelineMode = "flat"
    split_strategy: SplitStrategy = "stratified_shuffle"
    split_group_column: str | None = None
    models: list[ModelKey] = Field(default_factory=lambda: ["rf", "xgb", "lgbm"])
    hierarchical: HierarchicalSettings = Field(default_factory=HierarchicalSettings)
    paths: Paths
    ground_truth: GroundTruth
    validation: Validation

    @field_validator("split_strategy", mode="before")
    @classmethod
    def split_strategy_known(cls, value: str) -> SplitStrategy:
        if not isinstance(value, str):
            raise TypeError("split_strategy must be a string")
        return normalize_split_strategy(value)

    @field_validator("feature_set")
    @classmethod
    def feature_set_known(cls, v: str) -> str:
        if v not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {sorted(FEATURE_SETS)}")
        return v

    @model_validator(mode="after")
    def grouped_split_requires_group_column(self) -> Settings:
        if (
            self.split_strategy == CANONICAL_GROUPED_SPLIT_STRATEGY
            and not self.split_group_column
        ):
            raise ValueError(
                "split_group_column is required when split_strategy resolves to "
                "'grouped'"
            )
        return self

    def feature_columns(self) -> list[str]:
        return FEATURE_SETS[self.feature_set]


def load_settings(path: str | Path) -> Settings:
    """Load settings from a YAML file and validate."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Settings.model_validate(raw)
