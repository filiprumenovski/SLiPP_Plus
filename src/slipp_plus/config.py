"""Configuration loader backed by pydantic.

Reads ``configs/day1.yaml`` (or any passed path) and returns a typed settings
object used throughout the pipeline.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from .composite.config import CompositeSettings
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

class FlatModelHyperparameters(BaseModel):
    """Tunable per-family hyperparameters for flat-mode RF/XGB/LGBM/CatBoost.

    Defaults reproduce the values that were inlined in ``train.py:_build_model``
    before HPO was introduced. Anything not configured here keeps the historical
    sklearn / xgboost / lightgbm / catboost defaults.
    """

    # RandomForest
    rf_n_estimators: int = Field(default=100, ge=10, le=2000)
    rf_max_depth: int | None = Field(default=None, ge=1, le=64)
    rf_min_samples_leaf: int = Field(default=1, ge=1, le=64)
    rf_max_features: float | None = Field(default=None, gt=0.0, le=1.0)
    # XGBoost
    xgb_max_depth: int = Field(default=6, ge=1, le=16)
    xgb_n_estimators: int = Field(default=100, ge=10, le=2000)
    xgb_learning_rate: float = Field(default=0.3, gt=0.0, le=1.0)
    xgb_subsample: float = Field(default=1.0, gt=0.0, le=1.0)
    xgb_colsample_bytree: float = Field(default=1.0, gt=0.0, le=1.0)
    xgb_min_child_weight: float = Field(default=1.0, ge=0.0)
    xgb_reg_alpha: float = Field(default=0.0, ge=0.0)
    xgb_reg_lambda: float = Field(default=1.0, ge=0.0)
    # LightGBM
    lgbm_num_leaves: int = Field(default=31, ge=2, le=512)
    lgbm_n_estimators: int = Field(default=100, ge=10, le=2000)
    lgbm_learning_rate: float = Field(default=0.1, gt=0.0, le=1.0)
    lgbm_min_data_in_leaf: int = Field(default=20, ge=1, le=200)
    lgbm_feature_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    lgbm_bagging_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    # CatBoost (added when ``cat`` is present in the configured models list)
    cat_depth: int = Field(default=6, ge=1, le=16)
    cat_iterations: int = Field(default=200, ge=10, le=2000)
    cat_learning_rate: float = Field(default=0.05, gt=0.0, le=1.0)
    cat_l2_leaf_reg: float = Field(default=3.0, ge=0.0)


FeatureSet = Literal[
    "v14",
    "v14+v22",
    "v14+aa",
    "v14+shell",
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
    "v14+aa+tunnel_shape",
    "v14+shell+tunnel_shape",
    "v14+aa12+tunnel_shape",
    "v14+aa16+tunnel_shape",
    "v14+aa20+shell6+tunnel_shape",
    "v14+aa20+shell6+tunnel_shape3",
    "v49+tunnel_shape3",
    "v49+tunnel_shape_hydro4",
    "v49+tunnel_shape",
    "v49+tunnel_chem",
    "v49+tunnel_geom",
    "v_graph_tunnel",
    "v_caver_t12",
]
ModelKey = Literal["rf", "xgb", "lgbm", "cat"]
SplitStrategy = Literal["stratified_shuffle", "grouped"]
PipelineMode = Literal["flat", "hierarchical", "composite"]
Stage1Source = Literal["ensemble", "trained"]
NonlipidSource = Literal["dedicated_head", "flat_fallback"]
SpecialistGateMode = Literal["heuristic", "utility"]
LipidFamilyMode = Literal["softmax", "binary_ovr"]


def normalize_split_strategy(value: str) -> SplitStrategy:
    """Normalize configured split strategy names.

    Parameters
    ----------
    value
        Raw split strategy value from YAML.

    Returns
    -------
    SplitStrategy
        Canonical split strategy literal.

    Raises
    ------
    ValueError
        If ``value`` is not a supported split strategy or known alias.
    """

    if value in GROUPED_SPLIT_STRATEGY_ALIASES:
        return CANONICAL_GROUPED_SPLIT_STRATEGY
    if value in SUPPORTED_SPLIT_STRATEGIES:
        return value  # type: ignore[return-value]
    raise ValueError(f"split_strategy must be one of {sorted(SUPPORTED_SPLIT_STRATEGIES)}")


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


class SpecialistRuleSettings(BaseModel):
    name: str = "ste_specialist"
    positive_label: str = "STE"
    neighbor_labels: tuple[str, ...] = ("PLM", "COA", "OLA", "MYR")
    top_k: int = Field(default=4, ge=2, le=len(CLASS_10))
    min_positive_proba: float | None = Field(default=None, ge=0.0, le=1.0)
    max_margin: float | None = Field(default=None, ge=0.0)

    @field_validator("positive_label")
    @classmethod
    def specialist_positive_label_known(cls, value: str) -> str:
        if value not in CLASS_10:
            raise ValueError(f"unknown positive_label {value!r}; expected one of {CLASS_10}")
        return value

    @field_validator("neighbor_labels")
    @classmethod
    def specialist_neighbor_labels_known(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("neighbor_labels cannot be empty")
        unknown = sorted(set(value) - set(CLASS_10))
        if unknown:
            raise ValueError(f"unknown neighbor_labels: {unknown}")
        if len(set(value)) != len(value):
            raise ValueError("neighbor_labels cannot contain duplicates")
        return value

    @model_validator(mode="after")
    def specialist_positive_not_neighbor(self) -> SpecialistRuleSettings:
        if self.positive_label in self.neighbor_labels:
            raise ValueError("positive_label cannot also appear in neighbor_labels")
        return self

    def to_rule(self):
        from .hierarchical_postprocess import OneVsNeighborsRule

        return OneVsNeighborsRule(
            name=self.name,
            positive_label=self.positive_label,
            neighbor_labels=self.neighbor_labels,
            top_k=self.top_k,
            min_positive_proba=self.min_positive_proba,
            max_margin=self.max_margin,
        )


class XGBHyperparameters(BaseModel):
    """Tunable XGB hyperparameters for one stage of the hierarchical pipeline.

    Defaults intentionally match the values that were inlined at every XGB call
    site before HPO was introduced. Existing experiments must continue to
    reproduce bit-identically when no overrides are supplied.
    """

    max_depth: int = Field(default=5, ge=1, le=16)
    n_estimators: int = Field(default=250, ge=10, le=2000)
    learning_rate: float = Field(default=0.05, gt=0.0, le=1.0)
    subsample: float = Field(default=1.0, gt=0.0, le=1.0)
    colsample_bytree: float = Field(default=1.0, gt=0.0, le=1.0)
    min_child_weight: float = Field(default=1.0, ge=0.0)
    reg_alpha: float = Field(default=0.0, ge=0.0)
    reg_lambda: float = Field(default=1.0, ge=0.0)
    gamma: float = Field(default=0.0, ge=0.0)

    def to_xgb_kwargs(self) -> dict[str, float | int]:
        """Render as keyword arguments for ``xgboost.XGBClassifier``."""

        return {
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "gamma": self.gamma,
        }


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
    specialist_rule: SpecialistRuleSettings | None = None
    specialist_rules: list[SpecialistRuleSettings] | None = None
    # Per-stage XGB hyperparameters. Defaults reproduce the historical
    # hand-tuned values; HPO drivers override these per-trial.
    stage1_xgb: XGBHyperparameters = Field(default_factory=XGBHyperparameters)
    lipid_family_xgb: XGBHyperparameters = Field(default_factory=XGBHyperparameters)
    nonlipid_xgb: XGBHyperparameters = Field(default_factory=XGBHyperparameters)
    specialist_xgb: XGBHyperparameters = Field(default_factory=XGBHyperparameters)
    boundary_head_xgb: XGBHyperparameters = Field(default_factory=XGBHyperparameters)

    @model_validator(mode="after")
    def specialist_rule_or_rules_not_both(self) -> HierarchicalSettings:
        if self.specialist_rule is not None and self.specialist_rules:
            raise ValueError(
                "specify exactly one of `specialist_rule` (single) or "
                "`specialist_rules` (list); not both"
            )
        return self

    def resolved_specialist_rule(self):
        """Single-rule shim for legacy call sites.

        Returns the first resolved rule. New code should prefer
        :meth:`resolved_specialist_rules` which always returns a list.
        """

        rules = self.resolved_specialist_rules()
        return rules[0] if rules else None

    def resolved_specialist_rules(self) -> list:
        """Return the configured specialist rules in declaration order.

        Falls back to a single default STE rule (using
        :attr:`ste_threshold`) when neither `specialist_rule` nor
        `specialist_rules` is set, preserving historical behavior.
        """

        if self.specialist_rules:
            return [setting.to_rule() for setting in self.specialist_rules]
        if self.specialist_rule is not None:
            return [self.specialist_rule.to_rule()]
        return [SpecialistRuleSettings(min_positive_proba=self.ste_threshold).to_rule()]


class Settings(BaseModel):
    config_path: Path | None = None
    config_sha256: str | None = None
    seed_base: int = 42
    n_iterations: int = 25
    test_fraction: float = 0.10
    feature_set: FeatureSet = "v14"
    pipeline_mode: PipelineMode = "flat"
    split_strategy: SplitStrategy = "stratified_shuffle"
    split_group_column: str | None = None
    models: list[ModelKey] = Field(default_factory=lambda: ["rf", "xgb", "lgbm"])
    flat_hyperparameters: FlatModelHyperparameters = Field(
        default_factory=FlatModelHyperparameters
    )
    hierarchical: HierarchicalSettings = Field(default_factory=HierarchicalSettings)
    composite: CompositeSettings = Field(default_factory=CompositeSettings)
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
        if self.split_strategy == CANONICAL_GROUPED_SPLIT_STRATEGY and not self.split_group_column:
            raise ValueError(
                "split_group_column is required when split_strategy resolves to 'grouped'"
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
    raw["config_path"] = p
    raw["config_sha256"] = sha256(p.read_bytes()).hexdigest()
    return Settings.model_validate(raw)
