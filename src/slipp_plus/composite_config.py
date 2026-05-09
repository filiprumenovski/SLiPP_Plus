"""Typed configuration for the Model V2 composite/MoE topology."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .constants import CLASS_10, FEATURE_SETS

BackboneKind = Literal["teacher_ensemble", "family_encoder"]
HeadKind = Literal["multiclass", "binary", "lipid_family", "nonlipid_family"]
ExpertKind = Literal["one_vs_neighbors", "binary_boundary", "local_multiclass"]
GateKind = Literal["heuristic", "utility", "candidate"]
CombineKind = Literal["neighbor_rescue", "pair_swap", "closed_world_redistribute"]


class CompositeBackboneSettings(BaseModel):
    kind: BackboneKind = "teacher_ensemble"
    feature_families: tuple[str, ...] = Field(default_factory=tuple)
    teacher_init: str | None = None
    class_weight_multipliers: dict[str, float] = Field(default_factory=dict)

    @field_validator("class_weight_multipliers")
    @classmethod
    def class_weight_multipliers_known(cls, value: dict[str, float]) -> dict[str, float]:
        unknown = sorted(set(value) - set(CLASS_10))
        if unknown:
            raise ValueError(f"unknown class_weight_multipliers labels: {unknown}")
        non_positive = {label: weight for label, weight in value.items() if weight <= 0}
        if non_positive:
            raise ValueError(f"class_weight_multipliers must be positive: {non_positive}")
        return value


class CompositeHeadSettings(BaseModel):
    name: str
    kind: HeadKind
    labels: tuple[str, ...]
    feature_set: str | None = None

    @field_validator("labels")
    @classmethod
    def labels_known(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = sorted(set(value) - set(CLASS_10) - {"lipid", "nonlipid"})
        if unknown:
            raise ValueError(f"unknown composite head labels: {unknown}")
        return value

    @field_validator("feature_set")
    @classmethod
    def feature_set_known(cls, value: str | None) -> str | None:
        if value is not None and value not in FEATURE_SETS:
            raise ValueError(f"unknown composite head feature_set: {value!r}")
        return value


class CompositeExpertSettings(BaseModel):
    name: str
    kind: ExpertKind
    labels: tuple[str, ...]
    gate: GateKind = "heuristic"
    combine: CombineKind = "closed_world_redistribute"
    feature_set: str | None = None
    margin: float = Field(default=0.20, ge=0.0)
    max_rank: int | None = Field(default=None, ge=2, le=len(CLASS_10))

    @field_validator("labels")
    @classmethod
    def expert_labels_known(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) < 2:
            raise ValueError("composite experts require at least two labels")
        unknown = sorted(set(value) - set(CLASS_10))
        if unknown:
            raise ValueError(f"unknown composite expert labels: {unknown}")
        if len(set(value)) != len(value):
            raise ValueError("composite expert labels cannot contain duplicates")
        return value

    @field_validator("feature_set")
    @classmethod
    def expert_feature_set_known(cls, value: str | None) -> str | None:
        if value is not None and value not in FEATURE_SETS:
            raise ValueError(f"unknown composite expert feature_set: {value!r}")
        return value


class CompositeSettings(BaseModel):
    """Model V2 composite topology config.

    Phase A supports the teacher-ensemble backbone and maps the topology onto
    the existing staged runtime. ``family_encoder`` is reserved for the next
    backbone replacement slice.
    """

    backbone: CompositeBackboneSettings = Field(default_factory=CompositeBackboneSettings)
    heads: list[CompositeHeadSettings] = Field(default_factory=list)
    experts: list[CompositeExpertSettings] = Field(default_factory=list)
    execution_order: list[str] = Field(default_factory=list)
    teacher_predictions_path: Path | None = None
