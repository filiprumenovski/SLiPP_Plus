"""Topology construction for the Model V2 composite/MoE runtime."""

from __future__ import annotations

from typing import Any

from ..constants import CLASS_10
from ..hierarchical_experiment import LIPID_LABELS, NONLIPID_LABELS
from .config import (
    CompositeBackboneSettings,
    CompositeExpertSettings,
    CompositeHeadSettings,
    CompositeSettings,
)


def default_composite_from_hierarchical(settings: Any) -> CompositeSettings:
    """Map the current hierarchical config to an explicit composite topology."""

    hierarchy = settings.hierarchical
    heads = [
        CompositeHeadSettings(
            name="global_10",
            kind="multiclass",
            labels=tuple(CLASS_10),
            feature_set=settings.feature_set,
        ),
        CompositeHeadSettings(
            name="binary_lipid",
            kind="binary",
            labels=("nonlipid", "lipid"),
            feature_set=settings.feature_set,
        ),
        CompositeHeadSettings(
            name="lipid_family",
            kind="lipid_family",
            labels=tuple(LIPID_LABELS),
            feature_set=hierarchy.lipid_family_feature_set or settings.feature_set,
        ),
        CompositeHeadSettings(
            name="nonlipid_family",
            kind="nonlipid_family",
            labels=tuple(NONLIPID_LABELS),
            feature_set=hierarchy.nonlipid_feature_set or settings.feature_set,
        ),
    ]

    specialist_rule = hierarchy.resolved_specialist_rule()
    experts = [
        CompositeExpertSettings(
            name=specialist_rule.name,
            kind="one_vs_neighbors",
            labels=(specialist_rule.positive_label, *specialist_rule.neighbor_labels),
            gate=hierarchy.specialist_gate,
            combine="neighbor_rescue",
            feature_set=hierarchy.specialist_feature_set or settings.feature_set,
            max_rank=specialist_rule.top_k,
        )
    ]
    for item in hierarchy.boundary_heads:
        experts.append(
            CompositeExpertSettings(
                name=item.name,
                kind="binary_boundary",
                labels=(item.positive_label, *item.negative_labels),
                gate="heuristic",
                combine="pair_swap",
                feature_set=item.feature_set or settings.feature_set,
                margin=item.margin,
                max_rank=item.max_rank,
            )
        )

    execution_order = [head.name for head in heads] + [expert.name for expert in experts]
    return CompositeSettings(
        backbone=CompositeBackboneSettings(
            kind="teacher_ensemble",
            teacher_init="rf_xgb_lgbm",
        ),
        heads=heads,
        experts=experts,
        execution_order=execution_order,
    )


def resolve_composite_topology(settings: Any) -> CompositeSettings:
    """Return explicit composite topology, defaulting to hierarchical parity."""

    composite = settings.composite
    if composite.heads or composite.experts or composite.execution_order:
        return composite
    return default_composite_from_hierarchical(settings)


def composite_topology_metadata(settings: Any) -> dict[str, Any]:
    """Serialize the resolved composite topology for artifact sidecars.

    Parameters
    ----------
    settings
        Loaded experiment settings containing either explicit composite config
        or a hierarchical config that can be mapped to composite parity.

    Returns
    -------
    dict[str, Any]
        Metadata payload with composite schema version and JSON-serializable
        topology settings.
    """

    topology = resolve_composite_topology(settings)
    return {
        "composite_version": 1,
        "topology": topology.model_dump(mode="json"),
    }
