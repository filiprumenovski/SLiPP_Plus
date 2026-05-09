"""Composite/MoE training entry point.

Phase A runs the explicit composite topology through the existing staged
teacher-ensemble runtime for parity. Later phases can replace the backbone
behind this entry point without changing the evaluation artifact contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from ..artifact_schema import read_artifact_schema_sidecar, write_artifact_schema_sidecar
from ..config import Settings
from ..constants import HIERARCHICAL_PREDICTIONS_NAME
from ..hierarchical_pipeline import run_hierarchical_training
from .topology import composite_topology_metadata, resolve_composite_topology


def _assert_supported_phase_a(settings: Settings) -> None:
    topology = resolve_composite_topology(settings)
    if topology.backbone.kind != "teacher_ensemble":
        raise NotImplementedError(
            "pipeline_mode='composite' currently supports backbone.kind="
            "'teacher_ensemble'. The family_encoder backbone belongs to the "
            "next MoE phase."
        )
    unsupported_gates = [expert.name for expert in topology.experts if expert.gate == "candidate"]
    if unsupported_gates:
        raise NotImplementedError(
            f"candidate gates are not implemented in composite Phase A: {unsupported_gates}"
        )


def _phase_a_hierarchical_settings(settings: Settings) -> Settings:
    """Translate the supported MoE topology into the parity executor config."""

    topology = resolve_composite_topology(settings)
    raw = settings.model_dump(mode="python")
    raw["pipeline_mode"] = "hierarchical"
    hierarchy = raw.setdefault("hierarchical", {})

    one_vs_neighbors = [expert for expert in topology.experts if expert.kind == "one_vs_neighbors"]
    if one_vs_neighbors:
        # Multi-specialist support: declaration order is preserved so that rule
        # priority (e.g. STE first) is deterministic. Phase A previously
        # supported only one specialist; the constraint is now lifted because
        # OLA / PLM rescues stack on top of STE.
        gate_modes = {expert.gate for expert in one_vs_neighbors}
        feature_sets = {expert.feature_set for expert in one_vs_neighbors}
        if len(gate_modes) > 1:
            raise NotImplementedError(
                "composite Phase A requires a single specialist gate mode across "
                f"experts; got {sorted(str(g) for g in gate_modes)}"
            )
        if len(feature_sets) > 1:
            raise NotImplementedError(
                "composite Phase A requires a single specialist feature_set across "
                f"experts; got {sorted(str(s) for s in feature_sets)}"
            )
        hierarchy["specialist_gate"] = next(iter(gate_modes))
        hierarchy["specialist_feature_set"] = next(iter(feature_sets))
        hierarchy["specialist_rules"] = [
            {
                "name": expert.name,
                "positive_label": expert.labels[0],
                "neighbor_labels": list(expert.labels[1:]),
                "top_k": expert.max_rank or len(expert.labels),
                "min_positive_proba": settings.hierarchical.ste_threshold,
            }
            for expert in one_vs_neighbors
        ]
        hierarchy.pop("specialist_rule", None)

    boundary_heads = []
    for expert in topology.experts:
        if expert.kind != "binary_boundary":
            continue
        boundary_heads.append(
            {
                "name": expert.name,
                "positive_label": expert.labels[0],
                "negative_labels": list(expert.labels[1:]),
                "margin": expert.margin,
                "max_rank": expert.max_rank,
                "feature_set": expert.feature_set,
            }
        )
    hierarchy["boundary_heads"] = boundary_heads

    return Settings.model_validate(raw)


def _annotate_joblib(path: Path, metadata: dict[str, Any]) -> None:
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise TypeError(f"composite bundle payload must be a dict: {path}")
    payload.update(metadata)
    joblib.dump(payload, path)


def run_composite_training(settings: Settings) -> dict[str, Path]:
    """Train the Phase-A composite/MoE topology and preserve prediction parity."""

    if settings.composite.backbone.kind == "family_encoder":
        from .family_train import run_family_encoder_training

        return run_family_encoder_training(settings)

    if settings.composite.teacher_predictions_path is not None:
        from .pair_moe import run_pair_moe_training

        return run_pair_moe_training(settings)

    _assert_supported_phase_a(settings)
    metadata = composite_topology_metadata(settings)

    hierarchical_settings = _phase_a_hierarchical_settings(settings)
    out = run_hierarchical_training(hierarchical_settings)

    bundle_path = out.get("hierarchical_bundle")
    if bundle_path is not None:
        _annotate_joblib(
            bundle_path,
            {
                "composite_mode": True,
                **metadata,
            },
        )

    predictions_path = settings.paths.processed_dir / "predictions" / HIERARCHICAL_PREDICTIONS_NAME
    sidecar = read_artifact_schema_sidecar(predictions_path) or {}
    sidecar.update(
        {
            "artifact_type": "composite_predictions",
            "pipeline_mode": "composite",
            **metadata,
        }
    )
    write_artifact_schema_sidecar(predictions_path, sidecar)

    return {
        **out,
        "composite_predictions": predictions_path,
        "composite_bundle": bundle_path,
    }
