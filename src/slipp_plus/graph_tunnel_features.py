"""Cheap alpha-sphere graph tunnel-depth features.

This is the Tier 2 tunnel proxy: build a touching-sphere graph from fpocket
alpha spheres, then run Dijkstra from the deepest pocket sphere to the nearest
expanded-cloud boundary. It is deliberately lightweight and independent of
CAVER.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .constants import GRAPH_TUNNEL_FEATURES_3 as _REGISTRY_GRAPH_TUNNEL_FEATURES_3
from .plm_ste_features import _axial_profile

GRAPH_TUNNEL_FEATURES_3: list[str] = [
    "tunnel_length",
    "tunnel_bottleneck_radius",
    "tunnel_length_over_axial_length",
]

_EPS = 1e-6
_DEFAULT_THROAT_RADIUS = 6.0
_TOUCH_EPS = 0.25


@dataclass(frozen=True)
class AlphaSphere:
    coord: np.ndarray
    radius: float
    pocket_number: int
    atom_name: str


def _empty_features() -> dict[str, float]:
    return {
        "tunnel_length": 0.0,
        "tunnel_bottleneck_radius": 0.0,
        "tunnel_length_over_axial_length": 0.0,
    }


def _parse_alpha_sphere_line(
    line: str,
    *,
    default_pocket_number: int | None = None,
) -> AlphaSphere | None:
    if not line.startswith(("ATOM", "HETATM")):
        return None
    parts = line.split()
    if len(parts) < 10:
        return None
    try:
        atom_name = parts[2].upper()
        pocket_number = int(parts[4]) if default_pocket_number is None else default_pocket_number
        coord = np.asarray([float(parts[5]), float(parts[6]), float(parts[7])], dtype=float)
        radius = float(parts[-1])
    except (ValueError, IndexError):
        return None
    if not np.isfinite(coord).all() or not math.isfinite(radius) or radius <= 0.0:
        return None
    return AlphaSphere(coord=coord, radius=radius, pocket_number=pocket_number, atom_name=atom_name)


def load_alpha_spheres(
    path: Path, *, default_pocket_number: int | None = None
) -> list[AlphaSphere]:
    spheres: list[AlphaSphere] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            sphere = _parse_alpha_sphere_line(line, default_pocket_number=default_pocket_number)
            if sphere is not None:
                spheres.append(sphere)
    if not spheres:
        raise ValueError(f"{path}: no alpha spheres parsed")
    return spheres


def _axial_length(coords: np.ndarray) -> float:
    if len(coords) < 2:
        return 0.0
    axial = _axial_profile(coords)
    if axial is not None:
        return float(axial["axial_length"])
    centered = coords - coords.mean(axis=0)
    distances = np.linalg.norm(centered[:, None, :] - centered[None, :, :], axis=-1)
    return float(distances.max())


def _touching_graph(coords: np.ndarray, radii: np.ndarray, touch_eps: float) -> csr_matrix:
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    touching = distances <= (radii[:, None] + radii[None, :] + touch_eps)
    np.fill_diagonal(touching, False)
    rows, cols = np.nonzero(touching)
    return csr_matrix((distances[rows, cols], (rows, cols)), shape=(len(coords), len(coords)))


def _fallback_boundary_targets(
    coords: np.ndarray,
    pocket_mask: np.ndarray,
    pocket_coords: np.ndarray,
) -> np.ndarray:
    pocket_indices = np.flatnonzero(pocket_mask)
    if len(pocket_indices) <= 1:
        return pocket_indices
    centroid = pocket_coords.mean(axis=0)
    radial = np.linalg.norm(coords[pocket_indices] - centroid, axis=1)
    cutoff = float(np.quantile(radial, 0.80))
    targets = pocket_indices[radial >= cutoff]
    if len(targets) == 0:
        targets = np.asarray([pocket_indices[int(np.argmax(radial))]], dtype=int)
    return targets


def _reconstruct_path(predecessors: np.ndarray, source: int, target: int) -> list[int]:
    path = [target]
    current = target
    seen = {target}
    while current != source:
        current = int(predecessors[current])
        if current < 0 or current in seen:
            return []
        seen.add(current)
        path.append(current)
    path.reverse()
    return path


def graph_tunnel_features_for_pocket(
    pocket_spheres: list[AlphaSphere],
    all_spheres: list[AlphaSphere] | None = None,
    *,
    throat_radius: float = _DEFAULT_THROAT_RADIUS,
    touch_eps: float = _TOUCH_EPS,
) -> dict[str, float]:
    """Return Dijkstra tunnel-depth proxy features for one pocket.

    fpocket output in this repo does not expose per-alpha-sphere SAS as a
    numeric field. We therefore treat nearby alpha spheres assigned to other
    pockets as the surface/throat boundary. If no such boundary nodes are
    available, the most radial 20% of the pocket's own alpha spheres are used
    as a conservative boundary fallback.
    """

    if not pocket_spheres:
        return _empty_features()
    if all_spheres is None:
        all_spheres = pocket_spheres

    pocket_coords = np.asarray([sphere.coord for sphere in pocket_spheres], dtype=float)
    pocket_numbers = {sphere.pocket_number for sphere in pocket_spheres}

    all_coords_raw = np.asarray([sphere.coord for sphere in all_spheres], dtype=float)
    distances_to_pocket = np.linalg.norm(
        all_coords_raw[:, None, :] - pocket_coords[None, :, :],
        axis=-1,
    )
    expanded_raw_mask = distances_to_pocket.min(axis=1) <= throat_radius

    expanded_spheres = [
        sphere for sphere, keep in zip(all_spheres, expanded_raw_mask, strict=True) if keep
    ]
    if not expanded_spheres:
        expanded_spheres = pocket_spheres

    coords = np.asarray([sphere.coord for sphere in expanded_spheres], dtype=float)
    radii = np.asarray([sphere.radius for sphere in expanded_spheres], dtype=float)
    pocket_mask = np.asarray(
        [sphere.pocket_number in pocket_numbers for sphere in expanded_spheres],
        dtype=bool,
    )
    if int(pocket_mask.sum()) == 0:
        return _empty_features()

    target_indices = np.flatnonzero(~pocket_mask)
    if len(target_indices) == 0:
        target_indices = _fallback_boundary_targets(coords, pocket_mask, pocket_coords)
    if len(target_indices) == 0:
        return _empty_features()

    graph = _touching_graph(coords, radii, touch_eps)
    dist_from_targets = dijkstra(graph, directed=False, indices=target_indices)
    if dist_from_targets.ndim == 1:
        nearest_target_dist = dist_from_targets
    else:
        nearest_target_dist = np.min(dist_from_targets, axis=0)

    source_candidates = np.flatnonzero(pocket_mask)
    non_target_candidates = np.setdiff1d(source_candidates, target_indices, assume_unique=False)
    if len(non_target_candidates) > 0:
        source_candidates = non_target_candidates
    finite_candidates = source_candidates[np.isfinite(nearest_target_dist[source_candidates])]
    if len(finite_candidates) == 0:
        return _empty_features()

    source = int(finite_candidates[np.argmax(nearest_target_dist[finite_candidates])])
    dist_from_source, predecessors = dijkstra(
        graph,
        directed=False,
        indices=source,
        return_predecessors=True,
    )
    finite_targets = target_indices[np.isfinite(dist_from_source[target_indices])]
    if len(finite_targets) == 0:
        return _empty_features()
    target = int(finite_targets[np.argmin(dist_from_source[finite_targets])])
    tunnel_length = float(dist_from_source[target])
    path = _reconstruct_path(predecessors, source, target)
    bottleneck = 0.0 if not path else float(np.min(radii[np.asarray(path, dtype=int)]))

    axial = _axial_length(pocket_coords)
    ratio = tunnel_length / (axial + _EPS) if axial > 0.0 else 0.0
    if not math.isfinite(ratio):
        ratio = 0.0
    return {
        "tunnel_length": tunnel_length if math.isfinite(tunnel_length) else 0.0,
        "tunnel_bottleneck_radius": bottleneck if math.isfinite(bottleneck) else 0.0,
        "tunnel_length_over_axial_length": ratio,
    }


def _cast_features(features: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for column in GRAPH_TUNNEL_FEATURES_3:
        try:
            value = float(features.get(column, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        if not math.isfinite(value):
            value = 0.0
        out[column] = value
    return out


def _class_code_from_pdb_ligand(pdb_ligand: str) -> str:
    return pdb_ligand.split("/", 1)[0]


def _pdb_stem(pdb_ligand: str) -> str:
    return Path(pdb_ligand).stem


def _process_group(task: dict[str, object]) -> dict[str, object]:
    rows = list(task["rows"])  # type: ignore[arg-type]
    structure_dir = Path(str(task["structure_dir"]))
    label = str(task["label"])
    stem = str(task["stem"])

    warnings: list[str] = []
    all_path = structure_dir / f"{stem}_pockets.pqr"
    try:
        all_spheres = load_alpha_spheres(all_path)
    except (OSError, ValueError) as exc:
        warnings.append(f"{label}: all-pocket alpha-sphere parse failed: {exc}")
        all_spheres = []

    pocket_cache: dict[int, dict[str, float]] = {}
    enriched: list[dict[str, object]] = []
    for row in rows:
        pocket_number = int(row["matched_pocket_number"])
        if pocket_number not in pocket_cache:
            vert_path = structure_dir / "pockets" / f"pocket{pocket_number}_vert.pqr"
            try:
                pocket_spheres = load_alpha_spheres(
                    vert_path,
                    default_pocket_number=pocket_number,
                )
                source_spheres = all_spheres if all_spheres else pocket_spheres
                features = graph_tunnel_features_for_pocket(pocket_spheres, source_spheres)
                pocket_cache[pocket_number] = _cast_features(features)
            except (OSError, ValueError) as exc:
                warnings.append(f"{label}/pocket{pocket_number}: graph tunnel failed: {exc}")
                pocket_cache[pocket_number] = _empty_features()
        out = dict(row)
        out.update(pocket_cache[pocket_number])
        enriched.append(out)
    return {"rows": enriched, "warnings": warnings}


def _run_tasks_and_write(
    tasks: list[dict[str, object]],
    base: pd.DataFrame,
    output_path: Path,
    workers: int,
) -> dict[str, object]:
    if workers <= 1 or len(tasks) == 1:
        results = [_process_group(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_process_group, tasks))

    all_rows: list[dict[str, object]] = []
    warnings: list[str] = []
    for result in results:
        all_rows.extend(result["rows"])  # type: ignore[arg-type]
        warnings.extend(result["warnings"])  # type: ignore[arg-type]
    for warning in warnings:
        logging.warning(warning)

    enriched = pd.DataFrame(all_rows)
    if "_row_order" in enriched.columns:
        enriched = enriched.sort_values("_row_order").reset_index(drop=True)
    if len(enriched) != len(base):
        raise ValueError(
            f"row drift after v_graph_tunnel join: got {len(enriched)}, expected {len(base)}"
        )

    for column in GRAPH_TUNNEL_FEATURES_3:
        if column not in enriched.columns:
            raise ValueError(f"missing graph tunnel feature column: {column}")
        numeric = pd.to_numeric(enriched[column], errors="raise")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f"{column}: non-finite values present after build")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output_path, index=False)
    return {
        "rows": len(enriched),
        "structures": len(tasks),
        "warnings": len(warnings),
        "output_path": output_path,
        "output_size_mb": output_path.stat().st_size / (1024 * 1024),
    }


def build_training_v_graph_tunnel_parquet(
    base_parquet: Path = Path("processed/v_sterol/full_pockets.parquet"),
    source_pdbs_root: Path = Path("data/structures/source_pdbs"),
    output_path: Path = Path("processed/v_graph_tunnel/full_pockets.parquet"),
    workers: int = 6,
) -> dict[str, object]:
    base = pd.read_parquet(base_parquet)
    if "pdb_ligand" not in base.columns or "matched_pocket_number" not in base.columns:
        raise ValueError(f"{base_parquet}: expected pdb_ligand + matched_pocket_number columns")

    tasks: list[dict[str, object]] = []
    for pdb_ligand, frame in base.groupby("pdb_ligand", sort=False):
        class_code = _class_code_from_pdb_ligand(str(pdb_ligand))
        stem = _pdb_stem(str(pdb_ligand))
        tasks.append(
            {
                "rows": frame.to_dict(orient="records"),
                "structure_dir": str(source_pdbs_root / class_code / f"{stem}_out"),
                "label": str(pdb_ligand),
                "stem": stem,
            }
        )
    return _run_tasks_and_write(tasks, base, output_path, workers)


def build_holdout_v_graph_tunnel_parquet(
    base_parquet: Path,
    structures_root: Path,
    output_path: Path,
    workers: int = 6,
) -> dict[str, object]:
    base = pd.read_parquet(base_parquet)
    if "structure_id" not in base.columns or "matched_pocket_number" not in base.columns:
        raise ValueError(f"{base_parquet}: expected structure_id + matched_pocket_number columns")

    tasks: list[dict[str, object]] = []
    for structure_id, frame in base.groupby("structure_id", sort=False):
        stem = str(structure_id)
        tasks.append(
            {
                "rows": frame.to_dict(orient="records"),
                "structure_dir": str(structures_root / f"{stem}_out"),
                "label": stem,
                "stem": stem,
            }
        )
    return _run_tasks_and_write(tasks, base, output_path, workers)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    training = sub.add_parser(
        "training", help="Build processed/v_graph_tunnel/full_pockets.parquet."
    )
    training.add_argument(
        "--base-parquet", type=Path, default=Path("processed/v_sterol/full_pockets.parquet")
    )
    training.add_argument(
        "--source-pdbs-root", type=Path, default=Path("data/structures/source_pdbs")
    )
    training.add_argument(
        "--output", type=Path, default=Path("processed/v_graph_tunnel/full_pockets.parquet")
    )
    training.add_argument("--workers", type=int, default=6)

    holdout = sub.add_parser("holdout", help="Build a v_graph_tunnel holdout parquet.")
    holdout.add_argument("--base-parquet", type=Path, required=True)
    holdout.add_argument("--structures-root", type=Path, required=True)
    holdout.add_argument("--output", type=Path, required=True)
    holdout.add_argument("--workers", type=int, default=6)

    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )
    try:
        if args.command == "training":
            summary = build_training_v_graph_tunnel_parquet(
                base_parquet=args.base_parquet,
                source_pdbs_root=args.source_pdbs_root,
                output_path=args.output,
                workers=args.workers,
            )
        elif args.command == "holdout":
            summary = build_holdout_v_graph_tunnel_parquet(
                base_parquet=args.base_parquet,
                structures_root=args.structures_root,
                output_path=args.output,
                workers=args.workers,
            )
        else:
            raise ValueError(f"unknown command {args.command}")
    except Exception as exc:
        logging.error("v_graph_tunnel build failed: %s", exc)
        return 1

    print(f"Rows: {summary['rows']}")
    print(f"Structures: {summary['structures']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Output: {summary['output_path']} ({summary['output_size_mb']:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())


assert GRAPH_TUNNEL_FEATURES_3 == _REGISTRY_GRAPH_TUNNEL_FEATURES_3, (
    "graph_tunnel_features.GRAPH_TUNNEL_FEATURES_3 drifted from constants.GRAPH_TUNNEL_FEATURES_3"
)
