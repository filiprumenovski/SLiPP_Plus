"""Sterol-targeted feature extraction from fpocket outputs.

Adds 38 chemistry-refined and pocket-geometry features to the v49 base:

* 28 residue-shell counts across 7 chemistry groups x 4 shells.
* 4 polar/hydrophobic ratio features per shell.
* 5 alpha-sphere PCA geometry features (eigenvalues + elongation + planarity).
* 1 pocket burial feature vs. protein C-alpha centroid.

Designed to materially sharpen CLR (cholesterol) and STE (steryl ester) signal
while reusing the existing v49 pocket matching and helpers from
``aromatic_aliphatic.py``.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

from .aromatic_aliphatic import (
    _closest_heavy_atom_distance,
    _compute_centroid,
    _parse_pqr_coordinates,
    _shell_index,
    _symmetric_jeffreys_log_ratio,
)

# ---------------------------------------------------------------------------
# Chemistry groups (strict residue-code sets)
# ---------------------------------------------------------------------------
# GLY is intentionally not in any group: no side chain -> no discriminative
# sterol-contact signal. Every other canonical AA lives in exactly one group.
CHEMISTRY_GROUPS: dict[str, frozenset[str]] = {
    "aromatic_pi": frozenset({"PHE", "TRP"}),
    "aromatic_polar": frozenset({"TYR", "HIS"}),
    "bulky_hydrophobic": frozenset({"LEU", "ILE", "VAL", "MET"}),
    "small_special": frozenset({"ALA", "PRO", "CYS"}),
    "polar_neutral": frozenset({"SER", "THR", "ASN", "GLN"}),
    "cationic": frozenset({"LYS", "ARG"}),
    "anionic": frozenset({"ASP", "GLU"}),
}

# Deterministic ordering used for column generation and feature concatenation.
CHEMISTRY_GROUP_ORDER: list[str] = [
    "aromatic_pi",
    "aromatic_polar",
    "bulky_hydrophobic",
    "small_special",
    "polar_neutral",
    "cationic",
    "anionic",
]

# Reverse lookup: residue code -> group name.
_RESIDUE_TO_GROUP: dict[str, str] = {
    residue: group for group, residues in CHEMISTRY_GROUPS.items() for residue in residues
}

SHELLS: tuple[int, int, int, int] = (1, 2, 3, 4)

STEROL_CHEMISTRY_SHELL_COLS: list[str] = [
    f"{group}_count_shell{shell}" for group in CHEMISTRY_GROUP_ORDER for shell in SHELLS
] + [f"polar_hydrophobic_ratio_shell{shell}" for shell in SHELLS]

POCKET_GEOMETRY_COLS: list[str] = [
    "pocket_lam1",
    "pocket_lam2",
    "pocket_lam3",
    "pocket_elongation",
    "pocket_planarity",
    "pocket_burial",
]

STEROL_FEATURES_38: list[str] = STEROL_CHEMISTRY_SHELL_COLS + POCKET_GEOMETRY_COLS

_COUNT_COLUMNS: list[str] = [
    f"{group}_count_shell{shell}" for group in CHEMISTRY_GROUP_ORDER for shell in SHELLS
]
_RATIO_COLUMNS: list[str] = [f"polar_hydrophobic_ratio_shell{shell}" for shell in SHELLS]

_EPS: float = 1e-6


# ---------------------------------------------------------------------------
# Low-level extractors
# ---------------------------------------------------------------------------
def _collect_vert_coords(vert_path: Path) -> np.ndarray:
    """Return all alpha-sphere XYZ coordinates from a fpocket vert.pqr file."""
    coords: list[list[float]] = []
    with vert_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            coord = _parse_pqr_coordinates(line)
            if coord is not None:
                coords.append(coord)
    if not coords:
        raise ValueError(f"{vert_path.name}: no alpha-sphere coordinates")
    return np.asarray(coords, dtype=float)


def _protein_ca_stats(protein_pdb_path: Path) -> tuple[np.ndarray, float]:
    """Return (centroid, max_ca_spread) computed from C-alpha atoms.

    ``max_ca_spread`` is the maximum distance from any C-alpha to the centroid,
    i.e. the bounding radius of the protein skeleton. Gives the
    ``pocket_burial`` feature a stable 0 = buried, ~1 = surface normalization.
    """

    coords: list[list[float]] = []
    with protein_pdb_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            # Columns 13-16 hold the atom name; strip to match " CA "/"CA".
            if line[12:16].strip() != "CA":
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            coords.append([x, y, z])
    if not coords:
        raise ValueError(f"{protein_pdb_path.name}: no C-alpha atoms found")
    array = np.asarray(coords, dtype=float)
    centroid = array.mean(axis=0)
    max_spread = float(np.linalg.norm(array - centroid, axis=1).max())
    return centroid, max_spread


def _pocket_pca_features(vert_coords: np.ndarray) -> dict[str, float]:
    """PCA-based elongation/planarity from alpha-sphere coords."""
    if len(vert_coords) < 3:
        return {
            "pocket_lam1": 0.0,
            "pocket_lam2": 0.0,
            "pocket_lam3": 0.0,
            "pocket_elongation": 1.0,
            "pocket_planarity": 1.0,
        }
    cov = np.cov(vert_coords.T)
    eigs = np.linalg.eigvalsh(cov)
    # eigvalsh returns ascending; clamp tiny negatives from numerical noise.
    eigs = np.clip(eigs, 0.0, None)
    lam3, lam2, lam1 = float(eigs[0]), float(eigs[1]), float(eigs[2])
    return {
        "pocket_lam1": lam1,
        "pocket_lam2": lam2,
        "pocket_lam3": lam3,
        "pocket_elongation": lam1 / (lam3 + _EPS),
        "pocket_planarity": lam2 / (lam3 + _EPS),
    }


def _chemistry_shell_counts(atm_path: Path, pocket_centroid: np.ndarray) -> dict[str, int]:
    """Count residues per (chemistry group, shell) bucket for a pocket."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(atm_path.stem, atm_path)
    counts: dict[str, list[int]] = {group: [0, 0, 0, 0] for group in CHEMISTRY_GROUP_ORDER}
    for residue in structure.get_residues():
        residue_name = residue.get_resname().strip().upper()
        group = _RESIDUE_TO_GROUP.get(residue_name)
        if group is None:
            continue
        distance = _closest_heavy_atom_distance(residue, pocket_centroid)
        if distance is None:
            continue
        shell = _shell_index(distance)
        if shell is None:
            continue
        counts[group][shell] += 1
    flat: dict[str, int] = {}
    for group in CHEMISTRY_GROUP_ORDER:
        for shell in SHELLS:
            flat[f"{group}_count_shell{shell}"] = counts[group][shell - 1]
    return flat


def _polar_hydrophobic_ratios(counts: dict[str, int]) -> dict[str, float]:
    """Symmetric Jeffreys-smoothed log-ratio of polar to hydrophobic counts."""
    ratios: dict[str, float] = {}
    for shell in SHELLS:
        polar = (
            counts[f"aromatic_polar_count_shell{shell}"]
            + counts[f"polar_neutral_count_shell{shell}"]
        )
        hydrophobic = (
            counts[f"bulky_hydrophobic_count_shell{shell}"]
            + counts[f"aromatic_pi_count_shell{shell}"]
        )
        ratios[f"polar_hydrophobic_ratio_shell{shell}"] = _symmetric_jeffreys_log_ratio(
            polar, hydrophobic
        )
    return ratios


def extract_pocket_sterol_features(
    atm_path: Path,
    vert_path: Path,
    protein_pdb_path: Path,
) -> dict[str, float]:
    """Return all 38 sterol-targeted features for a single pocket.

    Reuses ``aromatic_aliphatic._compute_centroid``, ``_shell_index``, and
    ``_closest_heavy_atom_distance`` helpers so shell logic stays consistent
    with the v49 pipeline.
    """

    pocket_centroid = _compute_centroid(vert_path)
    counts = _chemistry_shell_counts(atm_path, pocket_centroid)
    ratios = _polar_hydrophobic_ratios(counts)

    vert_coords = _collect_vert_coords(vert_path)
    pca = _pocket_pca_features(vert_coords)

    protein_centroid, max_spread = _protein_ca_stats(protein_pdb_path)
    if max_spread <= _EPS:
        burial = 0.0
    else:
        burial = float(np.linalg.norm(vert_coords.mean(axis=0) - protein_centroid) / max_spread)

    features: dict[str, float] = {}
    features.update({key: int(value) for key, value in counts.items()})
    features.update(ratios)
    features.update(pca)
    features["pocket_burial"] = burial
    return features


# ---------------------------------------------------------------------------
# Worker / task plumbing
# ---------------------------------------------------------------------------
def _cast_row_features(features: dict[str, float]) -> dict[str, float]:
    """Ensure counts stay ints and floats stay finite floats."""
    out: dict[str, float] = {}
    for column in _COUNT_COLUMNS:
        out[column] = int(features[column])
    for column in _RATIO_COLUMNS:
        value = float(features[column])
        if not math.isfinite(value):
            value = 0.0
        out[column] = value
    for column in POCKET_GEOMETRY_COLS:
        value = float(features[column])
        if not math.isfinite(value):
            value = 0.0
        out[column] = value
    return out


def _empty_features() -> dict[str, float]:
    features: dict[str, float] = {column: 0 for column in _COUNT_COLUMNS}
    features.update({column: 0.0 for column in _RATIO_COLUMNS})
    features.update({column: 0.0 for column in POCKET_GEOMETRY_COLS})
    features["pocket_elongation"] = 1.0
    features["pocket_planarity"] = 1.0
    return features


def _process_group(task: dict[str, object]) -> dict[str, object]:
    """Extract 38 features for every row in a pdb_ligand group.

    Expects ``task`` to carry:

    * ``rows``: list of base parquet rows (dicts) sharing a pdb_ligand / structure_id.
    * ``structure_dir``: fpocket ``<stem>_out`` directory path.
    * ``protein_pdb``: raw protein PDB path (ATOM CA records).
    * ``label``: identifier used for warnings/logs.
    """

    rows = list(task["rows"])  # type: ignore[arg-type]
    structure_dir = Path(str(task["structure_dir"]))
    protein_pdb = Path(str(task["protein_pdb"]))
    label = str(task["label"])

    warnings: list[str] = []
    try:
        protein_centroid, max_spread = _protein_ca_stats(protein_pdb)
    except (OSError, ValueError) as exc:
        warnings.append(f"{label}: failed protein CA stats: {exc}")
        protein_centroid = np.zeros(3, dtype=float)
        max_spread = 0.0

    pocket_cache: dict[int, dict[str, float]] = {}
    enriched: list[dict[str, object]] = []
    for row in rows:
        pocket_number = int(row["matched_pocket_number"])
        if pocket_number not in pocket_cache:
            atm_path = structure_dir / "pockets" / f"pocket{pocket_number}_atm.pdb"
            vert_path = structure_dir / "pockets" / f"pocket{pocket_number}_vert.pqr"
            if not atm_path.exists() or not vert_path.exists():
                warnings.append(
                    f"{label}: missing pocket{pocket_number} files (atm={atm_path.exists()}, "
                    f"vert={vert_path.exists()})"
                )
                pocket_cache[pocket_number] = _empty_features()
            else:
                try:
                    vert_coords = _collect_vert_coords(vert_path)
                    pocket_centroid = vert_coords.mean(axis=0)
                    counts = _chemistry_shell_counts(atm_path, _compute_centroid(vert_path))
                    ratios = _polar_hydrophobic_ratios(counts)
                    pca = _pocket_pca_features(vert_coords)
                    if max_spread > _EPS:
                        burial = float(
                            np.linalg.norm(pocket_centroid - protein_centroid) / max_spread
                        )
                    else:
                        burial = 0.0
                    merged: dict[str, float] = {}
                    merged.update({k: int(v) for k, v in counts.items()})
                    merged.update(ratios)
                    merged.update(pca)
                    merged["pocket_burial"] = burial
                    pocket_cache[pocket_number] = merged
                except (OSError, ValueError) as exc:
                    warnings.append(f"{label}/pocket{pocket_number}: extraction failed: {exc}")
                    pocket_cache[pocket_number] = _empty_features()
        features = _cast_row_features(pocket_cache[pocket_number])
        enriched_row = dict(row)
        enriched_row.update(features)
        enriched.append(enriched_row)
    return {"rows": enriched, "warnings": warnings}


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------
def _class_code_from_pdb_ligand(pdb_ligand: str) -> str:
    return pdb_ligand.split("/", 1)[0]


def _pdb_stem(pdb_ligand: str) -> str:
    return Path(pdb_ligand).stem


def build_training_v_sterol_parquet(
    base_parquet: Path,
    source_pdbs_root: Path,
    output_path: Path,
    workers: int = 6,
) -> dict[str, object]:
    """Extend the v49 training parquet with 38 sterol-targeted features."""

    base = pd.read_parquet(base_parquet)
    if "pdb_ligand" not in base.columns or "matched_pocket_number" not in base.columns:
        raise ValueError(f"{base_parquet}: expected pdb_ligand + matched_pocket_number columns")

    groups = [
        (pdb_ligand, frame.copy()) for pdb_ligand, frame in base.groupby("pdb_ligand", sort=False)
    ]

    tasks: list[dict[str, object]] = []
    for pdb_ligand, frame in groups:
        class_code = _class_code_from_pdb_ligand(pdb_ligand)
        stem = _pdb_stem(pdb_ligand)
        structure_dir = source_pdbs_root / class_code / f"{stem}_out"
        protein_pdb = source_pdbs_root / pdb_ligand
        tasks.append(
            {
                "rows": frame.to_dict(orient="records"),
                "structure_dir": str(structure_dir),
                "protein_pdb": str(protein_pdb),
                "label": pdb_ligand,
            }
        )

    return _run_tasks_and_write(tasks, base, output_path, workers=workers)


def build_holdout_v_sterol_parquet(
    base_parquet: Path,
    structures_root: Path,
    output_path: Path,
    workers: int = 6,
) -> dict[str, object]:
    """Extend a v49 holdout parquet with 38 sterol-targeted features.

    ``structures_root`` is the directory containing ``<stem>.pdb`` and
    ``<stem>_out/`` fpocket outputs (e.g. ``processed/v49/structures/apo_pdb_holdout``).
    """

    base = pd.read_parquet(base_parquet)
    if "structure_id" not in base.columns or "matched_pocket_number" not in base.columns:
        raise ValueError(f"{base_parquet}: expected structure_id + matched_pocket_number columns")

    groups = [
        (structure_id, frame.copy())
        for structure_id, frame in base.groupby("structure_id", sort=False)
    ]

    tasks: list[dict[str, object]] = []
    for structure_id, frame in groups:
        stem = str(structure_id)
        structure_dir = structures_root / f"{stem}_out"
        protein_pdb = structures_root / f"{stem}.pdb"
        tasks.append(
            {
                "rows": frame.to_dict(orient="records"),
                "structure_dir": str(structure_dir),
                "protein_pdb": str(protein_pdb),
                "label": stem,
            }
        )

    return _run_tasks_and_write(tasks, base, output_path, workers=workers)


def _run_tasks_and_write(
    tasks: list[dict[str, object]],
    base: pd.DataFrame,
    output_path: Path,
    workers: int,
) -> dict[str, object]:
    if workers is None or workers <= 1 or len(tasks) == 1:
        results = [_process_group(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_process_group, tasks))

    all_rows: list[dict[str, object]] = []
    all_warnings: list[str] = []
    for result in results:
        all_rows.extend(result["rows"])  # type: ignore[arg-type]
        all_warnings.extend(result["warnings"])  # type: ignore[arg-type]

    for warning in all_warnings:
        logging.warning(warning)

    enriched = pd.DataFrame(all_rows)
    if "_row_order" in enriched.columns:
        enriched = enriched.sort_values("_row_order").reset_index(drop=True)
    if len(enriched) != len(base):
        raise ValueError(
            f"row drift after v_sterol join: got {len(enriched)}, expected {len(base)}"
        )

    missing = [column for column in STEROL_FEATURES_38 if column not in enriched.columns]
    if missing:
        raise ValueError(f"output missing sterol feature columns: {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output_path, index=False)

    return {
        "rows": len(enriched),
        "structures": len(tasks),
        "warnings": len(all_warnings),
        "output_path": output_path,
        "output_size_mb": output_path.stat().st_size / (1024 * 1024),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    training = sub.add_parser("training", help="Build processed/v_sterol/full_pockets.parquet.")
    training.add_argument(
        "--base-parquet",
        type=Path,
        default=Path("processed/v49/full_pockets.parquet"),
        help="Input v49 full_pockets parquet.",
    )
    training.add_argument(
        "--source-pdbs-root",
        type=Path,
        default=Path("data/structures/source_pdbs"),
        help="Root containing <CLASS>/<stem>.pdb and <stem>_out/ fpocket outputs.",
    )
    training.add_argument(
        "--output",
        type=Path,
        default=Path("processed/v_sterol/full_pockets.parquet"),
        help="Destination parquet path.",
    )
    training.add_argument("--workers", type=int, default=6)

    holdout = sub.add_parser("holdout", help="Build a v_sterol holdout parquet.")
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
            summary = build_training_v_sterol_parquet(
                base_parquet=args.base_parquet,
                source_pdbs_root=args.source_pdbs_root,
                output_path=args.output,
                workers=args.workers,
            )
        elif args.command == "holdout":
            summary = build_holdout_v_sterol_parquet(
                base_parquet=args.base_parquet,
                structures_root=args.structures_root,
                output_path=args.output,
                workers=args.workers,
            )
        else:
            raise ValueError(f"unknown command {args.command}")
    except Exception as exc:
        logging.error("v_sterol build failed: %s", exc)
        return 1

    print(f"Rows: {summary['rows']}")
    print(f"Structures: {summary['structures']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Output: {summary['output_path']} ({summary['output_size_mb']:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
