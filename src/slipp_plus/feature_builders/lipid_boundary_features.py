"""Lipid-subclass boundary feature extraction from matched fpocket pockets.

``v_lipid_boundary`` is an experimental branch layered on top of ``v_sterol``.
It targets the current lipid-class error modes: STE -> PLM, PLM <-> MYR,
OLA -> PLM/CLR, and lipid -> COA leakage. The features are deliberately
sign-stable: PCA axis direction is oriented by polar density at the two ends,
and directional shape summaries use ratios / absolute spread rather than raw
signed slopes.
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..constants import FEATURE_SETS, LIPID_BOUNDARY_FEATURES_22
from ..schemas import validate_holdout, validate_training
from .aromatic_aliphatic import _parse_pqr_coordinates, _shell_index
from .plm_ste_features import (
    _bin_of_t,
    _closest_heavy_atom_info,
    _parse_protein_chains,
)

HYDROPHOBE_RESIDUES: frozenset[str] = frozenset(
    {"ALA", "VAL", "LEU", "ILE", "MET", "PRO", "CYS", "PHE", "TRP", "TYR"}
)
AROMATIC_RESIDUES: frozenset[str] = frozenset({"PHE", "TYR", "TRP", "HIS"})
BETA_BRANCHED_RESIDUES: frozenset[str] = frozenset({"VAL", "ILE", "THR"})
DONOR_RESIDUES: frozenset[str] = frozenset(
    {"ARG", "ASN", "GLN", "HIS", "LYS", "SER", "THR", "TRP", "TYR", "CYS"}
)
ACCEPTOR_RESIDUES: frozenset[str] = frozenset(
    {"ASN", "ASP", "GLN", "GLU", "HIS", "SER", "THR", "TYR", "CYS", "MET"}
)
CATIONIC_RESIDUES: frozenset[str] = frozenset({"ARG", "LYS", "HIS"})
ANIONIC_RESIDUES: frozenset[str] = frozenset({"ASP", "GLU"})
CHARGED_RESIDUES: frozenset[str] = CATIONIC_RESIDUES | ANIONIC_RESIDUES
POLAR_RESIDUES: frozenset[str] = DONOR_RESIDUES | ACCEPTOR_RESIDUES | CHARGED_RESIDUES

P_LOOP_RE = re.compile(r"G.{3,5}G[KS][ST]?")
_EPS = 1e-6


def _collect_vert_coords(vert_path: Path) -> np.ndarray:
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


def _empty_features() -> dict[str, float]:
    return {
        "lb_axis_length": 0.0,
        "lb_radius_mean": 0.0,
        "lb_radius_std": 0.0,
        "lb_radius_range": 0.0,
        "lb_endpoint_radius_ratio": 1.0,
        "lb_center_radius_ratio": 1.0,
        "lb_linearity": 1.0,
        "lb_planar_spread": 1.0,
        "lb_tube_hydrophobe_fraction": 0.0,
        "lb_tube_gly_fraction": 0.0,
        "lb_tube_aromatic_fraction": 0.0,
        "lb_tube_beta_branched_fraction": 0.0,
        "lb_nonpolar_end_hydrophobe_count": 0.0,
        "lb_polar_end_donor_count": 0.0,
        "lb_polar_end_acceptor_count": 0.0,
        "lb_polar_end_charged_count": 0.0,
        "lb_polar_end_aromatic_count": 0.0,
        "lb_anchor_charge_balance": 0.0,
        "lb_p_loop_like_motif_count": 0.0,
        "lb_cationic_anchor_density": 0.0,
        "lb_phosphate_anchor_score": 0.0,
        "lb_gly_rich_anchor_fraction": 0.0,
    }


def _cast_features(features: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    defaults = _empty_features()
    for column in LIPID_BOUNDARY_FEATURES_22:
        try:
            value = float(features.get(column, defaults[column]))
        except (TypeError, ValueError):
            value = defaults[column]
        if not math.isfinite(value):
            value = defaults[column]
        if value < 0.0 and column not in {"lb_anchor_charge_balance"}:
            value = 0.0
        out[column] = value
    return out


def _axial_context(vert_coords: np.ndarray) -> dict[str, Any] | None:
    if len(vert_coords) < 5:
        return None
    centroid = vert_coords.mean(axis=0)
    centered = vert_coords - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    lam3, lam2, lam1 = float(eigvals[0]), float(eigvals[1]), float(eigvals[2])
    axis = eigvecs[:, -1]
    norm = float(np.linalg.norm(axis))
    if norm <= _EPS:
        return None
    axis = axis / norm
    t = centered @ axis
    axial_length = float(t.max() - t.min())
    if axial_length <= _EPS:
        return None
    radial = np.linalg.norm(centered - np.outer(t, axis), axis=1)
    bin_edges = np.linspace(float(t.min()), float(t.max()), 6)
    bin_idx = np.searchsorted(bin_edges, t, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, 4).astype(int)
    return {
        "centroid": centroid,
        "axis": axis,
        "t": t,
        "radial": radial,
        "bin_idx": bin_idx,
        "bin_edges": bin_edges,
        "lam1": lam1,
        "lam2": lam2,
        "lam3": lam3,
        "axial_length": axial_length,
    }


def _contact_records(
    chains: list[dict[str, object]],
    centroid: np.ndarray,
    axis: np.ndarray,
    bin_edges: np.ndarray,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for chain in chains:
        residues: list[object] = chain["residues"]  # type: ignore[assignment]
        chain_id: str = chain["chain_id"]  # type: ignore[assignment]
        for residue in residues:
            info = _closest_heavy_atom_info(residue, centroid)
            if info is None:
                continue
            distance, coord = info
            shell = _shell_index(distance)
            if shell is None:
                continue
            resname = residue.get_resname().strip().upper()  # type: ignore[attr-defined]
            t = float(np.dot(coord - centroid, axis))
            records.append(
                {
                    "residue": residue,
                    "resname": resname,
                    "chain_id": chain_id,
                    "res_id": tuple(residue.get_id()),  # type: ignore[attr-defined]
                    "distance": distance,
                    "shell": shell,
                    "coord": coord,
                    "t": t,
                    "bin": _bin_of_t(t, bin_edges),
                }
            )
    return records


def _polar_density(records: list[dict[str, object]], bin_id: int) -> float:
    in_bin = [r for r in records if int(r["bin"]) == bin_id]
    if not in_bin:
        return 0.0
    polar = sum(1 for r in in_bin if str(r["resname"]) in POLAR_RESIDUES)
    return float(polar / len(in_bin))


def _orient_context(
    ctx: dict[str, Any],
    records: list[dict[str, object]],
) -> tuple[dict[str, Any], list[dict[str, object]], int, int]:
    """Orient PCA axis so bin 0 is the more polar end.

    If the ends tie, no flip is applied and downstream geometry stays
    sign-invariant.
    """

    density0 = _polar_density(records, 0)
    density4 = _polar_density(records, 4)
    if density4 <= density0:
        return ctx, records, 0, 4

    oriented = dict(ctx)
    oriented["axis"] = -ctx["axis"]
    oriented["t"] = -ctx["t"]
    oriented["bin_idx"] = 4 - ctx["bin_idx"]

    flipped_records: list[dict[str, object]] = []
    for record in records:
        out = dict(record)
        out["t"] = -float(record["t"])
        out["bin"] = 4 - int(record["bin"])
        flipped_records.append(out)
    return oriented, flipped_records, 0, 4


def _bin_means(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    means: list[float] = []
    fallback = float(values.mean()) if len(values) else 0.0
    for i in range(5):
        mask = bins == i
        means.append(float(values[mask].mean()) if mask.any() else fallback)
    return np.asarray(means, dtype=float)


def _geometry_features(ctx: dict[str, Any]) -> dict[str, float]:
    radial = np.asarray(ctx["radial"], dtype=float)
    bins = np.asarray(ctx["bin_idx"], dtype=int)
    r_bin = _bin_means(radial, bins)
    endpoint_mean = float((r_bin[0] + r_bin[4]) / 2.0)
    center_mean = float(r_bin[2])
    max_endpoint = float(max(r_bin[0], r_bin[4]))
    min_endpoint = float(min(r_bin[0], r_bin[4]))
    axis_length = float(ctx["axial_length"])
    return {
        "lb_axis_length": axis_length,
        "lb_radius_mean": float(radial.mean()) if len(radial) else 0.0,
        "lb_radius_std": float(radial.std()) if len(radial) else 0.0,
        "lb_radius_range": float(radial.max() - radial.min()) if len(radial) else 0.0,
        "lb_endpoint_radius_ratio": max_endpoint / (min_endpoint + _EPS),
        "lb_center_radius_ratio": center_mean / (endpoint_mean + _EPS),
        "lb_linearity": float(axis_length / (axis_length + radial.mean() + _EPS)),
        "lb_planar_spread": float(math.sqrt(ctx["lam2"] / (ctx["lam3"] + _EPS))),
    }


def _fraction(records: list[dict[str, object]], residues: frozenset[str]) -> float:
    if not records:
        return 0.0
    return float(sum(1 for r in records if str(r["resname"]) in residues) / len(records))


def _chemistry_features(
    records: list[dict[str, object]],
    *,
    polar_end_bin: int,
    nonpolar_end_bin: int,
) -> dict[str, float]:
    tube_records = [r for r in records if int(r["bin"]) in {1, 2, 3}]
    polar_end = [r for r in records if int(r["bin"]) == polar_end_bin]
    nonpolar_end = [r for r in records if int(r["bin"]) == nonpolar_end_bin]

    cationic = sum(1 for r in polar_end if str(r["resname"]) in CATIONIC_RESIDUES)
    anionic = sum(1 for r in polar_end if str(r["resname"]) in ANIONIC_RESIDUES)
    donor = sum(1 for r in polar_end if str(r["resname"]) in DONOR_RESIDUES)
    acceptor = sum(1 for r in polar_end if str(r["resname"]) in ACCEPTOR_RESIDUES)
    charged = sum(1 for r in polar_end if str(r["resname"]) in CHARGED_RESIDUES)
    aromatic = sum(1 for r in polar_end if str(r["resname"]) in AROMATIC_RESIDUES)
    gly = sum(1 for r in polar_end if str(r["resname"]) == "GLY")
    polar_total = len(polar_end)

    cationic_density = cationic / (polar_total + 1.0)
    gly_fraction = gly / polar_total if polar_total else 0.0
    phosphate_score = (2.0 * cationic + donor + acceptor + gly) / (polar_total + 1.0)

    return {
        "lb_tube_hydrophobe_fraction": _fraction(tube_records, HYDROPHOBE_RESIDUES),
        "lb_tube_gly_fraction": _fraction(tube_records, frozenset({"GLY"})),
        "lb_tube_aromatic_fraction": _fraction(tube_records, AROMATIC_RESIDUES),
        "lb_tube_beta_branched_fraction": _fraction(tube_records, BETA_BRANCHED_RESIDUES),
        "lb_nonpolar_end_hydrophobe_count": float(
            sum(1 for r in nonpolar_end if str(r["resname"]) in HYDROPHOBE_RESIDUES)
        ),
        "lb_polar_end_donor_count": float(donor),
        "lb_polar_end_acceptor_count": float(acceptor),
        "lb_polar_end_charged_count": float(charged),
        "lb_polar_end_aromatic_count": float(aromatic),
        "lb_anchor_charge_balance": float((cationic - anionic) / (cationic + anionic + 1.0)),
        "lb_cationic_anchor_density": float(cationic_density),
        "lb_phosphate_anchor_score": float(phosphate_score),
        "lb_gly_rich_anchor_fraction": float(gly_fraction),
    }


def _p_loop_like_motif_count(
    chains: list[dict[str, object]],
    contact_keys_6a: set[tuple[str, tuple]],
) -> float:
    count = 0
    for chain in chains:
        sequence: str = chain["sequence"]  # type: ignore[assignment]
        residues: list[object] = chain["residues"]  # type: ignore[assignment]
        chain_id: str = chain["chain_id"]  # type: ignore[assignment]
        for match in P_LOOP_RE.finditer(sequence):
            motif_residues = residues[match.start() : match.end()]
            if any(
                (chain_id, tuple(residue.get_id())) in contact_keys_6a  # type: ignore[attr-defined]
                for residue in motif_residues
            ):
                count += 1
    return float(count)


def features_from_coordinates_and_chains(
    vert_coords: np.ndarray,
    chains: list[dict[str, object]],
) -> dict[str, float]:
    """Compute all lipid-boundary features from alpha spheres and protein chains."""

    ctx = _axial_context(vert_coords)
    if ctx is None:
        return _empty_features()

    records = _contact_records(chains, ctx["centroid"], ctx["axis"], ctx["bin_edges"])
    ctx, records, polar_end_bin, nonpolar_end_bin = _orient_context(ctx, records)
    geom = _geometry_features(ctx)
    chem = _chemistry_features(
        records,
        polar_end_bin=polar_end_bin,
        nonpolar_end_bin=nonpolar_end_bin,
    )
    contact_keys_6a = {
        (str(r["chain_id"]), r["res_id"])  # type: ignore[misc]
        for r in records
        if float(r["distance"]) <= 6.0
    }
    out: dict[str, float] = {}
    out.update(geom)
    out.update(chem)
    out["lb_p_loop_like_motif_count"] = _p_loop_like_motif_count(chains, contact_keys_6a)
    return _cast_features(out)


def extract_pocket_lipid_boundary_features(
    vert_path: Path,
    protein_pdb_path: Path,
) -> dict[str, float]:
    chains = _parse_protein_chains(protein_pdb_path)
    vert_coords = _collect_vert_coords(vert_path)
    return features_from_coordinates_and_chains(vert_coords, chains)


def _process_group(task: dict[str, object]) -> dict[str, object]:
    rows = list(task["rows"])  # type: ignore[arg-type]
    structure_dir = Path(str(task["structure_dir"]))
    protein_pdb = Path(str(task["protein_pdb"]))
    label = str(task["label"])

    warnings: list[str] = []
    try:
        chains = _parse_protein_chains(protein_pdb)
    except (OSError, ValueError) as exc:
        warnings.append(f"{label}: failed to parse protein pdb: {exc}")
        chains = []

    pocket_cache: dict[int, dict[str, float]] = {}
    enriched: list[dict[str, object]] = []
    for row in rows:
        pocket_number = int(row["matched_pocket_number"])
        if pocket_number not in pocket_cache:
            vert_path = structure_dir / "pockets" / f"pocket{pocket_number}_vert.pqr"
            if not chains or not vert_path.exists():
                warnings.append(
                    f"{label}: missing inputs for pocket{pocket_number} "
                    f"(chains={bool(chains)}, vert={vert_path.exists()})"
                )
                pocket_cache[pocket_number] = _empty_features()
            else:
                try:
                    vert_coords = _collect_vert_coords(vert_path)
                    pocket_cache[pocket_number] = features_from_coordinates_and_chains(
                        vert_coords, chains
                    )
                except (OSError, ValueError) as exc:
                    warnings.append(f"{label}/pocket{pocket_number}: extraction failed: {exc}")
                    pocket_cache[pocket_number] = _empty_features()
        enriched_row = dict(row)
        enriched_row.update(pocket_cache[pocket_number])
        enriched.append(enriched_row)
    return {"rows": enriched, "warnings": warnings}


def _class_code_from_pdb_ligand(pdb_ligand: str) -> str:
    return pdb_ligand.split("/", 1)[0]


def _pdb_stem(pdb_ligand: str) -> str:
    return Path(pdb_ligand).stem


def _ensure_matched_pocket_number(
    base: pd.DataFrame,
    structural_join: pd.DataFrame | None,
) -> pd.DataFrame:
    if "matched_pocket_number" in base.columns:
        return base
    if structural_join is None or "matched_pocket_number" not in structural_join.columns:
        raise ValueError(
            "base parquet lacks matched_pocket_number and no structural join provides it"
        )
    if len(base) != len(structural_join):
        raise ValueError(
            f"structural join row count mismatch: base={len(base)} join={len(structural_join)}"
        )
    if (
        "pdb_ligand" in base.columns
        and "pdb_ligand" in structural_join.columns
        and not base["pdb_ligand"]
        .reset_index(drop=True)
        .equals(structural_join["pdb_ligand"].reset_index(drop=True))
    ):
        raise ValueError("structural join pdb_ligand order does not match base parquet")
    out = base.copy()
    out["matched_pocket_number"] = structural_join["matched_pocket_number"].to_numpy()
    return out


def _run_tasks_and_write(
    tasks: list[dict[str, object]],
    base: pd.DataFrame,
    output_path: Path,
    workers: int,
    *,
    validate_feature_set: bool,
    holdout: bool,
    reports_dir: Path | None,
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
            f"row drift after v_lipid_boundary join: got {len(enriched)}, expected {len(base)}"
        )

    missing = [c for c in LIPID_BOUNDARY_FEATURES_22 if c not in enriched.columns]
    if missing:
        raise ValueError(f"output missing v_lipid_boundary feature columns: {missing}")
    for column in LIPID_BOUNDARY_FEATURES_22:
        numeric = pd.to_numeric(enriched[column], errors="raise")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f"{column}: non-finite values present after build")

    if validate_feature_set:
        if holdout:
            validate_holdout(enriched, FEATURE_SETS["v_lipid_boundary"])
        else:
            validate_training(enriched, FEATURE_SETS["v_lipid_boundary"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output_path, index=False)

    if reports_dir is not None:
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / "feature_build.md"
        with report_path.open("w", encoding="utf-8") as handle:
            handle.write("# v_lipid_boundary feature build\n\n")
            handle.write(f"- Rows: {len(enriched)}\n")
            handle.write(f"- Structures: {len(tasks)}\n")
            handle.write(f"- New features: {len(LIPID_BOUNDARY_FEATURES_22)}\n")
            handle.write(f"- Warnings: {len(all_warnings)}\n")
            handle.write(f"- Output: `{output_path}`\n")
            if all_warnings:
                handle.write("\n## Warnings\n\n")
                for warning in all_warnings[:200]:
                    handle.write(f"- {warning}\n")
                if len(all_warnings) > 200:
                    handle.write(f"- ... {len(all_warnings) - 200} more warnings omitted\n")

    return {
        "rows": len(enriched),
        "structures": len(tasks),
        "warnings": len(all_warnings),
        "warnings_list": all_warnings,
        "output_path": output_path,
        "output_size_mb": output_path.stat().st_size / (1024 * 1024),
    }


def build_training_v_lipid_boundary_parquet(
    base_parquet: Path,
    source_pdbs_root: Path,
    output_path: Path,
    structural_join_parquet: Path | None = Path("processed/v49/full_pockets.parquet"),
    reports_dir: Path | None = Path("reports/v_lipid_boundary"),
    workers: int = 6,
    validate_output: bool = True,
) -> dict[str, object]:
    """Extend the v_sterol training parquet with lipid-boundary features."""

    base = pd.read_parquet(base_parquet)
    join = (
        pd.read_parquet(structural_join_parquet)
        if structural_join_parquet is not None and structural_join_parquet.exists()
        else None
    )
    base = _ensure_matched_pocket_number(base, join)
    if "pdb_ligand" not in base.columns:
        raise ValueError(f"{base_parquet}: expected pdb_ligand column")

    tasks: list[dict[str, object]] = []
    for pdb_ligand, frame in base.groupby("pdb_ligand", sort=False):
        class_code = _class_code_from_pdb_ligand(str(pdb_ligand))
        stem = _pdb_stem(str(pdb_ligand))
        tasks.append(
            {
                "rows": frame.to_dict(orient="records"),
                "structure_dir": str(source_pdbs_root / class_code / f"{stem}_out"),
                "protein_pdb": str(source_pdbs_root / str(pdb_ligand)),
                "label": str(pdb_ligand),
            }
        )

    return _run_tasks_and_write(
        tasks,
        base,
        output_path,
        workers,
        validate_feature_set=validate_output,
        holdout=False,
        reports_dir=reports_dir,
    )


def build_holdout_v_lipid_boundary_parquet(
    base_parquet: Path,
    structures_root: Path,
    output_path: Path,
    reports_dir: Path | None = Path("reports/v_lipid_boundary"),
    workers: int = 6,
    validate_output: bool = True,
) -> dict[str, object]:
    """Extend a v_sterol holdout parquet with lipid-boundary features."""

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
                "protein_pdb": str(structures_root / f"{stem}.pdb"),
                "label": stem,
            }
        )

    return _run_tasks_and_write(
        tasks,
        base,
        output_path,
        workers,
        validate_feature_set=validate_output,
        holdout=True,
        reports_dir=reports_dir,
    )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    training = sub.add_parser(
        "training", help="Build processed/v_lipid_boundary/full_pockets.parquet."
    )
    training.add_argument(
        "--base-parquet", type=Path, default=Path("processed/v_sterol/full_pockets.parquet")
    )
    training.add_argument(
        "--source-pdbs-root", type=Path, default=Path("data/structures/source_pdbs")
    )
    training.add_argument(
        "--structural-join-parquet", type=Path, default=Path("processed/v49/full_pockets.parquet")
    )
    training.add_argument(
        "--output", type=Path, default=Path("processed/v_lipid_boundary/full_pockets.parquet")
    )
    training.add_argument("--reports-dir", type=Path, default=Path("reports/v_lipid_boundary"))
    training.add_argument("--workers", type=int, default=6)
    training.add_argument("--skip-validation", action="store_true")

    holdout = sub.add_parser("holdout", help="Build a v_lipid_boundary holdout parquet.")
    holdout.add_argument("--base-parquet", type=Path, required=True)
    holdout.add_argument("--structures-root", type=Path, required=True)
    holdout.add_argument("--output", type=Path, required=True)
    holdout.add_argument("--reports-dir", type=Path, default=Path("reports/v_lipid_boundary"))
    holdout.add_argument("--workers", type=int, default=6)
    holdout.add_argument("--skip-validation", action="store_true")

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
            summary = build_training_v_lipid_boundary_parquet(
                base_parquet=args.base_parquet,
                source_pdbs_root=args.source_pdbs_root,
                output_path=args.output,
                structural_join_parquet=args.structural_join_parquet,
                reports_dir=args.reports_dir,
                workers=args.workers,
                validate_output=not args.skip_validation,
            )
        elif args.command == "holdout":
            summary = build_holdout_v_lipid_boundary_parquet(
                base_parquet=args.base_parquet,
                structures_root=args.structures_root,
                output_path=args.output,
                reports_dir=args.reports_dir,
                workers=args.workers,
                validate_output=not args.skip_validation,
            )
        else:  # pragma: no cover
            raise ValueError(f"unknown command {args.command}")
    except Exception as exc:
        logging.error("v_lipid_boundary build failed: %s", exc)
        return 1
    logging.info("v_lipid_boundary build OK: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
