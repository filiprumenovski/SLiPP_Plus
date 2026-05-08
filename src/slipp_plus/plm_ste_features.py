"""Palmitate-vs-steryl disambiguation features.

Adds 16 pocket-level features tuned to separate palmitate (PLM; linear 16C
acyl chain) from steryl ester (STE; sterol headgroup + acyl tail). Built on
top of ``v_sterol`` and reuses the same shell / centroid / chemistry group
conventions so columns concatenate cleanly onto
``processed/v_sterol/full_pockets.parquet``.

Feature groups (in canonical order, see
``constants.PALMITATE_VS_STERYL_EXTRA_16``):

* Group A (4) — CRAC / CARC sequence motifs contacting the pocket.
* Group B (7) — axial profile derived from the alpha-sphere PCA principal
    axis (length, radius variation, gauge-invariant gradient magnitude,
    tapering, bottleneck, aspect).
* Group C (5) — polar-anchor chemistry at the axial end with the highest
  polar-residue density.
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

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

from .aromatic_aliphatic import (
    _parse_pqr_coordinates,
    _shell_index,
)
from .sterol_features import (
    _RESIDUE_TO_GROUP,
    CHEMISTRY_GROUP_ORDER,
)

# ---------------------------------------------------------------------------
# Feature columns (authoritative order; mirror
# constants.PALMITATE_VS_STERYL_EXTRA_16)
# ---------------------------------------------------------------------------
PALMITATE_VS_STERYL_EXTRA_16: list[str] = [
    # Group A - motifs
    "crac_count",
    "carc_count",
    "any_sterol_motif",
    "motif_residue_density",
    # Group B - axial profile
    "axial_length",
    "axial_radius_std",
    "axial_radius_gradient",
    "fatend_ratio",
    "bottleneck_position",
    "thick_end_asymmetry",
    "cross_section_aspect",
    # Group C - polar-anchor chemistry
    "polar_end_cationic_count",
    "polar_end_aromatic_polar_count",
    "polar_end_neutral_polar_count",
    "anchor_charge_balance",
    "anchor_chemistry_entropy",
]

# Backward-compatible alias kept to avoid broader churn in the current slice.
PLM_STE_EXTRA_16 = PALMITATE_VS_STERYL_EXTRA_16

_INT_COLS: frozenset[str] = frozenset(
    {
        "crac_count",
        "carc_count",
        "any_sterol_motif",
        "polar_end_cationic_count",
        "polar_end_aromatic_polar_count",
        "polar_end_neutral_polar_count",
    }
)

# Polar residue groups used to decide "polar end" of the axis.
_POLAR_GROUPS: frozenset[str] = frozenset(
    {
        "aromatic_polar",
        "polar_neutral",
        "cationic",
        "anionic",
    }
)

# 3-letter -> 1-letter for the canonical 20 AAs.
# Non-standard / HETATM residues are represented as 'X' in the sequence so
# that the linear positional mapping is preserved without accidentally
# matching motif residue classes.
_THREE_TO_ONE: dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

_CRAC_RE = re.compile(r"[LV].{1,5}[Y].{1,5}[RK]")
_CARC_RE = re.compile(r"[RK].{1,5}[YF].{1,5}[LV]")

_EPS: float = 1e-6
_LOG7: float = math.log(7.0)


# ---------------------------------------------------------------------------
# Low-level helpers
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


def _closest_heavy_atom_info(
    residue: object, centroid: np.ndarray
) -> tuple[float, np.ndarray] | None:
    """Return (distance, coord) for the heavy atom closest to ``centroid``."""
    best_dist: float | None = None
    best_coord: np.ndarray | None = None
    for atom in residue:  # type: ignore[attr-defined]
        if atom.element.strip().upper() == "H":
            continue
        coord = np.asarray(atom.get_coord(), dtype=float)
        dist = float(np.linalg.norm(coord - centroid))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_coord = coord
    if best_dist is None or best_coord is None:
        return None
    return best_dist, best_coord


def _parse_protein_chains(
    protein_pdb_path: Path,
) -> list[dict[str, object]]:
    """Parse a protein PDB into per-chain residue records.

    Each chain dict has:

    * ``chain_id``: str
    * ``sequence``: 1-letter sequence string (unknowns -> 'X')
    * ``residues``: list of residue objects in linear order, aligned 1:1
      with ``sequence``
    """

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(protein_pdb_path.stem, protein_pdb_path)
    chains_out: list[dict[str, object]] = []
    seen_chain_ids: set[str] = set()
    # Iterate the first model only; multi-model ensembles would otherwise
    # duplicate residues and inflate motif counts.
    try:
        model = next(structure.get_models())
    except StopIteration:
        return chains_out

    for chain in model.get_chains():
        chain_id = chain.id
        if chain_id in seen_chain_ids:
            continue
        seen_chain_ids.add(chain_id)
        residues: list[object] = []
        letters: list[str] = []
        for residue in chain.get_residues():
            hetflag = residue.id[0]
            # Skip waters / heteroatoms (hetflag != ' ').
            if hetflag != " ":
                continue
            name = residue.get_resname().strip().upper()
            letters.append(_THREE_TO_ONE.get(name, "X"))
            residues.append(residue)
        if not residues:
            continue
        chains_out.append(
            {
                "chain_id": chain_id,
                "sequence": "".join(letters),
                "residues": residues,
            }
        )
    return chains_out


def _axial_profile(
    vert_coords: np.ndarray,
) -> dict[str, object] | None:
    """Compute axial profile features and return projection context.

    Returns a dict with the seven Group B feature values plus ``axis``,
    ``centroid``, and ``bin_edges`` (np.ndarray of length 6), or ``None``
    when the pocket has fewer than 5 alpha-spheres.
    """

    if len(vert_coords) < 5:
        return None

    centroid = vert_coords.mean(axis=0)
    centered = vert_coords - centroid
    # Cov uses columns as features; transpose so cov is 3x3.
    cov = np.cov(centered.T)
    # eigvalsh / eigh return ascending eigenvalues.
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    lam3, lam2, _lam1 = float(eigvals[0]), float(eigvals[1]), float(eigvals[2])
    axis = eigvecs[:, -1]  # principal axis (largest eigenvalue)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= _EPS:
        return None
    axis = axis / axis_norm

    t = centered @ axis
    t_min = float(t.min())
    t_max = float(t.max())
    axial_length = t_max - t_min
    if axial_length <= _EPS:
        return None

    radial_vec = centered - np.outer(t, axis)
    radial = np.linalg.norm(radial_vec, axis=1)

    bin_edges = np.linspace(t_min, t_max, 6)
    # searchsorted with side='right' maps t_min to bin 0..1; we want bin 0.
    bin_idx = np.searchsorted(bin_edges, t, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, 4).astype(int)

    bin_means: list[float] = []
    bin_has: list[bool] = []
    for b in range(5):
        mask = bin_idx == b
        if mask.any():
            bin_means.append(float(radial[mask].mean()))
            bin_has.append(True)
        else:
            bin_means.append(float("nan"))
            bin_has.append(False)

    global_mean_radial = float(radial.mean())
    # Carry forward. Leading empty bins use global mean as fallback.
    last_valid: float | None = None
    for b in range(5):
        if bin_has[b]:
            last_valid = bin_means[b]
        else:
            if last_valid is None:
                bin_means[b] = global_mean_radial
            else:
                bin_means[b] = last_valid
    R = np.asarray(bin_means, dtype=float)

    axial_radius_std = float(R.std())
    # PCA eigenvector sign is arbitrary, so expose only the slope magnitude.
    x = np.arange(1.0, 6.0)
    slope = float(abs(np.polyfit(x, R, 1)[0]))
    max_r = float(R.max())
    min_r = float(R.min())
    fatend_ratio = max_r / (min_r + _EPS)
    bottleneck_position = float(np.argmin(R)) / 4.0
    mean_r = float(R.mean())
    thick_end_asymmetry = float(abs(R[0] - R[4]) / (mean_r + _EPS))
    aspect = math.sqrt(lam2 / lam3) if lam3 > _EPS else 100.0
    cross_section_aspect = float(np.clip(aspect, 1.0, 100.0))

    return {
        "axial_length": float(axial_length),
        "axial_radius_std": axial_radius_std,
        "axial_radius_gradient": slope,
        "fatend_ratio": fatend_ratio,
        "bottleneck_position": bottleneck_position,
        "thick_end_asymmetry": thick_end_asymmetry,
        "cross_section_aspect": cross_section_aspect,
        "axis": axis,
        "centroid": centroid,
        "bin_edges": bin_edges,
    }


def _empty_axial_features() -> dict[str, float]:
    return {
        "axial_length": 0.0,
        "axial_radius_std": 0.0,
        "axial_radius_gradient": 0.0,
        "fatend_ratio": 1.0,
        "bottleneck_position": 0.5,
        "thick_end_asymmetry": 0.0,
        "cross_section_aspect": 1.0,
    }


def _empty_anchor_features() -> dict[str, float]:
    return {
        "polar_end_cationic_count": 0,
        "polar_end_aromatic_polar_count": 0,
        "polar_end_neutral_polar_count": 0,
        "anchor_charge_balance": 0.0,
        "anchor_chemistry_entropy": 0.0,
    }


def _empty_motif_features() -> dict[str, float]:
    return {
        "crac_count": 0,
        "carc_count": 0,
        "any_sterol_motif": 0,
        "motif_residue_density": 0.0,
    }


def _bin_of_t(t: float, bin_edges: np.ndarray) -> int:
    idx = int(np.searchsorted(bin_edges, t, side="right") - 1)
    if idx < 0:
        idx = 0
    if idx > 4:
        idx = 4
    return idx


# ---------------------------------------------------------------------------
# Per-pocket extractor
# ---------------------------------------------------------------------------
def _compute_motif_and_anchor_features(
    chains: list[dict[str, object]],
    pocket_centroid: np.ndarray,
    axial_ctx: dict[str, object] | None,
) -> dict[str, float]:
    """Compute Group A + Group C features for a single pocket."""

    # --- First pass: per-residue distance / chem group / axial bin ---
    # We cache closest-heavy-atom info per residue for this pocket so we
    # don't re-walk atoms multiple times.
    contacting_residues_all: list[dict[str, object]] = []
    for chain in chains:
        residues: list[object] = chain["residues"]  # type: ignore[assignment]
        for residue in residues:
            info = _closest_heavy_atom_info(residue, pocket_centroid)
            if info is None:
                continue
            distance, coord = info
            shell = _shell_index(distance)
            if shell is None:
                continue
            # Shells are 0-indexed; "shells 1+2" per AGENTS convention
            # correspond to index 0 (<3A) and 1 (3-6A), i.e. distance <= 6A.
            if shell > 1:
                continue
            resname = residue.get_resname().strip().upper()  # type: ignore[attr-defined]
            group = _RESIDUE_TO_GROUP.get(resname)
            contacting_residues_all.append(
                {
                    "residue": residue,
                    "resname": resname,
                    "group": group,
                    "coord": coord,
                    "chain_id": chain["chain_id"],
                    "res_id": residue.get_id(),  # type: ignore[attr-defined]
                }
            )

    total_contacting = len(contacting_residues_all)

    # --- Group A: CRAC / CARC motif scan ---
    crac_count = 0
    carc_count = 0
    motif_residue_ids: set[tuple[str, tuple]] = set()
    for chain in chains:
        residues: list[object] = chain["residues"]  # type: ignore[assignment]
        sequence: str = chain["sequence"]  # type: ignore[assignment]
        chain_id: str = chain["chain_id"]  # type: ignore[assignment]
        for pattern, is_crac in ((_CRAC_RE, True), (_CARC_RE, False)):
            for match in pattern.finditer(sequence):
                start, end = match.span()  # [start, end)
                motif_residues = residues[start:end]
                # Contacting iff any residue in motif has dist <= 6A.
                contacting = False
                for mres in motif_residues:
                    info = _closest_heavy_atom_info(mres, pocket_centroid)
                    if info is None:
                        continue
                    dist = info[0]
                    if dist <= 6.0:
                        contacting = True
                        break
                if not contacting:
                    continue
                if is_crac:
                    crac_count += 1
                else:
                    carc_count += 1
                for mres in motif_residues:
                    motif_residue_ids.add(
                        (chain_id, tuple(mres.get_id()))  # type: ignore[attr-defined]
                    )

    any_sterol_motif = 1 if (crac_count + carc_count) > 0 else 0

    if total_contacting > 0 and motif_residue_ids:
        numer = sum(
            1
            for row in contacting_residues_all
            if (row["chain_id"], tuple(row["res_id"])) in motif_residue_ids
        )
        motif_residue_density = float(numer) / float(total_contacting)
    else:
        motif_residue_density = 0.0

    motif_features = {
        "crac_count": crac_count,
        "carc_count": carc_count,
        "any_sterol_motif": any_sterol_motif,
        "motif_residue_density": motif_residue_density,
    }

    # --- Group C: polar-anchor chemistry at the more-polar axial end ---
    if axial_ctx is None:
        anchor_features = _empty_anchor_features()
        return {**motif_features, **anchor_features}

    axis: np.ndarray = axial_ctx["axis"]  # type: ignore[assignment]
    centroid: np.ndarray = axial_ctx["centroid"]  # type: ignore[assignment]
    bin_edges: np.ndarray = axial_ctx["bin_edges"]  # type: ignore[assignment]

    # Annotate each contacting residue with its axial bin.
    bin_records: list[dict[str, object]] = []
    for row in contacting_residues_all:
        coord = row["coord"]
        t = float(np.dot(coord - centroid, axis))  # type: ignore[operator]
        b = _bin_of_t(t, bin_edges)
        bin_records.append({**row, "bin": b, "t": t})

    def _polar_density(bin_id: int) -> tuple[float, int]:
        residues_in_bin = [r for r in bin_records if r["bin"] == bin_id]
        if not residues_in_bin:
            return 0.0, 0
        polar = sum(1 for r in residues_in_bin if r["group"] in _POLAR_GROUPS)
        return polar / len(residues_in_bin), len(residues_in_bin)

    density0, _count0 = _polar_density(0)
    density4, _count4 = _polar_density(4)
    # Tie-break deterministically toward bin 0 when equal.
    polar_end_bin = 0 if density0 >= density4 else 4

    end_residues = [r for r in bin_records if r["bin"] == polar_end_bin]
    group_counts: dict[str, int] = {group: 0 for group in CHEMISTRY_GROUP_ORDER}
    for row in end_residues:
        group = row["group"]
        if group is not None:
            group_counts[group] += 1

    cationic = group_counts["cationic"]
    anionic = group_counts["anionic"]
    aromatic_polar = group_counts["aromatic_polar"]
    polar_neutral = group_counts["polar_neutral"]

    anchor_charge_balance = float((cationic - anionic) / (cationic + anionic + 1.0))

    total_group_residues = sum(group_counts.values())
    if total_group_residues > 0:
        entropy = 0.0
        for count in group_counts.values():
            if count <= 0:
                continue
            p = count / total_group_residues
            entropy -= p * math.log(p)
        anchor_chemistry_entropy = float(entropy / _LOG7)
    else:
        anchor_chemistry_entropy = 0.0

    anchor_features = {
        "polar_end_cationic_count": cationic,
        "polar_end_aromatic_polar_count": aromatic_polar,
        "polar_end_neutral_polar_count": polar_neutral,
        "anchor_charge_balance": anchor_charge_balance,
        "anchor_chemistry_entropy": anchor_chemistry_entropy,
    }

    return {**motif_features, **anchor_features}


def extract_pocket_plm_ste_features(
    atm_path: Path,
    vert_path: Path,
    protein_pdb_path: Path,
) -> dict[str, float]:
    """Return all 16 PLM-vs-STE features for a single pocket.

    ``atm_path`` is accepted for signature symmetry with
    :func:`sterol_features.extract_pocket_sterol_features` but is not read
    here — all residue information is sourced from the raw protein PDB so
    the sequence-level motif scan has the full chain context.
    """

    del atm_path  # kept for API symmetry
    chains = _parse_protein_chains(protein_pdb_path)
    vert_coords = _collect_vert_coords(vert_path)
    pocket_centroid = vert_coords.mean(axis=0)
    axial_ctx = _axial_profile(vert_coords)

    if axial_ctx is not None:
        axial_features = {
            key: axial_ctx[key]
            for key in (
                "axial_length",
                "axial_radius_std",
                "axial_radius_gradient",
                "fatend_ratio",
                "bottleneck_position",
                "thick_end_asymmetry",
                "cross_section_aspect",
            )
        }
    else:
        axial_features = _empty_axial_features()

    motif_and_anchor = _compute_motif_and_anchor_features(chains, pocket_centroid, axial_ctx)

    features: dict[str, float] = {}
    features.update(motif_and_anchor)
    features.update(axial_features)
    return features


# ---------------------------------------------------------------------------
# Worker plumbing (parallel, grouped by structure)
# ---------------------------------------------------------------------------
def _cast_row_features(features: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for column in PALMITATE_VS_STERYL_EXTRA_16:
        value = features[column]
        if column in _INT_COLS:
            try:
                iv = int(value)
            except (TypeError, ValueError):
                iv = 0
            out[column] = iv
        else:
            try:
                fv = float(value)
            except (TypeError, ValueError):
                fv = 0.0
            if not math.isfinite(fv):
                fv = 0.0
            out[column] = fv
    return out


def _empty_features() -> dict[str, float]:
    merged: dict[str, float] = {}
    merged.update(_empty_motif_features())
    merged.update(_empty_axial_features())
    merged.update(_empty_anchor_features())
    return _cast_row_features(merged)


def _process_group(task: dict[str, object]) -> dict[str, object]:
    """Extract 16 features for every row in a pdb_ligand / structure group."""

    rows = list(task["rows"])  # type: ignore[arg-type]
    structure_dir = Path(str(task["structure_dir"]))
    protein_pdb = Path(str(task["protein_pdb"]))
    label = str(task["label"])

    warnings: list[str] = []
    chains: list[dict[str, object]] | None = None
    try:
        chains = _parse_protein_chains(protein_pdb)
    except (OSError, ValueError) as exc:
        warnings.append(f"{label}: failed to parse protein pdb: {exc}")
        chains = None

    pocket_cache: dict[int, dict[str, float]] = {}
    enriched: list[dict[str, object]] = []
    for row in rows:
        pocket_number = int(row["matched_pocket_number"])
        if pocket_number not in pocket_cache:
            vert_path = structure_dir / "pockets" / f"pocket{pocket_number}_vert.pqr"
            atm_path = structure_dir / "pockets" / f"pocket{pocket_number}_atm.pdb"
            if chains is None or not vert_path.exists():
                warnings.append(
                    f"{label}: missing inputs for pocket{pocket_number} "
                    f"(chains={chains is not None}, vert={vert_path.exists()})"
                )
                pocket_cache[pocket_number] = _empty_features()
            else:
                try:
                    vert_coords = _collect_vert_coords(vert_path)
                    pocket_centroid = vert_coords.mean(axis=0)
                    axial_ctx = _axial_profile(vert_coords)
                    if axial_ctx is not None:
                        axial_features = {
                            key: axial_ctx[key]
                            for key in (
                                "axial_length",
                                "axial_radius_std",
                                "axial_radius_gradient",
                                "fatend_ratio",
                                "bottleneck_position",
                                "thick_end_asymmetry",
                                "cross_section_aspect",
                            )
                        }
                    else:
                        axial_features = _empty_axial_features()

                    motif_and_anchor = _compute_motif_and_anchor_features(
                        chains, pocket_centroid, axial_ctx
                    )
                    merged: dict[str, float] = {}
                    merged.update(motif_and_anchor)
                    merged.update(axial_features)
                    pocket_cache[pocket_number] = _cast_row_features(merged)
                except (OSError, ValueError) as exc:
                    warnings.append(f"{label}/pocket{pocket_number}: extraction failed: {exc}")
                    pocket_cache[pocket_number] = _empty_features()
            # Silence unused-variable lint in the rare case atm_path is unused;
            # keep the reference so future debugging can log both file paths.
            _ = atm_path
        features = pocket_cache[pocket_number]
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


def build_training_v_plm_ste_parquet(
    base_parquet: Path,
    source_pdbs_root: Path,
    output_path: Path,
    workers: int = 6,
) -> dict[str, object]:
    """Extend the v_sterol training parquet with 16 PLM-vs-STE features."""

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


def build_holdout_v_plm_ste_parquet(
    base_parquet: Path,
    structures_root: Path,
    output_path: Path,
    workers: int = 6,
) -> dict[str, object]:
    """Extend a v_sterol holdout parquet with 16 PLM-vs-STE features.

    ``structures_root`` contains ``<stem>.pdb`` and ``<stem>_out/`` fpocket
    outputs (e.g. ``processed/v_sterol/structures/apo_pdb_holdout``).
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
            f"row drift after v_plm_ste join: got {len(enriched)}, expected {len(base)}"
        )

    missing = [column for column in PALMITATE_VS_STERYL_EXTRA_16 if column not in enriched.columns]
    if missing:
        raise ValueError(f"output missing v_plm_ste feature columns: {missing}")

    # Finite/NaN sanity check on the 16 new columns.
    for column in PALMITATE_VS_STERYL_EXTRA_16:
        series = enriched[column]
        if series.isna().any():
            raise ValueError(f"{column}: NaN values present after build")
        numeric = pd.to_numeric(series, errors="raise")
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f"{column}: non-finite values present after build")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output_path, index=False)

    return {
        "rows": len(enriched),
        "structures": len(tasks),
        "warnings": len(all_warnings),
        "warnings_list": all_warnings,
        "output_path": output_path,
        "output_size_mb": output_path.stat().st_size / (1024 * 1024),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    training = sub.add_parser("training", help="Build processed/v_plm_ste/full_pockets.parquet.")
    training.add_argument(
        "--base-parquet",
        type=Path,
        default=Path("processed/v_sterol/full_pockets.parquet"),
        help="Input v_sterol full_pockets parquet.",
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
        default=Path("processed/v_plm_ste/full_pockets.parquet"),
        help="Destination parquet path.",
    )
    training.add_argument("--workers", type=int, default=6)

    holdout = sub.add_parser("holdout", help="Build a v_plm_ste holdout parquet.")
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
            summary = build_training_v_plm_ste_parquet(
                base_parquet=args.base_parquet,
                source_pdbs_root=args.source_pdbs_root,
                output_path=args.output,
                workers=args.workers,
            )
        elif args.command == "holdout":
            summary = build_holdout_v_plm_ste_parquet(
                base_parquet=args.base_parquet,
                structures_root=args.structures_root,
                output_path=args.output,
                workers=args.workers,
            )
        else:
            raise ValueError(f"unknown command {args.command}")
    except Exception as exc:
        logging.error("v_plm_ste build failed: %s", exc)
        return 1

    print(f"Rows: {summary['rows']}")
    print(f"Structures: {summary['structures']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Output: {summary['output_path']} ({summary['output_size_mb']:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Re-export the canonical constant via the constants module for symmetry.
# ``constants.PALMITATE_VS_STERYL_EXTRA_16`` is the authoritative import path
# for the rest of the codebase; this local list is kept in lockstep.
from .constants import PALMITATE_VS_STERYL_EXTRA_16 as _PALMITATE_VS_STERYL_EXTRA_16  # noqa: E402

assert PALMITATE_VS_STERYL_EXTRA_16 == _PALMITATE_VS_STERYL_EXTRA_16, (
    "plm_ste_features.PALMITATE_VS_STERYL_EXTRA_16 drifted from "
    "constants.PALMITATE_VS_STERYL_EXTRA_16"
)
