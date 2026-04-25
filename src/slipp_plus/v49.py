"""Build a v49 training parquet by joining Day 2 pocket-shell features."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from scipy.optimize import linear_sum_assignment

from .aromatic_aliphatic import (
    _classify_residue_name,
    _closest_heavy_atom_distance,
    _compute_centroid,
    _shell_index,
)
from .constants import AA20, AROMATIC_ALIPHATIC_12, SELECTED_17
from .ingest import _read_training_csv
from .schemas import validate_training

INFO_FIELD_MAP: dict[str, str] = {
    "Volume": "pock_vol",
    "Number of Alpha Spheres": "nb_AS",
    "Mean alpha sphere radius": "mean_as_ray",
    "Mean alp. sph. solvent access": "mean_as_solv_acc",
    "Apolar alpha sphere proportion": "apol_as_prop",
    "Mean local hydrophobic density": "mean_loc_hyd_dens",
    "Hydrophobicity score": "hydrophobicity_score",
    "Volume score": "volume_score",
    "Polarity score": "polarity_score",
    "Charge score": "charge_score",
    "Flexibility": "flex",
    "Proportion of polar atoms": "prop_polar_atm",
    "Alpha sphere density": "as_density",
    "Cent. of mass - Alpha Sphere max dist": "as_max_dst",
    "Polar SASA": "surf_pol_vdw14",
    "Apolar SASA": "surf_apol_vdw14",
    "Total SASA": "surf_vdw14",
}
POCKET_HEADER = re.compile(r"^Pocket\s+(?P<index>\d+)\s*:$")


def build_training_v49_parquet(
    training_csv: Path,
    fpocket_root: Path,
    output_path: Path,
    workers: int | None = None,
) -> dict[str, object]:
    """Create a training parquet with the 49-column Day 2 feature set."""

    base = _read_training_csv(training_csv).reset_index(drop=True)
    base.insert(0, "_row_order", np.arange(len(base), dtype=np.int64))
    groups = [(pdb_ligand, frame.copy()) for pdb_ligand, frame in base.groupby("pdb_ligand", sort=False)]

    tasks = [
        {
            "pdb_ligand": pdb_ligand,
            "rows": frame.to_dict(orient="records"),
            "fpocket_root": str(fpocket_root),
        }
        for pdb_ligand, frame in groups
    ]

    if workers == 1 or len(tasks) == 1:
        matched_groups = [_match_training_group(task) for task in tasks]
    else:
        max_workers = workers
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            matched_groups = list(executor.map(_match_training_group, tasks))

    matched = pd.concat(matched_groups, ignore_index=True).sort_values("_row_order").reset_index(drop=True)
    if len(matched) != len(base):
        raise ValueError(f"row drift after v49 join: got {len(matched)}, expected {len(base)}")

    feature_columns = SELECTED_17 + AA20 + AROMATIC_ALIPHATIC_12
    validate_training(matched, feature_columns)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matched.to_parquet(output_path, index=False)

    exact_aa = int((matched["match_aa_l1"] == 0).sum())
    return {
        "rows": len(matched),
        "structures": len(groups),
        "exact_aa_matches": exact_aa,
        "exact_aa_rate": exact_aa / len(matched),
        "mean_descriptor_cost": float(matched["match_desc_cost"].mean()),
        "max_descriptor_cost": float(matched["match_desc_cost"].max()),
        "output_path": output_path,
        "output_size_mb": output_path.stat().st_size / (1024 * 1024),
    }


def _match_training_group(task: dict[str, object]) -> pd.DataFrame:
    pdb_ligand = str(task["pdb_ligand"])
    fpocket_root = Path(str(task["fpocket_root"]))
    training_rows = pd.DataFrame(task["rows"])

    candidates = _build_structure_candidates(fpocket_root, pdb_ligand)
    if len(candidates) < len(training_rows):
        raise ValueError(
            f"{pdb_ligand}: only {len(candidates)} candidate pockets for "
            f"{len(training_rows)} training rows"
        )

    aa_cost = _aa_cost_matrix(training_rows, candidates)
    descriptor_cost = _descriptor_cost_matrix(training_rows, candidates)
    total_cost = aa_cost * 1000.0 + descriptor_cost

    row_idx, col_idx = linear_sum_assignment(total_cost)
    if len(row_idx) != len(training_rows):
        raise ValueError(
            f"{pdb_ligand}: assignment covered {len(row_idx)} rows, expected {len(training_rows)}"
        )

    selected = candidates.iloc[col_idx].reset_index(drop=True)
    out = training_rows.iloc[row_idx].reset_index(drop=True).copy()
    out["matched_pocket_number"] = selected["pocket_number"].to_numpy(dtype=np.int64)
    out["match_aa_l1"] = aa_cost[row_idx, col_idx]
    out["match_desc_cost"] = descriptor_cost[row_idx, col_idx]
    for column in AROMATIC_ALIPHATIC_12:
        out[column] = selected[column].to_numpy()
    return out


def _build_structure_candidates(fpocket_root: Path, pdb_ligand: str) -> pd.DataFrame:
    structure_dir = fpocket_root / pdb_ligand.replace(".pdb", "_out")
    info_path = structure_dir / f"{Path(pdb_ligand).stem}_info.txt"
    info_df = _parse_info_file(info_path)

    rows: list[dict[str, object]] = []
    for pocket_number in info_df["pocket_number"].tolist():
        atm_path = structure_dir / "pockets" / f"pocket{pocket_number}_atm.pdb"
        vert_path = structure_dir / "pockets" / f"pocket{pocket_number}_vert.pqr"
        if not atm_path.exists() or not vert_path.exists():
            continue
        rows.append(
            {
                "pocket_number": pocket_number,
                **_extract_pocket_aa_and_shell_features(atm_path, vert_path),
            }
        )

    features_df = pd.DataFrame(rows)
    if features_df.empty:
        raise ValueError(f"{pdb_ligand}: no candidate pockets with atm/vert pairs")
    return info_df.merge(features_df, on="pocket_number", how="inner", validate="one_to_one")


def _parse_info_file(info_path: Path) -> pd.DataFrame:
    if not info_path.exists():
        raise FileNotFoundError(f"missing fpocket info file: {info_path}")

    rows: list[dict[str, object]] = []
    current: dict[str, object] | None = None

    with info_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            match = POCKET_HEADER.match(line)
            if match is not None:
                if current is not None:
                    rows.append(current)
                current = {"pocket_number": int(match.group("index"))}
                continue
            if current is None or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().replace("\t", "")
            mapped = INFO_FIELD_MAP.get(key)
            if mapped is None:
                continue
            try:
                current[mapped] = float(value.strip().replace("\t", ""))
            except ValueError:
                continue

    if current is not None:
        rows.append(current)

    frame = pd.DataFrame(rows)
    missing_columns = ["pocket_number", *SELECTED_17]
    missing = [column for column in missing_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{info_path}: missing descriptor columns {missing}")
    return frame[missing_columns].copy()


def _extract_pocket_aa_and_shell_features(atm_path: Path, vert_path: Path) -> dict[str, object]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(atm_path.stem, atm_path)
    centroid = _compute_centroid(vert_path)

    aa_counts = {column: 0 for column in AA20}
    aromatic_counts = [0, 0, 0, 0]
    aliphatic_counts = [0, 0, 0, 0]

    for residue in structure.get_residues():
        residue_name = residue.get_resname().strip().upper()
        if residue_name in aa_counts:
            aa_counts[residue_name] += 1

        residue_group = _classify_residue_name(residue_name)
        if residue_group is None:
            continue
        distance = _closest_heavy_atom_distance(residue, centroid)
        if distance is None:
            continue
        shell_index = _shell_index(distance)
        if shell_index is None:
            continue
        if residue_group == "aromatic":
            aromatic_counts[shell_index] += 1
        else:
            aliphatic_counts[shell_index] += 1

    ratios = [
        float(aromatic / (aliphatic + 1.0))
        for aromatic, aliphatic in zip(aromatic_counts, aliphatic_counts, strict=True)
    ]
    return {
        **aa_counts,
        "aromatic_count_shell1": aromatic_counts[0],
        "aromatic_count_shell2": aromatic_counts[1],
        "aromatic_count_shell3": aromatic_counts[2],
        "aromatic_count_shell4": aromatic_counts[3],
        "aliphatic_count_shell1": aliphatic_counts[0],
        "aliphatic_count_shell2": aliphatic_counts[1],
        "aliphatic_count_shell3": aliphatic_counts[2],
        "aliphatic_count_shell4": aliphatic_counts[3],
        "aromatic_aliphatic_ratio_shell1": ratios[0],
        "aromatic_aliphatic_ratio_shell2": ratios[1],
        "aromatic_aliphatic_ratio_shell3": ratios[2],
        "aromatic_aliphatic_ratio_shell4": ratios[3],
    }


def _aa_cost_matrix(training_rows: pd.DataFrame, candidates: pd.DataFrame) -> np.ndarray:
    train_aa = training_rows[AA20].to_numpy(dtype=np.float64)
    cand_aa = candidates[AA20].to_numpy(dtype=np.float64)
    return np.abs(train_aa[:, None, :] - cand_aa[None, :, :]).sum(axis=2)


def _descriptor_cost_matrix(training_rows: pd.DataFrame, candidates: pd.DataFrame) -> np.ndarray:
    train_desc = training_rows[SELECTED_17].to_numpy(dtype=np.float64)
    cand_desc = candidates[SELECTED_17].to_numpy(dtype=np.float64)
    scale = cand_desc.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return (np.abs(train_desc[:, None, :] - cand_desc[None, :, :]) / scale).sum(axis=2)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-csv",
        type=Path,
        default=Path("reference/SLiPP_2024-main/training_pockets.csv"),
        help="Path to the authors' training_pockets.csv.",
    )
    parser.add_argument(
        "--fpocket-root",
        type=Path,
        required=True,
        help="Root containing class-scoped fpocket outputs, e.g. ADN/pdb1BX4_out.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination parquet path.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes. Defaults to ProcessPoolExecutor default.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    try:
        result = build_training_v49_parquet(
            training_csv=args.training_csv,
            fpocket_root=args.fpocket_root,
            output_path=args.output,
            workers=args.workers,
        )
    except Exception as exc:
        logging.error("v49 build failed: %s", exc)
        return 1

    print(f"Rows: {result['rows']}")
    print(f"Structures: {result['structures']}")
    print(f"Exact AA matches: {result['exact_aa_matches']} ({result['exact_aa_rate']:.2%})")
    print(f"Mean descriptor cost: {result['mean_descriptor_cost']:.3f}")
    print(f"Output: {result['output_path']} ({result['output_size_mb']:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
