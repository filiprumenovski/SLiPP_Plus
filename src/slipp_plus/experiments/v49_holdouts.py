"""Build v49 holdout parquets by downloading structures and running fpocket."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import AA20, AROMATIC_ALIPHATIC_12, SELECTED_17
from ..ingest import _read_holdout_xlsx
from ..schemas import validate_holdout
from ..feature_builders.v49 import (
    _descriptor_cost_matrix,
    _extract_pocket_aa_and_shell_features,
    _parse_info_file,
)

RCSB_PDB_URL = "https://files.rcsb.org/download/{structure_id}.pdb"
ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"


def build_v49_holdouts(
    apo_xlsx: Path,
    af_xlsx: Path,
    output_dir: Path,
    fpocket_bin: Path,
    jobs: int = 4,
) -> dict[str, object]:
    """Download, pocket, enrich, and write the two v49 holdout parquets."""

    output_dir.mkdir(parents=True, exist_ok=True)
    structures_root = output_dir / "structures"
    apo_root = structures_root / "apo_pdb_holdout"
    af_root = structures_root / "alphafold_holdout"
    apo_root.mkdir(parents=True, exist_ok=True)
    af_root.mkdir(parents=True, exist_ok=True)

    apo_df = _read_holdout_xlsx(apo_xlsx, id_col="PDB_ID")
    af_df = _read_holdout_xlsx(af_xlsx, id_col="UniProt ID code")

    _download_apo_structures(apo_df["structure_id"].tolist(), apo_root, jobs=jobs)
    _download_alphafold_structures(af_df["structure_id"].tolist(), af_root, jobs=jobs)

    _run_fpocket_dir(apo_root, fpocket_bin=fpocket_bin, jobs=jobs)
    _run_fpocket_dir(af_root, fpocket_bin=fpocket_bin, jobs=jobs)

    apo_out = _attach_v49_holdout_features(apo_df, apo_root)
    af_out = _attach_v49_holdout_features(af_df, af_root)

    feature_columns = SELECTED_17 + AA20 + AROMATIC_ALIPHATIC_12
    validate_holdout(apo_out, feature_columns)
    validate_holdout(af_out, feature_columns)

    apo_path = output_dir / "apo_pdb_holdout.parquet"
    af_path = output_dir / "alphafold_holdout.parquet"
    apo_out.to_parquet(apo_path, index=False)
    af_out.to_parquet(af_path, index=False)

    return {
        "apo_rows": len(apo_out),
        "af_rows": len(af_out),
        "apo_output": apo_path,
        "af_output": af_path,
        "apo_match_cost_mean": float(apo_out["match_desc_cost"].mean()),
        "af_match_cost_mean": float(af_out["match_desc_cost"].mean()),
    }


def _download_apo_structures(structure_ids: list[str], root: Path, jobs: int) -> None:
    tasks = [
        (RCSB_PDB_URL.format(structure_id=structure_id), root / f"{structure_id}.pdb")
        for structure_id in sorted(set(structure_ids))
    ]
    _download_many(tasks, jobs=jobs)


def _download_alphafold_structures(accessions: list[str], root: Path, jobs: int) -> None:
    unique = sorted(set(accessions))
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        urls = list(executor.map(_resolve_alphafold_pdb_url, unique))
    tasks = [(url, root / f"{accession}.pdb") for accession, url in zip(unique, urls, strict=True)]
    _download_many(tasks, jobs=jobs)


def _resolve_alphafold_pdb_url(accession: str) -> str:
    with urllib.request.urlopen(ALPHAFOLD_API_URL.format(accession=accession)) as response:
        payload = json.load(response)
    if not payload:
        raise ValueError(f"{accession}: AlphaFold API returned no entries")
    pdb_url = payload[0].get("pdbUrl")
    if not pdb_url:
        raise ValueError(f"{accession}: AlphaFold API returned no pdbUrl")
    return str(pdb_url)


def _download_many(tasks: list[tuple[str, Path]], jobs: int) -> None:
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        list(executor.map(_download_file_if_missing, tasks))


def _download_file_if_missing(task: tuple[str, Path]) -> None:
    url, destination = task
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)


def _run_fpocket_dir(root: Path, fpocket_bin: Path, jobs: int) -> None:
    pdb_paths = sorted(root.glob("*.pdb"))
    if not pdb_paths:
        raise ValueError(f"no pdb files found under {root}")
    tasks = [(fpocket_bin, pdb_path) for pdb_path in pdb_paths]
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        list(executor.map(_run_fpocket_one, tasks))


def _run_fpocket_one(task: tuple[Path, Path]) -> None:
    fpocket_bin, pdb_path = task
    out_dir = pdb_path.with_name(f"{pdb_path.stem}_out")
    if (out_dir / "pockets").exists():
        return
    subprocess.run(
        [str(fpocket_bin), "-f", pdb_path.name],
        cwd=pdb_path.parent,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _attach_v49_holdout_features(base: pd.DataFrame, structure_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in base.to_dict(orient="records"):
        structure_id = str(row["structure_id"])
        candidates = _build_holdout_candidates(structure_root, structure_id)
        row_df = pd.DataFrame([row])
        desc_cost = _descriptor_cost_matrix(row_df, candidates)
        best_index = int(np.argmin(desc_cost[0]))
        best = candidates.iloc[best_index]
        enriched = dict(row)
        enriched["matched_pocket_number"] = int(best["pocket_number"])
        enriched["match_desc_cost"] = float(desc_cost[0, best_index])
        for column in AA20:
            enriched[column] = int(best[column])
        for column in AROMATIC_ALIPHATIC_12:
            enriched[column] = float(best[column]) if "ratio" in column else int(best[column])
        rows.append(enriched)
    return pd.DataFrame(rows)


def _build_holdout_candidates(structure_root: Path, structure_id: str) -> pd.DataFrame:
    structure_dir = structure_root / f"{structure_id}_out"
    info_df = _parse_info_file(structure_dir / f"{structure_id}_info.txt")
    rows: list[dict[str, object]] = []
    for pocket_number in info_df["pocket_number"].tolist():
        atm_path = structure_dir / "pockets" / f"pocket{pocket_number}_atm.pdb"
        vert_path = structure_dir / "pockets" / f"pocket{pocket_number}_vert.pqr"
        if not atm_path.exists() or not vert_path.exists():
            continue
        row = {"pocket_number": pocket_number}
        row.update(_extract_pocket_aa_and_shell_features(atm_path, vert_path))
        rows.append(row)
    features_df = pd.DataFrame(rows)
    if features_df.empty:
        raise ValueError(f"{structure_id}: no candidate pockets with atm/vert pairs")
    return info_df.merge(features_df, on="pocket_number", how="inner", validate="one_to_one")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apo-xlsx",
        type=Path,
        default=Path("data/raw/supplementary/ci5c01076_si_003.xlsx"),
        help="Supporting File 2 path.",
    )
    parser.add_argument(
        "--af-xlsx",
        type=Path,
        default=Path("data/raw/supplementary/ci5c01076_si_004.xlsx"),
        help="Supporting File 3 path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Processed output directory, usually processed/v49.",
    )
    parser.add_argument(
        "--fpocket-bin",
        type=Path,
        default=Path("/tmp/fpocket-codex/bin/fpocket"),
        help="Path to the fpocket binary.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Concurrent download/fpocket jobs.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    try:
        result = build_v49_holdouts(
            apo_xlsx=args.apo_xlsx,
            af_xlsx=args.af_xlsx,
            output_dir=args.output_dir,
            fpocket_bin=args.fpocket_bin,
            jobs=args.jobs,
        )
    except Exception as exc:
        logging.error("v49 holdout build failed: %s", exc)
        return 1

    print(f"apo_rows: {result['apo_rows']}")
    print(f"af_rows: {result['af_rows']}")
    print(f"apo_match_cost_mean: {result['apo_match_cost_mean']:.3f}")
    print(f"af_match_cost_mean: {result['af_match_cost_mean']:.3f}")
    print(f"apo_output: {result['apo_output']}")
    print(f"af_output: {result['af_output']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
