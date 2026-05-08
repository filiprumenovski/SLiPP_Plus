"""Run dpocket over ligand-bound PDB structures. Day 7+ from-scratch path.

Day 1 does not use this module. The authors' repo ships
``reference/SLiPP_2024-main/training_pockets.csv`` with all 17 descriptors
pre-extracted for the 1,786 training structures, and the SF2/SF3 xlsx holdouts
carry descriptors too. ``src/slipp_plus/ingest.py`` reads these directly.

Implementation sketch for Day 7+:

1. For each ligand-bound PDB, call ``biobb_vs.fpocket.fpocket_run`` or invoke
   ``dpocket`` via subprocess.
2. Parse the ``dpocket_fpocketnpocket.txt`` output, one row per detected pocket.
3. Rename columns using ``reference/SLiPP_2024-main/slipp.py:NAME_CONVERSION``:
   ``volume`` -> ``pock_vol``, ``alpha_sphere_density`` -> ``as_density``, etc.
4. Classify each pocket as ``ligand_pocket`` if its COM is within 8 A of the
   ligand COM AND >= 10 protein residues lie within 8 A of the ligand; else
   ``pseudo_pocket``. (Matches paper Methods p.17.)
5. Write per-structure parquets to ``data/pockets/{pdb_id}_{ligand_code}.parquet``.

Install deps via ``uv sync --extra scratch`` + system fpocket >= 4.0.
"""

from __future__ import annotations

from pathlib import Path


def extract_pockets(structures_dir: Path, output_dir: Path) -> None:
    """Placeholder for the Day 7+ dpocket/fpocket extraction path.

    Parameters
    ----------
    structures_dir
        Directory that will contain raw ligand-bound PDB structures.
    output_dir
        Destination for extracted pocket descriptor parquets.

    Raises
    ------
    NotImplementedError
        Always raised because the current reproducible path consumes the
        authors' pre-extracted descriptor tables.
    """

    raise NotImplementedError(
        "Day 7+ from-scratch reproduction. Day 1 uses the pre-extracted "
        "descriptor tables from the authors' supplementary materials."
    )
