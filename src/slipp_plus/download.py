"""PDB + AlphaFold model download. Day 7+ from-scratch path.

Day 1 does not use this module; the paper's curated descriptor tables ship in
the repo and ``src/slipp_plus/ingest.py`` consumes them directly.

Implementation sketch for Day 7+:

1. Parse SF1 xlsx (``ci5c01076_si_002.xlsx``) into a DataFrame of
   ``(pdb_id, ligand_code, chain_id)`` rows.
2. For each PDB id, call ``Bio.PDB.PDBList().retrieve_pdb_file(..., file_format='pdb')``.
   Retry 3x on network failure; log to ``reports/download_log.csv``.
3. For AlphaFold (SF3), fetch ``https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v4.pdb``.
4. Emit structures to ``data/structures/`` and ``data/alphafold/`` respectively.

Install deps via ``uv sync --extra scratch`` before invoking.
"""

from __future__ import annotations

from pathlib import Path


def download_all(output_dir: Path) -> None:
    """Placeholder for the Day 7+ raw-structure download workflow.

    Parameters
    ----------
    output_dir:
        Intended destination root for downloaded PDB and AlphaFold structures.
        The parameter is accepted now so the CLI contract is stable before the
        scratch workflow is implemented.

    Returns
    -------
    None
        This function does not currently complete successfully.

    Raises
    ------
    NotImplementedError
        Always raised in the Day 1 release because SLiPP++ consumes the
        pre-curated descriptor tables shipped with the repository.
    """
    raise NotImplementedError(
        "Day 7+ from-scratch reproduction. Day 1 uses the pre-curated tables "
        "shipped in reference/SLiPP_2024-main/ and data/raw/supplementary/."
    )
