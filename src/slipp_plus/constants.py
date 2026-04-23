"""Canonical vocabularies and feature orderings for SLiPP++.

All constants here are the single source of truth. Column order and class-code
spelling come from `reference/SLiPP_2024-main/slipp.py:SELECTED_PARAM` and
the paper's Methods section.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
# Paper canonical 17 descriptors (exact order from slipp.py:SELECTED_PARAM).
# Note: paper Methods paragraph lists 16 by name but the authors' code uses 17
# (mean_as_ray is the 17th; the text omits it). We follow the code.
SELECTED_17: list[str] = [
    "pock_vol",
    "nb_AS",
    "mean_as_ray",
    "mean_as_solv_acc",
    "apol_as_prop",
    "mean_loc_hyd_dens",
    "hydrophobicity_score",
    "volume_score",
    "polarity_score",
    "charge_score",
    "flex",
    "prop_polar_atm",
    "as_density",
    "as_max_dst",
    "surf_pol_vdw14",
    "surf_apol_vdw14",
    "surf_vdw14",
]

# Free extras already sitting in the CSV. Flagged via configs/day1.yaml:feature_set.
EXTRA_VDW22: list[str] = [
    "surf_pol_vdw22",
    "surf_apol_vdw22",
]

AA20: list[str] = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]

FEATURE_SETS: dict[str, list[str]] = {
    "v14": SELECTED_17,
    "v14+v22": SELECTED_17 + EXTRA_VDW22,
    "v14+aa": SELECTED_17 + AA20,
    "v14+v22+aa": SELECTED_17 + EXTRA_VDW22 + AA20,
}

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
LIPID_CODES: frozenset[str] = frozenset({"CLR", "MYR", "PLM", "STE", "OLA"})
NONLIPID_CODES: frozenset[str] = frozenset({"ADN", "B12", "BGC", "COA"})
PP_CODE: str = "PP"

# Sorted so downstream arrays/plots have deterministic class ordering.
CLASS_10: list[str] = sorted(LIPID_CODES | NONLIPID_CODES | {PP_CODE})

# Mapping from the reference CSV's `lig` column values to our 10-class labels.
# The CSV uses lig="none" for pseudo-pockets.
LIG_TO_CLASS: dict[str, str] = {code: code for code in (LIPID_CODES | NONLIPID_CODES)}
LIG_TO_CLASS["none"] = PP_CODE


def is_lipid(class_code: str) -> bool:
    """Binary collapse: True iff the 10-class code is one of the 5 lipids."""
    return class_code in LIPID_CODES
