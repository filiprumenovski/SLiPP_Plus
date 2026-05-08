"""Chemistry-informed derived features from existing pocket descriptors.

Computes ~26 biophysically motivated features from the base v_sterol columns:
AA composition fractions, size-normalized densities, surface chemistry ratios,
spatial gradients, and shape-chemistry interaction terms.

All features are computable from columns already present in full_pockets.parquet
with no new fpocket runs or external data required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------

AA20: list[str] = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]

# Biophysical AA groupings
_HYDROPHOBIC_AA: list[str] = ["LEU", "ILE", "VAL", "ALA", "PHE", "MET", "TRP"]
_CHARGED_AA: list[str] = ["ARG", "LYS", "ASP", "GLU", "HIS"]
_AROMATIC_AA: list[str] = ["PHE", "TYR", "TRP", "HIS"]
_POLAR_UNCHARGED_AA: list[str] = ["SER", "THR", "ASN", "GLN", "TYR", "CYS"]
_SMALL_AA: list[str] = ["GLY", "ALA", "SER", "CYS", "PRO"]
_BRANCHED_ALIPHATIC_AA: list[str] = ["ILE", "LEU", "VAL"]

DERIVED_FEATURE_COLS: list[str] = [
    # --- AA composition fractions (6) ---
    "hydrophobic_aa_frac",
    "charged_aa_frac",
    "aromatic_aa_frac",
    "polar_uncharged_aa_frac",
    "small_aa_frac",
    "branched_aliphatic_aa_frac",
    # --- Size-normalized densities (6) ---
    "vol_per_as",
    "aromatic_density",
    "aliphatic_density",
    "cation_density",
    "anion_density",
    "residue_density",
    # --- Surface chemistry ratios (3) ---
    "polar_surface_frac",
    "surf_pol_vdw22_frac",
    "net_charge_per_vol",
    # --- Spatial gradients (4) ---
    "bulky_hydro_gradient",
    "polar_gradient",
    "aromatic_gradient",
    "charge_gradient",
    # --- Shape-chemistry interactions (5) ---
    "shape_anisotropy",
    "elongation_x_hydro",
    "planarity_x_aromatic_density",
    "burial_x_polar_surface",
    "vol_x_polar_surface",
    # --- Charge asymmetry (2) ---
    "charge_balance",
    "charge_density_per_vol",
]


def _safe_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Element-wise ratio with zero-safe denominator."""
    denom = np.maximum(np.abs(denominator), eps)
    return numerator / denom


def _gradient(
    inner: np.ndarray,
    outer: np.ndarray,
) -> np.ndarray:
    """Signed gradient: (inner - outer) / (inner + outer + 1)."""
    return (inner - outer) / (inner + outer + 1.0)


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ~26 derived features to a pocket DataFrame in-place.

    Expects all v_sterol columns to be present. Returns the DataFrame
    with new columns appended (does not drop existing columns).
    """
    df = df.copy()

    total_aa = df[AA20].sum(axis=1).clip(lower=1).astype(np.float64)

    # --- AA composition fractions ---
    df["hydrophobic_aa_frac"] = df[_HYDROPHOBIC_AA].sum(axis=1) / total_aa
    df["charged_aa_frac"] = df[_CHARGED_AA].sum(axis=1) / total_aa
    df["aromatic_aa_frac"] = df[_AROMATIC_AA].sum(axis=1) / total_aa
    df["polar_uncharged_aa_frac"] = df[_POLAR_UNCHARGED_AA].sum(axis=1) / total_aa
    df["small_aa_frac"] = df[_SMALL_AA].sum(axis=1) / total_aa
    df["branched_aliphatic_aa_frac"] = df[_BRANCHED_ALIPHATIC_AA].sum(axis=1) / total_aa

    # --- Size-normalized densities ---
    nb_as = df["nb_AS"].clip(lower=1).astype(np.float64)
    df["vol_per_as"] = df["pock_vol"].values / nb_as.values

    aro_total = sum(df[f"aromatic_count_shell{s}"].values for s in [1, 2, 3, 4])
    ali_total = sum(df[f"aliphatic_count_shell{s}"].values for s in [1, 2, 3, 4])
    cat_total = sum(df[f"cationic_count_shell{s}"].values for s in [1, 2, 3, 4])
    ani_total = sum(df[f"anionic_count_shell{s}"].values for s in [1, 2, 3, 4])

    df["aromatic_density"] = aro_total / nb_as.values
    df["aliphatic_density"] = ali_total / nb_as.values
    df["cation_density"] = cat_total / nb_as.values
    df["anion_density"] = ani_total / nb_as.values
    df["residue_density"] = total_aa.values / nb_as.values

    # --- Surface chemistry ratios ---
    df["polar_surface_frac"] = _safe_ratio(
        df["surf_pol_vdw14"].values,
        df["surf_vdw14"].values,
    )
    # vdw22 surfaces may be absent in holdout data
    if "surf_pol_vdw22" in df.columns and "surf_apol_vdw22" in df.columns:
        surf22_total = (
            df["surf_pol_vdw22"].values + df["surf_apol_vdw22"].values
        )
        df["surf_pol_vdw22_frac"] = _safe_ratio(
            df["surf_pol_vdw22"].values,
            surf22_total,
        )
    else:
        df["surf_pol_vdw22_frac"] = df["polar_surface_frac"].values  # fallback
    df["net_charge_per_vol"] = df["charge_score"].values / df["pock_vol"].clip(lower=1.0).values

    # --- Spatial gradients ---
    # Inner = shell 1+2, outer = shell 3+4
    bulky_inner = (
        df["bulky_hydrophobic_count_shell1"].values
        + df["bulky_hydrophobic_count_shell2"].values
    ).astype(np.float64)
    bulky_outer = (
        df["bulky_hydrophobic_count_shell3"].values
        + df["bulky_hydrophobic_count_shell4"].values
    ).astype(np.float64)
    df["bulky_hydro_gradient"] = _gradient(bulky_inner, bulky_outer)

    polar_inner = (
        df["polar_neutral_count_shell1"].values
        + df["polar_neutral_count_shell2"].values
    ).astype(np.float64)
    polar_outer = (
        df["polar_neutral_count_shell3"].values
        + df["polar_neutral_count_shell4"].values
    ).astype(np.float64)
    df["polar_gradient"] = _gradient(polar_inner, polar_outer)

    aro_inner = (
        df["aromatic_count_shell1"].values
        + df["aromatic_count_shell2"].values
    ).astype(np.float64)
    aro_outer = (
        df["aromatic_count_shell3"].values
        + df["aromatic_count_shell4"].values
    ).astype(np.float64)
    df["aromatic_gradient"] = _gradient(aro_inner, aro_outer)

    charge_inner = (
        df["cationic_count_shell1"].values
        + df["cationic_count_shell2"].values
        + df["anionic_count_shell1"].values
        + df["anionic_count_shell2"].values
    ).astype(np.float64)
    charge_outer = (
        df["cationic_count_shell3"].values
        + df["cationic_count_shell4"].values
        + df["anionic_count_shell3"].values
        + df["anionic_count_shell4"].values
    ).astype(np.float64)
    df["charge_gradient"] = _gradient(charge_inner, charge_outer)

    # --- Shape-chemistry interactions ---
    df["shape_anisotropy"] = _safe_ratio(
        df["pocket_lam2"].values,
        df["pocket_lam1"].values,
    )
    df["elongation_x_hydro"] = (
        df["pocket_elongation"].values * df["hydrophobicity_score"].values
    )
    df["planarity_x_aromatic_density"] = (
        df["pocket_planarity"].values * df["aromatic_density"].values
    )
    df["burial_x_polar_surface"] = (
        df["pocket_burial"].values * df["polar_surface_frac"].values
    )
    df["vol_x_polar_surface"] = (
        df["pock_vol"].values * df["polar_surface_frac"].values
    )

    # --- Charge asymmetry ---
    df["charge_balance"] = _gradient(
        cat_total.astype(np.float64),
        ani_total.astype(np.float64),
    )
    df["charge_density_per_vol"] = (
        (cat_total + ani_total).astype(np.float64)
        / df["pock_vol"].clip(lower=1.0).values
    )

    # Validate no NaN introduced
    for col in DERIVED_FEATURE_COLS:
        if df[col].isna().any():
            raise ValueError(f"NaN values in derived feature column: {col}")

    return df
