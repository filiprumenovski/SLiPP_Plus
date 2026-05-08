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

# Stripped down fpocket features, removing noisy heuristics and redundant proxies.
SELECTED_8_LEAN: list[str] = [
    "pock_vol",
    "nb_AS",
    "mean_as_solv_acc",
    "prop_polar_atm",
    "as_density",
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
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]

# Signal-ranked AA subsets from the 105-feature audit, sorted by seed-0
# permutation lipid5 drop. These are locked before encoder retraining to avoid
# post-hoc feature picking.
AA_SIGNAL_12: list[str] = [
    "LEU",
    "HIS",
    "TYR",
    "MET",
    "ARG",
    "ALA",
    "GLY",
    "VAL",
    "ASP",
    "GLU",
    "CYS",
    "ILE",
]

AA_SIGNAL_16: list[str] = [
    *AA_SIGNAL_12,
    "THR",
    "TRP",
    "LYS",
    "PHE",
]

AROMATIC_ALIPHATIC_12: list[str] = [
    "aromatic_count_shell1",
    "aromatic_count_shell2",
    "aromatic_count_shell3",
    "aromatic_count_shell4",
    "aliphatic_count_shell1",
    "aliphatic_count_shell2",
    "aliphatic_count_shell3",
    "aliphatic_count_shell4",
    "aromatic_aliphatic_ratio_shell1",
    "aromatic_aliphatic_ratio_shell2",
    "aromatic_aliphatic_ratio_shell3",
    "aromatic_aliphatic_ratio_shell4",
]

SHELL_SIGNAL_6: list[str] = [
    "aliphatic_count_shell3",
    "aromatic_count_shell4",
    "aliphatic_count_shell2",
    "aliphatic_count_shell4",
    "aromatic_aliphatic_ratio_shell4",
    "aromatic_aliphatic_ratio_shell3",
]

AROMATIC_ALIPHATIC_RATIOS_4: list[str] = [
    "aromatic_aliphatic_ratio_shell1",
    "aromatic_aliphatic_ratio_shell2",
    "aromatic_aliphatic_ratio_shell3",
    "aromatic_aliphatic_ratio_shell4",
]

AROMATIC_ALIPHATIC_NORMALIZED_12: list[str] = [
    "target_residue_count_shell1",
    "target_residue_count_shell2",
    "target_residue_count_shell3",
    "target_residue_count_shell4",
    "aromatic_fraction_shell1",
    "aromatic_fraction_shell2",
    "aromatic_fraction_shell3",
    "aromatic_fraction_shell4",
    "aliphatic_fraction_shell1",
    "aliphatic_fraction_shell2",
    "aliphatic_fraction_shell3",
    "aliphatic_fraction_shell4",
]

# v_sterol — chemistry-refined residue-shell counts (7 groups x 4 shells) +
# (aromatic_polar + polar_neutral) / (bulky_hydrophobic + aromatic_pi + 1)
# per-shell ratios. 7*4 + 4 = 32 columns. Authoritative source for ordering
# lives in ``sterol_features.STEROL_CHEMISTRY_SHELL_COLS``; keep in lockstep.
STEROL_CHEMISTRY_SHELL_COLS: list[str] = [
    "aromatic_pi_count_shell1",
    "aromatic_pi_count_shell2",
    "aromatic_pi_count_shell3",
    "aromatic_pi_count_shell4",
    "aromatic_polar_count_shell1",
    "aromatic_polar_count_shell2",
    "aromatic_polar_count_shell3",
    "aromatic_polar_count_shell4",
    "bulky_hydrophobic_count_shell1",
    "bulky_hydrophobic_count_shell2",
    "bulky_hydrophobic_count_shell3",
    "bulky_hydrophobic_count_shell4",
    "small_special_count_shell1",
    "small_special_count_shell2",
    "small_special_count_shell3",
    "small_special_count_shell4",
    "polar_neutral_count_shell1",
    "polar_neutral_count_shell2",
    "polar_neutral_count_shell3",
    "polar_neutral_count_shell4",
    "cationic_count_shell1",
    "cationic_count_shell2",
    "cationic_count_shell3",
    "cationic_count_shell4",
    "anionic_count_shell1",
    "anionic_count_shell2",
    "anionic_count_shell3",
    "anionic_count_shell4",
    "polar_hydrophobic_ratio_shell1",
    "polar_hydrophobic_ratio_shell2",
    "polar_hydrophobic_ratio_shell3",
    "polar_hydrophobic_ratio_shell4",
]

# v_sterol — pocket geometry: alpha-sphere PCA eigenvalues, elongation,
# planarity, plus burial relative to protein C-alpha centroid.
POCKET_GEOMETRY_COLS: list[str] = [
    "pocket_lam1",
    "pocket_lam2",
    "pocket_lam3",
    "pocket_elongation",
    "pocket_planarity",
    "pocket_burial",
]

# v_plm_ste — 16 extra features specifically designed to disambiguate
# palmitate from steryl ester. Ordering is authoritative.
# Group A (4): CRAC/CARC sequence motifs contacting the pocket.
# Group B (7): axial profile features from alpha-sphere PCA.
# Group C (5): polar-anchor chemistry at the polar end of the axial profile.
# Keep in lockstep with ``plm_ste_features.PALMITATE_VS_STERYL_EXTRA_16``.
PALMITATE_VS_STERYL_EXTRA_16: list[str] = [
    "crac_count",
    "carc_count",
    "any_sterol_motif",
    "motif_residue_density",
    "axial_length",
    "axial_radius_std",
    "axial_radius_gradient",
    "fatend_ratio",
    "bottleneck_position",
    "thick_end_asymmetry",
    "cross_section_aspect",
    "polar_end_cationic_count",
    "polar_end_aromatic_polar_count",
    "polar_end_neutral_polar_count",
    "anchor_charge_balance",
    "anchor_chemistry_entropy",
]

# Backward-compatible alias kept to avoid broader churn in the current slice.
PLM_STE_EXTRA_16 = PALMITATE_VS_STERYL_EXTRA_16

# v_lipid_boundary - 22 features for the observed lipid subclass boundaries:
# acyl-chain geometry, axial-bin sleeve chemistry, polar anchor chemistry, and
# COA-like phosphate-anchor leakage proxies.
LIPID_BOUNDARY_FEATURES_22: list[str] = [
    "lb_axis_length",
    "lb_radius_mean",
    "lb_radius_std",
    "lb_radius_range",
    "lb_endpoint_radius_ratio",
    "lb_center_radius_ratio",
    "lb_linearity",
    "lb_planar_spread",
    "lb_tube_hydrophobe_fraction",
    "lb_tube_gly_fraction",
    "lb_tube_aromatic_fraction",
    "lb_tube_beta_branched_fraction",
    "lb_nonpolar_end_hydrophobe_count",
    "lb_polar_end_donor_count",
    "lb_polar_end_acceptor_count",
    "lb_polar_end_charged_count",
    "lb_polar_end_aromatic_count",
    "lb_anchor_charge_balance",
    "lb_p_loop_like_motif_count",
    "lb_cationic_anchor_density",
    "lb_phosphate_anchor_score",
    "lb_gly_rich_anchor_fraction",
]

# v_tunnel - CAVER-derived tunnel geometry and tunnel-lining chemistry.
TUNNEL_FEATURES_15: list[str] = [
    "tunnel_count",
    "tunnel_primary_length",
    "tunnel_primary_bottleneck_radius",
    "tunnel_primary_avg_radius",
    "tunnel_primary_curvature",
    "tunnel_primary_throughput",
    "tunnel_primary_hydrophobicity",
    "tunnel_primary_charge",
    "tunnel_primary_aromatic_fraction",
    "tunnel_max_length",
    "tunnel_total_length",
    "tunnel_min_bottleneck",
    "tunnel_branching_factor",
    "tunnel_length_over_axial",
    "tunnel_extends_beyond_pocket",
]

TUNNEL_MISSINGNESS_3: list[str] = [
    "tunnel_pocket_context_present",
    "tunnel_caver_profile_present",
    "tunnel_has_tunnel",
]

TUNNEL_FEATURES_18: list[str] = TUNNEL_FEATURES_15 + TUNNEL_MISSINGNESS_3

# Compact tunnel slices for paper-facing ablations. These were selected from
# aligned v_tunnel LGBM/XGB screens to avoid shipping the full redundant tunnel
# block as the headline model.
TUNNEL_SHAPE_AVAIL_6: list[str] = [
    "tunnel_has_tunnel",
    "tunnel_caver_profile_present",
    "tunnel_count",
    "tunnel_primary_bottleneck_radius",
    "tunnel_length_over_axial",
    "tunnel_extends_beyond_pocket",
]

TUNNEL_SHAPE_SIGNAL_3: list[str] = [
    "tunnel_caver_profile_present",
    "tunnel_count",
    "tunnel_primary_bottleneck_radius",
]

TUNNEL_CHEM_5: list[str] = [
    "tunnel_has_tunnel",
    "tunnel_caver_profile_present",
    "tunnel_primary_hydrophobicity",
    "tunnel_primary_charge",
    "tunnel_primary_aromatic_fraction",
]

TUNNEL_GEOM_9: list[str] = [
    "tunnel_has_tunnel",
    "tunnel_caver_profile_present",
    "tunnel_count",
    "tunnel_primary_length",
    "tunnel_primary_bottleneck_radius",
    "tunnel_primary_curvature",
    "tunnel_total_length",
    "tunnel_min_bottleneck",
    "tunnel_length_over_axial",
]

# v_graph_tunnel - cheap alpha-sphere graph tunnel-depth proxy.
GRAPH_TUNNEL_FEATURES_3: list[str] = [
    "tunnel_length",
    "tunnel_bottleneck_radius",
    "tunnel_length_over_axial_length",
]

# v_caver_t12 - persisted-output-first CAVER tunnel features covering
# Tier 1-2 geometry/profile signals.
CAVER_T12_FEATURES_17: list[str] = [
    "caver_tunnel_count",
    "caver_primary_length",
    "caver_total_length",
    "caver_primary_bottleneck_radius",
    "caver_median_bottleneck_radius",
    "caver_primary_mean_radius",
    "caver_primary_radius_std",
    "caver_primary_radius_min",
    "caver_primary_radius_max",
    "caver_primary_radius_skewness",
    "caver_primary_straightness",
    "caver_primary_mean_curvature",
    "caver_primary_max_curvature",
    "caver_primary_high_curvature_count",
    "caver_primary_alignment_angle_deg",
    "caver_primary_bottleneck_count",
    "caver_primary_length_over_axial",
]

FEATURE_SETS: dict[str, list[str]] = {
    "v14": SELECTED_17,
    "v14+v22": SELECTED_17 + EXTRA_VDW22,
    "v14+aa": SELECTED_17 + AA20,
    "v14+shell": SELECTED_17 + AROMATIC_ALIPHATIC_12,
    "v14+v22+aa": SELECTED_17 + EXTRA_VDW22 + AA20,
    "v49": SELECTED_17 + AA20 + AROMATIC_ALIPHATIC_12,
    "v61": SELECTED_17 + AA20 + AROMATIC_ALIPHATIC_12 + AROMATIC_ALIPHATIC_NORMALIZED_12,
    "v_sterol": (
        SELECTED_17
        + AA20
        + AROMATIC_ALIPHATIC_12
        + STEROL_CHEMISTRY_SHELL_COLS
        + POCKET_GEOMETRY_COLS
    ),
}
FEATURE_SETS["v_plm_ste"] = FEATURE_SETS["v_sterol"] + PALMITATE_VS_STERYL_EXTRA_16
FEATURE_SETS["v_lipid_boundary"] = FEATURE_SETS["v_sterol"] + LIPID_BOUNDARY_FEATURES_22
FEATURE_SETS["v_tunnel"] = FEATURE_SETS["v_sterol"] + TUNNEL_FEATURES_18
FEATURE_SETS["v14+aa+tunnel_shape"] = FEATURE_SETS["v14+aa"] + TUNNEL_SHAPE_AVAIL_6
FEATURE_SETS["v14+shell+tunnel_shape"] = FEATURE_SETS["v14+shell"] + TUNNEL_SHAPE_AVAIL_6
FEATURE_SETS["v14+aa12+tunnel_shape"] = SELECTED_17 + AA_SIGNAL_12 + TUNNEL_SHAPE_AVAIL_6
FEATURE_SETS["v14+aa16+tunnel_shape"] = SELECTED_17 + AA_SIGNAL_16 + TUNNEL_SHAPE_AVAIL_6
FEATURE_SETS["v14+aa20+shell6+tunnel_shape"] = (
    SELECTED_17 + AA20 + SHELL_SIGNAL_6 + TUNNEL_SHAPE_AVAIL_6
)
FEATURE_SETS["v14+aa20+shell6+tunnel_shape3"] = (
    SELECTED_17 + AA20 + SHELL_SIGNAL_6 + TUNNEL_SHAPE_SIGNAL_3
)
FEATURE_SETS["v49+tunnel_shape3"] = FEATURE_SETS["v49"] + TUNNEL_SHAPE_SIGNAL_3
FEATURE_SETS["v49+tunnel_shape"] = FEATURE_SETS["v49"] + TUNNEL_SHAPE_AVAIL_6
FEATURE_SETS["v49+tunnel_chem"] = FEATURE_SETS["v49"] + TUNNEL_CHEM_5
FEATURE_SETS["v49+tunnel_geom"] = FEATURE_SETS["v49"] + TUNNEL_GEOM_9
FEATURE_SETS["v_graph_tunnel"] = FEATURE_SETS["v_sterol"] + GRAPH_TUNNEL_FEATURES_3
FEATURE_SETS["v_caver_t12"] = FEATURE_SETS["v_sterol"] + CAVER_T12_FEATURES_17

# v_sterol ablation families used to test whether extra surfaces / derived
# features add signal or just dimensionality.
#
# `v_sterol_v2` is kept as a backward-compatible alias for the holdout-safe
# derived-only slice because the repo's current reports/configs already use it.
# The full historical "115-feature" variant is exposed explicitly as
# `v_sterol+vdw22+derived` so the raw-vdw22 effect can be isolated instead of
# being conflated with the derived family.
DERIVED_FEATURES_26: list[str] = [
    # AA composition fractions
    "hydrophobic_aa_frac",
    "charged_aa_frac",
    "aromatic_aa_frac",
    "polar_uncharged_aa_frac",
    "small_aa_frac",
    "branched_aliphatic_aa_frac",
    # Size-normalized densities
    "vol_per_as",
    "aromatic_density",
    "aliphatic_density",
    "cation_density",
    "anion_density",
    "residue_density",
    # Surface chemistry ratios
    "polar_surface_frac",
    "surf_pol_vdw22_frac",
    "net_charge_per_vol",
    # Spatial gradients
    "bulky_hydro_gradient",
    "polar_gradient",
    "aromatic_gradient",
    "charge_gradient",
    # Shape-chemistry interactions
    "shape_anisotropy",
    "elongation_x_hydro",
    "planarity_x_aromatic_density",
    "burial_x_polar_surface",
    "vol_x_polar_surface",
    # Charge asymmetry
    "charge_balance",
    "charge_density_per_vol",
]
FEATURE_SETS["v_sterol+vdw22"] = FEATURE_SETS["v_sterol"] + EXTRA_VDW22
FEATURE_SETS["v_sterol+derived"] = FEATURE_SETS["v_sterol"] + DERIVED_FEATURES_26
FEATURE_SETS["v_sterol+vdw22+derived"] = (
    FEATURE_SETS["v_sterol"] + EXTRA_VDW22 + DERIVED_FEATURES_26
)
FEATURE_SETS["v_sterol_v2"] = FEATURE_SETS["v_sterol+derived"]

# v_sterol_lean - Pruned feature set (76 features) to prevent overfitting in the hierarchy.
# Removes raw AA counts, redundant shell counts, and noisy fpocket heuristics.
FEATURE_SETS["v_sterol_lean"] = (
    SELECTED_8_LEAN
    + AROMATIC_ALIPHATIC_RATIOS_4
    + STEROL_CHEMISTRY_SHELL_COLS
    + POCKET_GEOMETRY_COLS
    + DERIVED_FEATURES_26
)

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


# ---------------------------------------------------------------------------
# Artifact filenames (single source of truth to avoid duplication)
# ---------------------------------------------------------------------------
HIERARCHICAL_PREDICTIONS_NAME: str = "hierarchical_lipid_predictions.parquet"
HIERARCHICAL_REPORT_NAME: str = "hierarchical_lipid_report.md"
HIERARCHICAL_METRICS_NAME: str = "hierarchical_lipid_metrics.parquet"
HIERARCHICAL_BUNDLE_NAME: str = "hierarchical_bundle.joblib"
