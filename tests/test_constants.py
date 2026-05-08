from slipp_plus.constants import (
    AA20,
    AA_SIGNAL_12,
    AA_SIGNAL_16,
    CLASS_10,
    DERIVED_FEATURES_26,
    EXTRA_VDW22,
    FEATURE_SETS,
    LIG_TO_CLASS,
    LIPID_BOUNDARY_FEATURES_22,
    LIPID_CODES,
    NONLIPID_CODES,
    PP_CODE,
    SELECTED_17,
    SHELL_SIGNAL_6,
    TUNNEL_CHEM_5,
    TUNNEL_GEOM_9,
    TUNNEL_SHAPE_AVAIL_6,
    TUNNEL_SHAPE_SIGNAL_3,
    is_lipid,
)


def test_class10_size_and_membership():
    assert len(CLASS_10) == 10
    assert set(CLASS_10) == LIPID_CODES | NONLIPID_CODES | {PP_CODE}


def test_selected17_is_17_unique_strings():
    assert len(SELECTED_17) == 17
    assert len(set(SELECTED_17)) == 17


def test_aa20_is_20():
    assert len(AA20) == 20
    assert len(set(AA20)) == 20


def test_signal_ranked_subset_shapes() -> None:
    assert len(AA_SIGNAL_12) == 12
    assert len(AA_SIGNAL_16) == 16
    assert len(SHELL_SIGNAL_6) == 6
    assert len(TUNNEL_SHAPE_SIGNAL_3) == 3
    assert set(AA_SIGNAL_12) < set(AA20)
    assert set(AA_SIGNAL_16) < set(AA20)
    assert set(AA_SIGNAL_12) < set(AA_SIGNAL_16)
    assert set(SHELL_SIGNAL_6) < set(FEATURE_SETS["v14+shell"])
    assert set(TUNNEL_SHAPE_SIGNAL_3) < set(TUNNEL_SHAPE_AVAIL_6)


def test_v14_shell_registry_shape() -> None:
    shell12 = FEATURE_SETS["v49"][len(SELECTED_17) + len(AA20) :]
    assert FEATURE_SETS["v14+shell"] == SELECTED_17 + shell12
    assert len(FEATURE_SETS["v14+shell"]) == 29


def test_lig_to_class_maps_none_to_pp():
    assert LIG_TO_CLASS["none"] == PP_CODE
    for code in LIPID_CODES | NONLIPID_CODES:
        assert LIG_TO_CLASS[code] == code


def test_is_lipid_matches_set():
    for c in CLASS_10:
        assert is_lipid(c) == (c in LIPID_CODES)


def test_v_lipid_boundary_registry_shape():
    assert len(LIPID_BOUNDARY_FEATURES_22) == 22
    assert len(set(LIPID_BOUNDARY_FEATURES_22)) == 22
    assert FEATURE_SETS["v_lipid_boundary"][-22:] == LIPID_BOUNDARY_FEATURES_22
    assert len(FEATURE_SETS["v_lipid_boundary"]) == len(FEATURE_SETS["v_sterol"]) + 22


def test_compact_tunnel_registry_shapes() -> None:
    assert len(TUNNEL_SHAPE_AVAIL_6) == 6
    assert len(TUNNEL_CHEM_5) == 5
    assert len(TUNNEL_GEOM_9) == 9
    assert len(set(TUNNEL_SHAPE_AVAIL_6)) == len(TUNNEL_SHAPE_AVAIL_6)
    assert len(set(TUNNEL_CHEM_5)) == len(TUNNEL_CHEM_5)
    assert len(set(TUNNEL_GEOM_9)) == len(TUNNEL_GEOM_9)
    assert FEATURE_SETS["v14+aa+tunnel_shape"] == FEATURE_SETS["v14+aa"] + TUNNEL_SHAPE_AVAIL_6
    assert FEATURE_SETS["v14+shell+tunnel_shape"] == (
        FEATURE_SETS["v14+shell"] + TUNNEL_SHAPE_AVAIL_6
    )
    assert FEATURE_SETS["v14+aa12+tunnel_shape"] == (
        SELECTED_17 + AA_SIGNAL_12 + TUNNEL_SHAPE_AVAIL_6
    )
    assert FEATURE_SETS["v14+aa16+tunnel_shape"] == (
        SELECTED_17 + AA_SIGNAL_16 + TUNNEL_SHAPE_AVAIL_6
    )
    assert FEATURE_SETS["v14+aa20+shell6+tunnel_shape"] == (
        SELECTED_17 + AA20 + SHELL_SIGNAL_6 + TUNNEL_SHAPE_AVAIL_6
    )
    assert FEATURE_SETS["v49+tunnel_shape3"] == FEATURE_SETS["v49"] + TUNNEL_SHAPE_SIGNAL_3
    assert FEATURE_SETS["v49+tunnel_shape"] == FEATURE_SETS["v49"] + TUNNEL_SHAPE_AVAIL_6
    assert FEATURE_SETS["v49+tunnel_chem"] == FEATURE_SETS["v49"] + TUNNEL_CHEM_5
    assert FEATURE_SETS["v49+tunnel_geom"] == FEATURE_SETS["v49"] + TUNNEL_GEOM_9


def test_v_sterol_ablation_registry_shapes() -> None:
    assert FEATURE_SETS["v_sterol+vdw22"] == FEATURE_SETS["v_sterol"] + EXTRA_VDW22
    assert FEATURE_SETS["v_sterol+derived"] == FEATURE_SETS["v_sterol"] + DERIVED_FEATURES_26
    assert FEATURE_SETS["v_sterol+vdw22+derived"] == (
        FEATURE_SETS["v_sterol"] + EXTRA_VDW22 + DERIVED_FEATURES_26
    )
    assert FEATURE_SETS["v_sterol_v2"] == FEATURE_SETS["v_sterol+derived"]
