from slipp_plus.constants import (
    AA20,
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


def test_v_sterol_ablation_registry_shapes() -> None:
    assert FEATURE_SETS["v_sterol+vdw22"] == FEATURE_SETS["v_sterol"] + EXTRA_VDW22
    assert FEATURE_SETS["v_sterol+derived"] == FEATURE_SETS["v_sterol"] + DERIVED_FEATURES_26
    assert FEATURE_SETS["v_sterol+vdw22+derived"] == (
        FEATURE_SETS["v_sterol"] + EXTRA_VDW22 + DERIVED_FEATURES_26
    )
    assert FEATURE_SETS["v_sterol_v2"] == FEATURE_SETS["v_sterol+derived"]
