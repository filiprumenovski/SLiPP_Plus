from slipp_plus.constants import (
    AA20,
    CLASS_10,
    LIG_TO_CLASS,
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
