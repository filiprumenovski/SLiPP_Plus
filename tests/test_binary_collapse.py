"""Parity with the paper's TP/FP/TN/FN definitions (Methods p.19)."""

from __future__ import annotations

import numpy as np

from slipp_plus.constants import CLASS_10, LIPID_CODES
from slipp_plus.evaluate import binary_collapse


def _ci(code: str) -> int:
    return CLASS_10.index(code)


def test_all_correct_lipid():
    lipid_int = _ci("CLR")
    y_true = np.array([lipid_int] * 5)
    y_pred = np.array([lipid_int] * 5)
    proba = np.zeros((5, len(CLASS_10)))
    proba[:, lipid_int] = 1.0
    m = binary_collapse(y_true, y_pred, proba)
    assert m["tp"] == 5 and m["fn"] == 0 and m["fp"] == 0 and m["tn"] == 0
    assert m["sensitivity"] == 1.0
    assert m["precision"] == 1.0
    assert m["f1"] == 1.0


def test_all_correct_nonlipid():
    nl_int = _ci("ADN")
    y_true = np.array([nl_int] * 5)
    y_pred = np.array([nl_int] * 5)
    proba = np.zeros((5, len(CLASS_10)))
    proba[:, nl_int] = 1.0
    m = binary_collapse(y_true, y_pred, proba)
    assert m["tn"] == 5 and m["tp"] == 0 and m["fp"] == 0 and m["fn"] == 0
    assert m["specificity"] == 1.0


def test_mixed_confusion():
    clr, ste, adn, pp = _ci("CLR"), _ci("STE"), _ci("ADN"), _ci("PP")
    # 2 lipid correct (clr->ste still collapses to lipid); 1 lipid as nonlipid (fn);
    # 1 nonlipid as lipid (fp); 1 nonlipid correct
    y_true = np.array([clr, clr, ste, adn, pp])
    y_pred = np.array([clr, ste, adn, clr, pp])  # third: lipid predicted as adn (fn); fourth: pp predicted as clr (fp)
    proba = np.zeros((5, len(CLASS_10)))
    for i, p in enumerate(y_pred):
        proba[i, p] = 1.0
    m = binary_collapse(y_true, y_pred, proba)
    assert m["tp"] == 2  # clr->clr, clr->ste (both lipid predictions of lipid)
    assert m["fn"] == 1  # ste->adn
    assert m["fp"] == 1  # adn->clr
    assert m["tn"] == 1  # pp->pp


def test_lipid_codes_subset_of_class10():
    assert LIPID_CODES.issubset(CLASS_10)
