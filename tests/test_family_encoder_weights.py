from __future__ import annotations

import numpy as np
import pytest

from slipp_plus.composite.backbone_family_encoder import _class_weights
from slipp_plus.constants import CLASS_10


def test_class_weight_multiplier_boosts_target_class_after_normalization() -> None:
    y_train = np.array(
        [CLASS_10.index("PP")] * 40
        + [CLASS_10.index("PLM")] * 8
        + [CLASS_10.index("STE")] * 2,
        dtype=np.int64,
    )

    base = _class_weights(y_train)
    boosted = _class_weights(y_train, {"STE": 2.0})

    ste = CLASS_10.index("STE")
    plm = CLASS_10.index("PLM")
    assert boosted[ste] / boosted[plm] == pytest.approx(2.0 * base[ste] / base[plm])
    assert boosted.mean() == pytest.approx(1.0)
