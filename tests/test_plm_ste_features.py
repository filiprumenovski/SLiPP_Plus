from __future__ import annotations

import numpy as np
import pytest

from slipp_plus.constants import PALMITATE_VS_STERYL_EXTRA_16, PLM_STE_EXTRA_16
from slipp_plus.plm_ste_features import _axial_profile


def _vert_coords() -> np.ndarray:
    return np.asarray(
        [
            [-5.0, 0.0, 0.0],
            [-3.0, 0.5, 0.0],
            [-1.0, 0.2, 0.4],
            [1.0, -0.2, 0.2],
            [3.0, 0.5, -0.1],
            [5.0, 0.0, 0.0],
        ],
        dtype=float,
    )


def test_axial_profile_gradient_is_sign_stable_under_reflection() -> None:
    coords = _vert_coords()
    reflected = coords * np.asarray([-1.0, 1.0, 1.0])

    original = _axial_profile(coords)
    mirrored = _axial_profile(reflected)

    assert original is not None
    assert mirrored is not None
    assert original["axial_radius_gradient"] >= 0.0
    assert mirrored["axial_radius_gradient"] >= 0.0
    assert original["axial_radius_gradient"] == pytest.approx(
        mirrored["axial_radius_gradient"], rel=1e-8, abs=1e-8
    )
    assert original["axial_length"] == pytest.approx(
        mirrored["axial_length"], rel=1e-8, abs=1e-8
    )
    assert original["thick_end_asymmetry"] == pytest.approx(
        mirrored["thick_end_asymmetry"], rel=1e-8, abs=1e-8
    )


def test_palmitate_vs_steryl_constant_keeps_backward_compatible_alias() -> None:
    assert PALMITATE_VS_STERYL_EXTRA_16 == PLM_STE_EXTRA_16
    assert "axial_radius_gradient" in PALMITATE_VS_STERYL_EXTRA_16
