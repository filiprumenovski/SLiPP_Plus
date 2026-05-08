from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd

from slipp_plus.feature_families import (
    fit_family_scalers,
    materialize_family_arrays,
    resolve_family_specs,
)


def test_family_materialization_marks_optional_absent() -> None:
    specs = resolve_family_specs(["paper17", "caver_t12"])
    data = {column: [1.0, 2.0, 3.0] for column in specs[0].columns}
    frame = pd.DataFrame(data)

    scalers = fit_family_scalers(frame, np.array([0, 1]), specs)
    arrays, masks = materialize_family_arrays(frame, specs, scalers)

    assert arrays["paper17"].shape == (3, len(specs[0].columns))
    assert arrays["caver_t12"].shape == (3, len(specs[1].columns))
    assert masks[:, 0].tolist() == [1.0, 1.0, 1.0]
    assert masks[:, 1].tolist() == [0.0, 0.0, 0.0]


def test_compact_tunnel_families_resolve() -> None:
    specs = resolve_family_specs(["paper17", "tunnel_shape", "tunnel_chem", "tunnel_geom"])

    assert [spec.name for spec in specs] == [
        "paper17",
        "tunnel_shape",
        "tunnel_chem",
        "tunnel_geom",
    ]
    assert specs[1].optional
    assert specs[2].optional
    assert specs[3].optional


def test_signal_pruned_families_resolve() -> None:
    specs = resolve_family_specs(
        ["paper17", "aa_signal12", "aa_signal16", "shell_signal6", "tunnel_shape_signal3"]
    )

    assert [spec.name for spec in specs] == [
        "paper17",
        "aa_signal12",
        "aa_signal16",
        "shell_signal6",
        "tunnel_shape_signal3",
    ]
    assert specs[-1].optional


def test_family_encoder_forward_shape() -> None:
    code = """
import numpy as np
from slipp_plus.backbone_family_encoder import FamilyEncoderNet, predict_family_encoder_proba
arrays = {
    "paper17": np.ones((4, 17), dtype=np.float32),
    "aa20": np.ones((4, 20), dtype=np.float32),
}
masks = np.ones((4, 2), dtype=np.float32)
model = FamilyEncoderNet({"paper17": 17, "aa20": 20}, token_dim=8, hidden_dim=16, dropout=0.0)
proba = predict_family_encoder_proba(model, arrays, masks, batch_size=2)
assert proba.shape == (4, 10)
np.testing.assert_allclose(proba.sum(axis=1), np.ones(4), atol=1e-6)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
