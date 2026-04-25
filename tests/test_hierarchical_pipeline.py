from __future__ import annotations

import numpy as np

from slipp_plus.hierarchical_experiment import combine_hierarchical_softprobs


def test_combine_hierarchical_softprobs_rows_sum_to_one() -> None:
    n = 4
    p_lipid = np.linspace(0.1, 0.9, n)
    lipid = np.full((n, 5), 0.2, dtype=np.float64)
    nonlipid = np.array(
        [
            [0.5, 0.2, 0.1, 0.1, 0.1],
            [0.1, 0.5, 0.1, 0.2, 0.1],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.05, 0.05, 0.3, 0.3, 0.3],
        ],
        dtype=np.float64,
    )
    out = combine_hierarchical_softprobs(p_lipid, lipid, nonlipid)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-9)
