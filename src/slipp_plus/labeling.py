"""Assign 10-class labels to pockets emitted by dpocket. Day 7+ path.

Day 1 consumes pre-labeled data in ``reference/.../training_pockets.csv``
where the ``lig`` column already carries the 9 ligand codes
(``ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | STE``) or ``"none"`` for
pseudo-pockets. ``ingest.py`` maps that to ``class_10``.

For Day 7+ from-scratch reproduction, labeling rules:

- Pocket with ligand COM within 8 A -> ``class_10 = <ligand 3-letter code>``.
- Pocket with no ligand overlap -> ``class_10 = "PP"``.
- Apply surface-binding filter: exclude ligand pockets where fewer than 10
  protein residues lie within 8 A of the ligand COM (paper Methods p.17).
- Do not include heme (HEM) in the final training set (paper protocol).

See ``src/slipp_plus/constants.py`` for vocabularies.
"""

from __future__ import annotations


def assign_labels(*args: object, **kwargs: object) -> None:
    raise NotImplementedError(
        "Day 7+ from-scratch reproduction. Day 1 consumes pre-labeled data "
        "via src/slipp_plus/ingest.py."
    )
