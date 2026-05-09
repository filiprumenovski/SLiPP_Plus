# Detector Bakeoff

Benchmarks two pocket detectors (fpocket and P2Rank) on the SLiPP++ training set
of 1,780 ligand-bound PDBs, scoring each detector's ranked pockets against the
known ligand binding site.

## What this measures

For every (structure, detector) pair we compute, per predicted pocket:

- `DCC` — distance from pocket center to the ligand center of mass (closest copy).
- `DCA` — distance from pocket center to the nearest heavy atom of the ligand
  (closest copy).
- `hit_dcc_4A` / `hit_dca_4A` — boolean, threshold 4 Å (standard in the pocket
  literature: PRANK, DeepPocket).

For every (detector, ligand_class) and for (detector, "ALL") the summary reports:

- `top{1,3,5}_dcc` / `top{1,3,5}_dca` — fraction of structures where at least
  one of the top-K predictions is a hit at 4 Å.
- `mean_rank_first_dcc_hit` — mean rank of the first DCC ≤ 4 Å pocket
  (null-valued for structures without any hit; the mean drops nulls).
- `n_no_hit` — structures with zero DCC hits among the top 10 pockets.

The fpocket rank is the order in `<stem>_info.txt` (fpocket's default score
ordering). The fpocket center is the alpha-sphere centroid from
`pockets/pocketN_vert.pqr` via `aromatic_aliphatic._compute_centroid`.
The P2Rank rank is the `rank` column in its `*_predictions.csv`.

Ligand identity per structure is taken from the parent directory name in
`data/structures/source_pdbs/<CLASS>/`. We extract every HETATM record whose
residue name matches `<CLASS>` exactly, grouping by (chain, resseq, icode) into
distinct copies and taking the minimum distance across copies.

## Structure plan

The CLI walks `--fpocket-root/<CLASS>/*.pdb`. For each PDB it pairs:

- `fpocket_dir = <CLASS>/<stem>_out/`
- `p2rank_csv = <p2rank-root>/<stem>_predictions.csv` (skipped if missing)

P2Rank predictions missing are logged and omitted from that structure's rows;
the bakeoff continues for the fpocket side.

## How to re-run

```bash
uv run python -m slipp_plus.detector_bakeoff \
  --fpocket-root data/structures/source_pdbs \
  --p2rank-root processed/p2rank/train_out \
  --output reports/detector_bakeoff/training_scores.parquet \
  --summary-output reports/detector_bakeoff/training_summary.parquet \
  --summary-md reports/detector_bakeoff/training_summary.md \
  --workers 6
```

Outputs:

- `training_scores.parquet` — one row per (structure, detector, pocket_rank).
- `training_summary.parquet` — aggregated hit rates by (detector, ligand_class).
- `training_summary.md` — the same table rendered as markdown for the write-up.
