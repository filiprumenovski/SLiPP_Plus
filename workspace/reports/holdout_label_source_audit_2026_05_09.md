# Holdout Label Source Audit, 2026-05-09

## Question

Why did direct holdout scoring of exp-028 disagree with
`reports/compact_shape3_shell6_chem_weighted_10_20_70/metrics.md`?

## Finding

The root holdout files and component-specific holdout feature files have the
same row counts and row identities, but they are not in the same row order.
Directly comparing `class_binary` by row position gives apparent mismatches:

| holdout | root file | component file | rows | root lipids | component lipids | differing labels |
|---|---|---|---:|---:|---:|---:|
| apo-PDB | `processed/apo_pdb_holdout.parquet` | `processed/v49_tunnel_shape3/apo_pdb_holdout.parquet` | 117 | 67 | 67 | 10 |
| AlphaFold | `processed/alphafold_holdout.parquet` | `processed/v49_tunnel_shape3/alphafold_holdout.parquet` | 149 | 91 | 91 | 30 |

After aligning by `structure_id` and `ligand`, the labels agree exactly:

| holdout | aligned missing rows | aligned differing labels |
|---|---:|---:|
| apo-PDB | 0 | 0 |
| AlphaFold | 0 | 0 |

## Metric Impact

Current compact ensemble scripts previously evaluated holdouts against
`component_dirs[0] / {apo_pdb,alphafold}_holdout.parquet`. That was not
semantically wrong for exp-028, but it made ad hoc direct scoring against root
holdouts easy to get wrong unless labels were aligned by identity first.

| model | label source | apo-PDB F1/AUROC | AlphaFold F1/AUROC |
|---|---|---:|---:|
| exp-028 | component holdout labels | `0.717 / 0.801` | `0.724 / 0.855` |
| exp-028 | root labels aligned by identity | `0.717 / 0.801` | `0.724 / 0.855` |
| exp-031 diagnostic | aligned labels | `0.729 / 0.761` | `0.735 / 0.762` |

## Decision

Record as `exp-034-holdout-label-source-audit`.

Update the compact ensemble script to prefer canonical root labels when present,
aligning them to component prediction row order by `structure_id` and `ligand`.
This prevents future ad hoc row-order scoring mistakes while keeping the
official exp-028 holdout metrics unchanged.

The important scientific signal survives the audit: the corrected exp-031 grid
still finds a holdout-scored diagnostic that beats exp-028 on both holdout F1
scores, but exp-031 remains a holdout-scored diagnostic rather than a deployable
model-selection protocol.
