# Holdout Label Source Audit, 2026-05-09

## Question

Why did direct holdout scoring of exp-028 disagree with
`reports/compact_shape3_shell6_chem_weighted_10_20_70/metrics.md`?

## Finding

The root holdout files and component-specific holdout feature files have the
same row counts and row identities, but their `class_binary` columns disagree.

| holdout | root file | component file | rows | root lipids | component lipids | differing labels |
|---|---|---|---:|---:|---:|---:|
| apo-PDB | `processed/apo_pdb_holdout.parquet` | `processed/v49_tunnel_shape3/apo_pdb_holdout.parquet` | 117 | 67 | 67 | 10 |
| AlphaFold | `processed/alphafold_holdout.parquet` | `processed/v49_tunnel_shape3/alphafold_holdout.parquet` | 149 | 91 | 91 | 30 |

The root labels are ligand-derived from the ingestion contract:
`class_binary = ligand in {CLR, MYR, OLA, PLM, STE}`. The component file labels
are not semantically consistent with that rule. Examples:

| holdout | structure_id | ligand | root class_binary | component class_binary |
|---|---|---|---:|---:|
| apo-PDB | `3WDT` | BGC | 0 | 1 |
| apo-PDB | `3WEA` | OLA | 1 | 0 |
| apo-PDB | `4RDY` | MYR | 1 | 0 |
| apo-PDB | `4YBV` | COA | 0 | 1 |
| AlphaFold | `P21589` | ADN | 0 | 1 |
| AlphaFold | `P22059` | CLR | 1 | 0 |
| AlphaFold | `Q99835` | CLR | 1 | 0 |
| AlphaFold | `Q9I4H2` | COA | 0 | 1 |

## Metric Impact

Current compact ensemble scripts evaluate holdouts against
`component_dirs[0] / {apo_pdb,alphafold}_holdout.parquet`, so the official
exp-028 report uses the component labels.

| model | label source | apo-PDB F1/AUROC | AlphaFold F1/AUROC |
|---|---|---:|---:|
| exp-028 | component holdout labels | `0.717 / 0.801` | `0.724 / 0.855` |
| exp-028 | root holdout labels | `0.667 / 0.733` | `0.605 / 0.661` |
| exp-031 diagnostic | component holdout labels | `0.733 / 0.756` | `0.728 / 0.732` |
| exp-031 diagnostic | root holdout labels | `0.748 / 0.765` | `0.740 / 0.734` |

## Decision

Record as `exp-034-holdout-label-source-audit`.

Do not promote or demote deployable configs solely from old compact holdout
metrics until the label source is reconciled. For future holdout reporting,
prefer canonical root labels or regenerate component holdout feature files from
the root holdout tables while preserving row order and `class_binary`.

The important scientific signal survives the audit: exp-031 still beats exp-028
under both label sources, but exp-031 remains a holdout-scored diagnostic rather
than a deployable model-selection protocol.
