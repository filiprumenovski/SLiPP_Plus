# Datasheet

This datasheet documents the tabular data used by SLiPP++ as checked out in this repository. It is intentionally file-derived: counts and checksums below were computed from the local artifacts, not copied from older project notes.

## Motivation

SLiPP++ reformulates the Chou et al. lipid-pocket prediction dataset from a binary lipid-vs-nonlipid task into a 10-class task over `ADN`, `B12`, `BGC`, `CLR`, `COA`, `MYR`, `OLA`, `PLM`, `PP`, and `STE`.

## Files

| File | Role | SHA-256 |
|---|---|---|
| `reference/SLiPP_2024-main/training_pockets.csv` | Training and split-generation table | `4d27636b4381dc3c1b9e27451db5b788e6b16f13919c4ed36f8c2ba108097711` |
| `data/raw/supplementary/ci5c01076_si_003.xlsx` | apo-PDB holdout source workbook | `ff07b4ae9e428dcfcca64b825c4feb3a5a2ce4630ae4eeca5bc60f9076e303bd` |
| `data/raw/supplementary/ci5c01076_si_004.xlsx` | AlphaFold holdout source workbook | `ff7b41071e475c52b840f905c5515097bf849ce4d4c9de2dbf54643203264fdf` |

## Training Set Composition

`training_pockets.csv` has 15,219 rows and 64 columns. The class counts enforced by the ingestion gate are:

| Class | Rows |
|---|---:|
| ADN | 414 |
| B12 | 373 |
| BGC | 526 |
| CLR | 358 |
| COA | 2,020 |
| MYR | 424 |
| OLA | 329 |
| PLM | 718 |
| PP | 9,905 |
| STE | 152 |

Known imbalance: `STE` is the smallest class with 152 rows, while `PP` has 9,905 rows. This imbalance is part of the scientific problem and should be preserved in audit logs and ablation notes.

## Holdout Sets

The holdout ID lists are committed under `data/holdouts/` and are derived from the current supplementary workbooks:

| Holdout | Source sheet | ID column | Unique IDs |
|---|---|---|---:|
| apo-PDB | `ci5c01076_si_003.xlsx::Table S2` | `PDB_ID` | 117 |
| AlphaFold | `ci5c01076_si_004.xlsx::Table S3` | `UniProt ID code` | 149 |

The source workbook ligand-label counts are:

| Holdout | Ligand counts |
|---|---|
| apo-PDB | `ADN=15`, `BGC=11`, `CLR=18`, `COA=12`, `HEM=12`, `MYR=18`, `OLA=13`, `PLM=14`, `STE=4` |
| AlphaFold | `ADN=16`, `BGC=12`, `CLR=23`, `COA=19`, `HEM=11`, `MYR=26`, `OLA=16`, `PLM=23`, `STE=3` |

These counts are lower than older handoff notes that mention 131 apo-PDB and 177 AlphaFold structures. The committed CSVs follow the files present in this checkout.

## Feature Columns

The Day 1 paper-aligned feature set is the 17-column `SELECTED_17` order in `src/slipp_plus/constants.py`. Additional feature families (`v22`, amino-acid counts, sterol chemistry, tunnel features, lipid-boundary features) are derived artifacts and should be documented with the config and experiment registry entry that generated them.

## Provenance

The training CSV comes from the public Dassama Lab SLiPP 2024 repository, restored at `reference/SLiPP_2024-main/`. The supplementary workbooks are stored under `data/raw/supplementary/` and correspond to the Chou et al. validation tables used for apo-PDB and AlphaFold holdout comparisons.

Primary scientific reference: Chou et al. 2024, DOI `10.1101/2024.01.26.577452`.

## License And Use Constraints

The SLiPP++ code is MIT licensed. Upstream dataset licensing and reuse terms should be checked against the Chou et al. publication, supplementary material, and the Dassama Lab source repository before redistribution outside this project.

## Maintenance Notes

Do not delete experiment logs, registry entries, or report directories when updating this datasheet. Negative results and superseded runs are part of the audit trail for this project.
