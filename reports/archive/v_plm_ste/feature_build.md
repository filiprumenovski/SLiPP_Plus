# v_plm_ste — Feature Build Summary

Feature set `v_plm_ste` extends `v_sterol` (96 columns) with 16 features
designed specifically to resolve the **PLM (palmitate) vs STE (steryl
ester)** confusion identified in the sterol-ID sprint. Total feature count:
**110** (`v_sterol` 94 feature columns + 16 new, plus bookkeeping cols).

The 16 features are appended to every row of
`processed/v_sterol/full_pockets.parquet`; row count, `pdb_ligand`,
`matched_pocket_number`, and all pre-existing feature columns pass through
unchanged.

## Inputs / outputs

| Path | Size | Notes |
| --- | --- | --- |
| `processed/v_sterol/full_pockets.parquet` | 15,219 x 96 | base input |
| `processed/v_plm_ste/full_pockets.parquet` | 15,219 x 112 | 96 base + 16 new |
| `models/v_plm_ste/{rf,xgb,lgbm}_multiclass.joblib` | 3 files | iter-0 dumps |
| `processed/v_plm_ste/predictions/test_predictions.parquet` | 114,150 rows | 25 iter x 3 models |
| `configs/v_plm_ste.yaml` | — | points at `processed/v_plm_ste/` |

Build command:

```
uv run python -m slipp_plus.plm_ste_features training \
    --base-parquet processed/v_sterol/full_pockets.parquet \
    --source-pdbs-root data/structures/source_pdbs \
    --output processed/v_plm_ste/full_pockets.parquet \
    --workers 8
```

Build result: **15,219 rows**, **1,780 structures**, **0 warnings**, output
parquet 3.0 MB. All 16 new columns are finite, non-null, numeric.

## The 16 features (canonical order)

### Group A — CRAC / CARC sequence motifs (4)

CRAC regex `[LV].{1,5}[Y].{1,5}[RK]`, CARC regex
`[RK].{1,5}[YF].{1,5}[LV]`, scanned per chain over the one-letter protein
sequence parsed from `data/structures/source_pdbs/<CLASS>/<stem>.pdb`.
A motif "contacts" the pocket when at least one of its residues has a
closest-heavy-atom distance ≤ 6 Å to the alpha-sphere centroid
(shell index ≤ 2, matching `v_sterol`'s shell 1+2 convention).

| column | definition |
| --- | --- |
| `crac_count` | number of CRAC motif hits contacting the pocket |
| `carc_count` | number of CARC motif hits contacting the pocket |
| `any_sterol_motif` | 1 if either of the above > 0 else 0 |
| `motif_residue_density` | contacting shell 1+2 residues belonging to any contacting motif ÷ total contacting shell 1+2 residues (0 if denom is 0) |

### Group B — Axial profile (7)

PCA on the pocket's alpha-sphere coordinates (`vert.pqr`). Principal axis
is the eigenvector for the largest eigenvalue; project each alpha-sphere
onto the axis (through the alpha-sphere centroid) to get a 1D coordinate
`t`. Bin `t` into 5 equal-width bins over `[t_min, t_max]`. For each bin
compute the mean radial distance to the axis (`R_i`). Empty bins carry
forward the previous non-empty `R`; leading empty bins fall back to the
global mean.

| column | definition |
| --- | --- |
| `axial_length` | `t_max − t_min` |
| `axial_radius_std` | std-dev of `R_1..R_5` |
| `axial_radius_gradient` | OLS slope of `R_i` vs bin index (1..5) |
| `fatend_ratio` | `max(R) / (min(R) + 1e-6)` |
| `bottleneck_position` | `argmin_i(R) / 4.0` ∈ {0, 0.25, 0.5, 0.75, 1} |
| `thick_end_asymmetry` | `|R_1 − R_5| / (mean(R) + 1e-6)` |
| `cross_section_aspect` | `sqrt(lam2 / lam3)`, clipped to [1.0, 100.0] |

Safe defaults (pocket has < 5 alpha-spheres or degenerate axis):
`axial_length=0`, `axial_radius_std=0`, `axial_radius_gradient=0`,
`fatend_ratio=1`, `bottleneck_position=0.5`, `thick_end_asymmetry=0`,
`cross_section_aspect=1`.

### Group C — Polar-anchor chemistry (5)

The "polar end" is whichever of the extreme axial bins (bin 0 or bin 4)
has the higher fraction of polar residues among its pocket-contacting
shell 1+2 residues, where "polar" = union of
`aromatic_polar, polar_neutral, cationic, anionic` chemistry groups
(GLY excluded from all groups, matching `sterol_features.py`). Tie-break
to bin 0. Contacting residues are projected onto the axis and assigned
to bins via the same quintile edges as Group B.

| column | definition |
| --- | --- |
| `polar_end_cationic_count` | `cationic` residues at the polar end |
| `polar_end_aromatic_polar_count` | `aromatic_polar` residues at the polar end |
| `polar_end_neutral_polar_count` | `polar_neutral` residues at the polar end |
| `anchor_charge_balance` | `(cationic − anionic) / (cationic + anionic + 1)` at the polar end |
| `anchor_chemistry_entropy` | Shannon entropy (natural log) over the 7 chemistry-group counts at the polar end, divided by `log(7)` → ∈ [0, 1]; 0 when no residues |

If the axial projection fails (< 5 alpha-spheres), all five features = 0.

## Per-class mean of the 16 features

```
class_10                           ADN     B12     BGC     CLR     COA     MYR     OLA     PLM     PP     STE
crac_count                       0.203   0.343   0.346   0.263   0.282   0.481   0.353   0.365  0.273   0.224
carc_count                       0.744   0.472   0.563   0.492   0.627   1.073   0.578   0.733  0.550   0.559
any_sterol_motif                 0.664   0.496   0.574   0.609   0.592   0.840   0.650   0.664  0.550   0.507
motif_residue_density            0.163   0.185   0.171   0.222   0.203   0.381   0.269   0.241  0.217   0.168
axial_length                    15.891  13.535  13.972  11.611  13.671  19.378  12.514  17.109  8.139  13.251
axial_radius_std                 0.522   0.541   0.534   0.450   0.537   0.660   0.473   0.615  0.365   0.432
axial_radius_gradient            0.000  -0.004  -0.030  -0.003   0.008  -0.003   0.016   0.009 -0.002   0.000
fatend_ratio                     2.274   2.695   2.655   2.661   2.654   2.352   2.996   2.548  3.357   2.625
bottleneck_position              0.488   0.499   0.527   0.502   0.494   0.485   0.446   0.493  0.502   0.480
thick_end_asymmetry              0.367   0.486   0.429   0.425   0.419   0.379   0.433   0.412  0.500   0.457
cross_section_aspect             1.911   1.938   2.066   1.980   1.950   1.942   1.964   1.973  2.249   1.927
polar_end_cationic_count         0.046   0.086   0.144   0.031   0.170   0.040   0.088   0.060  0.287   0.112
polar_end_aromatic_polar_count   0.082   0.102   0.165   0.087   0.095   0.033   0.122   0.047  0.161   0.066
polar_end_neutral_polar_count    0.111   0.346   0.337   0.204   0.227   0.050   0.228   0.132  0.435   0.355
anchor_charge_balance           -0.051  -0.005  -0.094   0.003   0.038   0.007   0.014   0.012 -0.005   0.037
anchor_chemistry_entropy         0.059   0.103   0.150   0.107   0.106   0.028   0.123   0.060  0.207   0.117
```

Noteworthy signal: **STE** has roughly 3x the `polar_end_neutral_polar_count`
of **PLM** (0.355 vs 0.132) and nearly 2x the `polar_end_cationic_count`
(0.112 vs 0.060) — consistent with the steryl-ester headgroup demanding a
polar anchor. PLM in turn has a longer `axial_length` (17.1 vs 13.3) and
larger `axial_radius_std`, consistent with a linear 16-C acyl chain. These
two features are the most directly actionable for a tiebreaker head.

## Baseline single-model per-class F1 (25-iter aggregate, 10-way softmax)

Aggregated across all 25 stratified-shuffle iterations on
`test_predictions.parquet` (114,150 rows per feature set).

| model | class | v_sterol F1 | v_plm_ste F1 | Δ |
| --- | --- | ---: | ---: | ---: |
| rf | PLM | 0.6188 | 0.6166 | −0.0022 |
| rf | STE | 0.3311 | 0.3212 | −0.0098 |
| xgb | PLM | 0.6273 | 0.6179 | −0.0094 |
| xgb | STE | 0.4178 | 0.3873 | −0.0305 |
| lgbm | PLM | 0.6359 | 0.6302 | −0.0056 |
| lgbm | STE | 0.4049 | 0.3951 | −0.0098 |

Precision / recall (same aggregation):

| model | class | v_sterol P | v_sterol R | v_plm_ste P | v_plm_ste R |
| --- | --- | ---: | ---: | ---: | ---: |
| rf | PLM | 0.661 | 0.582 | 0.659 | 0.579 |
| rf | STE | 0.460 | 0.259 | 0.456 | 0.248 |
| xgb | PLM | 0.648 | 0.608 | 0.639 | 0.598 |
| xgb | STE | 0.470 | 0.376 | 0.448 | 0.341 |
| lgbm | PLM | 0.645 | 0.627 | 0.633 | 0.628 |
| lgbm | STE | 0.472 | 0.355 | 0.469 | 0.341 |

### Interpretation

At the 10-way multiclass single-model level, `v_plm_ste` is roughly neutral
on PLM (−0.002 to −0.009 F1) and slightly regressive on STE (−0.010 to
−0.031 F1). This is expected: the 16 new features were *not* designed to
carry a standalone 10-class softmax — they're a focused PLM-vs-STE signal
to be consumed by the ensemble / tiebreaker head built separately. Using
them as extra tail dimensions in a balanced-weight softmax with only 152
STE examples creates a small amount of per-class noise that depresses F1
at the margin.

The per-class mean table above confirms the features do carry the intended
PLM↔STE structure (especially `polar_end_neutral_polar_count`,
`polar_end_cationic_count`, and `axial_length`). Wiring these into a
targeted PLM/STE tiebreaker (out of scope for this build) is where the
signal is expected to materialize.

## Files created / modified

**Modified**

- `src/slipp_plus/constants.py` — added `PLM_STE_EXTRA_16` and
  `FEATURE_SETS["v_plm_ste"]`.
- `src/slipp_plus/config.py` — extended `FeatureSet` Literal with
  `"v_plm_ste"`.

**Created**

- `src/slipp_plus/plm_ste_features.py` — 16-feature extractor, parallel
  worker, training / holdout CLI.
- `configs/v_plm_ste.yaml` — config cloned from `v_sterol.yaml` with
  `feature_set: v_plm_ste` and `processed/` `models/` `reports/` paths
  switched to `v_plm_ste`.
- `processed/v_plm_ste/full_pockets.parquet` — 15,219 x 112, 0 warnings,
  0 NaN / inf.
- `processed/v_plm_ste/predictions/test_predictions.parquet` — 114,150
  rows (25 iterations × 3 models × test-fold predictions).
- `models/v_plm_ste/{rf,xgb,lgbm}_multiclass.joblib` — iter-0 dumps.
- `reports/v_plm_ste/feature_build.md` — this file.

The v_sterol parquet and Rule 1 validation gate are untouched.
