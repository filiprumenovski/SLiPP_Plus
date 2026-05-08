# Ablation Matrix

Generated from `experiments/registry.yaml`. Metrics are copied from registry entries and should be refreshed whenever registry metrics change.

| experiment_id | feature_set | binary_f1 +/- std | macro_f1_10 +/- std | apo_pdb_f1 | alphafold_f1 | key_finding | superseded_by |
|---|---|---|---|---|---|---|---|
| exp-001-day1-v14 | v14 | 0.860 ± 0.017 | 0.650 ± 0.020 | 0.746 | 0.732 | Day 1 baseline. | exp-002-v49-baseline |
| exp-002-v49-baseline | v49 | 0.895 ± 0.015 | 0.725 ± 0.017 | 0.709 | 0.753 | +20 AA counts + 12 aromatic/aliphatic shell features. | exp-005-v_sterol-ensemble |
| exp-003-v49-ensemble-clr-ste-tb | v49 | 0.898 ± 0.016 | 0.731 ± 0.016 | - | - | CLR/STE tiebreaker fired only 2 times across 25 iterations (0.08%). | - |
| exp-004-v61-baseline | v61 | 0.894 ± 0.016 | 0.726 ± 0.016 | 0.745 | 0.69 | v49 + 12 normalized shell features. | - |
| exp-005-v_sterol-ensemble | v_sterol | 0.899 ± 0.015 | 0.734 ± 0.016 | - | - | Chemistry-refined residue shells (28 chem + 4 ratio + 5 PCA + 1 burial = 38 new features). | exp-007-v_sterol-plm_ste-tiebreaker |
| exp-006-v_plm_ste-features | v_plm_ste | - | ~0.730 | - | - | ABANDONED. | - |
| exp-007-v_sterol-plm_ste-tiebreaker | v_sterol | 0.899 ± 0.015 | 0.738 ± 0.015 | 0.716 | 0.725 | WINNING CONFIG. | exp-009-v_sterol-boundary-refactor |
| exp-008-detector-bakeoff | - | - | - | - | - | fpocket vs P2Rank on 1,752 ligand-bound structures. | - |
| exp-009-v_sterol-boundary-refactor | v_sterol | 0.899 ± 0.016 | 0.754 ± 0.016 | - | - | Boundary-head refactor promoted grouped STE-vs-neighbors rescue into the shared local-arbitration architecture. | exp-011-family-plus-moe |
| exp-010-binary-ovr-lipid-family | v_sterol | 0.896 ± 0.017 | 0.726 ± 0.018 | 0.679 | 0.708 | NEGATIVE RESULT. | - |
| exp-011-family-plus-moe | v_sterol | 0.901 ± 0.016 | 0.762 ± 0.015 | - | - | Current champion. | exp-012-compact-tunnel-shape |
| exp-012-compact-tunnel-shape | v49+tunnel_shape | 0.902 ± 0.017 | 0.766 ± 0.019 | 0.696 | 0.62 | Release-facing compact leader. | exp-014-v49-tunnel-shape3 |
| exp-013-v49-shell6-tunnel-shape | v14+aa20+shell6+tunnel_shape | 0.900 ± 0.017 | 0.766 ± 0.019 | 0.711 | 0.703 | MIXED/TIE RESULT. | - |
| **exp-014-v49-tunnel-shape3** | v49+tunnel_shape3 | 0.900 ± 0.015 | 0.768 ± 0.018 | 0.667 | 0.724 | NEW INTERNAL LEADER, SMALL EFFECT. | - |
