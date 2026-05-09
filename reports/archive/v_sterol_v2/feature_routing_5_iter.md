# v_sterol_v2 feature-routing quick check (5 iterations)

Goal: test whether the hierarchical regression of `v_sterol_v2` can be fixed quickly by routing smaller feature subsets to specific stages while keeping the richer derived stack where it helps.

All runs used the same `processed/v_sterol_v2` parquet content copied into isolated 5-iteration working directories with the same seed base (`42`).

## Compared variants

| variant | change |
|---|---|
| baseline | plain `v_sterol_v2` hierarchy |
| full_routed | `lipid_family_feature_set=v_sterol`, `specialist_feature_set=v_sterol`, OLA/PLM boundary kept on `v_sterol_v2` |
| specialist_routed | `specialist_feature_set=v_sterol`, family head left on `v_sterol_v2` |
| family_v49 | `lipid_family_feature_set=v49`, specialist left on `v_sterol_v2` |
| nonlipid_routed | `nonlipid_feature_set=v_sterol`, lipid family + specialist left on `v_sterol_v2` |

## Mean metrics across 5 iterations

| variant | macro-F1 (10) | macro-F1 (5 lipids) | binary F1 | binary AUROC | CLR F1 | MYR F1 | OLA F1 | PLM F1 | STE F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.7318 | 0.6195 | 0.8912 | 0.9850 | 0.7166 | 0.6524 | 0.5143 | 0.6373 | 0.5767 |
| full_routed | 0.7268 | 0.6091 | **0.8970** | **0.9850** | 0.6843 | 0.6426 | 0.4943 | 0.6323 | 0.5922 |
| specialist_routed | 0.7309 | 0.6174 | 0.8929 | 0.9850 | 0.7166 | 0.6524 | 0.5126 | **0.6375** | 0.5678 |
| family_v49 | 0.7242 | 0.6036 | 0.8917 | 0.9850 | 0.6853 | 0.6362 | 0.4845 | 0.6156 | **0.5964** |
| nonlipid_routed | 0.7254 | 0.6172 | 0.8919 | 0.9850 | 0.7080 | 0.6478 | **0.5160** | 0.6374 | 0.5767 |

## Decision

Keep the routing infrastructure, but do **not** promote any of these quick routed variants over the baseline `v_sterol_v2` hierarchy.

Observed pattern:

- Routing can trade objectives cleanly: the full-routed variant improves binary F1 and modestly improves STE F1, but loses too much macro-F1 and lipid macro-F1.
- Routing only the specialist is almost neutral, which means the derived-feature regression is not coming primarily from the STE specialist in this 5-iteration panel.
- Routing the lipid-family head to smaller bases (`v_sterol` or `v49`) hurts CLR/OLA/PLM too much even when it helps STE.
- Routing only the non-lipid head recovers a small OLA bump but still does not beat the baseline overall.

Interpretation: simple stage pruning is not enough. The derived-stack regression in the hierarchy is likely tied to how the stage objectives interact, not just to a single over-wide stage feature matrix.

## Outputs

- `reports/v_sterol_v2_baseline_5/`
- `reports/v_sterol_v2_stage_routed_5/`
- `reports/v_sterol_v2_specialist_routed_5/`
- `reports/v_sterol_v2_family_v49_5/`
- `reports/v_sterol_v2_nonlipid_routed_5/`