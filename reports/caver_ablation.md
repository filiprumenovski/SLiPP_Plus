# CAVER/Tunnel Marginal-Value Ablation

Date: 2026-05-08

## Question

Handoff item 8.6 asks whether CAVER-derived tunnel features add publishable
signal over the no-tunnel baseline.

## Evidence Used

This report consolidates existing persisted ablations; no new model training was
run.

Inputs:

| evidence | source |
|---|---|
| v_sterol-aligned tunnel screen | `reports/v_tunnel_ablation/summary.md` |
| compact family-encoder ladder | `reports/compact_publishable/summary.md` |
| experiment registry | `experiments/registry.yaml` |

## v_sterol-Aligned Screen

The direct v_sterol-aligned screen uses the same 25 canonical splits and
compares v_sterol-only features against compact CAVER/tunnel subsets.

| learner | best tunnel subset | v_sterol-only lipid5 | tunnel lipid5 | delta | macro10 delta | binary F1 effect | interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| LGBM | `tunnel_shape_avail6` | 0.597 +/- 0.028 | 0.608 +/- 0.031 | +0.011 | +0.004 | -0.003 | small positive, mostly CLR/OLA |
| XGB | `tunnel_shape_avail6` | 0.594 +/- 0.028 | 0.609 +/- 0.030 | +0.015 | +0.008 | -0.001 | small positive, mostly CLR/OLA |

The full 18-column tunnel block is not the best variant:

| learner | full tunnel block lipid5 | delta vs v_sterol-only | conclusion |
|---|---:|---:|---|
| LGBM | 0.606 +/- 0.030 | +0.009 | positive but below compact shape subset |
| XGB | 0.600 +/- 0.037 | +0.006 | positive but below compact shape/chem subsets |

## Compact Family-Encoder Ladder

The compact release ladder shows the same pattern at the family-encoder level.

| comparison | lipid5 macro-F1 delta | interpretation |
|---|---:|---|
| `paper17+aa20+tunnel_shape` vs `paper17+aa20` | +0.015 | tunnel lift without shell12 |
| `v49+tunnel_shape` vs `v49` | +0.017 | best single compact tunnel lift |
| `v49+tunnel_shape3` vs `v49+tunnel_shape` | +0.002 | three screened tunnel signals slightly improve internal validation |
| `v_tunnel+moe` vs `v49+tunnel_shape` | -0.001 | high-complexity full tunnel/MoE path does not beat compact tunnel shape |

The latest subset sweeps add a deployment-relevant caveat: tunnel/chem blends
can be tuned for either internal validation or external holdout stability.
`exp-019` is the current internal leader but is holdout-regressive; `exp-021`
is the best holdout-balanced compact blend found so far, but its internal
lipid5 macro-F1 is lower.

## Conclusion

CAVER/tunnel features do help, but the effect is modest and subset-dependent.
The publishable answer is not "use all tunnel descriptors"; it is:

- compact tunnel shape features add about `+0.011` to `+0.017` lipid macro-F1
  over matched no-tunnel baselines;
- the full raw tunnel block is redundant and weaker than screened compact
  subsets;
- gains are concentrated in CLR/OLA and do not solve STE;
- external holdout behavior is not monotonic, so internal leader selection must
  keep holdout tradeoffs visible.

This closes handoff item 8.6 from existing artifacts.
