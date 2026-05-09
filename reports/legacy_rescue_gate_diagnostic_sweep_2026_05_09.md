# Legacy Rescue Gate Diagnostic Sweep, 2026-05-09

## Question

Can alternate logistic-gate feature sets, rewrite modes, or thresholds beat the
current script-backed deployable gate (`exp-035`)?

## Sweep

This was a diagnostic sweep over saved prediction artifacts only. It inspected
holdout labels to rank candidates, so its winner is not deployable by itself.

- Gate feature sets: `compact`, `simple`, `rich`, `no_base`
- Logistic `C`: `0.03`, `0.1`, `0.3`, `1`, `3`, `10`, `30`
- Rewrite modes: `mean`, `paper17`, `vsterol`, `maxlegacy`, `blend_base25`,
  `blend_base50`
- Thresholds: `0.90`, `0.95`, `0.99`

Full sweep rows are in `reports/legacy_rescue_gate_diagnostic_sweep.csv`.

## Best Diagnostic Candidate

`compact`, `C=0.3`, `rewrite_mode=maxlegacy`, `threshold=0.90`.

| metric | value |
|---|---:|
| internal binary F1 | `0.895` |
| internal lipid5 macro-F1 | `0.658` |
| internal fire rate | `2.9%` |
| apo-PDB F1/AUROC | `0.756 / 0.791` |
| apo-PDB fire rate | `17.9%` |
| AlphaFold F1/AUROC | `0.807 / 0.829` |
| AlphaFold fire rate | `20.1%` |

The first-class rerun at the default script `C=1.0` gives the same top-line
holdout F1s:

| metric | value |
|---|---:|
| internal binary F1 | `0.895 +/- 0.015` |
| internal lipid5 macro-F1 | `0.658` |
| apo-PDB F1/AUROC | `0.756 / 0.791` |
| AlphaFold F1/AUROC | `0.807 / 0.829` |

## Decision

Record as `exp-037-legacy-rescue-maxlegacy-diagnostic`.

This beats `exp-035` externally but is not deployable yet because the
`maxlegacy` rewrite and `0.90` threshold were found by a holdout-scored
diagnostic sweep. The next useful ablation is an internal-only selection rule
that can choose this more recall-heavy rewrite policy without looking at
holdout labels.
