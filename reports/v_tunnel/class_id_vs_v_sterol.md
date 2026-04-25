# v_tunnel vs v_sterol — class ID check

Question: did the current `v_tunnel` build improve lipid subclass identification, or did it only preserve binary lipid-vs-nonlipid performance?

Source tables:
- `reports/v_tunnel/metrics_table.md`
- `reports/v_sterol/metrics_table.md`

All values below are test-split mean F1 across 25 iterations.

## Headline

`v_tunnel` did **not** deliver a broad class-ID gain over `v_sterol`.

- Binary detection was slightly worse for the boosted models.
- Overall macro-F1 was flat to slightly worse.
- A few subclass boundaries improved, mainly `CLR` and sometimes `STE` / `OLA`.
- Those gains were offset by losses in `MYR` and `PLM`.

The honest read is that the current tunnel feature table changes where the model wins and loses, but it does not improve subclass identification as a whole.

## Aggregate deltas

`Δ = v_tunnel - v_sterol`

| model | macro-F1 (10) | macro-F1 (5 lipids) | binary F1 | binary AUROC |
|---|---|---|---|---|
| rf | -0.011 | -0.008 | -0.007 | -0.001 |
| xgb | +0.000 | -0.001 | -0.004 | +0.000 |
| lgbm | -0.002 | -0.001 | -0.004 | -0.001 |

Interpretation:
- No model shows a meaningful gain in lipid-only macro-F1.
- `xgb` is effectively flat overall, not better.
- `rf` regresses.
- `lgbm` regresses slightly despite a few local per-class gains.

## Per-class F1 deltas

`Δ = v_tunnel - v_sterol`

| model | ADN | B12 | BGC | CLR | COA | MYR | OLA | PLM | PP | STE |
|---|---|---|---|---|---|---|---|---|---|---|
| rf | -0.007 | -0.024 | -0.022 | -0.018 | -0.017 | -0.001 | -0.016 | -0.006 | -0.003 | +0.000 |
| xgb | +0.003 | -0.006 | +0.009 | +0.019 | -0.002 | -0.023 | +0.008 | -0.005 | +0.001 | -0.005 |
| lgbm | -0.004 | -0.001 | -0.009 | +0.008 | +0.000 | -0.021 | +0.008 | -0.016 | +0.001 | +0.017 |

## Lipid-only view

This is the part that matters for subclass ID.

| model | CLR | MYR | OLA | PLM | STE | lipid macro-F1 Δ |
|---|---|---|---|---|---|---|
| rf | -0.018 | -0.001 | -0.016 | -0.006 | +0.000 | -0.008 |
| xgb | +0.019 | -0.023 | +0.008 | -0.005 | -0.005 | -0.001 |
| lgbm | +0.008 | -0.021 | +0.008 | -0.016 | +0.017 | -0.001 |

Interpretation:
- `CLR` is the clearest winner: `xgb +0.019`, `lgbm +0.008`.
- `STE` improves only for `lgbm`, and by a moderate amount (`+0.017`).
- `OLA` improves slightly for both boosted models (`+0.008`).
- `MYR` gets worse for both boosted models (`xgb -0.023`, `lgbm -0.021`).
- `PLM` also weakens, especially for `lgbm` (`-0.016`).

So yes, there is some gain in class ID, but it is narrow and class-specific rather than global.

## Strongest movements

Largest positive subclass-ID shifts:

1. `xgb CLR`: `+0.019`
2. `lgbm STE`: `+0.017`
3. `lgbm CLR`: `+0.008`
4. `xgb OLA`: `+0.008`
5. `lgbm OLA`: `+0.008`

Largest negative subclass-ID shifts:

1. `xgb MYR`: `-0.023`
2. `lgbm MYR`: `-0.021`
3. `rf CLR`: `-0.018`
4. `lgbm PLM`: `-0.016`
5. `rf OLA`: `-0.016`

## Bottom line

If the question is "did tunnel features help identify lipid subclasses better?", the answer is:

- **Not overall.**
- **Yes locally**, mostly for `CLR`, with some support for `STE` and `OLA` in specific boosted models.
- The current `v_tunnel` build does **not** justify a claim of general class-ID improvement because the gains are canceled by `MYR` and `PLM` regressions.

This makes the current `v_tunnel` result useful as a diagnostic signal, not as a winning feature set.