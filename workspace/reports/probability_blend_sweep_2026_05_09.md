# Probability Blend Sweep, 2026-05-09

## Question

Can saved prediction artifacts be recombined to beat the current deployable
recommendation, `exp-028-compact-shape3-shell6-chem-weighted`, without new
training?

Current deployable anchor:

| metric | exp-028 |
|---|---:|
| binary F1 | `0.903 +/- 0.016` |
| binary AUROC | `0.989 +/- 0.003` |
| macro10 F1 | `0.769 +/- 0.019` |
| lipid5 macro-F1 | `0.670 +/- 0.032` |
| apo-PDB F1/AUROC | `0.717 / 0.801` |
| AlphaFold F1/AUROC | `0.724 / 0.855` |

## Search

Two cheap probability-space sweeps were run over existing artifacts only:

1. Anchor blends: `w * exp-028 + (1 - w) * candidate`, `w in {0.00, 0.05, ..., 1.00}`.
2. Pairwise saved-artifact blends across 21 compatible prediction sets, followed
   by a focused four-component refinement over:
   `compact_shape3_shape6_ensemble`, `v_sterol`, `exp-028`, and
   `compact_shape6_shell6shape3_hydro4_geom_chem_ensemble`.

No models were retrained.

## Best Internal Candidate

The strongest internal-validation blend found was:

`0.20*compact_shape3_shape6_ensemble + 0.20*v_sterol + 0.10*exp-028 + 0.50*compact_shape6_shell6shape3_hydro4_geom_chem_ensemble`

| metric | value |
|---|---:|
| binary F1 | `0.908 +/- 0.015` |
| binary AUROC | `0.990 +/- 0.003` |
| macro10 F1 | `0.781 +/- 0.018` |
| lipid5 macro-F1 | `0.687 +/- 0.031` |
| CLR F1 | `0.778` |
| MYR F1 | `0.713` |
| OLA F1 | `0.632` |
| PLM F1 | `0.659` |
| STE F1 | `0.652` |
| apo-PDB F1/AUROC | `0.643 / 0.766` |
| AlphaFold F1/AUROC | `0.536 / 0.738` |

This is a new internal-validation leader versus exp-019
(`lipid5 0.684`, `macro10 0.778`, `binary F1 0.906`) and exp-028
(`lipid5 0.670`, `macro10 0.769`, `binary F1 0.903`).

It is not deployable: both external holdouts collapse, especially AlphaFold.
This reinforces the existing pattern that internal lipid macro-F1 can be
improved by adding mass from the old five-way internal leader, but that signal
does not transfer externally.

## Best Balanced Holdout Diagnostic

Among candidates with at least `0.650` internal lipid5 macro-F1, the strongest
holdout-mean blend was:

`0.35*paper17_family_encoder + 0.65*v_sterol`

| metric | value |
|---|---:|
| binary F1 | `0.901 +/- 0.015` |
| binary AUROC | `0.989 +/- 0.003` |
| macro10 F1 | `0.759 +/- 0.015` |
| lipid5 macro-F1 | `0.657 +/- 0.028` |
| STE F1 | `0.626` |
| apo-PDB F1/AUROC | `0.739 / 0.808` |
| AlphaFold F1/AUROC | `0.711 / 0.863` |
| holdout mean F1 | `0.725` |

This beats exp-028 on apo-PDB F1 and holdout mean, but it loses AlphaFold F1
and internal lipid5 macro-F1. It is useful as a domain-shift clue, not a
replacement: the legacy paper17 encoder carries an external-recall signal that
is mostly incompatible with the best internal subclass model.

## Decision

Do not replace exp-028 as the deployable recommendation.

Record `exp-030-probability-blend-internal-leader` as an internal-positive,
holdout-negative ablation. The next high-impact path should not be more blind
probability averaging; it should explain why legacy paper17/v_sterol blends
recover apo-PDB while compact internal leaders fail on AlphaFold.
