# Compact Weight Local Refinement

Question: after promoting `exp-028`, does a nearby 0.05-resolution weighted
blend over the seven compact components beat it?

## Search

Starting point:

- `shape3`: `0.10`
- `shell6_shape`: `0.20`
- `chem`: `0.70`

Local search constraints:

- Multiples of `0.05`.
- `shape3`, `shell6_shape`, and `chem` remain present.
- `chem >= 0.45`.
- Optional combined mass for `shape6`, `shell6_shape3`, `hydro4`, and `geom`
  capped at `0.25`.
- Candidate must reach holdout mean at least `0.716` before internal metrics are
  computed.

## Result

Only the already-promoted `0.10 / 0.20 / 0.70` `shape3 / shell6_shape / chem`
blend survived the holdout threshold:

| blend | weights | lipid5 | binary F1 | STE | apo-PDB F1 | AlphaFold F1 | holdout mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `shape3 + shell6_shape + chem` | `0.10 / 0.20 / 0.70` | `0.670` | `0.903` | `0.644` | `0.717` | `0.724` | `0.720` |

## Decision

Closed as a negative refinement. The current exp-028 weights are locally stable
under this bounded 0.05 grid; adding small mass from `shape6`, `shell6_shape3`,
`hydro4`, or `geom` did not improve the holdout-balanced objective.
