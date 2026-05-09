# Domain-Shift Component Rescue Ablation

This diagnostic asks whether any individual compact component assigns high lipid probability to `exp-019` holdout false negatives. If a component rescues many false negatives at `p_lipid >= 0.5`, a gated ensemble might recover transfer; if none do, the miss is shared across the component family.

## Component-Level Rescue Rates

| holdout | component | FN mean p_lipid | TP mean p_lipid | FN >=0.5 | FN >=0.4 | FN >=0.3 |
|---|---|---:|---:|---:|---:|---:|
| alphafold | `shape3` | 0.353 | 0.778 | 0.250 | 0.438 | 0.604 |
| alphafold | `chem` | 0.313 | 0.788 | 0.208 | 0.312 | 0.521 |
| alphafold | `shell6_shape` | 0.336 | 0.782 | 0.188 | 0.375 | 0.583 |
| alphafold | `shell6_shape3` | 0.295 | 0.760 | 0.104 | 0.250 | 0.500 |
| alphafold | `hydro4` | 0.257 | 0.752 | 0.042 | 0.208 | 0.375 |
| alphafold | `shape6` | 0.250 | 0.757 | 0.042 | 0.208 | 0.312 |
| alphafold | `geom` | 0.232 | 0.706 | 0.000 | 0.104 | 0.292 |
| apo_pdb | `shell6_shape` | 0.345 | 0.864 | 0.226 | 0.452 | 0.548 |
| apo_pdb | `chem` | 0.281 | 0.867 | 0.226 | 0.290 | 0.484 |
| apo_pdb | `shape6` | 0.257 | 0.853 | 0.161 | 0.226 | 0.355 |
| apo_pdb | `shape3` | 0.277 | 0.843 | 0.129 | 0.323 | 0.484 |
| apo_pdb | `geom` | 0.232 | 0.792 | 0.065 | 0.129 | 0.419 |
| apo_pdb | `hydro4` | 0.230 | 0.818 | 0.065 | 0.194 | 0.387 |
| apo_pdb | `shell6_shape3` | 0.232 | 0.841 | 0.032 | 0.194 | 0.355 |

## Key Signal

- No component rescues a large share of `exp-019` false negatives at the deployable `0.5` threshold.
- `shell6_shape` and `chem` are the best rescue components on apo-PDB, matching the holdout-balanced ensemble result.
- On AlphaFold, rescue is diffuse and weak: even the best components rescue only about one in five false negatives at `0.5`, so the transfer gap is shared across compact variants.

## apo_pdb: Most Rescuable exp-019 False Negatives

| row | exp019_p | best_component | best_component_p |
|---:|---:|---|---:|
| 42 | 0.466 | `shell6_shape` | 0.720 |
| 19 | 0.491 | `shell6_shape` | 0.649 |
| 27 | 0.367 | `shell6_shape` | 0.640 |
| 96 | 0.416 | `shell6_shape` | 0.633 |
| 22 | 0.415 | `shape6` | 0.616 |
| 72 | 0.465 | `shape3` | 0.614 |
| 64 | 0.469 | `shape3` | 0.597 |
| 41 | 0.420 | `geom` | 0.554 |
| 100 | 0.327 | `chem` | 0.548 |
| 73 | 0.399 | `hydro4` | 0.533 |
| 91 | 0.425 | `shape6` | 0.531 |
| 88 | 0.294 | `shape3` | 0.494 |
| 1 | 0.336 | `shell6_shape` | 0.493 |
| 84 | 0.299 | `shell6_shape` | 0.472 |
| 52 | 0.299 | `shape3` | 0.443 |

## alphafold: Most Rescuable exp-019 False Negatives

| row | exp019_p | best_component | best_component_p |
|---:|---:|---|---:|
| 6 | 0.444 | `shape3` | 0.758 |
| 114 | 0.457 | `shape3` | 0.711 |
| 119 | 0.465 | `shape3` | 0.663 |
| 122 | 0.452 | `shell6_shape3` | 0.657 |
| 33 | 0.477 | `shell6_shape` | 0.651 |
| 65 | 0.398 | `chem` | 0.644 |
| 62 | 0.446 | `shape3` | 0.629 |
| 42 | 0.392 | `chem` | 0.628 |
| 43 | 0.498 | `chem` | 0.617 |
| 57 | 0.322 | `shape3` | 0.604 |
| 7 | 0.380 | `shape3` | 0.592 |
| 128 | 0.376 | `chem` | 0.579 |
| 85 | 0.337 | `shape3` | 0.533 |
| 13 | 0.345 | `chem` | 0.532 |
| 71 | 0.274 | `shell6_shape` | 0.529 |
