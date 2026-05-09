# Compact Subset Ensemble Sweep

Equal-probability subset sweep over seven existing compact family-encoder prediction artifacts.

## Key Signal

- Best internal subset remains `shape6+shell6_shape3+hydro4+geom+chem`: lipid5 0.684, apo-PDB 0.649, AlphaFold 0.623.
- Best holdout-balanced subset is `shell6_shape+chem`: lipid5 0.673, apo-PDB 0.717, AlphaFold 0.698.
- `hydro4` and `geom` help internal validation but are associated with weaker holdout F1 in the top internal blends.
- `shell6_shape` is the strongest holdout anchor; pairing it with `chem` beats the current internal leader on both holdouts.

## Top Internal Lipid5

| components | n | lipid5 | binary_f1 | STE | apo_f1 | alphafold_f1 | holdout_mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `shape6+shell6_shape3+hydro4+geom+chem` | 5 | 0.684 | 0.906 | 0.645 | 0.649 | 0.623 | 0.636 |
| `shape3+shell6_shape+shell6_shape3+geom+chem` | 5 | 0.683 | 0.904 | 0.652 | 0.678 | 0.690 | 0.684 |
| `shell6_shape3+hydro4+geom+chem` | 4 | 0.682 | 0.905 | 0.648 | 0.636 | 0.628 | 0.632 |
| `shape3+shape6+shell6_shape3+geom` | 4 | 0.682 | 0.905 | 0.647 | 0.655 | 0.657 | 0.656 |
| `shape3+shell6_shape3+hydro4+geom+chem` | 5 | 0.682 | 0.905 | 0.645 | 0.631 | 0.667 | 0.649 |
| `shape3+shell6_shape+geom+chem` | 4 | 0.682 | 0.905 | 0.651 | 0.661 | 0.681 | 0.671 |
| `shape6+shell6_shape+shell6_shape3+hydro4+geom+chem` | 6 | 0.682 | 0.906 | 0.650 | 0.667 | 0.643 | 0.655 |
| `shape3+shell6_shape+shell6_shape3+hydro4+geom` | 5 | 0.682 | 0.905 | 0.656 | 0.655 | 0.671 | 0.663 |
| `shape3+shape6+shell6_shape3+hydro4+geom+chem` | 6 | 0.682 | 0.906 | 0.646 | 0.643 | 0.633 | 0.638 |
| `shape3+shape6+shell6_shape3+geom+chem` | 5 | 0.682 | 0.906 | 0.645 | 0.667 | 0.662 | 0.664 |
| `shape3+shell6_shape+shell6_shape3+hydro4+geom+chem` | 6 | 0.682 | 0.906 | 0.652 | 0.678 | 0.671 | 0.675 |
| `shape3+shell6_shape+shell6_shape3+geom` | 4 | 0.682 | 0.905 | 0.652 | 0.667 | 0.685 | 0.676 |

## Top Holdout Mean

| components | n | lipid5 | binary_f1 | STE | apo_f1 | alphafold_f1 | holdout_mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `shell6_shape+chem` | 2 | 0.673 | 0.902 | 0.646 | 0.717 | 0.698 | 0.707 |
| `shape3+shell6_shape` | 2 | 0.674 | 0.902 | 0.644 | 0.695 | 0.711 | 0.703 |
| `shape3+shell6_shape+chem` | 3 | 0.675 | 0.903 | 0.646 | 0.706 | 0.693 | 0.700 |
| `shape3+shell6_shape+shell6_shape3` | 3 | 0.677 | 0.904 | 0.648 | 0.712 | 0.685 | 0.698 |
| `shape3+shape6+chem` | 3 | 0.676 | 0.904 | 0.632 | 0.701 | 0.685 | 0.693 |
| `shape6+shell6_shape` | 2 | 0.674 | 0.902 | 0.653 | 0.723 | 0.662 | 0.692 |
| `shape3+shape6+shell6_shape` | 3 | 0.678 | 0.903 | 0.649 | 0.712 | 0.671 | 0.692 |
| `shape3+shell6_shape+shell6_shape3+chem` | 4 | 0.680 | 0.905 | 0.648 | 0.701 | 0.680 | 0.691 |
| `shape6+shell6_shape+chem` | 3 | 0.676 | 0.904 | 0.645 | 0.706 | 0.671 | 0.689 |
| `shape3+shape6+shell6_shape+chem` | 4 | 0.676 | 0.904 | 0.642 | 0.706 | 0.667 | 0.686 |
| `shape3+shape6+shell6_shape+hydro4` | 4 | 0.675 | 0.905 | 0.649 | 0.701 | 0.671 | 0.686 |
| `shape3+shape6+shell6_shape+shell6_shape3+chem` | 5 | 0.679 | 0.904 | 0.646 | 0.701 | 0.671 | 0.686 |

## Top STE

| components | n | lipid5 | binary_f1 | STE | apo_f1 | alphafold_f1 | holdout_mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `shape6+shell6_shape3` | 2 | 0.677 | 0.904 | 0.665 | 0.667 | 0.619 | 0.643 |
| `shell6_shape+shell6_shape3+geom` | 3 | 0.679 | 0.904 | 0.660 | 0.655 | 0.657 | 0.656 |
| `shape6+shell6_shape+shell6_shape3` | 3 | 0.678 | 0.903 | 0.660 | 0.690 | 0.648 | 0.669 |
| `shell6_shape3+hydro4+geom` | 3 | 0.680 | 0.904 | 0.660 | 0.649 | 0.637 | 0.643 |
| `shape6+shell6_shape3+hydro4` | 3 | 0.681 | 0.905 | 0.659 | 0.649 | 0.629 | 0.639 |
| `shape6+shell6_shape+shell6_shape3+hydro4` | 4 | 0.681 | 0.905 | 0.659 | 0.678 | 0.648 | 0.663 |
| `shape3+shape6+shell6_shape3+hydro4` | 4 | 0.681 | 0.905 | 0.659 | 0.655 | 0.657 | 0.656 |
| `shape6+shell6_shape+shell6_shape3+geom` | 4 | 0.678 | 0.905 | 0.657 | 0.655 | 0.633 | 0.644 |
