# v_sterol holdout validation — ensemble + PLM/STE tiebreaker

_Iteration-0 holdout validation using the saved v_sterol RF/XGB/LGBM multiclass bundles, mean-probability ensembling, and a PLM-vs-STE binary head trained on seed_00.parquet with margin < 0.99._

| Holdout | Condition | N | N lipid | F1 | AUROC | precision | sensitivity | specificity | ΔF1 vs base | ΔAUROC vs base |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| apo_pdb | ensemble | 117 | 67 | 0.679 | 0.812 | 0.844 | 0.567 | 0.860 | +0.000 | +0.000 |
| apo_pdb | ensemble + plm_ste_tiebreaker | 117 | 67 | 0.679 | 0.812 | 0.844 | 0.567 | 0.860 | +0.000 | +0.000 |
| alphafold | ensemble | 149 | 91 | 0.708 | 0.864 | 0.962 | 0.560 | 0.966 | +0.000 | +0.000 |
| alphafold | ensemble + plm_ste_tiebreaker | 149 | 91 | 0.708 | 0.864 | 0.962 | 0.560 | 0.966 | +0.000 | +0.000 |

_Binary holdout F1/AUROC are invariant here by construction: the PLM/STE arbiter only redistributes mass within lipid subclasses, so summed lipid probability is unchanged. The discriminating holdout check is the pair-only PLM/STE section below._

## PLM/STE pair-only holdout metrics

| Holdout | Condition | Pair rows | PLM F1 | STE F1 | PLM correct | STE correct | PLM→STE | STE→PLM | Fired rows |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| apo_pdb | ensemble | 18 | 0.750 | 0.400 | 9 | 1 | 0 | 1 | 0 |
| apo_pdb | ensemble + plm_ste_tiebreaker | 18 | 0.750 | 0.400 | 9 | 1 | 0 | 1 | 1 |
| alphafold | ensemble | 26 | 0.485 | 0.000 | 8 | 0 | 0 | 2 | 0 |
| alphafold | ensemble + plm_ste_tiebreaker | 26 | 0.485 | 0.000 | 8 | 0 | 0 | 2 | 1 |

## Interpretation

On the pair-only holdout rows, the PLM/STE tiebreaker does not reduce STE F1 on either holdout.

Binary metrics are listed for completeness and remain unchanged (apo ΔF1=+0.000, AlphaFold ΔF1=+0.000); the real signal is STE F1 on pair rows (apo Δ=+0.000, AlphaFold Δ=+0.000).
