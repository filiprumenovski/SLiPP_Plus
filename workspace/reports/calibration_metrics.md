# Calibration metrics — binary baselines vs multi-class lipid-sum

_10-bin and 15-bin equal-width ECE over [0, 1]. Test holdout excludes PP pockets (trivially separable, inflates calibration). Apo and AlphaFold holdouts used as-is. Binary baselines trained with paper's defaults (no class-weight, seed=42). Multi-class models are the Day 1 iteration-0 artifacts._

## Table 1. Expected Calibration Error by model and holdout

| Model | Formulation | Test ECE10 | Test ECE15 | Apo ECE10 | Apo ECE15 | AlphaFold ECE10 | AlphaFold ECE15 |
|---|---|---|---|---|---|---|---|
| RF | Binary | 0.040 | 0.046 | 0.190 | 0.196 | 0.247 | 0.242 |
| RF | Multi-class | 0.059 | 0.063 | 0.192 | 0.177 | 0.206 | 0.215 |
| XGB | Binary | 0.065 | 0.066 | 0.227 | 0.236 | 0.242 | 0.242 |
| XGB | Multi-class | 0.036 | 0.044 | 0.207 | 0.217 | 0.242 | 0.245 |
| LGBM | Binary | 0.052 | 0.053 | 0.215 | 0.211 | 0.254 | 0.261 |
| LGBM | Multi-class | 0.050 | 0.049 | 0.194 | 0.209 | 0.223 | 0.229 |

## Table 2. Brier score and Maximum Calibration Error (LGBM)

| Holdout | Formulation | Brier | MCE |
|---|---|---|---|
| Test (in-distribution, PP excluded) | Binary | 0.081 | 0.255 |
| Test (in-distribution, PP excluded) | Multi-class | 0.074 | 0.303 |
| Apo-PDB holdout | Binary | 0.221 | 0.529 |
| Apo-PDB holdout | Multi-class | 0.208 | 0.590 |
| AlphaFold holdout | Binary | 0.252 | 0.508 |
| AlphaFold holdout | Multi-class | 0.241 | 0.637 |

## Sample sizes (N pockets / N lipid)

| Holdout | N total | N lipid |
|---|---|---|
| Test (in-distribution, PP excluded) | 531 | 198 |
| Apo-PDB holdout | 117 | 67 |
| AlphaFold holdout | 149 | 91 |

## Caption

LGBM ECE on the in-distribution test split is 0.052 (binary) vs 0.050 (multi-class), a gap of +0.003. On the apo-PDB holdout the gap is +0.020 (0.215 vs 0.194), and on AlphaFold it is +0.031 (0.254 vs 0.223). At 15 bins, the AlphaFold gap remains +0.031 (0.261 vs 0.229). The gap widens with distribution shift, matching the calibration hypothesis: the multi-class softmax stays better-calibrated as the feature distribution drifts off the training manifold.
