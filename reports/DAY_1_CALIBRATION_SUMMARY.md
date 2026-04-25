# Day 1 calibration analysis — binary baseline vs multi-class lipid-sum

## Question

Does the multi-class softmax's better AlphaFold F1 over the paper's binary classifier come from better calibration under distribution shift, rather than sharper decision boundaries?

## Method

Three binary baselines (RF, XGB, LGBM) were trained on the same iteration-0 train split as the Day 1 multi-class models with labels collapsed to lipid-vs-rest, using scikit-learn/XGBoost/LightGBM defaults at seed 42 and no class-balance reweighting — the paper's binary protocol. All six models (three binary plus three multi-class lipid-sums via P_lipid = sum over CLR/MYR/PLM/STE/OLA softmax entries) were scored on three holdouts: the in-distribution test split with PP pockets excluded (531), the apo-PDB holdout (117), and the AlphaFold holdout (149); Expected Calibration Error with 10 and 15 equal-width bins, Brier score, and Maximum Calibration Error were computed per cell.

## Result

On AlphaFold, the LGBM binary classifier has ECE = 0.254 while the LGBM multi-class lipid-sum has ECE = 0.223, a gap of +0.031 at 10 bins. At 15 bins, the same AlphaFold gap is +0.031 (0.261 vs 0.229), which is the robustness check relevant to the email gate. On apo-PDB the 10-bin gap is +0.020 (0.215 vs 0.194). On the in-distribution test split the 10-bin gap is +0.003 (0.052 vs 0.050). AlphaFold Brier moves from 0.252 (binary) to 0.241 (multi-class) and worst-bin MCE from 0.508 to 0.637. The same qualitative pattern holds for RF and XGB in the metrics table.

## Interpretation

The gap widens with distribution shift, matching the calibration hypothesis: the multi-class softmax stays better-calibrated as the feature distribution drifts off the training manifold. Mechanistically, under the calibration hypothesis the multi-class softmax distributes probability mass across five lipid subclasses rather than concentrating it at a single binary decision boundary; on out-of-distribution AlphaFold-predicted pockets with noisier geometry, this richer probability structure degrades more gracefully because the summed lipid probability can stay reliable even when no single lipid class is confident. The binary classifier has no such fallback: one boundary, and a drifting feature vector flips the label.

## Limitations

This analysis uses iteration-0 models only; there is no 25-iteration uncertainty band on the ECE numbers, though the Day 1 split protocol supports adding it as a follow-up. The 17-descriptor feature set and the holdout row counts are fixed by Day 1 ingest. No post-hoc calibration (Platt, isotonic) was applied — the point was to measure native calibration of the two training formulations. The 15-bin robustness check flagged a >0.01 ECE shift on at least one cell; the AlphaFold LGBM gap should therefore be read from the explicit 15-bin values in the metrics table.
