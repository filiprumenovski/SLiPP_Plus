# Covariate-Shift Threshold Ablation, 2026-05-09

## Question

Can unlabeled holdout probability distributions make the binary threshold
selection transfer better than plain internal validation?

This is a holdout-safe domain-adaptation diagnostic: holdout labels are not used
to select thresholds.

## Method

For each holdout independently:

1. Use exp-028 prediction probabilities as a shared feature space.
2. Train a source-vs-target domain classifier to distinguish internal split rows
   from unlabeled holdout rows.
3. Convert internal source rows to density-ratio-like weights using
   `p(target | x) / p(source | x)`, clipped to `[0.05, 20.0]`.
4. Select the binary lipid threshold that maximizes weighted internal F1.
5. Apply that selected threshold once to the target holdout.

Two source scopes were checked:

- `all25`: all 25 internal split prediction rows.
- `iter0`: iteration-0 internal test rows only, matching the iteration-0 holdout
  prediction artifact.

## Results

Unweighted internal selection chooses threshold `0.54` with internal binary F1
`0.904`.

| holdout | source scope | domain AUROC | selected threshold | weighted internal F1 | holdout F1/AUROC |
|---|---|---:|---:|---:|---:|
| apo-PDB | all25 | `0.932` | `0.51` | `0.838` | `0.667 / 0.733` |
| apo-PDB | iter0 | `0.925` | `0.50` | `0.755` | `0.667 / 0.733` |
| AlphaFold | all25 | `0.967` | `0.51` | `0.801` | `0.605 / 0.661` |
| AlphaFold | iter0 | `0.951` | `0.48` | `0.688` | `0.605 / 0.661` |

## Interpretation

The high domain AUROCs confirm that exp-028 probability space contains strong
source-vs-holdout shift. However, using that shift to reweight internal rows
does not select the low thresholds that helped in the holdout-only diagnostic.

The selected thresholds remain near `0.5`, and target F1 falls below the
deployable exp-028 default (`0.717` apo-PDB, `0.724` AlphaFold). The weighted
source rows still do not behave like labeled target lipids; external misses are
not solved by probability-space covariate reweighting alone.

## Decision

Record as `exp-033-covariate-shift-threshold-negative`.

Do not spend more time on probability-only threshold adaptation unless it is
paired with a richer target representation or an explicit external-data
extension. The domain classifier is useful as evidence of shift, not as a
threshold-selection mechanism.
