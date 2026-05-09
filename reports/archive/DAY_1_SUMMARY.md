# Day 1 summary (template)

_This file is overwritten by `make all` with real numbers pulled from `raw_metrics.parquet`._

## What shipped

- 10-class softmax reformulation of SLiPP over `{ADN, B12, BGC, CLR, COA, MYR, OLA, PLM, PP, STE}`.
- RF, XGB, LGBM heads, class-balanced weighting, 25 stratified shuffle splits.
- Pre-curated training data (`reference/SLiPP_2024-main/training_pockets.csv`) consumed directly; no dpocket rerun required for Day 1.
- Holdouts (apo-PDB, AlphaFold) scored from descriptors shipped in supporting-file xlsx.
- Rule 1 validation gate on ingest (total + per-class counts exact match to paper).

## Headline numbers (filled in after `make all`)

- Test binary F1: _TBD_ (paper: 0.869)
- Test binary AUROC: _TBD_ (paper: 0.970)
- Apo-PDB binary F1: _TBD_ (paper: 0.726)
- AlphaFold binary F1: _TBD_ (paper: 0.643)
- **10-class macro-F1: _TBD_** (new)
- **5-lipid macro-F1: _TBD_** (new headline)

See `reports/metrics_table.md` for the full breakdown and `reports/*.png` for figures.

## Figures

- `confusion_matrix.png` — 10 × 10 row-normalized, counts annotated.
- `per_class_roc.png` — one-vs-rest ROC, lipid classes drawn thicker.
- `pca_colored_by_pred.png` — paper's Fig 2B redone with predicted class coloring.
- `metrics_comparison.png` — Day 1 vs paper Table 1 bar chart with 25-iter error bars.

## What Day 2 unlocks

The 10-class softmax probabilities become a 10-dimensional embedding per pocket. Day 2 concatenates `[17 fpocket descriptors || 10 class probs]` as input to a second-stage classifier with new VdW-resolved descriptors. Day 1's model effectively becomes an embedding layer for the rest of the roadmap.
