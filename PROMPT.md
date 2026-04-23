# SLiPP++ Day 1: Multi-Class Reformulation

**Author:** Filip Rumenovski
**Date:** 2026-04-23
**Status:** Scope-locked. Day 1 of a four-week roadmap.
**Target:** Claude Code / Codex / any coding agent with bash + python.

---

## 1. Objective

Refit the SLiPP classifier from Chou et al. 2024 (bioRxiv 2024.01.26.577452 v3) on their exact curated dataset as a **10-class softmax problem** instead of the published binary lipid-vs-rest formulation. No new features, no new pocket detector, no new training data. Only the loss head changes.

The hypothesis: the 17 fpocket descriptors carry lipid subclass signal that the binary loss discards. The published PCA in their Fig 2B shows CLR, MYR, PLM, STE, OLA occupy overlapping but distinct regions of descriptor space. A multi-class softmax exposes that structure as a supervised signal instead of leaving it in the residual.

The deliverable: a reproducible CLI that produces (a) a confusion matrix resolving the 5 lipid subclasses, (b) a binary-collapsed F1 directly comparable to their Table 1, (c) per-class ROC curves, (d) a short writeup for Dassama suitable for same-day outreach.

## 2. Why this is Day 1 and not Day 0 or Day 7

Day 0 would be a weekend of reading. Skip.

Day 7 would be VdW feature engineering plus LJ probe maps. That is a meaningful methods contribution but needs real implementation time and cannot be delivered as a "I thought about this overnight" signal to Dassama.

Day 1 is the free win. Zero new physics, zero new code for feature extraction, zero new data. The entire change fits in the `fit()` call of an sklearn estimator. It is specifically the kind of change that demonstrates reading-with-intent and extracts a result in hours.

## 3. Scope

**IN:**
- Regenerate the 83,807-pocket dataset from Supporting Files 1, 2, 3 of the preprint using fpocket/dpocket per the paper's Methods.
- Assign 10-class labels: `{ADN, B12, BGC, COA, CLR, MYR, PLM, STE, OLA, PP}` where PP denotes pseudo-pocket.
- Train three multi-class classifiers: RandomForest (matches paper), XGBoost, LightGBM.
- Evaluate on (a) the 10% stratified test split matching the paper's protocol, (b) the 131-protein apo PDB holdout (Supporting File 2), (c) the 177-protein AlphaFold holdout (Supporting File 3).
- Report binary-collapsed metrics parallel to their Table 1, plus macro-F1, per-class F1, and full confusion matrix.
- Run 25 stratified shuffle split iterations to match their uncertainty quantification.

**OUT:**
- Any new feature beyond the 17 dpocket descriptors.
- Replacing fpocket with P2Rank, DeepPocket, or anything else.
- Hyperparameter tuning beyond sklearn defaults (their paper found defaults win).
- Training set expansion.
- Any GNN, transformer, or learned-representation architecture.
- Rewriting the pocket detection step.

Anything on the OUT list is Day 2 through Day 28. Do not scope-creep.

## 4. Data sources

1. **Supporting Files 1, 2, 3** from bioRxiv 2024.01.26.577452 v3 (May 26, 2025 posting). These contain the PDB IDs for ligand-bound training structures, apo PDB holdout, and AlphaFold holdout respectively. Download from the preprint supplementary.
2. **PDB structures** fetched via RCSB REST API or `Bio.PDB.PDBList` using the IDs from Supporting File 1.
3. **AlphaFold models** fetched from the AlphaFold DB (`https://alphafold.ebi.ac.uk`) using UniProt IDs from Supporting File 3.
4. **Reference code** at https://github.com/dassamalab/SLiPP_2024. Use as sanity check for pocket extraction parameters. Do not depend on their training scripts; reimplement cleanly.

If Supporting Files are PDFs or awkwardly formatted, parse with pandas/tabula and validate with Pandera before proceeding.

## 5. The 17 dpocket descriptors

Keep these exact names, exact order:

```
pock_vol, nb_AS, surf_vdw, surf_pol_vdw, surf_apol_vdw,
hydrophobicity_score, mean_loc_hyd_dens, apol_as_prop, prop_polar_atm,
mean_as_ray, mean_as_solv_acc, as_density, as_max_dst,
volume_score, polarity_score, charge_score, flex
```

These are the columns output by `dpocket` when run on a ligand-bound PDB structure. Confirm column names in the raw dpocket output; some fpocket versions use `as_density` vs `as_dens` and `mean_as_ray` vs `mean_as_rad`. Standardize on the names in the paper.

## 6. Pipeline

### 6.1 Environment

```bash
# System
fpocket >= 4.0  # install from source: https://github.com/Discngine/fpocket
conda install -c bioconda fpocket  # alternate

# Python 3.11
pip install \
  polars==1.* \
  pandera==0.20.* \
  pydantic==2.* \
  scikit-learn==1.3.1 \  # match paper exactly
  xgboost==2.* \
  lightgbm==4.* \
  biopython==1.83 \
  matplotlib==3.8.* \
  seaborn==0.13.* \
  tqdm \
  pyyaml
```

Pin scikit-learn to 1.3.1 to match the paper. Do not use 1.5+; RF defaults changed.

### 6.2 Directory structure

```
slipp_plus/
├── configs/
│   └── day1.yaml              # paths, seeds, model hyperparams
├── data/
│   ├── raw/
│   │   ├── supporting_file_1.csv
│   │   ├── supporting_file_2.csv
│   │   └── supporting_file_3.csv
│   ├── structures/             # downloaded PDBs
│   ├── alphafold/              # downloaded AF models
│   └── pockets/                # dpocket outputs
├── processed/
│   ├── train_pockets.parquet   # 17 descriptors + 10-class label + metadata
│   ├── test_pockets.parquet
│   ├── apo_pdb_holdout.parquet
│   └── alphafold_holdout.parquet
├── models/
│   ├── rf_multiclass.joblib
│   ├── xgb_multiclass.joblib
│   └── lgbm_multiclass.joblib
├── reports/
│   ├── metrics_table.md        # parallels paper's Table 1
│   ├── confusion_matrix.png
│   ├── per_class_roc.png
│   └── pca_colored_by_pred.png
├── src/
│   ├── download.py             # fetch PDBs + AF models from Supporting Files
│   ├── pocket_extraction.py    # run dpocket, parse 17 descriptors
│   ├── labeling.py             # assign 10-class labels from ligand HETATM codes
│   ├── schemas.py              # Pandera + Pydantic validation
│   ├── train.py                # stratified CV, 25 iterations, 3 models
│   ├── evaluate.py             # binary collapse, per-class, confusion matrix
│   └── figures.py              # all plots
├── CLAUDE.md                   # this file, condensed
└── Makefile                    # one-shot reproduction: `make all`
```

### 6.3 Step-by-step

**Step 1: Ingest Supporting Files.**
Parse Supporting Files 1-3 into Polars DataFrames. Validate with Pandera schema requiring columns `pdb_id`, `ligand_code` (3-letter HETATM), `chain_id` (where present). Write to `data/raw/*.csv` canonical form.

**Step 2: Download structures.**
For each row in Supporting File 1, fetch the PDB with `Bio.PDB.PDBList().retrieve_pdb_file()`. Retry 3 times on failure. Skip IDs that 404. Log to `reports/download_log.csv`.

For Supporting File 3, fetch AlphaFold models via `https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb`.

**Step 3: Run dpocket.**
For each ligand-bound structure, run dpocket with the ligand selection specified by the 3-letter code. Parse the `dpocket_fpocketnpocket.txt` output (one row per detected pocket). Classify each row as:
- `ligand_pocket` if it overlaps the ligand COM within 8 Å
- `pseudo_pocket` otherwise

Apply the paper's surface-binding filter: exclude ligand pockets where fewer than 10 residues lie within 8 Å of the ligand COM.

Save to `data/pockets/{pdb_id}_{ligand_code}.parquet` with the 17 descriptors and a class label.

**Step 4: Assign 10-class labels.**
```python
LIPID_CODES = {"CLR", "MYR", "PLM", "STE", "OLA"}
NONLIPID_CODES = {"ADN", "B12", "BGC", "COA"}
# PP assigned at Step 3 for pseudo-pockets
```

Label column: `class_10` takes one of 10 string values. For binary-collapse evaluation, also compute `class_binary = 1 if class_10 in LIPID_CODES else 0`.

**Step 5: Validation layer (Rule 1).**
Before any model sees the data, assert with Pandera:
- Every row has exactly 17 numeric descriptor columns with no NaN.
- Every row has a valid `class_10` label from the 10-value vocabulary.
- Total row count is within 5 percent of 83,807 (paper's reported total).
- Lipid pocket count is within 5 percent of 1,981.
- Non-lipid pocket count is within 5 percent of 3,333.
- PP count is within 5 percent of 78,481 (from their text; paper also says 90,232 at a different stage, reconcile in code comments).

If validation fails, abort. Do not proceed to training with a mismatched dataset.

**Step 6: Train/test split.**
Stratified shuffle split 90:10 on `class_10` (not on binary) to preserve class proportions across splits. Seed the split. Run 25 independent iterations matching the paper. Persist all splits to `processed/splits/seed_{i}.parquet` for reproducibility.

**Step 7: Train.**
For each of 25 iterations, for each of 3 model classes:
- `RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)` with sklearn defaults
- `XGBClassifier(objective="multi:softprob", num_class=10, random_state=seed)`
- `LGBMClassifier(objective="multiclass", num_class=10, random_state=seed)`

Class imbalance: use `class_weight="balanced"` for RF and LightGBM; use `sample_weight` computed via `sklearn.utils.class_weight.compute_sample_weight("balanced", y)` for XGBoost.

Store fitted models only for the first iteration (for later interpretation). Store metrics for all 25.

**Step 8: Evaluate.**

For each iteration and each model, compute on the test split:
- Per-class: precision, recall, F1, support.
- Macro-F1 across all 10 classes.
- Macro-F1 across 5 lipid classes only (the headline multi-class number).
- Binary-collapsed metrics: AUROC, accuracy, F1, sensitivity, specificity, precision. Collapse rule: `pred_binary = 1 if predicted class in LIPID_CODES else 0`.
- Full 10 by 10 confusion matrix.

Aggregate across 25 iterations as mean +/- std for each metric.

Apply the two holdouts (apo PDB, AlphaFold) only with the first-iteration model. Report binary metrics to parallel their Table 1.

**Step 9: Figures.**

- `confusion_matrix.png`: normalized 10 by 10 heatmap, log scale for readability, diagonal emphasized.
- `per_class_roc.png`: one-vs-rest ROC curves for all 10 classes on one axes.
- `pca_colored_by_pred.png`: redo their Fig 2B but color by predicted class. Expected outcome: visible subclass islands where their plot showed overlapping clouds.
- `metrics_comparison.png`: bar chart comparing their Table 1 numbers to the binary-collapsed multi-class numbers. Include error bars from the 25-iteration CV.

**Step 10: Report.**

Write `reports/metrics_table.md` with three sections:
1. Binary-collapsed metrics on their test set. Direct comparison to their Table 1 line 1.
2. Binary-collapsed metrics on apo PDB and AlphaFold holdouts. Direct comparison to their Table 1 lines 2-3.
3. Multi-class metrics (new). Macro-F1 overall, macro-F1 lipids-only, per-class F1 table, confusion matrix summary.

Write `reports/DAY_1_SUMMARY.md` as a one-page writeup: what was done, the numbers, the figure references, what Day 2 unlocks.

## 7. Expected results (for sanity-checking the coding agent)

If the reimplementation is correct:

- **Binary test F1**: approximately 0.87 +/- 0.02 (paper reports 0.869). If wildly off, fpocket parameters or labeling are wrong.
- **Binary AUROC on test**: ~0.97 (paper reports 0.970).
- **Binary F1 on AlphaFold holdout**: ~0.64 (paper reports 0.643). Regression to this floor is expected.
- **Multi-class macro-F1 (10 classes)**: predicting this is the interesting number. Expect 0.55-0.70 based on class imbalance. Rare classes (e.g., STE if low-count) will drag macro-F1 down.
- **Lipid-only macro-F1 (5 classes)**: expect 0.60-0.75. This is the headline number for Day 1.
- **XGBoost / LightGBM over RF**: expect 1-4 points of binary F1 improvement. Non-trivial gain from a drop-in.

If XGBoost beats RF on binary F1 by 2+ points AND the 5-lipid macro-F1 is above 0.60, the Day 1 result is publishable as a methods refinement on its own.

## 8. DO / DO NOT

**DO:**
- Match scikit-learn 1.3.1 exactly to reproduce paper's baseline.
- Validate pocket count totals against paper's stated counts before training.
- Seed all RNGs explicitly and persist seeds to disk.
- Compute metrics with 25-iteration mean +/- std, not single-split numbers.
- Write the binary-collapsed evaluation function once and apply it identically to test, apo PDB, and AlphaFold holdouts.
- Commit the exact dpocket command line to `CLAUDE.md` so the pocket extraction step is reproducible.

**DO NOT:**
- Add any descriptor beyond the 17. Aromatic/aliphatic splits, LJ probes, shells: all Day 2+.
- Replace fpocket. Day 2+.
- Tune hyperparameters. Paper already did this; defaults won.
- Collapse PP into nLBP for training. Keep 10 classes. PP is meaningfully different from ligand-bound-but-not-lipid pockets and the multi-class model should learn that.
- Use 1.5+ sklearn RF defaults; they changed.
- Include hemes in training. Paper excluded them from the main training set; Day 1 matches paper protocol exactly. Heme handling is Day 5+.
- Attempt to fix BstC-class fragmentation by merging fpocket outputs. Wrong day.
- Report only overall accuracy. Accuracy is misleading at 93 percent PP base rate; report F1 and macro-F1.
- Send the Dassama email before metrics are in hand and the confusion matrix figure exists.

## 9. Reproducibility

Single command to rebuild everything from zero:

```bash
make all  # downloads data, runs fpocket, trains 3 models x 25 seeds, generates reports
```

Wall-clock budget on a 16-core workstation:
- Download: 30-60 min (network-bound).
- dpocket over 1,786 structures: 15-45 min.
- 25 iterations x 3 models: 10-30 min.
- Figures and reports: 2 min.

Total: under two hours end-to-end. This is the entire point of Day 1.

## 10. References

1. Chou JC-C, Chatterjee P, Decosto CM, Dassama LMK. A machine learning model for the proteome-wide prediction of lipid-interacting proteins. *bioRxiv* 2024.01.26.577452 v3 (May 26, 2025). https://doi.org/10.1101/2024.01.26.577452
2. Le Guilloux V, Schmidtke P, Tuffery P. Fpocket: An open source platform for ligand pocket detection. *BMC Bioinformatics* 2009;10:168.
3. Schmidtke P, Le Guilloux V, Maupetit J, Tuffery P. fpocket: online tools for protein ensemble pocket detection and tracking. *Nucleic Acids Res* 2010;38:W582-589.
4. Pedregosa F et al. Scikit-learn: Machine Learning in Python. *JMLR* 2011;12:2825-2830.
5. Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. *KDD* 2016.
6. Ke G et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS* 2017.
7. Berman HM et al. The Protein Data Bank. *Nucleic Acids Res* 2000;28:235-242.
8. Jumper J et al. Highly accurate protein structure prediction with AlphaFold. *Nature* 2021;596:583-589.
9. Varoquaux G, Buitinck L, Louppe G et al. Scikit-learn: Machine learning without learning the machinery. *GetMobile* 2015;19:29-33. (For `class_weight="balanced"` semantics.)
10. Andrio P et al. BioExcel Building Blocks (biobb_vs), a software library for interoperable biomolecular simulation workflows. *Sci Data* 2019;6:169. (Alternate fpocket wrapper used in original paper.)

SLiPP reference implementation: https://github.com/dassamalab/SLiPP_2024

## 11. What Day 2 unlocks (for planning only, do not scope-creep into Day 1)

The multi-class softmax probabilities become a feature-space representation. On Day 2, concatenating `[17 fpocket descriptors || 10 class probabilities]` becomes the input to a second-stage classifier with the new VdW-resolved descriptors (aromatic/aliphatic shells, head/mid/tail zones, pocket anisotropy tensors). The Day 1 model effectively becomes an embedding layer. This framing is what makes the four-week roadmap end in a paper rather than a footnote.

---

## Appendix A: Dassama outreach email draft

**Subject:** Quick follow-up on SLiPP — multi-class extension, one-day result

Dr. Dassama,

Thanks again for the conversation about the PTM-lipid intersection work. I have been reading the v3 SLiPP preprint closely and wanted to share something I ran today.

The binary lipid-vs-rest formulation in the current model discards subclass structure that the 17 dpocket descriptors appear to carry. Your Fig 2B shows CLR, MYR, PLM, STE, and OLA occupying distinguishable regions of the PCA space, but the binary loss gives the model no gradient to separate them. I reformulated the classifier as a 10-class softmax over {ADN, B12, BGC, COA, CLR, MYR, PLM, STE, OLA, PP}, retrained on the exact training set from Supporting Files 1-3 with scikit-learn 1.3.1 to match your baseline, and compared RF, XGBoost, and LightGBM heads.

The binary-collapsed F1 on your test split is [X.XXX +/- X.XXX] across 25 stratified splits, compared to the 0.869 reported in Table 1. The new number is the 5-lipid macro-F1 of [X.XXX], which is the first time this dataset has produced lipid-subclass predictions. Confusion matrix attached. CLR separates cleanly from the fatty-acid classes; MYR/PLM/STE show the expected aliphatic-chain-length confusion which is a meaningful signal on its own.

This is Day 1 of a roadmap I have been sketching toward a SLiPP successor with physically-grounded VdW descriptors (aromatic/aliphatic dispersion decomposition, head/mid/tail polarity zones, Lennard-Jones probe maps for the heme false-positive problem you flag in Fig 7C-D). The multi-class reformulation is the cheapest piece and I wanted to share it before building further, in case the direction is of interest or conflicts with anything in the lab's pipeline.

Happy to share the code, the figures, and the full metrics table. Would also welcome a brief conversation about the next steps when timing works.

Best,
Filip Rumenovski
Wayne State University / Fragmatics
[contact]

---

## Appendix B: Hand-off notes for the coding agent

You are reimplementing Chou et al. 2024 (https://doi.org/10.1101/2024.01.26.577452) with one change: the classifier head is 10-class instead of binary. Everything else matches the paper. Your job is to produce the artifacts listed in Section 6.3 Step 9-10 and the reports in Section 6.3 Step 10.

Before writing code, read:
- Section 3 (Scope) and Section 8 (DO / DO NOT) in this document.
- The Methods section of the preprint.
- The repo at https://github.com/dassamalab/SLiPP_2024 for pocket extraction parameters only.

When you finish Step 5 (validation), stop and report the row counts. Do not proceed to training if counts are off by more than 5 percent. Over-eagerness here is how Day 1 becomes Day 3.

Any ambiguity in the paper (pocket overlap threshold, exact fpocket command line, etc.): default to the value that reproduces the paper's reported counts. The validation layer is the ground truth for "did I ingest the data correctly."

When you produce the final metrics table, include both the paper's numbers and your numbers side by side. If your binary-collapsed numbers are more than 0.03 F1 off from theirs, something is wrong in the pipeline, not in the reformulation. Debug the pipeline.
