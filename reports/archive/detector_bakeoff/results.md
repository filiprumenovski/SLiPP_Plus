# Detector Bakeoff Results: fpocket vs P2Rank

**Benchmark:** 1,752 unique ligand-bound training PDBs across 9 ligand classes
**Date:** 2026-04-23
**P2Rank version:** 2.4.2, default model, 6 threads, 7m 30s runtime
**fpocket version:** whatever produced the `_out/` directories in `data/structures/source_pdbs/`
**Hit criterion:** DCC ≤ 4 Å (pocket center to ligand COM) or DCA ≤ 4 Å (pocket center to nearest ligand heavy atom)

## Headline: fpocket wins this benchmark

| metric | fpocket | P2Rank | Δ (fp − p2r) |
|---|---:|---:|---:|
| Top-1 DCC ≤ 4 Å | **0.364** | 0.289 | +0.075 |
| Top-3 DCC ≤ 4 Å | **0.532** | 0.403 | +0.129 |
| Top-5 DCC ≤ 4 Å | **0.578** | 0.436 | +0.142 |
| Top-1 DCA ≤ 4 Å | **0.666** | 0.593 | +0.073 |
| Top-3 DCA ≤ 4 Å | **0.857** | 0.750 | +0.107 |
| Top-5 DCA ≤ 4 Å | **0.899** | 0.799 | +0.100 |
| Mean rank of first DCC hit | **2.25** | 3.28 | −1.03 |
| Structures with no DCC hit in top-10 | **701** (40%) | 924 (53%) | −223 |

**fpocket beats P2Rank on every aggregate metric.** Fpocket puts the correct pocket in position 2–3 on average; P2Rank in position 3–4, with more structures where the correct pocket is missed entirely in top-10.

## Per ligand class (top-1 DCC ≤ 4 Å)

| class | n | fpocket | P2Rank | Δ | comment |
|---|---:|---:|---:|---:|---|
| ADN | 179 | 0.492 | 0.207 | **+0.285** | fpocket crushes P2Rank |
| B12 | 60 | 0.317 | 0.217 | +0.100 | fpocket wins |
| BGC | 266 | 0.316 | 0.402 | **−0.086** | **only class where P2Rank wins top-1** |
| CLR | 116 | 0.250 | 0.164 | +0.086 | fpocket +17.2pp on top-3 |
| COA | 499 | 0.208 | 0.080 | **+0.128** | P2Rank essentially blind to CoA pockets |
| MYR | 138 | 0.435 | 0.399 | +0.036 | near tie |
| OLA | 138 | 0.341 | 0.326 | +0.014 | near tie |
| PLM | 341 | 0.534 | 0.494 | +0.040 | near tie |
| STE | 43 | 0.605 | 0.512 | +0.093 | fpocket wins |

Lipid subclasses of interest for SLiPP: CLR, MYR, OLA, PLM, STE. fpocket is as-good-or-better on all five. CLR is the largest gap among lipids (+17.2pp on top-3 DCC, +19.0pp on top-1 DCA).

## Per ligand class (top-3 DCC ≤ 4 Å)

| class | n | fpocket | P2Rank | Δ |
|---|---:|---:|---:|---:|
| ADN | 179 | 0.665 | 0.380 | +0.285 |
| B12 | 60 | 0.417 | 0.217 | +0.200 |
| BGC | 266 | 0.519 | 0.508 | +0.011 |
| CLR | 116 | 0.440 | 0.267 | +0.172 |
| COA | 499 | 0.345 | 0.143 | +0.202 |
| MYR | 138 | 0.725 | 0.725 | +0.000 |
| OLA | 138 | 0.565 | 0.442 | +0.123 |
| PLM | 341 | 0.654 | 0.591 | +0.063 |
| STE | 43 | 0.721 | 0.605 | +0.116 |

## Ranking quality: mean rank of first DCC ≤ 4 Å hit

| class | fpocket | P2Rank |
|---|---:|---:|
| ADN | 1.48 | 2.14 |
| B12 | 2.57 | 2.71 |
| BGC | 2.37 | 2.57 |
| CLR | 2.60 | **7.64** |
| COA | 3.00 | **5.67** |
| MYR | 2.20 | 2.88 |
| OLA | 2.28 | 2.70 |
| PLM | 1.74 | 2.07 |
| STE | 2.44 | 1.31 |
| ALL | 2.25 | 3.28 |

For **STE** P2Rank finds the pocket earlier in its ranking when it finds it at all (rank 1.31 vs fpocket's 2.44), but P2Rank misses more STE structures entirely (17 no-hit vs 11 for fpocket). For **CLR** and **COA** P2Rank both misses more often *and* ranks the true pocket much lower when it does find it.

## Interpretation

1. **For SLiPP's stated use case (lipid pocket identification), fpocket is the right detector.** P2Rank's CNN was trained on the general CHEN11/Holo4k benchmark sets, which are dominated by small drug-like ligands. Lipids and cofactors are underrepresented in its training distribution and it ranks their binding pockets lower.

2. **P2Rank is more selective.** fpocket emits ~56 pockets per structure, P2Rank emits ~15. P2Rank's ranking is better *within* the pockets it does emit — but lipid pockets often don't make its emission threshold at all.

3. **BGC (β-D-glucose) is the only class where P2Rank beats fpocket on top-1.** Glucose pockets look more like the drug-pocket training distribution than lipid pockets do.

4. **DCC vs DCA gap tells you about pocket-center geometry.** fpocket and P2Rank agree on pocket *location* (DCA) much more than on pocket *center* (DCC). P2Rank uses SAS-point aggregation, fpocket uses alpha-sphere centroid; both land somewhere on the pocket but not at the same point. For a SLiPP-style classifier that uses pocket descriptors, this matters less than the ranking gap does.

## Caveats

- **3 P2Rank structures emitted zero pockets** (pdb1Y7U, pdb2E6U, pdb4AL0 — all COA). These are counted as non-hits in P2Rank's denominator via subsequent scoring; the hit rate denominator for P2Rank is 1,749 vs fpocket's 1,752.
- **fpocket was run with the project's existing parameters** (whatever `data/structures/source_pdbs/*/*_out/` was generated with). P2Rank was run with its default model (no conservation scores). Adding `-c conservation_hmm` with an MSA could move P2Rank's numbers up materially.
- **Hit threshold 4 Å is the PRANK paper convention.** Loosening to 5 Å raises both rates; the relative ordering is preserved.
- **Ligand COM uses heavy atoms only from the exact HETATM residue name matching the class directory.** Multi-copy ligands are handled: DCC/DCA is the min across copies.

## Reproduce

```bash
# P2Rank batch (7.5 min on 6 threads)
tools/p2rank_2.4.2/prank predict -f processed/p2rank/train.ds -o processed/p2rank/train_out -threads 6

# Bakeoff (10 seconds)
uv run python -m slipp_plus.detector_bakeoff \
  --fpocket-root data/structures/source_pdbs \
  --p2rank-root processed/p2rank/train_out \
  --output reports/detector_bakeoff/training_scores.parquet \
  --summary-output reports/detector_bakeoff/training_summary.parquet \
  --summary-md reports/detector_bakeoff/training_summary.md \
  --workers 6
```

## Artifacts

- `reports/detector_bakeoff/training_scores.parquet` — 125,473 per-pocket rows (detector × structure × rank).
- `reports/detector_bakeoff/training_summary.parquet` — per-(detector, class) aggregate.
- `reports/detector_bakeoff/training_summary.md` — rendered markdown table.
- `processed/p2rank/train.ds`, `processed/p2rank/train_run.log` — P2Rank dataset file + log.
- `processed/p2rank/train_out/` — 1,752 `*.pdb_predictions.csv` + 1,752 `*.pdb_residues.csv`.
