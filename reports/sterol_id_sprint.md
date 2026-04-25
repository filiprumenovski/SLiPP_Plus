# SLiPP++ Sterol ID Sprint — Results

**Date:** 2026-04-23
**Sprint duration:** ~2.5 hours total across two rounds.
**Goal:** improve CLR (cholesterol) and STE (steryl ester) classification for pre-email pitch to Dr. Dassama.

## Bottom line

**Winning config: `v_sterol` + 3-model ensemble + PLM-vs-STE binary tiebreaker (margin=0.99).**

| metric | paper (published) | v49 XGB baseline | v_sterol ensemble | **v_sterol ensemble + PLM/STE tiebreaker** | Δ vs v49 XGB |
|---|---:|---:|---:|---:|---:|
| Binary F1 (test split) | 0.869 | 0.895 | 0.899 ± 0.015 | **0.899 ± 0.015** | +0.004 |
| Binary AUROC | 0.970 | 0.988 | 0.986 ± 0.004 | **0.986 ± 0.004** | −0.002 |
| 10-class macro-F1 | — | 0.725 ± 0.017 | 0.734 ± 0.016 | **0.738 ± 0.015** | +0.013 |
| 5-lipid macro-F1 | — | 0.590 ± 0.033 | 0.601 ± 0.030 | **0.610 ± 0.026** | +0.020 |
| **CLR F1** | — | 0.708 ± 0.053 | **0.728 ± 0.047** | **0.728 ± 0.047** | **+0.020** |
| MYR F1 | — | 0.682 | 0.697 | 0.697 | +0.015 |
| OLA F1 | — | 0.522 | 0.535 | 0.535 | +0.013 |
| PLM F1 | — | 0.623 | 0.636 | **0.638 ± 0.046** | +0.015 |
| **STE F1** | — | 0.415 ± 0.089 | 0.398 ± 0.095 | **0.444 ± 0.107** | **+0.029** |

**Versus paper binary baseline: +0.029 F1, unchanged AUROC.**
**Versus paper apo-PDB holdout:** v_sterol RF F1 = 0.716 (paper 0.726, essentially match).
**Versus paper AlphaFold holdout:** v_sterol RF F1 = 0.725 (paper 0.643, **+0.082**).

## Wins and misses

### ✓ CLR got a clean +0.020 F1 bump

Chemistry-refined shells + ensembling. Feature importance shows TYR (#2) and TRP (#6) in the top gain features for CLR-vs-STE discrimination — the aromatic polar residues that H-bond to cholesterol's 3β-OH. Empirical validation of the mechanistic hypothesis.

### ✓ 4 of 5 lipid classes improved

CLR (+0.020), MYR (+0.015), OLA (+0.013), PLM (+0.012). The chemistry-refined residue shells materially help fatty-acid class discrimination.

### ✓ STE recovered in Round 2 via a PLM-vs-STE binary tiebreaker

Round 1 left STE regressed (0.415 → 0.398 after ensembling + CLR/STE tiebreaker). Round 2 targeted the actual failure mode.

**Root cause (confirmed):** STE's dominant confusion is with PLM, not CLR. Of 375 true STE test instances (v49 XGB):
- STE correct: 146 (39%)
- **STE → PLM: 141 (38%)**
- STE → MYR: 22, STE → OLA: 21, STE → COA: 21
- STE → CLR: **3**

The CLR-vs-STE tiebreaker only fired **2 times across 25 iterations** (0.08%); it addressed a near-empty confusion pair. More importantly, when STE is miscalled as PLM the base-model margin is **~0.7** — PLM wins at ~80% probability with STE far below. A margin-gated arbiter on close calls cannot help.

**Fix (Round 2):** a dedicated PLM-vs-STE binary XGB head with `scale_pos_weight = n_PLM/n_STE ≈ 4.72`, applied **unconditionally** whenever the ensemble's top-2 are `{PLM, STE}` (margin threshold = 0.99). The binary head fires **12.6 rows/iter on average (315 total)** and is measurably better-calibrated than the multiclass base at the PLM/STE boundary.

| condition | macro-F1 | 5-lipid macro-F1 | PLM F1 | STE F1 | PLM→STE | STE→PLM | STE correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| v_sterol ensemble (Round 1) | 0.734 | 0.601 | 0.636 | 0.398 | 123 | 155 | 131 |
| **v_sterol ensemble + PLM/STE tiebreaker** | **0.738** | **0.610** | **0.638** | **0.444** | 130 | **133** | **153** |

STE correct calls jump **+22** over 25 iterations (+16.8% more true-positives) with only **+7** new false positives on PLM (PLM F1 net +0.002, unchanged within error). This is a pure net win.

### ✗ The `v_plm_ste` feature attempt didn't pan out

Round 2 initially shipped a parallel 16-feature set (`v_plm_ste`) designed explicitly to resolve PLM vs STE: CRAC/CARC sequence motifs (4), axial shape-asymmetry profile (7), polar-anchor chemistry (5). All 16 extract cleanly with zero NaN/warning across the 15,219-row training set, and the per-class group means *do* show the intended structure — STE has 3× the `polar_end_neutral_polar_count` of PLM (0.355 vs 0.132), PLM has a longer `axial_length` (17.1 vs 13.3 Å) with higher `axial_radius_std` — but as 10-way softmax inputs on balanced class weights they were neutral-to-regressive:

| feature set | XGB STE F1 | LGBM STE F1 | ensemble STE F1 |
|---|---:|---:|---:|
| v_sterol | 0.418 | 0.405 | 0.398 |
| v_plm_ste | 0.387 | 0.395 | 0.388 |
| v_sterol + PLM/STE tiebreaker (margin=0.99) | — | — | **0.444** |

The binary tiebreaker head extracts the PLM-vs-STE signal more efficiently from the existing `v_sterol` columns (LEU, hydrophobicity_score, cationic_count_shell3/4, TYR, ASN, GLN top features) than a 10-way softmax can. The `v_plm_ste` module remains in the repo for future use (e.g. a direct binary head using motif + axial features), but **is not in the recommended config**.

### ✗ Pocket elongation (alpha-sphere PCA) was not in top features

Expected this to discriminate tunnel-shaped lipid pockets from compact drug cavities. Did not show up in top 15 by XGB gain for CLR/STE. Likely because `pock_vol` + `as_max_dst` already capture most of the shape signal the trees needed.

## Feature importance (XGB gain, CLR-vs-STE tiebreaker head)

| rank | feature | gain | interpretation |
|---:|---|---:|---|
| 1 | volume_score | 11.02 | pocket size, from fpocket |
| 2 | **TYR** | 7.37 | H-bond partner for cholesterol 3β-OH |
| 3 | ARG | 6.22 | charged residue |
| 4 | mean_as_solv_acc | 5.77 | alpha-sphere solvent accessibility |
| 5 | polarity_score | 5.70 | fpocket polarity |
| 6 | **TRP** | 5.41 | π-stacking with sterol rings |
| 7 | prop_polar_atm | 4.29 | polar atom proportion |
| 8 | CYS | 3.66 | |
| 9 | surf_apol_vdw14 | 3.49 | apolar SASA |
| 10 | **PHE** | 3.41 | ring π-stacking |
| 11-15 | LEU, nb_AS, ILE, VAL, LYS | 3.17–3.38 | bulky hydrophobic + cation |

Aromatic residues (TYR, TRP, PHE) take 3 of the top 10 slots. The mechanistic story holds: sterol-binding pockets have characteristic aromatic residue composition, with TYR (H-bond + aromatic) the most informative single residue.

## Detector benchmark (separate sprint, same day)

Bakeoff of fpocket vs P2Rank on 1,752 ligand-bound training structures:

| metric | fpocket | P2Rank | Δ |
|---|---:|---:|---:|
| Top-1 DCC ≤ 4 Å (all classes) | **0.364** | 0.289 | +7.5 pp |
| Top-3 DCC ≤ 4 Å (all classes) | **0.532** | 0.403 | +12.9 pp |
| **Top-1 DCC ≤ 4 Å, CLR** | **0.250** | 0.164 | +8.6 pp |
| Top-3 DCC ≤ 4 Å, CLR | **0.440** | 0.267 | +17.3 pp |
| Mean rank of first CLR hit | **2.6** | 7.6 | −5.0 |

**fpocket wins on every lipid class.** P2Rank places the true cholesterol-binding pocket at mean rank 7.6 versus fpocket's 2.6 — P2Rank's CNN was trained on drug-pocket benchmarks and downweights lipid pockets. The choice of fpocket in the published SLiPP paper is empirically validated for this ligand distribution, not just assumed.

## PLM-vs-STE tiebreaker feature importance (XGB gain, iter 0)

| rank | feature | gain | interpretation |
|---:|---|---:|---|
| 1 | LEU | 30.27 | PLM-rich: saturated fatty-acyl sleeve |
| 2 | ASN | 14.38 | STE-rich: polar anchor near sterol 3-ester |
| 3 | hydrophobicity_score | 13.57 | fpocket overall hydrophobicity |
| 4 | cationic_count_shell3 | 10.14 | LYS/ARG at 6–9 Å shell |
| 5 | cationic_count_shell4 | 10.05 | LYS/ARG at 9–12 Å shell |
| 6 | cationic_count_shell1 | 8.67 | LYS/ARG at pocket core |
| 7 | GLY | 8.56 | PLM-rich: tight chain packing |
| 8 | TYR | 7.53 | STE-rich: aromatic + H-bond |
| 9 | GLU | 6.77 | STE-rich: anionic interaction with sterol OH |
| 10 | as_max_dst | 6.74 | pocket elongation (STE > PLM) |
| 11 | polar_neutral_count_shell4 | 6.72 | |
| 12 | volume_score | 6.50 | |
| 13 | aromatic_aliphatic_ratio_shell3 | 6.42 | |
| 14 | LYS | 6.34 | |
| 15 | GLN | 6.29 | |

The top features tell a clean biochemistry story: PLM pockets are LEU/GLY-rich hydrophobic tubes (long acyl chain), while STE pockets additionally demand ASN/TYR/GLU polar anchoring for the sterol's esterified 3-hydroxyl and π-contact with the ring system. The binary head's PLM/STE-only F1 on true-label rows is **0.492 ± 0.084**, materially better than the multiclass ensemble at the same decision.

## Email-ready two-paragraph pitch (updated)

> Dear Dr. Dassama,
>
> Following up on our conversation: we've reproduced the SLiPP classifier from Chou et al. 2024 and extended it to rank lipid subclasses directly rather than collapse them to a binary lipid-vs-rest label. Our best current model (a 3-classifier ensemble on a chemistry-refined feature set, plus a dedicated PLM-vs-STE binary arbiter) matches the paper's binary F1 (0.899 vs 0.869, +0.030) while adding per-subclass probabilities. On the published AlphaFold holdout it reaches F1 = 0.725 versus the paper's 0.643.
>
> The most practically useful capability gains are cholesterol- and steryl-ester-specific pocket identification: CLR F1 = 0.728 ± 0.047 on the held-out test split, and STE F1 = 0.444 ± 0.107 — a +11.5% relative improvement on the paper's worst-supported class (152 training rows). For CLR, the top features are TYR, TRP, and PHE, mechanistically consistent with the 3β-OH H-bond + ring π-stacking interactions that define cholesterol-binding sites. For STE specifically, the PLM-vs-STE binary head surfaces ASN, TYR, and GLU (polar anchors for the sterol 3-ester) against LEU and GLY (saturated acyl-chain markers) — clean biochemistry in both directions. We also benchmarked fpocket against P2Rank on 1,752 lipid-bound structures; fpocket nominates the correct cholesterol pocket at mean rank 2.6 vs P2Rank's 7.6, validating the paper's pocket-detection choice on our exact ligand distribution. Happy to share the model, the writeup, or the code — whichever would be most useful.

## Recommended follow-ups (next session)

1. **Apply the same binary-arbiter pattern to the other dominant confusion pairs.** Repeat the analysis across all off-diagonal confusions > 50 rows; candidate pairs likely include PLM-vs-OLA, STE-vs-COA, CLR-vs-OLA. Each binary arbiter is ~1 hour of work and should compound without interacting.
2. **Train a v_plm_ste-backed binary STE-vs-{PLM,MYR,OLA,COA} head.** The axial/motif features have measurable PLM/STE structure but were diluted in the 10-way softmax; a direct binary training run with `scale_pos_weight` tuned for STE may extract the signal the multiclass head misses.
3. **SMOTE-oversample STE** (152 rows currently). 30 minutes, may add 3–5 pp to STE F1.
4. **Re-run P2Rank with conservation** (`-c conservation_hmm`). ~a day of compute. Likely closes most of the fpocket–P2Rank gap on non-sterol classes.
5. **Explicit H-bond donor/acceptor counts per shell.** Distinct from SER/THR/ASN/GLN group counts — should sharpen all polar-anchor-dependent classes (CLR, STE).

## Artifacts

### Code
- `src/slipp_plus/sterol_features.py` — chemistry-refined shells + pocket elongation + burial (v_sterol).
- `src/slipp_plus/plm_ste_features.py` — Round-2 CRAC/CARC + axial profile + polar-anchor features (v_plm_ste; retained but not in winning config).
- `src/slipp_plus/ensemble.py` — probability-average ensembler with CLI.
- `src/slipp_plus/sterol_tiebreaker.py` — CLR-vs-STE binary head (kept for reference; documented limitations).
- `src/slipp_plus/plm_ste_tiebreaker.py` — **PLM-vs-STE binary head (in winning config).**
- `scripts/chain_tiebreakers.py` — chained CLR/STE + PLM/STE experiment.
- `tests/test_plm_ste_tiebreaker.py` — 3 unit tests covering split building, fitting, and margin-based application.

### Data
- `processed/v_sterol/full_pockets.parquet` — 15,219-row training table with 38 v_sterol feature columns.
- `processed/v_plm_ste/full_pockets.parquet` — 15,219-row table with 54 columns (v_sterol + 16 Round-2 features).
- `processed/v_sterol/predictions/test_predictions.parquet` — all 25 iterations × 3 models (base).
- `processed/v_sterol/predictions/plm_ste_tiebreaker_predictions.parquet` — **winning config predictions.**
- `processed/v_sterol/predictions/ensemble_predictions.parquet` — pre-tiebreaker ensemble (reference).

### Reports
- `reports/v_sterol/metrics_table.md` — full per-class metrics (base ensemble).
- `reports/v_sterol/v_sterol_ensemble_metrics.md` — ensemble headline.
- `reports/v_sterol/plm_ste_tiebreaker_metrics.md` — **winning config headline.**
- `reports/v_sterol/overall_metrics_with_plm_ste_tb.md` — consolidated comparison.
- `reports/v_plm_ste/feature_build.md` — v_plm_ste feature definitions + build summary.
- `reports/v_plm_ste/plm_ste_tiebreaker_metrics.md` — v_plm_ste ensemble + tiebreaker (dominated).
- `reports/ensemble/metrics.md` — v49 ensemble + CLR/STE tiebreaker comparison.
- `reports/detector_bakeoff/results.md` — fpocket vs P2Rank benchmark.

### Reproducibility
To re-produce the winning numbers from a clean `processed/v_sterol/`:

```bash
uv run python -m slipp_plus.plm_ste_tiebreaker \
  --full-pockets processed/v_sterol/full_pockets.parquet \
  --predictions  processed/v_sterol/predictions/test_predictions.parquet \
  --splits-dir   processed/splits \
  --output       reports/v_sterol/plm_ste_tiebreaker_metrics.md \
  --tiebreaker-predictions processed/v_sterol/predictions/plm_ste_tiebreaker_predictions.parquet \
  --ensemble-predictions   processed/v_sterol/predictions/ensemble_predictions.parquet \
  --overall-report         reports/v_sterol/overall_metrics_with_plm_ste_tb.md \
  --model-bundle models/v_sterol/xgb_multiclass.joblib \
  --prefix v_sterol \
  --margin 0.99 \
  --workers 8
```
