# BioDolphin Extension — Proposed Next Stage of SLiPP++

**Status:** Discovered during the SLiPP++ submission preparation, May 2026.
Documented here as the explicit next-stage extension of the
[Chou et al. 2024 SLiPP](https://doi.org/10.1101/2024.01.26.577452) work and
the SLiPP++ pipeline that this repository ships.

## TL;DR

The published SLiPP dataset uses a 27 April 2023 snapshot of the Protein Data
Bank, restricted to five lipid ligand codes (CLR, MYR, OLA, PLM, STE) chosen
because each had at least twenty PDB entries at that cutoff date. The
explicit Methods statement is that phospholipids, sphingolipids, and
glycerolipids were *omitted* "due to the limited number of available
structures."

Since then, the [BioDolphin database (Lin et al., *Nat. Comm. Chem.* 2024)](https://www.nature.com/articles/s42004-024-01384-z)
has aggregated the BioLiP2 + PDB curation through 6 September 2024 into a
single resource of **127,359 lipid–protein interaction entries** spanning
**14,891 unique PDB structures**, **6,464 unique proteins**, and **2,619
distinct lipid molecules**. Compared to the 780 PDB structures and 1,981
lipid pockets used in the SLiPP training set, this is roughly:

| Axis | Chou et al. 2024 | BioDolphin v1.1 | Multiplier |
|---|---:|---:|---:|
| PDB structures | 780 | 14,891 | ~19× |
| Pocket / interaction entries | 1,981 | 127,359 | ~64× |
| Distinct lipid molecules | 5 | 2,619 | ~524× |

The bottleneck on the SLiPP++ ten-class softmax — STE at 152 training rows
with per-class F1 0.398 prior to the boundary-rescue intervention — is
therefore *not* a property of nature. It is a property of the cutoff date
and the ≥20-entries filter that Chou et al. necessarily had to apply at the
time of their work. Two more years of PDB growth and BioDolphin's
aggregation have changed the data landscape.

This document proposes a clean extension of the SLiPP / SLiPP++ work that
takes advantage of that change. We frame it explicitly as a collaboration
proposal with the Dassama lab.

## Why this matters

The largest unresolved caveat in this repository's submission narrative is
the [holdout-overfitting trap](../reports/holdout_threshold_ablation.md):
internal-validation leaders (exp-019, five-way compact ensemble) regress on
the external holdouts (apo-PDB F1 0.649, AlphaFold F1 0.623) compared with
the deployable artifact (exp-021: apo-PDB 0.717, AlphaFold 0.715).

The deeper cause of the overfitting risk is that the model is being asked to
learn ten-class lipid sub-class structure from a training distribution where
STE has 152 rows, OLA has 329, and CLR has 358. With BioDolphin coverage
those counts plausibly grow by 3–10× without any modeling-side work, and
that re-balances the class-weighted training signal in a way no
hyperparameter search or stacking strategy can match.

Concretely:

- **Per-class F1 improvements in expectation.** STE at the 152-row scale is
  the per-class bottleneck across every SLiPP++ experiment. Historic
  per-row scaling on tabular GBDTs at this regime gives roughly +0.05 F1
  per doubling of training rows on the bottleneck class. A 2–4× expansion
  of STE coverage is therefore plausibly worth more than the entire
  HPO + stacking + multi-specialist scaffold combined.
- **Reduced overfitting risk.** A larger and more diverse training
  distribution makes the existing holdout-regression pathology *easier*
  to navigate, not harder. Multi-objective HPO also benefits: its
  Pareto-front structure is more meaningful when the training set is no
  longer the bottleneck.
- **Beyond the original five lipids.** BioDolphin includes sphingolipids,
  phospholipids, and glycerolipids that Chou et al. explicitly excluded
  for sample-size reasons. Even keeping the original ten-class softmax
  intact, those classes can serve as *hard negatives* in a binary
  lipid-vs-rest gate. Or the softmax itself can be widened to a fifteen-
  or twenty-class problem.
- **Story-changing rather than gilding.** The HPO and stacking work
  described in `tools/optuna_hpo.py` and `src/slipp_plus/stacking.py`
  amounts to optimization on the existing saturated data. A BioDolphin
  extension is a different kind of contribution: a corrected dataset of
  scale.

## The three things that need building

### 1. Data ingestion

BioDolphin v1.1 is publicly downloadable as CSV / TSV with REST API
access from `biodolphin.chemistry.gatech.edu`. The schema differs from
Chou et al.'s `training_pockets.csv`: BioDolphin records atom-level
intermolecular interactions via PLIP, not pre-extracted fpocket
descriptors. We therefore need:

- A pull script (similar in role to `src/slipp_plus/download.py`, which is
  presently a Day-7+ stub for the Chou-style raw-PDB workflow).
- A filter step that maps BioDolphin's lipid taxonomy back to the SLiPP++
  ten-class labels (CLR/MYR/OLA/PLM/STE) for backwards-compatible
  comparisons, with optional opt-in classes (sphingolipids, phospholipids,
  etc.) for the wider experiment.
- Per-PDB structure download (RCSB; a stub URL pattern already exists in
  `src/slipp_plus/v49_holdouts.py`) for the structures BioDolphin annotates
  but the local cache does not have.

### 2. Pocket extraction

Chou et al.'s pipeline uses fpocket on each PDB structure, then keeps any
pocket with ≥10 residues within 8 Å of the bound ligand center of mass.
BioDolphin already provides the bound-lipid identity and binding residues,
so the extraction is a structurally identical recipe applied to the larger
BioDolphin coverage. The repo's existing
[`src/slipp_plus/pocket_extraction.py`](../src/slipp_plus/pocket_extraction.py)
and [`src/slipp_plus/download.py`](../src/slipp_plus/download.py) stubs
contain the intended scaffold; they have been intentionally never exercised
because the curated `training_pockets.csv` was sufficient for the SLiPP++
binary-comparable submission.

After fpocket extraction, the SLiPP++ feature builders (`v49`, `v_sterol`,
`tunnel_shape*`, `boundary22`) can run unchanged. The feature
implementation does not depend on the dataset size; only on the per-pocket
PDB structure plus its fpocket output directory.

### 3. Sequence-identity-grouped CV

The current SLiPP++ split strategies include `grouped_uniprot_clustered`
([`src/slipp_plus/splits.py`](../src/slipp_plus/splits.py)). This was added
preemptively in anticipation of exactly this scaling: with 14,891 PDB
structures (vs. 780), homology between train and test pockets becomes
substantially more likely without principled grouping, and the
holdout-overfitting risk grows accordingly. The grouped CV protocol is
mandatory for any serious BioDolphin extension; the published SLiPP
numbers, by contrast, used stratified shuffle and are therefore
homology-leaked relative to a grouped-CV evaluation.

The grouped CV protocol is implemented in
[`src/slipp_plus/splits.py`](../src/slipp_plus/splits.py) under the
`grouped_uniprot_clustered` strategy and was added preemptively for
exactly this scaling regime.

## What we have already built that supports this extension

The optimization scaffold landed in this submission branch is *exactly the
right tool* for a BioDolphin-scale dataset:

- **Multi-objective Hyperband HPO** ([`tools/optuna_hpo.py`](../tools/optuna_hpo.py))
  becomes more useful with more training data, because the
  internal-vs-holdout Pareto front actually has room to move.
- **CatBoost as a fourth ensemble base learner** matters more on larger
  data because CatBoost's ordered-boosting inductive bias helps most when
  there is enough signal to stabilize gradient sampling.
- **The stacked meta-learner** ([`src/slipp_plus/stacking.py`](../src/slipp_plus/stacking.py))
  benefits from the larger-data regime for the same reason.
- **Multi-specialist support** in the configuration layer (`SpecialistRuleSettings`
  list, with the `composite_train.py` constraint lifted) makes it cheap to
  add OLA / PLM / additional-lipid specialists once those classes have
  enough rows to support a one-vs-neighbors rescue.

The optimization work is therefore not wasted by the data extension; it is
positioned to amplify it.

## Honest scope estimate

| Phase | Effort | Notes |
|---|---|---|
| BioDolphin ingest + filter to current 10 classes | 1–2 days | Schema mapping is the work. |
| fpocket extraction over the new structures | 3–5 days | Mostly compute time; the existing pipeline stub fits. |
| Feature builds + grouped CV retrain (existing 10 classes) | 2–3 days | Reuses the entire SLiPP++ feature stack. |
| Holdout regeneration + significance gating | 2 days | Recomputes apo-PDB and AlphaFold holdout F1 against the larger trained model. |
| Optional: widen the softmax to include sphingolipids / phospholipids | 1 week | Independent extension; not required for the first version. |

Total minimum-viable extension: **~2 weeks of focused work**, more than
half of which is compute time rather than engineering. Optional
softmax-widening adds another week.

## Collaboration framing

This finding was recovered during the SLiPP++ submission preparation as a
direct read of the Chou et al. Methods section against current public
databases. It is not novel as data: BioDolphin published the resource in
late 2024. But it is novel as a *connection* to the SLiPP problem, because
the SLiPP team's published cutoff predates BioDolphin and the SLiPP code
release does not currently use it.

The proposed extension is therefore offered to the Dassama lab as a
collaboration: the SLiPP++ codebase here ships a holdout-disciplined
ten-class softmax, modern HPO scaffolding, and a clean BioDolphin
ingestion path; the Dassama lab brings the original SLiPP curation
expertise, the lipid-class taxonomy, and the experimental validation of
predictions on novel pockets.

If the lab is interested, the natural next step is to schedule the data
ingestion sprint together so the lipid-taxonomy decisions (which
BioDolphin classes map to which SLiPP class; which novel classes to add)
are made by the people closest to the underlying chemistry.

## References

- Chou, J. C.-C. et al., *A machine learning model for the proteome-wide
  prediction of lipid-interacting proteins*. **bioRxiv** 2024 (preprint),
  *J. Chem. Inf. Model.* 2024 (peer-reviewed). DOI:
  [10.1101/2024.01.26.577452](https://doi.org/10.1101/2024.01.26.577452).
- Lin, J. et al., *BioDolphin as a comprehensive database of
  lipid-protein binding interactions*. **Nature Communications
  Chemistry** 7, 280 (2024). DOI:
  [10.1038/s42004-024-01384-z](https://doi.org/10.1038/s42004-024-01384-z).
  PMC mirror:
  [PMC11618342](https://pmc.ncbi.nlm.nih.gov/articles/PMC11618342/).
- SLiPP_2024 source repository (Dassama Lab): <https://github.com/dassamalab/SLiPP_2024>.
- BioDolphin web resource: `biodolphin.chemistry.gatech.edu` (consult the
  Nature Comm. Chem. paper for the canonical link).
