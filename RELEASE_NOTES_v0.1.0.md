# SLiPP++ v0.1.0 Release Notes

SLiPP++ v0.1.0 is the first publication-polish release candidate for the 10-class lipid-pocket reformulation of Chou et al. 2024.

## Highlights

- Reimplements the SLiPP Day 1 training path as a 10-class classifier over `ADN`, `B12`, `BGC`, `CLR`, `COA`, `MYR`, `OLA`, `PLM`, `PP`, and `STE`.
- Preserves paper-aligned binary-collapse evaluation for comparison to Chou et al. Table 1.
- Includes the current registry-backed experiment record, including negative and superseded experiments.
- Adds publication metadata: `LICENSE`, `CITATION.cff`, `CHANGELOG.md`, `CONTRIBUTING.md`, and this release note.
- Adds reproducibility notes, file checksums, holdout ID lists, and a datasheet.
- Adds GitHub Actions workflow definitions for CI and manual release artifact upload.
- Adds a slow Day 1 regression test gated behind `pytest --runslow` and `make test-slow`.
- Adds `examples/quickstart.py`, a temporary one-iteration Day 1 smoke run that prints binary F1.

## Data And Provenance

- Training CSV checksum and class counts are documented in `DATASHEET.md` and `reference/README.md`.
- Holdout ID lists are committed under `data/holdouts/`.
- The holdout counts in this checkout are 117 apo-PDB IDs and 149 AlphaFold UniProt IDs, derived directly from the current supplementary workbooks.

## Experiment Status

- `reports/ablation_matrix.md` summarizes the registry entries in `experiments/registry.yaml`.
- `experiments/queued.md` records partial holdout gaps and long-run ablations that should not be run inline without long-run approval.
- The registry currently marks `exp-012-compact-tunnel-shape` as `is_current_best: true`.

## Known Gaps Before Final Tag

- GitHub Actions status is not yet visible from this checkout, so the README CI badge has not been added.
- Repo-wide `make test` reaches pytest green but currently fails at the Ruff stage because of pre-existing lint issues in active worktree files.
- Long ablations for STE imbalance handling, ensemble-vs-single-model lift, tiebreaker on/off, and tunnel marginal value are queued but not completed in this release candidate.
- Composite pair/local MoE holdout inference is not implemented for the family-plus-MoE bundle.

## User Actions Before Public Release

1. Confirm CI is green on the release branch.
2. Review `DATASHEET.md`, `docs/methods.md`, and `reports/ablation_matrix.md` against the intended manuscript claims.
3. Decide whether queued long ablations are required before tagging.
4. Reserve a Zenodo DOI if this release will be archived.
5. Tag the release with `git tag v0.1.0` and push the tag when ready.
