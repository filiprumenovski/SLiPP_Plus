# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Optimization scaffold: multi-objective Optuna NSGA-II + Hyperband HPO driver (`tools/optuna_hpo.py`), CatBoost as a fourth flat-mode base learner, OOF stacked meta-learner module (`src/slipp_plus/stacking.py`), and configuration-layer support for multiple boundary specialists.
- Per-stage `XGBHyperparameters` / `FlatModelHyperparameters` dataclasses and HPO plumbing through every XGB call site without changing existing experiment results.
- `src/slipp_plus/publication_figures.py` and `tools/build_publication_figures.py`: a three-figure publication set rendered to `figures/` in PNG / PDF / SVG, including a direct upgrade over Chou et al. 2024 Fig. 7.
- `BIODOLPHIN_EXTENSION.md`: data-availability finding documenting the 19× PDB coverage gap between Chou et al. 2024 and BioDolphin v1.1, framed as the proposed next-stage extension.

### Changed

- `experiments/registry.yaml`: deployable artifact flipped from `exp-019` (internal leader, holdout-regressive) to `exp-021` (holdout-balanced, +0.07 apo-PDB and +0.09 AlphaFold F1). `exp-019` retains `is_internal_best` for audit purposes.
- `README.md` rewritten around the deployable artifact, a paper-comparison table, and the publication figure set.

### Removed

- LLM-handoff and stale-spec docs: `handoff.md`, `NEXT_AGENT_HANDOFF.md`, `refactoring_prompt.md`, `MODEL_V2_SPEC.md`, `CONTEXT.md`, and `RELEASE_NOTES_v0.1.0.md` (the latter duplicated CHANGELOG content with no matching git tag).

## [0.1.0] - 2026-05-08

### Added

- Reimplemented the Chou et al. 2024 lipid-binding pocket classifier as a 10-class softmax model over ADN, B12, BGC, CLR, COA, MYR, OLA, PLM, PP, and STE.
- Added the current `v_sterol` ensemble workflow with RF, XGB, and LGBM probability averaging.
- Added PLM/STE tiebreaker support for the current best publication candidate.
- Added hardened CAVER feature extraction and persisted-output validation paths.
- Added hierarchical lipid-pipeline experiments, holdout validation, and registry-backed experiment tracking.
- Added paper-aligned ingestion, schema validation, stratified split generation, training, evaluation, and figure commands.

### Changed

- Expanded the sterol feature pipeline and hardened CAVER outputs in commit `91bb104`.
- Switched centroid-distance matching to the closest-heavy-atom-to-centroid metric in commit `55c8edc`.

### Notes

- The project is currently prepared as a publication-polish release candidate. Generated experiment outputs and logs remain part of the research audit trail and should not be deleted during polish work.
