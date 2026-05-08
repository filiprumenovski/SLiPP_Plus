# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
