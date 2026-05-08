# CAVER HPC Packaging (Docker + Apptainer + SLURM)

This folder packages the `slipp_plus` CAVER pipeline into a reproducible container workflow suitable for SLURM clusters.

## Why this setup

- CAVER Java runtime and Python deps are pinned in one image.
- Runtime is cluster-friendly via Apptainer/Singularity (`.sif` image).
- SLURM scripts provide smoke and full pipeline entrypoints.
- Avoids local terminal/session issues that caused orphan workers and inconsistent Java resolution.

## Data model (self-sufficiency)

The package is intentionally **code+runtime self-sufficient, but not data self-sufficient**.

- Included in image: Python env, Java runtime, project code, CAVER binaries under `tools/caver`.
- Not included in image by default: large project data (`data/`, `processed/`, `models/`, `reports/`).
- Expected pattern: bind-mount your repo or data paths into the container at runtime.

This keeps images small and avoids rebuilding/pushing huge artifacts for every data refresh.

## Portable self-sufficient upload folder

If you want an "upload this folder and run" workflow, use:

```bash
./infra/caver_hpc/portable/prepare_bundle.sh
```

This creates `dist/slipp_caver_hpc_bundle/` with:

- project code (`src/`, `configs/`)
- CAVER assets (`tools/caver/`)
- lockfiles (`pyproject.toml`, `uv.lock`)
- runtime scripts (`infra/caver_hpc/...`)
- selected data payload (by default includes `processed/v_sterol`, `processed/v49`, `data/structures/source_pdbs`)

Then upload that folder to HPC and run:

```bash
cd slipp_caver_hpc_bundle
./infra/caver_hpc/portable/run_train.sh
```

The portable runner is config-driven via:

- `infra/caver_hpc/portable/config.env`

You can change workers/thresholds/paths there without editing scripts.

## 1) Optional: isolate in a worktree

From your main repo checkout:

```bash
git worktree add ../slipp_plus_hpc hpc/caver-packaging
cd ../slipp_plus_hpc
```

Do all container/HPC packaging work there.

## 2) Build Docker image locally

From repo root:

```bash
docker build -f infra/caver_hpc/Dockerfile -t slipp-caver:latest .
```

Quick test:

```bash
docker run --rm slipp-caver:latest python -m slipp_plus.tunnel_features --help
```

## 3) Build Apptainer image (.sif)

On a machine with Apptainer:

```bash
apptainer build slipp-caver.sif docker-daemon://slipp-caver:latest
```

If Docker daemon is unavailable on cluster, build/push from CI or local and build from registry:

```bash
apptainer build slipp-caver.sif docker://<registry>/<image>:<tag>
```

## 4) Submit SLURM smoke job

```bash
export REPO_ROOT=/path/on/cluster/slipp_plus
export SIF_PATH=/path/on/cluster/slipp-caver.sif
sbatch infra/caver_hpc/slurm/run_v_tunnel_smoke.sbatch
```

The script reads optional config from:

- `infra/caver_hpc/configs/smoke.env`

Override with:

```bash
export CONFIG_FILE=/path/to/your_smoke.env
```

Smoke outputs are written under:

- `processed/v_tunnel/smoke/alphafold_holdout_smoke_v_tunnel.parquet`
- `processed/v_tunnel/smoke/caver_analysis/`
- `processed/v_tunnel/smoke/caver_analysis_manifest.csv`

## 5) Submit full training CAVER build

```bash
export REPO_ROOT=/path/on/cluster/slipp_plus
export SIF_PATH=/path/on/cluster/slipp-caver.sif
export WORKERS=10
sbatch infra/caver_hpc/slurm/run_v_tunnel_train.sbatch
```

The script reads optional config from:

- `infra/caver_hpc/configs/train.env`

Override with:

```bash
export CONFIG_FILE=/path/to/your_train.env
```

### Tunables

The training sbatch script accepts these env overrides:

- `WORKERS` (defaults to `SLURM_CPUS_PER_TASK`)
- `BATCH_INDEX` and `BATCH_SIZE` (optional, 0-based structure batches; set both)
- `MAX_MISSING_STRUCTURE_FRAC` (default `0.02`)
- `MIN_CONTEXT_PRESENT_FRAC` (default `0.98`)
- `MIN_PROFILE_PRESENT_FRAC` (default `0.95`)
- `TMPDIR_BASE` (default `/scratch/$USER/$SLURM_JOB_ID`)
- `BASE_PARQUET`
- `SOURCE_PDBS_ROOT`
- `CAVER_JAR`
- `OUTPUT_PARQUET`
- `CACHE_DIR`
- `REPORTS_DIR`
- `JAVA_TOOL_OPTIONS`

### Batch/debug loop

Run 100 structures at a time by setting a batch index and batch size. Each batch still writes per-structure JSON cache entries, so a later full run reuses completed structures.

```bash
export REPO_ROOT=/path/on/cluster/slipp_plus
export SIF_PATH=/path/on/cluster/slipp-caver.sif
export BATCH_SIZE=100
export BATCH_INDEX=0
export OUTPUT_PARQUET=processed/v_tunnel/batches/batch_0.parquet
export REPORTS_DIR=reports/v_tunnel/batches/batch_0
sbatch infra/caver_hpc/slurm/run_v_tunnel_train.sbatch
```

Increment `BATCH_INDEX` for the next chunk. After enough batches are cached, run once without `BATCH_INDEX`/`BATCH_SIZE` to assemble `processed/v_tunnel/full_pockets.parquet` from the cache.

## 6) Operational notes

- Use module/entrypoint execution only (`python -m slipp_plus.tunnel_features ...`), not inline heredoc scripts for multiprocessing runs.
- Keep `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` to prevent CPU oversubscription.
- If stale per-structure cache schema causes failures, clear the cache directory or run with `--no-cache` once.
- For long runs, prefer cache enabled after code/config stabilizes.

## 7) Files in this folder

- `Dockerfile` - reproducible runtime image with Java + Python env + CAVER assets
- `entrypoint.sh` - strict shell entrypoint wrapper
- `.dockerignore` - keeps image context lean
- `configs/train.env` - config-driven full training knobs (workers, paths, thresholds)
- `configs/smoke.env` - config-driven smoke knobs
- `slurm/run_v_tunnel_smoke.sbatch` - small validation run
- `slurm/run_v_tunnel_train.sbatch` - full v_tunnel build
- `portable/prepare_bundle.sh` - builds self-sufficient upload folder
- `portable/run_train.sh` - one-command runner from uploaded bundle
- `portable/config.env` - config-driven portable run settings
