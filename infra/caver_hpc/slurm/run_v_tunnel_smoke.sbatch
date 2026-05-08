#!/usr/bin/env bash
#SBATCH --job-name=slipp-vtunnel-smoke
#SBATCH -q primary
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

# Optional config file (KEY=VALUE lines). Override with:
#   export CONFIG_FILE=/path/to/smoke.env
CONFIG_FILE="${CONFIG_FILE:-infra/caver_hpc/configs/smoke.env}"
if [[ -f "${CONFIG_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi

# Required env at submit time:
#   REPO_ROOT=/path/to/slipp_plus
#   SIF_PATH=/path/to/slipp-caver.sif
REPO_ROOT="${REPO_ROOT:?set REPO_ROOT to repository path on shared storage}"
SIF_PATH="${SIF_PATH:?set SIF_PATH to apptainer image (.sif)}"

SMOKE_N_STRUCTURES="${SMOKE_N_STRUCTURES:-4}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
SMOKE_INPUT_PARQUET="${SMOKE_INPUT_PARQUET:-processed/v49/alphafold_holdout.parquet}"
SMOKE_SUBSET_PARQUET="${SMOKE_SUBSET_PARQUET:-processed/v_tunnel/smoke/alphafold_holdout_smoke.parquet}"
SMOKE_OUTPUT_PARQUET="${SMOKE_OUTPUT_PARQUET:-processed/v_tunnel/smoke/alphafold_holdout_smoke_v_tunnel.parquet}"
STRUCTURES_ROOT="${STRUCTURES_ROOT:-processed/v49/structures/alphafold_holdout}"
ANALYSIS_OUTPUT_ROOT="${ANALYSIS_OUTPUT_ROOT:-processed/v_tunnel/smoke/caver_analysis}"
ANALYSIS_MANIFEST="${ANALYSIS_MANIFEST:-processed/v_tunnel/smoke/caver_analysis_manifest.csv}"
CAVER_JAR="${CAVER_JAR:-tools/caver/caver.jar}"
MAX_MISSING_STRUCTURE_FRAC="${MAX_MISSING_STRUCTURE_FRAC:-0.05}"
MIN_CONTEXT_PRESENT_FRAC="${MIN_CONTEXT_PRESENT_FRAC:-0.95}"
MIN_PROFILE_PRESENT_FRAC="${MIN_PROFILE_PRESENT_FRAC:-0.90}"

mkdir -p "${REPO_ROOT}/logs/slurm"
mkdir -p "${REPO_ROOT}/processed/v_tunnel/smoke"

export APPTAINERENV_OMP_NUM_THREADS=1
export APPTAINERENV_MKL_NUM_THREADS=1

TMPDIR_BASE="${TMPDIR_BASE:-${SLURM_TMPDIR:-/scratch/${USER}/${SLURM_JOB_ID}}}"
mkdir -p "${TMPDIR_BASE}/tmp"

echo "[$(date)] preparing smoke subset (${SMOKE_N_STRUCTURES} structures)"

apptainer exec \
  --cleanenv \
  --bind "${REPO_ROOT}:/workspace" \
  --bind "${TMPDIR_BASE}/tmp:/tmp" \
  "${SIF_PATH}" \
  bash -lc "
    cd /workspace
    python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path('/workspace')
inp = root / '${SMOKE_INPUT_PARQUET}'
out = root / '${SMOKE_SUBSET_PARQUET}'
n = int('${SMOKE_N_STRUCTURES}')
df = pd.read_parquet(inp)
ids = list(dict.fromkeys(df['structure_id'].astype(str)))[:n]
sub = df[df['structure_id'].astype(str).isin(ids)].copy()
sub.to_parquet(out, index=False)
print(f'smoke parquet: {out}')
print(f'structures: {ids}')
print(f'rows: {len(sub)}')
PY

    python -m slipp_plus.tunnel_features holdout \
      --base-parquet ${SMOKE_SUBSET_PARQUET} \
      --structures-root ${STRUCTURES_ROOT} \
      --caver-jar ${CAVER_JAR} \
      --output ${SMOKE_OUTPUT_PARQUET} \
      --analysis-output-root ${ANALYSIS_OUTPUT_ROOT} \
      --analysis-manifest ${ANALYSIS_MANIFEST} \
      --workers ${WORKERS} \
      --max-missing-structure-frac ${MAX_MISSING_STRUCTURE_FRAC} \
      --min-context-present-frac ${MIN_CONTEXT_PRESENT_FRAC} \
      --min-profile-present-frac ${MIN_PROFILE_PRESENT_FRAC}
  "

echo "[$(date)] smoke job done"
