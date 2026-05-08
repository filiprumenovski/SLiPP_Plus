#!/usr/bin/env bash
#SBATCH --job-name=slipp-vtunnel-train
#SBATCH -q primary
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

# Optional config file (KEY=VALUE lines). Override with:
#   export CONFIG_FILE=/path/to/train.env
CONFIG_FILE="${CONFIG_FILE:-infra/caver_hpc/configs/train.env}"
if [[ -f "${CONFIG_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi

# Required env at submit time:
#   REPO_ROOT=/path/to/slipp_plus
#   SIF_PATH=/path/to/slipp-caver.sif
REPO_ROOT="${REPO_ROOT:?set REPO_ROOT to repository path on shared storage}"
SIF_PATH="${SIF_PATH:?set SIF_PATH to apptainer image (.sif)}"

WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-10}}"
MAX_MISSING_STRUCTURE_FRAC="${MAX_MISSING_STRUCTURE_FRAC:-0.02}"
MIN_CONTEXT_PRESENT_FRAC="${MIN_CONTEXT_PRESENT_FRAC:-0.98}"
MIN_PROFILE_PRESENT_FRAC="${MIN_PROFILE_PRESENT_FRAC:-0.95}"
BASE_PARQUET="${BASE_PARQUET:-processed/v_sterol/full_pockets.parquet}"
SOURCE_PDBS_ROOT="${SOURCE_PDBS_ROOT:-data/structures/source_pdbs}"
CAVER_JAR="${CAVER_JAR:-tools/caver/caver.jar}"
OUTPUT_PARQUET="${OUTPUT_PARQUET:-processed/v_tunnel/full_pockets.parquet}"
CACHE_DIR="${CACHE_DIR:-processed/v_tunnel/structure_json}"
REPORTS_DIR="${REPORTS_DIR:-reports/v_tunnel}"
BATCH_INDEX="${BATCH_INDEX:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:--Xms256m}"

mkdir -p "${REPO_ROOT}/logs/slurm"
mkdir -p "${REPO_ROOT}/processed/v_tunnel"
mkdir -p "${REPO_ROOT}/reports/v_tunnel"

export APPTAINERENV_OMP_NUM_THREADS=1
export APPTAINERENV_MKL_NUM_THREADS=1
export APPTAINERENV_JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS}"

TMPDIR_BASE="${TMPDIR_BASE:-${SLURM_TMPDIR:-/scratch/${USER}/${SLURM_JOB_ID}}}"
mkdir -p "${TMPDIR_BASE}/tmp"

echo "[$(date)] starting v_tunnel training build"
echo "REPO_ROOT=${REPO_ROOT}"
echo "SIF_PATH=${SIF_PATH}"
echo "WORKERS=${WORKERS}"
if [[ -n "${BATCH_INDEX}" || -n "${BATCH_SIZE}" ]]; then
  if [[ -z "${BATCH_INDEX}" || -z "${BATCH_SIZE}" ]]; then
    echo "BATCH_INDEX and BATCH_SIZE must be set together" >&2
    exit 2
  fi
  echo "BATCH_INDEX=${BATCH_INDEX}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
fi

BATCH_ARGS=()
if [[ -n "${BATCH_INDEX}" ]]; then
  BATCH_ARGS+=(--batch-index "${BATCH_INDEX}" --batch-size "${BATCH_SIZE}")
fi

apptainer exec \
  --cleanenv \
  --bind "${REPO_ROOT}:/workspace" \
  --bind "${TMPDIR_BASE}/tmp:/tmp" \
  "${SIF_PATH}" \
  bash -lc "
    cd /workspace
    python -m slipp_plus.tunnel_features training \
      --base-parquet ${BASE_PARQUET} \
      --source-pdbs-root ${SOURCE_PDBS_ROOT} \
      --caver-jar ${CAVER_JAR} \
      --output ${OUTPUT_PARQUET} \
      --reports-dir ${REPORTS_DIR} \
      --workers ${WORKERS} \
      --cache-dir ${CACHE_DIR} \
      --max-missing-structure-frac ${MAX_MISSING_STRUCTURE_FRAC} \
      --min-context-present-frac ${MIN_CONTEXT_PRESENT_FRAC} \
      --min-profile-present-frac ${MIN_PROFILE_PRESENT_FRAC} \
      ${BATCH_ARGS[*]}
  "

echo "[$(date)] completed v_tunnel training build"
