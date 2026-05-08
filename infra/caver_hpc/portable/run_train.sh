#!/usr/bin/env bash
set -euo pipefail

# Self-contained training runner for uploaded portable bundle.
# Usage:
#   ./run_train.sh
# Optional:
#   CONFIG_FILE=/path/to/config.env ./run_train.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# portable/ -> caver_hpc/ -> infra/ -> bundle root
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${BUNDLE_ROOT}"

CONFIG_FILE="${CONFIG_FILE:-infra/caver_hpc/portable/config.env}"
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Missing config file: ${CONFIG_FILE}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${CONFIG_FILE}"

if [[ ! -f "pyproject.toml" || ! -f "uv.lock" ]]; then
  echo "Bundle is incomplete: expected pyproject.toml + uv.lock in ${BUNDLE_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${CAVER_JAR}" ]]; then
  echo "CAVER jar missing: ${CAVER_JAR}" >&2
  exit 2
fi
if [[ ! -f "${BASE_PARQUET}" ]]; then
  echo "Base parquet missing: ${BASE_PARQUET}" >&2
  exit 2
fi
if [[ ! -d "${SOURCE_PDBS_ROOT}" ]]; then
  echo "Structure root missing: ${SOURCE_PDBS_ROOT}" >&2
  exit 2
fi

if ! command -v java >/dev/null 2>&1; then
  echo "java not found in PATH. Install Java 17+ or provide module before run." >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found in PATH. Install uv before run." >&2
  exit 2
fi

mkdir -p "$(dirname "${OUTPUT_PARQUET}")"
mkdir -p "$(dirname "${CACHE_DIR}")"
mkdir -p reports/v_tunnel

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:-}"

echo "[portable] bundle root: ${BUNDLE_ROOT}"
echo "[portable] ensuring locked environment"
uv sync --locked --no-dev

CMD=(
  python -m slipp_plus.tunnel_features training
  --base-parquet "${BASE_PARQUET}"
  --source-pdbs-root "${SOURCE_PDBS_ROOT}"
  --caver-jar "${CAVER_JAR}"
  --output "${OUTPUT_PARQUET}"
  --reports-dir "${REPORTS_DIR}"
  --workers "${WORKERS}"
  --cache-dir "${CACHE_DIR}"
  --max-missing-structure-frac "${MAX_MISSING_STRUCTURE_FRAC}"
  --min-context-present-frac "${MIN_CONTEXT_PRESENT_FRAC}"
  --min-profile-present-frac "${MIN_PROFILE_PRESENT_FRAC}"
)

if [[ -n "${ANALYSIS_OUTPUT_ROOT:-}" ]]; then
  CMD+=(--analysis-output-root "${ANALYSIS_OUTPUT_ROOT}")
fi
if [[ -n "${ANALYSIS_MANIFEST:-}" ]]; then
  CMD+=(--analysis-manifest "${ANALYSIS_MANIFEST}")
fi
if [[ -n "${BATCH_INDEX:-}" || -n "${BATCH_SIZE:-}" ]]; then
  if [[ -z "${BATCH_INDEX:-}" || -z "${BATCH_SIZE:-}" ]]; then
    echo "BATCH_INDEX and BATCH_SIZE must be set together" >&2
    exit 2
  fi
  CMD+=(--batch-index "${BATCH_INDEX}" --batch-size "${BATCH_SIZE}")
fi

echo "[portable] launching training run"
uv run "${CMD[@]}"
echo "[portable] done"
