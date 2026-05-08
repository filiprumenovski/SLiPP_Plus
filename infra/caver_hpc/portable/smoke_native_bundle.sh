#!/usr/bin/env bash
# Portable v_tunnel smoke: same steps as slurm/run_v_tunnel_smoke.sbatch but uv+java on the host
# (no Apptainer). Run from Grid login or an interactive/debug compute shell.
#
# Usage (on HPC, after extracting bundle):
#   bash infra/caver_hpc/portable/smoke_native_bundle.sh
#
# Optional overrides:
#   BUNDLE=/path/to/slipp_caver_hpc_bundle

set -eo pipefail

BUNDLE="${BUNDLE:-}"
if [[ -z "${BUNDLE}" ]]; then
  if [[ -n "${BASH_SOURCE[0]-}" ]]; then
    _HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "${_HERE}/../../../pyproject.toml" ]]; then
      BUNDLE="$(cd "${_HERE}/../../.." && pwd)"
    fi
  fi
fi
if [[ -z "${BUNDLE}" ]]; then
  echo "Set BUNDLE to the unpacked slipp_caver_hpc_bundle directory (bundle root)." >&2
  exit 2
fi
cd "${BUNDLE}" || { echo "cd failed: ${BUNDLE}" >&2; exit 1; }

export REPO_ROOT="${BUNDLE}"
export PATH="${HOME}/miniconda3/bin:${HOME}/micromamba:${HOME}/.local/bin:${HOME}/bin:${PATH}"

if [[ -n "${CONFIG_FILE:-}" ]]; then
  _SMOKE_CFG="${CONFIG_FILE}"
else
  _SMOKE_CFG="${BUNDLE}/infra/caver_hpc/configs/smoke.env"
fi
if [[ "${_SMOKE_CFG}" != /* ]]; then
  _SMOKE_CFG="${BUNDLE}/${_SMOKE_CFG}"
fi
# shellcheck disable=SC1090
source "${_SMOKE_CFG}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "${SMOKE_SUBSET_PARQUET}")" "$(dirname "${SMOKE_OUTPUT_PARQUET}")" "${ANALYSIS_OUTPUT_ROOT}"

command -v java >/dev/null 2>&1 || { echo "java not found; load a JDK 17+ module first" >&2; exit 2; }
command -v uv >/dev/null 2>&1 || { echo "uv not found; install uv or extend PATH" >&2; exit 2; }

test -f "${SMOKE_INPUT_PARQUET}" || { echo "missing ${SMOKE_INPUT_PARQUET}" >&2; exit 2; }
test -f "${CAVER_JAR}" || { echo "missing ${CAVER_JAR}" >&2; exit 2; }
test -d "${STRUCTURES_ROOT}" || { echo "missing structures dir ${STRUCTURES_ROOT}" >&2; exit 2; }

echo "[$(date)] smoke native: bundle=${BUNDLE} host=$(hostname) structures=${SMOKE_N_STRUCTURES}"

uv sync --locked --no-dev

uv run python - << PY
from pathlib import Path
import pandas as pd

root = Path.cwd()
inp = root / "${SMOKE_INPUT_PARQUET}"
out = root / "${SMOKE_SUBSET_PARQUET}"
n = int("${SMOKE_N_STRUCTURES}")
df = pd.read_parquet(inp)
ids = list(dict.fromkeys(df["structure_id"].astype(str)))[:n]
sub = df[df["structure_id"].astype(str).isin(ids)].copy()
sub.to_parquet(out, index=False)
print(f"smoke parquet: {out}")
print(f"structures: {ids}")
print(f"rows: {len(sub)}")
PY

uv run python -m slipp_plus.tunnel_features holdout \
  --base-parquet "${SMOKE_SUBSET_PARQUET}" \
  --structures-root "${STRUCTURES_ROOT}" \
  --caver-jar "${CAVER_JAR}" \
  --output "${SMOKE_OUTPUT_PARQUET}" \
  --analysis-output-root "${ANALYSIS_OUTPUT_ROOT}" \
  --analysis-manifest "${ANALYSIS_MANIFEST}" \
  --workers "${WORKERS}" \
  --max-missing-structure-frac "${MAX_MISSING_STRUCTURE_FRAC}" \
  --min-context-present-frac "${MIN_CONTEXT_PRESENT_FRAC}" \
  --min-profile-present-frac "${MIN_PROFILE_PRESENT_FRAC}"

echo "[$(date)] smoke native done"
ls -lh "${SMOKE_OUTPUT_PARQUET}" "${ANALYSIS_MANIFEST}" 2>/dev/null || true
