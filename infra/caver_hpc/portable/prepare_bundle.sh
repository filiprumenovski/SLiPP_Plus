#!/usr/bin/env bash
set -euo pipefail

# Assemble a self-sufficient folder you can upload to HPC.
#
# Default output:
#   dist/slipp_caver_hpc_bundle/
#
# Usage:
#   ./infra/caver_hpc/portable/prepare_bundle.sh
#   BUNDLE_DIR=/tmp/slipp_bundle ./infra/caver_hpc/portable/prepare_bundle.sh
#
# Optional knobs:
#   INCLUDE_SOURCE_PDBS=1   # include data/structures/source_pdbs (default: 1)
#   INCLUDE_PROCESSED=1     # include processed/v_sterol + processed/v49 (default: 1)
#   INCLUDE_REGISTRY=1    # include experiments/registry.yaml (default: 1)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

BUNDLE_DIR="${BUNDLE_DIR:-dist/slipp_caver_hpc_bundle}"
INCLUDE_SOURCE_PDBS="${INCLUDE_SOURCE_PDBS:-1}"
INCLUDE_PROCESSED="${INCLUDE_PROCESSED:-1}"
INCLUDE_REGISTRY="${INCLUDE_REGISTRY:-1}"

echo "[bundle] repo root: ${REPO_ROOT}"
echo "[bundle] output dir: ${BUNDLE_DIR}"

rm -rf "${BUNDLE_DIR}"
mkdir -p "${BUNDLE_DIR}"

# Core runtime/code assets
cp pyproject.toml uv.lock README.md "${BUNDLE_DIR}/"
cp -R src "${BUNDLE_DIR}/"
cp -R configs "${BUNDLE_DIR}/"
mkdir -p "${BUNDLE_DIR}/tools"
cp -R tools/caver "${BUNDLE_DIR}/tools/"
mkdir -p "${BUNDLE_DIR}/infra"
cp -R infra/caver_hpc "${BUNDLE_DIR}/infra/"

# Required data assets for v_tunnel training
mkdir -p "${BUNDLE_DIR}/processed"
mkdir -p "${BUNDLE_DIR}/data/structures"

if [[ "${INCLUDE_PROCESSED}" == "1" ]]; then
  if [[ -d processed/v_sterol ]]; then
    cp -R processed/v_sterol "${BUNDLE_DIR}/processed/"
  fi
  if [[ -d processed/v49 ]]; then
    cp -R processed/v49 "${BUNDLE_DIR}/processed/"
  fi
fi

if [[ "${INCLUDE_SOURCE_PDBS}" == "1" ]]; then
  if [[ -d data/structures/source_pdbs ]]; then
    cp -R data/structures/source_pdbs "${BUNDLE_DIR}/data/structures/"
  fi
fi

mkdir -p "${BUNDLE_DIR}/logs/slurm"

if [[ "${INCLUDE_REGISTRY}" == "1" && -f experiments/registry.yaml ]]; then
  mkdir -p "${BUNDLE_DIR}/experiments"
  cp experiments/registry.yaml "${BUNDLE_DIR}/experiments/"
fi

# Ensure portable scripts are executable in bundle
chmod +x \
  "${BUNDLE_DIR}/infra/caver_hpc/portable/run_train.sh" \
  "${BUNDLE_DIR}/infra/caver_hpc/portable/prepare_bundle.sh" \
  "${BUNDLE_DIR}/infra/caver_hpc/slurm/run_v_tunnel_smoke.sbatch" \
  "${BUNDLE_DIR}/infra/caver_hpc/slurm/run_v_tunnel_train.sbatch" \
  "${BUNDLE_DIR}/infra/caver_hpc/entrypoint.sh"

cat > "${BUNDLE_DIR}/RUN_ME_FIRST.txt" <<'EOF'
Self-sufficient SLiPP CAVER bundle
=================================

1) Upload this entire folder to HPC.
2) On HPC, cd into this folder.
3) Ensure Java 17+ and uv are available:
     java -version
     uv --version
4) Edit infra/caver_hpc/portable/config.env if needed.
5) Run:
     ./infra/caver_hpc/portable/run_train.sh

Optional:
- Submit via SLURM scripts under infra/caver_hpc/slurm/
- Create tarball:
    tar -czf slipp_caver_hpc_bundle.tgz -C "$(dirname "$PWD")" "$(basename "$PWD")"
EOF

echo "[bundle] completed"
echo "[bundle] next: tar -czf slipp_caver_hpc_bundle.tgz -C \"$(dirname "${BUNDLE_DIR}")\" \"$(basename "${BUNDLE_DIR}")\""
