#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "" ]]; then
  cat <<'EOF'
No command provided.

Example:
  python -m slipp_plus.tunnel_features training \
    --base-parquet processed/v_sterol/full_pockets.parquet \
    --source-pdbs-root data/structures/source_pdbs \
    --caver-jar tools/caver/caver.jar \
    --output processed/v_tunnel/full_pockets.parquet \
    --workers 10
EOF
  exit 2
fi

# Preserve deterministic CPU behavior in cluster jobs unless the user overrides.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

exec "$@"
