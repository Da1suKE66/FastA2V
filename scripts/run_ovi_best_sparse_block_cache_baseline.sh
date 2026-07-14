#!/usr/bin/env bash
set -euo pipefail

if [[ "${FASTA2V_SPARSE_COMBO+x}" == "x" ]]; then
  echo "run_ovi_best_sparse_block_cache_baseline.sh fixes FASTA2V_SPARSE_COMBO=block_cache; external override is forbidden" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FASTA2V_SPARSE_COMBO="block_cache"
exec bash "${SCRIPT_DIR}/run_ovi_sparse_combo_baseline.sh" "$@"
