#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/env.sh"

RUN_DIR="${FASTA2V_CACHE_ROOT}/runs/ovi_dense_720x720_5s_50step"
mkdir -p "${RUN_DIR}"
cd "${REPO_ROOT}"

"${FASTA2V_OVI_ENV}/bin/python" scripts/preflight_ovi.py
"${FASTA2V_OVI_ENV}/bin/python" inference.py \
  --config-file configs/ovi_720x720_5s_dense.yaml \
  2>&1 | tee "${RUN_DIR}/stdout.log"
