#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/env.sh"

cd "${REPO_ROOT}"
exec "${FASTA2V_OVI_ENV}/bin/python" download_weights.py \
  --models 720x720_5s \
  --output-dir "${FASTA2V_CACHE_ROOT}/ckpts"

