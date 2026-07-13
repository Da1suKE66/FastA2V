#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/env.sh"

RUN_TAG="${FASTA2V_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid FASTA2V_RUN_TAG: ${RUN_TAG}" >&2
  exit 2
fi
RUN_PARENT="${FASTA2V_CACHE_ROOT}/runs/ovi_720ckpt_cfg_cache_smoke_20step"
RUN_DIR="${RUN_PARENT}/${RUN_TAG}"
mkdir -p "${RUN_PARENT}"
if ! mkdir "${RUN_DIR}"; then
  echo "Refusing to reuse existing run directory: ${RUN_DIR}" >&2
  exit 2
fi
export FASTA2V_RUN_DIR="${RUN_DIR}"
cd "${REPO_ROOT}"

"${FASTA2V_OVI_ENV}/bin/python" scripts/check_pre_run_gpu.py \
  --device-index 0 \
  --output "${RUN_DIR}/pre_run_gpu.json"
cp "${FASTA2V_CACHE_ROOT}/ovi-environment.freeze.txt" "${RUN_DIR}/environment.freeze.txt"
cp "${FASTA2V_CACHE_ROOT}/checkpoint_manifest.json" "${RUN_DIR}/checkpoint_manifest.json"
"${FASTA2V_OVI_ENV}/bin/python" scripts/preflight_ovi.py --output "${RUN_DIR}/preflight.json"
"${FASTA2V_OVI_ENV}/bin/python" inference.py \
  --config-file configs/ovi_720x720_5s_cfg_cache_smoke.yaml \
  2>&1 | tee "${RUN_DIR}/stdout.log"
"${FASTA2V_OVI_ENV}/bin/python" scripts/verify_ovi_output.py "${RUN_DIR}"
echo "Run artifacts: ${RUN_DIR}"
