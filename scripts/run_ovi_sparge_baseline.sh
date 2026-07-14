#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/env.sh"

PHYSICAL_GPU_ZERO_UUID="$(
  /usr/bin/nvidia-smi --id 0 --query-gpu=uuid --format=csv,noheader,nounits \
    | awk 'NF {print $1; exit}'
)"
if [[ ! "${PHYSICAL_GPU_ZERO_UUID}" =~ ^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$ ]]; then
  echo "Could not resolve physical GPU 0 UUID" >&2
  exit 2
fi
case "${CUDA_VISIBLE_DEVICES:-}" in
  ""|"0"|"${PHYSICAL_GPU_ZERO_UUID}") ;;
  *)
    echo "CUDA_VISIBLE_DEVICES does not select physical GPU 0" >&2
    exit 2
    ;;
esac
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU_ZERO_UUID}"

RUN_TAG="${FASTA2V_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid FASTA2V_RUN_TAG: ${RUN_TAG}" >&2
  exit 2
fi
RUN_PARENT="${FASTA2V_CACHE_ROOT}/runs/ovi_720ckpt_sparge_50step"
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
cp "${FASTA2V_CACHE_ROOT}/spargeattn-install.json" "${RUN_DIR}/spargeattn-install.json"
cp "${FASTA2V_CACHE_ROOT}/spargeattn-build.log" "${RUN_DIR}/spargeattn-build.log"
cp "${FASTA2V_CACHE_ROOT}/spargeattn-pre_run_gpu.json" \
  "${RUN_DIR}/spargeattn-install-pre_run_gpu.json"
"${FASTA2V_OVI_ENV}/bin/python" scripts/preflight_ovi.py \
  --attention-method sparge \
  --output "${RUN_DIR}/preflight.json"
"${FASTA2V_OVI_ENV}/bin/python" inference.py \
  --config-file configs/ovi_720x720_5s_sparge.yaml \
  2>&1 | tee "${RUN_DIR}/stdout.log"
"${FASTA2V_OVI_ENV}/bin/python" scripts/verify_ovi_output.py "${RUN_DIR}"
echo "Run artifacts: ${RUN_DIR}"
