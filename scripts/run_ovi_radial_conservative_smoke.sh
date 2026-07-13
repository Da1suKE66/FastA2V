#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/env.sh"

RUN_TAG="${FASTA2V_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid FASTA2V_RUN_TAG: ${RUN_TAG}" >&2
  exit 2
fi
RUN_PARENT="${FASTA2V_CACHE_ROOT}/runs/ovi_720ckpt_radial_conservative_smoke_20step"
RUN_DIR="${RUN_PARENT}/${RUN_TAG}"
mkdir -p "${RUN_PARENT}"
if ! mkdir "${RUN_DIR}"; then
  echo "Refusing to reuse existing run directory: ${RUN_DIR}" >&2
  exit 2
fi
export FASTA2V_RUN_DIR="${RUN_DIR}"
cd "${REPO_ROOT}"

PHYSICAL_GPU_ZERO_UUID="$(
  nvidia-smi --id 0 --query-gpu=uuid --format=csv,noheader,nounits \
    | awk 'NF {print $1; exit}'
)"
if [[ ! "${PHYSICAL_GPU_ZERO_UUID}" =~ ^GPU-[A-Za-z0-9-]+$ ]]; then
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
"${FASTA2V_OVI_ENV}/bin/python" scripts/check_pre_run_gpu.py \
  --device-index 0 \
  --output "${RUN_DIR}/pre_run_gpu.json"
RADIAL_COMMIT="$(tr -d '[:space:]' < third_party/radial-attention.commit)"
cp "${FASTA2V_CACHE_ROOT}/ovi-environment.freeze.txt" \
  "${RUN_DIR}/environment.freeze.txt"
cp "${FASTA2V_CACHE_ROOT}/checkpoint_manifest.json" \
  "${RUN_DIR}/checkpoint_manifest.json"
cp "${FASTA2V_CACHE_ROOT}/radialattn-install.json" \
  "${RUN_DIR}/radialattn-install.json"
cp "${FASTA2V_CACHE_ROOT}/radial-flashinfer-manifest.json" \
  "${RUN_DIR}/radial-flashinfer-manifest.json"
cp "${FASTA2V_CACHE_ROOT}/sources/radial-attention-${RADIAL_COMMIT}/radial_attn/attn_mask.py" \
  "${RUN_DIR}/radial-attention-source.py"
cp "${FASTA2V_CACHE_ROOT}/derived/radial-attention-${RADIAL_COMMIT}/radial_attn/attn_mask.py" \
  "${RUN_DIR}/radial-attention-derived.py"
cp third_party/radial-attention-optional-imports.patch \
  "${RUN_DIR}/radial-attention-optional-imports.patch"
"${FASTA2V_OVI_ENV}/bin/python" scripts/preflight_ovi.py \
  --attention-method radial \
  --output "${RUN_DIR}/preflight.json"
"${FASTA2V_OVI_ENV}/bin/python" inference.py \
  --config-file configs/ovi_720x720_5s_radial_conservative_smoke.yaml \
  2>&1 | tee "${RUN_DIR}/stdout.log"
"${FASTA2V_OVI_ENV}/bin/python" scripts/verify_ovi_output.py "${RUN_DIR}"
echo "Run artifacts: ${RUN_DIR}"
