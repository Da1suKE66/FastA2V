#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "This runner accepts configuration only through its fixed environment variables" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPARSE_PROFILE="${FASTA2V_SPARSE_PROFILE:-}"
SPARSE_COMBO="${FASTA2V_SPARSE_COMBO:-}"
case "${SPARSE_PROFILE}" in
  sparge_topk50|sparge_topk75|radial_conservative|radial_aggressive) ;;
  *)
    echo "FASTA2V_SPARSE_PROFILE must be set explicitly to sparge_topk50, sparge_topk75, radial_conservative, or radial_aggressive" >&2
    exit 2
    ;;
esac
case "${SPARSE_COMBO}" in
  cfg|block_cache) ;;
  *)
    echo "FASTA2V_SPARSE_COMBO must be fixed explicitly to cfg or block_cache" >&2
    exit 2
    ;;
esac

case "${SPARSE_PROFILE}:${SPARSE_COMBO}" in
  sparge_topk50:cfg)
    CONFIG_FILE="configs/ovi_720x720_5s_sparge_topk50_cfg.yaml"
    ;;
  sparge_topk75:cfg)
    CONFIG_FILE="configs/ovi_720x720_5s_sparge_topk75_cfg.yaml"
    ;;
  radial_conservative:cfg)
    CONFIG_FILE="configs/ovi_720x720_5s_radial_conservative_cfg.yaml"
    ;;
  radial_aggressive:cfg)
    CONFIG_FILE="configs/ovi_720x720_5s_radial_aggressive_cfg.yaml"
    ;;
  sparge_topk50:block_cache)
    CONFIG_FILE="configs/ovi_720x720_5s_sparge_topk50_block_cache.yaml"
    ;;
  sparge_topk75:block_cache)
    CONFIG_FILE="configs/ovi_720x720_5s_sparge_topk75_block_cache.yaml"
    ;;
  radial_conservative:block_cache)
    CONFIG_FILE="configs/ovi_720x720_5s_radial_conservative_block_cache.yaml"
    ;;
  radial_aggressive:block_cache)
    CONFIG_FILE="configs/ovi_720x720_5s_radial_aggressive_block_cache.yaml"
    ;;
  *)
    echo "Unsupported immutable sparse/combo protocol" >&2
    exit 2
    ;;
esac

# Keep invalid or incomplete invocations side-effect free. env.sh creates the
# cache hierarchy, so source it only after both immutable selectors pass.
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

case "${SPARSE_PROFILE}" in
  sparge_topk50|sparge_topk75)
    ATTENTION_METHOD="sparge"
    ;;
  radial_conservative|radial_aggressive)
    ATTENTION_METHOD="radial"
    source "${REPO_ROOT}/scripts/radial_env.sh"
    ;;
esac

RUN_TAG="${FASTA2V_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid FASTA2V_RUN_TAG: ${RUN_TAG}" >&2
  exit 2
fi
RUN_PARENT="${FASTA2V_CACHE_ROOT}/runs/ovi_720ckpt_${SPARSE_PROFILE}_${SPARSE_COMBO}_50step"
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
cp "${FASTA2V_CACHE_ROOT}/ovi-environment.freeze.txt" \
  "${RUN_DIR}/environment.freeze.txt"
cp "${FASTA2V_CACHE_ROOT}/checkpoint_manifest.json" \
  "${RUN_DIR}/checkpoint_manifest.json"

if [[ "${ATTENTION_METHOD}" == "sparge" ]]; then
  cp "${FASTA2V_CACHE_ROOT}/spargeattn-install.json" \
    "${RUN_DIR}/spargeattn-install.json"
  cp "${FASTA2V_CACHE_ROOT}/spargeattn-build.log" \
    "${RUN_DIR}/spargeattn-build.log"
  cp "${FASTA2V_CACHE_ROOT}/spargeattn-pre_run_gpu.json" \
    "${RUN_DIR}/spargeattn-install-pre_run_gpu.json"
else
  RADIAL_COMMIT="$(tr -d '[:space:]' < third_party/radial-attention.commit)"
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
fi

"${FASTA2V_OVI_ENV}/bin/python" scripts/preflight_ovi.py \
  --attention-method "${ATTENTION_METHOD}" \
  --output "${RUN_DIR}/preflight.json"
"${FASTA2V_OVI_ENV}/bin/python" inference.py \
  --config-file "${CONFIG_FILE}" \
  2>&1 | tee "${RUN_DIR}/stdout.log"
"${FASTA2V_OVI_ENV}/bin/python" scripts/verify_ovi_output.py "${RUN_DIR}"
echo "Run artifacts: ${RUN_DIR}"
