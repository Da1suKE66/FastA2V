#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/ltx2_env.sh"

PIPELINE="${FASTA2V_LTX2_PIPELINE:-distilled}"
METHOD="${FASTA2V_LTX2_METHOD:-dense}"
SEED="${FASTA2V_LTX2_SEED:-42}"
STEPS="${FASTA2V_LTX2_STEPS:-30}"
OFFLOAD="${FASTA2V_LTX2_OFFLOAD:-none}"
RUN_TAG="${FASTA2V_LTX2_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"

case "${PIPELINE}" in
  distilled)
    CHECKPOINT="${FASTA2V_LTX2_CHECKPOINT_ROOT}/ltx-2.3-22b-distilled-1.1.safetensors"
    PIPELINE_ARGS=(
      --spatial-upsampler
      "${FASTA2V_LTX2_CHECKPOINT_ROOT}/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    )
    ;;
  one-stage)
    CHECKPOINT="${FASTA2V_LTX2_CHECKPOINT_ROOT}/ltx-2.3-22b-dev.safetensors"
    PIPELINE_ARGS=()
    ;;
  *)
    echo "FASTA2V_LTX2_PIPELINE must be distilled or one-stage; got ${PIPELINE}." >&2
    exit 2
    ;;
esac

case "${METHOD}" in
  dense|sparge) ;;
  *)
    echo "FASTA2V_LTX2_METHOD must be dense or sparge; got ${METHOD}." >&2
    exit 2
    ;;
esac

if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid FASTA2V_LTX2_RUN_TAG: ${RUN_TAG}" >&2
  exit 2
fi
if [[ ! -x "${FASTA2V_LTX2_ENV}/bin/python" ]]; then
  echo "LTX Python environment is missing; run scripts/setup_ltx2_env.sh first." >&2
  exit 2
fi

RUN_DIR="${FASTA2V_LTX2_RUN_ROOT}/${PIPELINE}_${METHOD}/${RUN_TAG}"
mkdir -p "$(dirname "${RUN_DIR}")"
if ! mkdir "${RUN_DIR}"; then
  echo "Refusing to reuse run directory: ${RUN_DIR}" >&2
  exit 2
fi

if [[ -n "${FASTA2V_LTX2_PROMPTS_CSV:-}" ]]; then
  PROMPT_ARGS=(--prompts-csv "${FASTA2V_LTX2_PROMPTS_CSV}")
else
  PROMPT_ARGS=(
    --prompt-id speech_smoke
    --prompt
    "A single continuous medium close-up in a quiet studio. A woman faces the camera and clearly says, 'The morning train arrives at seven.' Her lips follow every word. The camera is locked off with no cuts. Synchronized audio contains clean female speech and faint indoor room tone."
  )
fi

SPARSE_ARGS=()
if [[ "${METHOD}" == "sparge" ]]; then
  SPARSE_ARGS=(
    --topk "${FASTA2V_LTX2_TOPK:-0.5}"
    --pvthreshd "${FASTA2V_LTX2_PVTHRESHD:-50}"
  )
  if [[ "${FASTA2V_LTX2_ALLOW_DENSE_FALLBACK:-0}" == "1" ]]; then
    SPARSE_ARGS+=(--allow-dense-fallback)
  fi
fi

cd "${REPO_ROOT}"
"${FASTA2V_LTX2_ENV}/bin/python" -m ltx2.inference \
  --pipeline "${PIPELINE}" \
  --method "${METHOD}" \
  --checkpoint "${CHECKPOINT}" \
  --gemma-root "${FASTA2V_LTX2_GEMMA_ROOT}" \
  --output-dir "${RUN_DIR}" \
  --results "${RUN_DIR}/results.jsonl" \
  --seed "${SEED}" \
  --steps "${STEPS}" \
  --offload "${OFFLOAD}" \
  "${PIPELINE_ARGS[@]}" \
  "${PROMPT_ARGS[@]}" \
  "${SPARSE_ARGS[@]}" \
  2>&1 | tee "${RUN_DIR}/stdout.log"

echo "LTX run artifacts: ${RUN_DIR}"
