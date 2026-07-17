#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/env.sh"
cd "${REPO_ROOT}"

STAGE_TAG="${FASTA2V_STAGE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-stage0}"
if [[ ! "${STAGE_TAG}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid FASTA2V_STAGE_TAG: ${STAGE_TAG}" >&2
  exit 2
fi

MATRIX="${REPO_ROOT}/configs/matrix/ovi_cfg_cache_ablation_v2_matrix.csv"
BASE_CONFIG="${REPO_ROOT}/configs/ovi_720x720_5s_cfg_cache_late_window_ablation.yaml"
PROMPTS="${REPO_ROOT}/prompts/ovi_cfg_ablation_v2_stage0.csv"
INPUT_ROOT="${FASTA2V_CACHE_ROOT}/protocol_inputs/ovi_cfg_ablation_v2/${STAGE_TAG}"
RUN_ROOT="${FASTA2V_CACHE_ROOT}/runs/ovi_cfg_ablation_v2"
OLD_ANCHOR_HASHES="${FASTA2V_OLD_ANCHOR_HASHES:-${REPO_ROOT}/docs/results/ovi_cfg_cache_v2_old_anchor_decoded_stream_hashes.json}"

if [[ -e "${INPUT_ROOT}" ]]; then
  echo "Refusing to reuse Stage 0 input directory: ${INPUT_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${OLD_ANCHOR_HASHES}" ]]; then
  echo "Missing old 9-26/r5 decoded-stream receipt: ${OLD_ANCHOR_HASHES}" >&2
  exit 2
fi

"${FASTA2V_OVI_ENV}/bin/python" scripts/generate_ovi_cfg_ablation_v2_configs.py \
  materialize-config \
  --base-config "${BASE_CONFIG}" \
  --matrix "${MATRIX}" \
  --prompt-csv "${PROMPTS}" \
  --output-dir "${INPUT_ROOT}" \
  --seeds 103 \
  --stages 0 \
  --config-ids dense,late_12_29_r1_null,current_9_26_r5_anchor,new_12_29_r5_repeat \
  --warmup-runs 0 \
  --measurement-runs 1

config_for() {
  local config_id="$1"
  local -a matches
  matches=("${INPUT_ROOT}/configs/"*"_${config_id}_"*.yaml)
  if [[ "${#matches[@]}" -ne 1 || ! -f "${matches[0]}" ]]; then
    echo "Expected one materialized config for ${config_id}, found ${#matches[@]}" >&2
    exit 2
  fi
  printf '%s\n' "${matches[0]}"
}

run_cell() {
  local ordinal="$1"
  local label="$2"
  local config_id="$3"
  local run_tag="${STAGE_TAG}-${ordinal}-${label}"
  bash scripts/run_ovi_cfg_ablation_v2_cell.sh \
    --config "$(config_for "${config_id}")" \
    --matrix "${MATRIX}" \
    --cell-id "${config_id}" \
    --seed 103 \
    --run-tag "${run_tag}" \
    --expected-measurements 1
}

# The order is a protocol requirement. Do not parallelize these six cells.
run_cell 01 D0 dense
run_cell 02 null-steps12-29-r1 late_12_29_r1_null
run_cell 03 anchor-steps09-26-r5 current_9_26_r5_anchor
run_cell 04 candidate-steps12-29-r5-rep1 new_12_29_r5_repeat
run_cell 05 candidate-steps12-29-r5-rep2 new_12_29_r5_repeat
run_cell 06 D1 dense

"${FASTA2V_OVI_ENV}/bin/python" scripts/gate_ovi_cfg_ablation_v2_stage0.py \
  --d0 "${RUN_ROOT}/${STAGE_TAG}-01-D0" \
  --null "${RUN_ROOT}/${STAGE_TAG}-02-null-steps12-29-r1" \
  --anchor "${RUN_ROOT}/${STAGE_TAG}-03-anchor-steps09-26-r5" \
  --repeat1 "${RUN_ROOT}/${STAGE_TAG}-04-candidate-steps12-29-r5-rep1" \
  --repeat2 "${RUN_ROOT}/${STAGE_TAG}-05-candidate-steps12-29-r5-rep2" \
  --d1 "${RUN_ROOT}/${STAGE_TAG}-06-D1" \
  --old-anchor-hashes "${OLD_ANCHOR_HASHES}" \
  --output "${INPUT_ROOT}/stage0_gate.json"

echo "Stage 0 passed: ${INPUT_ROOT}/stage0_gate.json"
