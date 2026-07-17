#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat >&2 <<'EOF'
Usage: run_ovi_cfg_ablation_v2_stage1.sh [--plan]

Runs the frozen Stage 1 order: for seed 103 and then seed 211, a fresh Dense
reference followed by the six five-step bins in ascending timestep order.
--plan prints the exact GPU invocation order without creating files.
EOF
}

PLAN_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --plan)
      PLAN_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

STAGE_TAG="${FASTA2V_STAGE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-stage1}"
if [[ ! "${STAGE_TAG}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid FASTA2V_STAGE_TAG: ${STAGE_TAG}" >&2
  exit 2
fi

CONFIG_IDS=(
  dense
  bin_00_04_r5
  bin_05_09_r5
  bin_10_14_r5
  bin_15_19_r5
  bin_20_24_r5
  bin_25_29_r5
)
SEEDS=(103 211)

print_plan() {
  local seed config_id run_tag
  local ordinal=0
  echo "STAGE_TAG ${STAGE_TAG}"
  echo "PROMPT_SET dev3"
  echo "EXPECTED_MEASUREMENTS_PER_RUN 3"
  for seed in "${SEEDS[@]}"; do
    for config_id in "${CONFIG_IDS[@]}"; do
      ordinal=$((ordinal + 1))
      printf -v run_tag '%s-%02d-s%s-%s' \
        "${STAGE_TAG}" "${ordinal}" "${seed}" "${config_id}"
      printf 'RUN %02d seed=%s config_id=%s run_tag=%s\n' \
        "${ordinal}" "${seed}" "${config_id}" "${run_tag}"
    done
  done
}

if [[ ${PLAN_ONLY} -eq 1 ]]; then
  print_plan
  exit 0
fi

source "${REPO_ROOT}/scripts/env.sh"
cd "${REPO_ROOT}"

MATRIX="${REPO_ROOT}/configs/matrix/ovi_cfg_cache_ablation_v2_matrix.csv"
BASE_CONFIG="${REPO_ROOT}/configs/ovi_720x720_5s_cfg_cache_late_window_ablation.yaml"
PROMPTS="${REPO_ROOT}/prompts/ovi_cfg_ablation_v2_dev3.csv"
INPUT_ROOT="${FASTA2V_CACHE_ROOT}/protocol_inputs/ovi_cfg_ablation_v2/${STAGE_TAG}"
RUN_ROOT="${FASTA2V_CACHE_ROOT}/runs/ovi_cfg_ablation_v2"
PYTHON="${FASTA2V_OVI_ENV}/bin/python"

for required in "${MATRIX}" "${BASE_CONFIG}" "${PROMPTS}"; do
  [[ -f "${required}" ]] || { echo "Missing Stage 1 input: ${required}" >&2; exit 2; }
done
[[ -x "${PYTHON}" ]] || { echo "Missing Ovi Python: ${PYTHON}" >&2; exit 2; }
if [[ -e "${INPUT_ROOT}" ]]; then
  echo "Refusing to reuse Stage 1 input directory: ${INPUT_ROOT}" >&2
  exit 2
fi

# Refuse all run-tag collisions before spending GPU time on the first cell.
ordinal=0
for seed in "${SEEDS[@]}"; do
  for config_id in "${CONFIG_IDS[@]}"; do
    ordinal=$((ordinal + 1))
    printf -v run_tag '%s-%02d-s%s-%s' \
      "${STAGE_TAG}" "${ordinal}" "${seed}" "${config_id}"
    if [[ -e "${RUN_ROOT}/${run_tag}" ]]; then
      echo "Refusing existing Stage 1 run directory: ${RUN_ROOT}/${run_tag}" >&2
      exit 2
    fi
  done
done

"${PYTHON}" -B scripts/generate_ovi_cfg_ablation_v2_configs.py \
  materialize-config \
  --base-config "${BASE_CONFIG}" \
  --matrix "${MATRIX}" \
  --prompt-csv "${PROMPTS}" \
  --output-dir "${INPUT_ROOT}" \
  --execution-stage 1 \
  --config-ids "$(IFS=,; echo "${CONFIG_IDS[*]}")" \
  --seeds 103,211 \
  --warmup-runs 0 \
  --measurement-runs 1

{
  print_plan
  echo "MATERIALIZATION_MANIFEST ${INPUT_ROOT}/manifest.json"
} > "${INPUT_ROOT}/stage1_execution_plan.tsv"

config_for() {
  local config_id="$1"
  local seed="$2"
  local -a matches
  shopt -s nullglob
  matches=("${INPUT_ROOT}/configs/"*"_${config_id}_"*"_seed${seed}.yaml")
  shopt -u nullglob
  if [[ "${#matches[@]}" -ne 1 || ! -f "${matches[0]}" ]]; then
    echo "Expected one Stage 1 config for ${config_id}/seed${seed}, found ${#matches[@]}" >&2
    exit 2
  fi
  printf '%s\n' "${matches[0]}"
}

# This order is frozen. Do not parallelize Stage 1 cells.
ordinal=0
for seed in "${SEEDS[@]}"; do
  for config_id in "${CONFIG_IDS[@]}"; do
    ordinal=$((ordinal + 1))
    printf -v run_tag '%s-%02d-s%s-%s' \
      "${STAGE_TAG}" "${ordinal}" "${seed}" "${config_id}"
    bash scripts/run_ovi_cfg_ablation_v2_cell.sh \
      --config "$(config_for "${config_id}" "${seed}")" \
      --matrix "${MATRIX}" \
      --cell-id "${config_id}" \
      --seed "${seed}" \
      --run-tag "${run_tag}" \
      --expected-measurements 3
  done
done

echo "Stage 1 complete: ${INPUT_ROOT}"
