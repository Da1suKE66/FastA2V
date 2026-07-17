#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat >&2 <<'EOF'
Usage: run_ovi_cfg_ablation_v2_stage2_screen.sh --stage1-tag TAG [--plan]

Requires all fourteen Stage 1 validation receipts, then runs the eight Stage 2
matrix cells on the Stage 0 prompt with seed 103 in canonical matrix order.
--plan prints the exact GPU invocation order without creating files.
EOF
}

PLAN_ONLY=0
STAGE1_TAG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage1-tag)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      STAGE1_TAG="$2"
      shift 2
      ;;
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

for tag_name in STAGE1_TAG; do
  if [[ -z "${!tag_name}" || ! "${!tag_name}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "Missing or invalid --stage1-tag: ${!tag_name:-}" >&2
    exit 2
  fi
done
STAGE_TAG="${FASTA2V_STAGE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-stage2-screen}"
if [[ ! "${STAGE_TAG}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid FASTA2V_STAGE_TAG: ${STAGE_TAG}" >&2
  exit 2
fi
if [[ "${STAGE_TAG}" == "${STAGE1_TAG}" ]]; then
  echo "Stage 2 screen tag must differ from Stage 1 tag" >&2
  exit 2
fi

CONFIG_IDS=(
  current_6_23_r3
  late_12_29_r2
  late_12_29_r3
  late_12_29_r4
  late_12_29_r5
  late_15_29_r5
  late_14_29_r8
  late_15_29_r15
)

print_plan() {
  local config_id run_tag
  local ordinal=0
  echo "STAGE_TAG ${STAGE_TAG}"
  echo "REQUIRES_STAGE1_TAG ${STAGE1_TAG}"
  echo "PROMPT_SET stage0"
  echo "EXPECTED_MEASUREMENTS_PER_RUN 1"
  for config_id in "${CONFIG_IDS[@]}"; do
    ordinal=$((ordinal + 1))
    printf -v run_tag '%s-%02d-s103-%s' "${STAGE_TAG}" "${ordinal}" "${config_id}"
    printf 'RUN %02d seed=103 config_id=%s run_tag=%s\n' \
      "${ordinal}" "${config_id}" "${run_tag}"
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
PROMPTS="${REPO_ROOT}/prompts/ovi_cfg_ablation_v2_stage0.csv"
INPUT_ROOT="${FASTA2V_CACHE_ROOT}/protocol_inputs/ovi_cfg_ablation_v2/${STAGE_TAG}"
RUN_ROOT="${FASTA2V_CACHE_ROOT}/runs/ovi_cfg_ablation_v2"
PYTHON="${FASTA2V_OVI_ENV}/bin/python"

for required in "${MATRIX}" "${BASE_CONFIG}" "${PROMPTS}"; do
  [[ -f "${required}" ]] || { echo "Missing Stage 2 screen input: ${required}" >&2; exit 2; }
done
[[ -x "${PYTHON}" ]] || { echo "Missing Ovi Python: ${PYTHON}" >&2; exit 2; }
if [[ -e "${INPUT_ROOT}" ]]; then
  echo "Refusing to reuse Stage 2 screen input directory: ${INPUT_ROOT}" >&2
  exit 2
fi

check_receipt() {
  local receipt="$1"
  local config_id="$2"
  local seed="$3"
  local source_stage="$4"
  "${PYTHON}" -B - "${receipt}" "${config_id}" "${seed}" "${source_stage}" <<'PY'
import json
from pathlib import Path
import sys

path, config_id, seed, source_stage = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
if not path.is_file():
    raise SystemExit(f"missing prerequisite receipt: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
validation = payload.get("validation") or {}
cell = validation.get("cell") or {}
counts = validation.get("record_counts") or {}
checks = (
    (payload.get("status") == "passed", "status"),
    (payload.get("cell_id") == config_id, "cell_id"),
    (payload.get("seed") == seed, "seed"),
    (cell.get("config_id") == config_id, "validated config_id"),
    (cell.get("stage") == str(source_stage), "source stage"),
    (counts.get("measurements") == 3, "measurement count"),
)
failed = [label for ok, label in checks if not ok]
if failed:
    raise SystemExit(f"invalid prerequisite receipt {path}: {', '.join(failed)}")
PY
}

# Stage 2 may start only after the complete Stage 1 plan has validated.
stage1_ids=(dense bin_00_04_r5 bin_05_09_r5 bin_10_14_r5 bin_15_19_r5 bin_20_24_r5 bin_25_29_r5)
ordinal=0
for seed in 103 211; do
  for config_id in "${stage1_ids[@]}"; do
    ordinal=$((ordinal + 1))
    printf -v prior_tag '%s-%02d-s%s-%s' \
      "${STAGE1_TAG}" "${ordinal}" "${seed}" "${config_id}"
    if [[ "${config_id}" == "dense" ]]; then
      source_stage=0
    else
      source_stage=1
    fi
    check_receipt \
      "${RUN_ROOT}/${prior_tag}/protocol_validation.json" \
      "${config_id}" "${seed}" "${source_stage}"
  done
done

# Refuse all collisions before launching the first screen cell.
ordinal=0
for config_id in "${CONFIG_IDS[@]}"; do
  ordinal=$((ordinal + 1))
  printf -v run_tag '%s-%02d-s103-%s' "${STAGE_TAG}" "${ordinal}" "${config_id}"
  if [[ -e "${RUN_ROOT}/${run_tag}" ]]; then
    echo "Refusing existing Stage 2 screen run directory: ${RUN_ROOT}/${run_tag}" >&2
    exit 2
  fi
done

"${PYTHON}" -B scripts/generate_ovi_cfg_ablation_v2_configs.py \
  materialize-config \
  --base-config "${BASE_CONFIG}" \
  --matrix "${MATRIX}" \
  --prompt-csv "${PROMPTS}" \
  --output-dir "${INPUT_ROOT}" \
  --execution-stage 2 \
  --config-ids "$(IFS=,; echo "${CONFIG_IDS[*]}")" \
  --seeds 103 \
  --warmup-runs 0 \
  --measurement-runs 1

{
  print_plan
  echo "MATERIALIZATION_MANIFEST ${INPUT_ROOT}/manifest.json"
} > "${INPUT_ROOT}/stage2_screen_execution_plan.tsv"

config_for() {
  local config_id="$1"
  local -a matches
  shopt -s nullglob
  matches=("${INPUT_ROOT}/configs/"*"_${config_id}_"*"_seed103.yaml")
  shopt -u nullglob
  if [[ "${#matches[@]}" -ne 1 || ! -f "${matches[0]}" ]]; then
    echo "Expected one Stage 2 screen config for ${config_id}, found ${#matches[@]}" >&2
    exit 2
  fi
  printf '%s\n' "${matches[0]}"
}

# Canonical matrix order; r15 is last and receives exactly its one screening sample.
ordinal=0
for config_id in "${CONFIG_IDS[@]}"; do
  ordinal=$((ordinal + 1))
  printf -v run_tag '%s-%02d-s103-%s' "${STAGE_TAG}" "${ordinal}" "${config_id}"
  bash scripts/run_ovi_cfg_ablation_v2_cell.sh \
    --config "$(config_for "${config_id}")" \
    --matrix "${MATRIX}" \
    --cell-id "${config_id}" \
    --seed 103 \
    --run-tag "${run_tag}" \
    --expected-measurements 1
done

echo "Stage 2 screen complete; review all eight outputs before expansion: ${INPUT_ROOT}"
