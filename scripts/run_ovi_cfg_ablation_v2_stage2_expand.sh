#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat >&2 <<'EOF'
Usage: run_ovi_cfg_ablation_v2_stage2_expand.sh \
  --screen-tag TAG --config-ids ID[,ID[,ID]] --confirm-screen-reviewed [--plan]

After all eight one-sample Stage 2 screen cells pass and have been reviewed,
expands one to three explicit cells to dev5 with seeds 103 and 211. Fresh Dense
dev5 references are always included. The review confirmation affirms that no
severe semantic/speech/sync/reconstruction failure (including r15) is advanced.
--plan prints the exact GPU invocation order without creating files.
EOF
}

PLAN_ONLY=0
SCREEN_TAG=""
RAW_CONFIG_IDS=""
SCREEN_REVIEWED=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --screen-tag)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      SCREEN_TAG="$2"
      shift 2
      ;;
    --config-ids)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      RAW_CONFIG_IDS="$2"
      shift 2
      ;;
    --confirm-screen-reviewed)
      SCREEN_REVIEWED=1
      shift
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

if [[ -z "${SCREEN_TAG}" || ! "${SCREEN_TAG}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Missing or invalid --screen-tag: ${SCREEN_TAG}" >&2
  exit 2
fi
if [[ ! "${RAW_CONFIG_IDS}" =~ ^[a-z0-9_]+(,[a-z0-9_]+){0,2}$ ]]; then
  echo "--config-ids must contain one to three canonical comma-separated IDs" >&2
  exit 2
fi
if [[ ${SCREEN_REVIEWED} -ne 1 ]]; then
  echo "--confirm-screen-reviewed is required before Stage 2 expansion" >&2
  exit 2
fi

CANONICAL_IDS=(
  current_6_23_r3
  late_12_29_r2
  late_12_29_r3
  late_12_29_r4
  late_12_29_r5
  late_15_29_r5
  late_14_29_r8
  late_15_29_r15
)
IFS=',' read -r -a requested_ids <<< "${RAW_CONFIG_IDS}"

for ((i = 0; i < ${#requested_ids[@]}; i++)); do
  config_id="${requested_ids[$i]}"
  known=0
  for allowed in "${CANONICAL_IDS[@]}"; do
    [[ "${config_id}" == "${allowed}" ]] && known=1
  done
  if [[ ${known} -ne 1 ]]; then
    echo "Unknown or non-Stage-2 --config-ids entry: ${config_id}" >&2
    exit 2
  fi
  for ((j = 0; j < i; j++)); do
    if [[ "${config_id}" == "${requested_ids[$j]}" ]]; then
      echo "Duplicate --config-ids entry: ${config_id}" >&2
      exit 2
    fi
  done
done

# Canonicalize selection so command-line order cannot change execution order.
SELECTED_IDS=()
for allowed in "${CANONICAL_IDS[@]}"; do
  for config_id in "${requested_ids[@]}"; do
    if [[ "${config_id}" == "${allowed}" ]]; then
      SELECTED_IDS+=("${allowed}")
    fi
  done
done

STAGE_TAG="${FASTA2V_STAGE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-stage2-expand}"
if [[ ! "${STAGE_TAG}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid FASTA2V_STAGE_TAG: ${STAGE_TAG}" >&2
  exit 2
fi
if [[ "${STAGE_TAG}" == "${SCREEN_TAG}" ]]; then
  echo "Stage 2 expansion tag must differ from screen tag" >&2
  exit 2
fi

print_plan() {
  local seed config_id run_tag
  local ordinal=0
  local run_ids=(dense "${SELECTED_IDS[@]}")
  echo "STAGE_TAG ${STAGE_TAG}"
  echo "REQUIRES_SCREEN_TAG ${SCREEN_TAG}"
  echo "SCREEN_REVIEW_CONFIRMED true"
  echo "PROMPT_SET dev5"
  echo "SELECTED_CONFIG_IDS $(IFS=,; echo "${SELECTED_IDS[*]}")"
  echo "EXPECTED_MEASUREMENTS_PER_RUN 5"
  for seed in 103 211; do
    for config_id in "${run_ids[@]}"; do
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
PROMPTS="${REPO_ROOT}/prompts/ovi_cfg_ablation_v2_dev5.csv"
INPUT_ROOT="${FASTA2V_CACHE_ROOT}/protocol_inputs/ovi_cfg_ablation_v2/${STAGE_TAG}"
RUN_ROOT="${FASTA2V_CACHE_ROOT}/runs/ovi_cfg_ablation_v2"
PYTHON="${FASTA2V_OVI_ENV}/bin/python"

for required in "${MATRIX}" "${BASE_CONFIG}" "${PROMPTS}"; do
  [[ -f "${required}" ]] || { echo "Missing Stage 2 expansion input: ${required}" >&2; exit 2; }
done
[[ -x "${PYTHON}" ]] || { echo "Missing Ovi Python: ${PYTHON}" >&2; exit 2; }
if [[ -e "${INPUT_ROOT}" ]]; then
  echo "Refusing to reuse Stage 2 expansion input directory: ${INPUT_ROOT}" >&2
  exit 2
fi

check_screen_receipt() {
  local receipt="$1"
  local config_id="$2"
  "${PYTHON}" -B - "${receipt}" "${config_id}" <<'PY'
import json
from pathlib import Path
import sys

path, config_id = Path(sys.argv[1]), sys.argv[2]
if not path.is_file():
    raise SystemExit(f"missing Stage 2 screen receipt: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
validation = payload.get("validation") or {}
cell = validation.get("cell") or {}
counts = validation.get("record_counts") or {}
checks = (
    (payload.get("status") == "passed", "status"),
    (payload.get("cell_id") == config_id, "cell_id"),
    (payload.get("seed") == 103, "seed"),
    (cell.get("config_id") == config_id, "validated config_id"),
    (cell.get("stage") == "2", "source stage"),
    (counts.get("measurements") == 1, "measurement count"),
)
failed = [label for ok, label in checks if not ok]
if failed:
    raise SystemExit(f"invalid Stage 2 screen receipt {path}: {', '.join(failed)}")
PY
}

# Expansion is impossible until every screen cell, not only selected cells, passed.
ordinal=0
for config_id in "${CANONICAL_IDS[@]}"; do
  ordinal=$((ordinal + 1))
  printf -v prior_tag '%s-%02d-s103-%s' \
    "${SCREEN_TAG}" "${ordinal}" "${config_id}"
  check_screen_receipt \
    "${RUN_ROOT}/${prior_tag}/protocol_validation.json" "${config_id}"
done

run_ids=(dense "${SELECTED_IDS[@]}")
ordinal=0
for seed in 103 211; do
  for config_id in "${run_ids[@]}"; do
    ordinal=$((ordinal + 1))
    printf -v run_tag '%s-%02d-s%s-%s' \
      "${STAGE_TAG}" "${ordinal}" "${seed}" "${config_id}"
    if [[ -e "${RUN_ROOT}/${run_tag}" ]]; then
      echo "Refusing existing Stage 2 expansion run directory: ${RUN_ROOT}/${run_tag}" >&2
      exit 2
    fi
  done
done

generator_ids=(dense "${SELECTED_IDS[@]}")
"${PYTHON}" -B scripts/generate_ovi_cfg_ablation_v2_configs.py \
  materialize-config \
  --base-config "${BASE_CONFIG}" \
  --matrix "${MATRIX}" \
  --prompt-csv "${PROMPTS}" \
  --output-dir "${INPUT_ROOT}" \
  --execution-stage 2 \
  --config-ids "$(IFS=,; echo "${generator_ids[*]}")" \
  --seeds 103,211 \
  --warmup-runs 0 \
  --measurement-runs 1

{
  print_plan
  echo "MATERIALIZATION_MANIFEST ${INPUT_ROOT}/manifest.json"
} > "${INPUT_ROOT}/stage2_expand_execution_plan.tsv"

config_for() {
  local config_id="$1"
  local seed="$2"
  local -a matches
  shopt -s nullglob
  matches=("${INPUT_ROOT}/configs/"*"_${config_id}_"*"_seed${seed}.yaml")
  shopt -u nullglob
  if [[ "${#matches[@]}" -ne 1 || ! -f "${matches[0]}" ]]; then
    echo "Expected one Stage 2 expansion config for ${config_id}/seed${seed}, found ${#matches[@]}" >&2
    exit 2
  fi
  printf '%s\n' "${matches[0]}"
}

# Dense first, then selected cells in canonical matrix order, for each seed block.
ordinal=0
for seed in 103 211; do
  for config_id in "${run_ids[@]}"; do
    ordinal=$((ordinal + 1))
    printf -v run_tag '%s-%02d-s%s-%s' \
      "${STAGE_TAG}" "${ordinal}" "${seed}" "${config_id}"
    bash scripts/run_ovi_cfg_ablation_v2_cell.sh \
      --config "$(config_for "${config_id}" "${seed}")" \
      --matrix "${MATRIX}" \
      --cell-id "${config_id}" \
      --seed "${seed}" \
      --run-tag "${run_tag}" \
      --expected-measurements 5
  done
done

echo "Stage 2 dev5 expansion complete: ${INPUT_ROOT}"
