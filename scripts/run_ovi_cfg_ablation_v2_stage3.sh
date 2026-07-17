#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/env.sh"
cd "${REPO_ROOT}"

usage() {
  cat >&2 <<'EOF'
Usage: run_ovi_cfg_ablation_v2_stage3.sh \
  --new-12-config-id ID --new-14-config-id ID \
  [--stage-tag TAG] [--selection-evidence FILE]
EOF
}

NEW_12_CONFIG_ID=""
NEW_14_CONFIG_ID=""
STAGE_TAG="${FASTA2V_STAGE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-stage3}"
SELECTION_EVIDENCE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --new-12-config-id)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      NEW_12_CONFIG_ID="$2"
      shift 2
      ;;
    --new-14-config-id)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      NEW_14_CONFIG_ID="$2"
      shift 2
      ;;
    --stage-tag)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      STAGE_TAG="$2"
      shift 2
      ;;
    --selection-evidence)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      SELECTION_EVIDENCE="$2"
      shift 2
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

if [[ -z "${NEW_12_CONFIG_ID}" || -z "${NEW_14_CONFIG_ID}" ]]; then
  echo "Both frozen candidate IDs are required" >&2
  usage
  exit 2
fi
if [[ ! "${STAGE_TAG}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid Stage-3 tag: ${STAGE_TAG}" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing Stage 3 from a dirty Git tree" >&2
  exit 2
fi

MATRIX="${REPO_ROOT}/configs/matrix/ovi_cfg_cache_ablation_v2_matrix.csv"
BASE_CONFIG="${REPO_ROOT}/configs/ovi_720x720_5s_cfg_cache_late_window_ablation.yaml"
PROMPTS="${REPO_ROOT}/prompts/ovi_cfg_cache_heldout_prompts.csv"
PROMPT_MANIFEST="${REPO_ROOT}/prompts/ovi_cfg_cache_heldout_prompt_manifest.csv"
STAGE_ROOT="${FASTA2V_CACHE_ROOT}/protocol_inputs/ovi_cfg_ablation_v2/${STAGE_TAG}"
MATERIALIZED_ROOT="${STAGE_ROOT}/materialized"
RUN_ROOT="${FASTA2V_CACHE_ROOT}/runs/ovi_cfg_ablation_v2"
FROZEN_RECEIPT="${STAGE_ROOT}/frozen_candidates.json"
BLIND_REVIEW_ROOT="${STAGE_ROOT}/stage3_blind_review"

if [[ -e "${STAGE_ROOT}" ]]; then
  echo "Refusing to reuse Stage-3 input directory: ${STAGE_ROOT}" >&2
  exit 2
fi
for required in "${MATRIX}" "${BASE_CONFIG}" "${PROMPTS}" "${PROMPT_MANIFEST}"; do
  [[ -f "${required}" ]] || { echo "Missing Stage-3 input: ${required}" >&2; exit 2; }
done
if [[ -n "${SELECTION_EVIDENCE}" && ! -f "${SELECTION_EVIDENCE}" ]]; then
  echo "Missing Stage-2 selection evidence: ${SELECTION_EVIDENCE}" >&2
  exit 2
fi

config_id_for_label() {
  case "$1" in
    dense) printf '%s\n' dense ;;
    old_12) printf '%s\n' current_6_23_r3 ;;
    new_12) printf '%s\n' "${NEW_12_CONFIG_ID}" ;;
    old_14) printf '%s\n' current_9_26_r5_anchor ;;
    new_14) printf '%s\n' "${NEW_14_CONFIG_ID}" ;;
    *) echo "Internal error: unknown Stage-3 label $1" >&2; exit 2 ;;
  esac
}

# The order below is protocol-frozen. Do not sort, parallelize, or regroup it.
RUN_PLAN=(
  "503:dense"
  "503:old_12"
  "503:new_12"
  "503:old_14"
  "503:new_14"
  "887:new_12"
  "887:old_14"
  "887:new_14"
  "887:dense"
  "887:old_12"
  "1291:new_14"
  "1291:dense"
  "1291:old_12"
  "1291:new_12"
  "1291:old_14"
)

# Refuse every run collision before freezing or materializing any held-out input.
mkdir -p "${RUN_ROOT}"
ordinal=0
for item in "${RUN_PLAN[@]}"; do
  ordinal=$((ordinal + 1))
  seed="${item%%:*}"
  label="${item#*:}"
  run_tag="${STAGE_TAG}-$(printf '%02d' "${ordinal}")-seed${seed}-${label}"
  if [[ -e "${RUN_ROOT}/${run_tag}" ]]; then
    echo "Refusing to reuse Stage-3 run directory: ${RUN_ROOT}/${run_tag}" >&2
    exit 2
  fi
done

mkdir "${STAGE_ROOT}"
FREEZE_ARGS=(
  freeze
  --matrix "${MATRIX}"
  --prompt-csv "${PROMPTS}"
  --prompt-manifest "${PROMPT_MANIFEST}"
  --new-12-config-id "${NEW_12_CONFIG_ID}"
  --new-14-config-id "${NEW_14_CONFIG_ID}"
  --stage-tag "${STAGE_TAG}"
  --run-root "${RUN_ROOT}"
  --output "${FROZEN_RECEIPT}"
)
if [[ -n "${SELECTION_EVIDENCE}" ]]; then
  FREEZE_ARGS+=(--selection-evidence "${SELECTION_EVIDENCE}")
fi
"${FASTA2V_OVI_ENV}/bin/python" -B \
  scripts/prepare_ovi_cfg_ablation_v2_stage3.py "${FREEZE_ARGS[@]}"

CONFIG_IDS="dense,current_6_23_r3,${NEW_12_CONFIG_ID},current_9_26_r5_anchor,${NEW_14_CONFIG_ID}"
"${FASTA2V_OVI_ENV}/bin/python" -B \
  scripts/generate_ovi_cfg_ablation_v2_configs.py materialize-config \
  --base-config "${BASE_CONFIG}" \
  --matrix "${MATRIX}" \
  --prompt-csv "${PROMPTS}" \
  --prompt-manifest "${PROMPT_MANIFEST}" \
  --output-dir "${MATERIALIZED_ROOT}" \
  --execution-stage 3 \
  --stages 0,2 \
  --config-ids "${CONFIG_IDS}" \
  --seeds 503,887,1291 \
  --warmup-runs 0 \
  --measurement-runs 1

config_for() {
  local config_id="$1"
  local seed="$2"
  local -a matches
  matches=("${MATERIALIZED_ROOT}/configs/"*"_${config_id}_"*"_seed${seed}.yaml")
  if [[ "${#matches[@]}" -ne 1 || ! -f "${matches[0]}" ]]; then
    echo "Expected one Stage-3 config for ${config_id}/seed${seed}, found ${#matches[@]}" >&2
    exit 2
  fi
  printf '%s\n' "${matches[0]}"
}

ordinal=0
for item in "${RUN_PLAN[@]}"; do
  ordinal=$((ordinal + 1))
  seed="${item%%:*}"
  label="${item#*:}"
  config_id="$(config_id_for_label "${label}")"
  run_tag="${STAGE_TAG}-$(printf '%02d' "${ordinal}")-seed${seed}-${label}"
  bash scripts/run_ovi_cfg_ablation_v2_cell.sh \
    --config "$(config_for "${config_id}" "${seed}")" \
    --matrix "${MATRIX}" \
    --cell-id "${config_id}" \
    --seed "${seed}" \
    --run-tag "${run_tag}" \
    --expected-measurements 8
done

"${FASTA2V_OVI_ENV}/bin/python" -B \
  scripts/prepare_ovi_cfg_ablation_v2_stage3.py blind-packet \
  --frozen-receipt "${FROZEN_RECEIPT}" \
  --run-root "${RUN_ROOT}" \
  --output-dir "${BLIND_REVIEW_ROOT}"

echo "Stage 3 machine runs passed; ASR, SyncNet, and human review remain pending."
echo "Frozen receipt: ${FROZEN_RECEIPT}"
echo "Reviewer packet: ${BLIND_REVIEW_ROOT}/packet"
echo "Private mapping: ${BLIND_REVIEW_ROOT}/private_mapping.json"
