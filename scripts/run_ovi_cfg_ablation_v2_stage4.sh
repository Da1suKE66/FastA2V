#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat >&2 <<'EOF'
Usage: run_ovi_cfg_ablation_v2_stage4.sh --new12-id ID --new14-id ID [options]

Options:
  --stage-tag TAG          Immutable run prefix (default: UTC timestamp-stage4)
  --blocks N               Balanced blocks; must be a multiple of 3 (default: 3)
  --warmup-runs N          Per workload per block, minimum 3 (default: 3)
  --measurement-runs N     Per workload per block, minimum 5 (default: 5)
  --plan-only              Print the strict plan without launching GPU work (default)
  --execute                Execute the complete strict-balanced plan

With the existing single-config cell runner, the minimum formal plan is nine
cell invocations: 27 warmups + 45 measurements = 72 generations. There is no
24-generation mode because it cannot satisfy balanced order across 3 blocks.
EOF
}

NEW12_ID=""
NEW14_ID=""
STAGE_TAG="$(date -u +%Y%m%dT%H%M%SZ)-stage4"
BLOCKS=3
WARMUPS=3
MEASUREMENTS=5
EXECUTE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --new12-id) [[ $# -ge 2 ]] || { usage; exit 2; }; NEW12_ID="$2"; shift 2 ;;
    --new14-id) [[ $# -ge 2 ]] || { usage; exit 2; }; NEW14_ID="$2"; shift 2 ;;
    --stage-tag) [[ $# -ge 2 ]] || { usage; exit 2; }; STAGE_TAG="$2"; shift 2 ;;
    --blocks) [[ $# -ge 2 ]] || { usage; exit 2; }; BLOCKS="$2"; shift 2 ;;
    --warmup-runs) [[ $# -ge 2 ]] || { usage; exit 2; }; WARMUPS="$2"; shift 2 ;;
    --measurement-runs) [[ $# -ge 2 ]] || { usage; exit 2; }; MEASUREMENTS="$2"; shift 2 ;;
    --plan-only) EXECUTE=0; shift ;;
    --execute) EXECUTE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "${NEW12_ID}" && -n "${NEW14_ID}" ]] || { usage; exit 2; }

PLAN_ARGS=(
  plan
  --stage-tag "${STAGE_TAG}"
  --new12-id "${NEW12_ID}"
  --new14-id "${NEW14_ID}"
  --blocks "${BLOCKS}"
  --warmup-runs "${WARMUPS}"
  --measurement-runs "${MEASUREMENTS}"
  --seed 103
)

if [[ ${EXECUTE} -eq 0 ]]; then
  python3 scripts/assemble_ovi_cfg_ablation_v2_report.py "${PLAN_ARGS[@]}"
  exit 0
fi

source "${REPO_ROOT}/scripts/env.sh"
PYTHON_BIN="${FASTA2V_OVI_ENV}/bin/python"
MATRIX="${REPO_ROOT}/configs/matrix/ovi_cfg_cache_ablation_v2_matrix.csv"
BASE_CONFIG="${REPO_ROOT}/configs/ovi_720x720_5s_cfg_cache_late_window_ablation.yaml"
PROMPTS="${REPO_ROOT}/prompts/ovi_cfg_ablation_v2_stage0.csv"
STAGE_ROOT="${FASTA2V_CACHE_ROOT}/protocol_inputs/ovi_cfg_ablation_v2/${STAGE_TAG}"
MATERIALIZATION="${STAGE_ROOT}/materialization"
PLAN_PATH="${STAGE_ROOT}/stage4_plan.json"
RUN_ROOT="${FASTA2V_CACHE_ROOT}/runs/ovi_cfg_ablation_v2"

if [[ -e "${STAGE_ROOT}" ]]; then
  echo "Refusing to reuse Stage 4 directory: ${STAGE_ROOT}" >&2
  exit 2
fi
mkdir -p "${STAGE_ROOT}"
"${PYTHON_BIN}" scripts/assemble_ovi_cfg_ablation_v2_report.py \
  "${PLAN_ARGS[@]}" --output "${PLAN_PATH}"

"${PYTHON_BIN}" scripts/generate_ovi_cfg_ablation_v2_configs.py \
  materialize-config \
  --base-config "${BASE_CONFIG}" \
  --matrix "${MATRIX}" \
  --prompt-csv "${PROMPTS}" \
  --output-dir "${MATERIALIZATION}" \
  --seeds 103 \
  --stages 0,2 \
  --config-ids "dense,${NEW12_ID},${NEW14_ID}" \
  --execution-stage 4 \
  --warmup-runs "${WARMUPS}" \
  --measurement-runs "${MEASUREMENTS}" \
  --benchmark-eligible

emit_plan_rows() {
  "${PYTHON_BIN}" -B -c \
    'import json,sys; p=json.load(open(sys.argv[1])); [print(e["ordinal"],e["block_index"],e["position"],e["workload"],e["config_id"],e["run_tag"],sep="\t") for e in p["execution"]]' \
    "${PLAN_PATH}"
}

# Refuse every collision before spending GPU time on the first balanced block.
while IFS=$'\t' read -r ordinal block position workload config_id run_tag; do
  if [[ -e "${RUN_ROOT}/${run_tag}" ]]; then
    echo "Refusing existing Stage 4 run directory: ${RUN_ROOT}/${run_tag}" >&2
    exit 2
  fi
done < <(emit_plan_rows)

config_for() {
  local config_id="$1"
  local -a matches
  matches=("${MATERIALIZATION}/configs/"*"_${config_id}_"*.yaml)
  if [[ "${#matches[@]}" -ne 1 || ! -f "${matches[0]}" ]]; then
    echo "Expected one Stage 4 config for ${config_id}, found ${#matches[@]}" >&2
    exit 2
  fi
  printf '%s\n' "${matches[0]}"
}

while IFS=$'\t' read -r ordinal block position workload config_id run_tag; do
  echo "Stage 4 launch ${ordinal}: block=${block} position=${position} workload=${workload} config=${config_id}"
  bash scripts/run_ovi_cfg_ablation_v2_cell.sh \
    --config "$(config_for "${config_id}")" \
    --matrix "${MATRIX}" \
    --cell-id "${config_id}" \
    --seed 103 \
    --run-tag "${run_tag}" \
  --expected-measurements "${MEASUREMENTS}"
done < <(emit_plan_rows)

REPORT_ARGS=(
  report
  --stage4-plan "${PLAN_PATH}"
  --run-root "${RUN_ROOT}"
  --output-json "${STAGE_ROOT}/machine_report.json"
  --output-markdown "${STAGE_ROOT}/machine_report.md"
)
if [[ -n "${FASTA2V_STAGE0_GATE:-}" ]]; then
  REPORT_ARGS+=(--stage0-gate "${FASTA2V_STAGE0_GATE}")
fi
if [[ -n "${FASTA2V_CANDIDATE_FREEZE:-}" ]]; then
  REPORT_ARGS+=(--candidate-freeze "${FASTA2V_CANDIDATE_FREEZE}")
fi
"${PYTHON_BIN}" scripts/assemble_ovi_cfg_ablation_v2_report.py "${REPORT_ARGS[@]}"

echo "Stage 4 strict-balanced benchmark complete: ${STAGE_ROOT}/machine_report.json"
