#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat >&2 <<'EOF'
Usage: run_ovi_cfg_ablation_v2_stage3_analysis.sh \
  --frozen-receipt FILE --run-root DIR --output-dir DIR \
  [--bootstrap-replicates N] [--bootstrap-seed N]

This is a CPU/ffmpeg post-processing step. It creates the twelve Dense-reference
comparison JSONs (old/new at 12/14 hits for all three held-out seeds), then
writes separate 12-hit and 14-hit prompt-cluster analyses and an equivalence
summary. It does not launch inference or any GPU experiment.
EOF
}

FROZEN_RECEIPT=""
RUN_ROOT=""
OUTPUT_DIR=""
BOOTSTRAP_REPLICATES=5000
BOOTSTRAP_SEED=20260717
while [[ $# -gt 0 ]]; do
  case "$1" in
    --frozen-receipt) [[ $# -ge 2 ]] || { usage; exit 2; }; FROZEN_RECEIPT="$2"; shift 2 ;;
    --run-root) [[ $# -ge 2 ]] || { usage; exit 2; }; RUN_ROOT="$2"; shift 2 ;;
    --output-dir) [[ $# -ge 2 ]] || { usage; exit 2; }; OUTPUT_DIR="$2"; shift 2 ;;
    --bootstrap-replicates) [[ $# -ge 2 ]] || { usage; exit 2; }; BOOTSTRAP_REPLICATES="$2"; shift 2 ;;
    --bootstrap-seed) [[ $# -ge 2 ]] || { usage; exit 2; }; BOOTSTRAP_SEED="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "${FROZEN_RECEIPT}" && -f "${FROZEN_RECEIPT}" ]] || { echo "Missing frozen receipt" >&2; exit 2; }
[[ -n "${RUN_ROOT}" && -d "${RUN_ROOT}" ]] || { echo "Missing Stage-3 run root" >&2; exit 2; }
[[ -n "${OUTPUT_DIR}" ]] || { echo "Missing output directory" >&2; exit 2; }
[[ "${BOOTSTRAP_REPLICATES}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid bootstrap replicate count" >&2; exit 2; }
[[ "${BOOTSTRAP_SEED}" =~ ^[0-9]+$ ]] || { echo "Invalid bootstrap seed" >&2; exit 2; }
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse Stage-3 analysis directory: ${OUTPUT_DIR}" >&2
  exit 2
fi

source "${REPO_ROOT}/scripts/env.sh"
PYTHON_BIN="${FASTA2V_OVI_ENV}/bin/python"
COMPARISON_DIR="${OUTPUT_DIR}/comparisons"
SUMMARY_DIR="${OUTPUT_DIR}/summary"
mkdir -p "${COMPARISON_DIR}"

while IFS=$'\t' read -r label seed config_id dense_tag candidate_tag; do
  dense_run="${RUN_ROOT}/${dense_tag}"
  candidate_run="${RUN_ROOT}/${candidate_tag}"
  [[ -d "${dense_run}" ]] || { echo "Missing Dense run: ${dense_run}" >&2; exit 2; }
  [[ -d "${candidate_run}" ]] || { echo "Missing candidate run: ${candidate_run}" >&2; exit 2; }
  output="${COMPARISON_DIR}/${label}-seed${seed}-vs-dense.json"
  "${PYTHON_BIN}" -B scripts/compare_ovi_cfg_ablation_v2.py compare \
    --dense-run "${dense_run}" \
    --candidate-run "${candidate_run}" \
    --split heldout \
    --seed "${seed}" \
    --candidate-id "${config_id}" \
    --comparison-id "${label}_vs_dense" \
    --output "${output}"
done < <("${PYTHON_BIN}" -B -c '
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
configs = p.get("configurations", {})
runs = {(int(r["seed"]), r["label"]): r["run_tag"] for r in p.get("planned_runs", [])}
for label in ("old_12", "new_12", "old_14", "new_14"):
    config_id = configs.get(label, {}).get("config_id")
    if not config_id:
        raise SystemExit(f"missing frozen config: {label}")
    for seed in (503, 887, 1291):
        print(label, seed, config_id, runs[(seed, "dense")], runs[(seed, label)], sep="\t")
' "${FROZEN_RECEIPT}")

COMPARISONS=("${COMPARISON_DIR}"/*.json)
if [[ "${#COMPARISONS[@]}" -ne 12 ]]; then
  echo "Expected exactly 12 Stage-3 comparison JSONs, found ${#COMPARISONS[@]}" >&2
  exit 2
fi
"${PYTHON_BIN}" -B scripts/summarize_ovi_cfg_ablation_v2_stage3.py \
  --frozen-receipt "${FROZEN_RECEIPT}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES}" \
  --bootstrap-seed "${BOOTSTRAP_SEED}" \
  --output-dir "${SUMMARY_DIR}" \
  "${COMPARISONS[@]}"

echo "Stage-3 held-out machine summary: ${SUMMARY_DIR}/stage3_equivalence_summary.json"
echo "ASR, SyncNet, and blind human review remain pending."
