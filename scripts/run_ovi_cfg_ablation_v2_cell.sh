#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/env.sh"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0

usage() {
  echo "Usage: $0 --config YAML --matrix CSV --cell-id ID --seed N --run-tag TAG [--expected-measurements N]" >&2
}

CONFIG=""
MATRIX=""
CELL_ID=""
SEED=""
RUN_TAG=""
EXPECTED_MEASUREMENTS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      CONFIG="$2"
      shift 2
      ;;
    --matrix)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      MATRIX="$2"
      shift 2
      ;;
    --cell-id)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      CELL_ID="$2"
      shift 2
      ;;
    --seed)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      SEED="$2"
      shift 2
      ;;
    --run-tag)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      RUN_TAG="$2"
      shift 2
      ;;
    --expected-measurements)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      EXPECTED_MEASUREMENTS="$2"
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

for value_name in CONFIG MATRIX CELL_ID SEED RUN_TAG; do
  if [[ -z "${!value_name}" ]]; then
    echo "Missing required argument: ${value_name}" >&2
    usage
    exit 2
  fi
done
if [[ ! "${CELL_ID}" =~ ^[a-z0-9_]+$ ]]; then
  echo "Invalid --cell-id: ${CELL_ID}" >&2
  exit 2
fi
if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "Invalid --seed: ${SEED}" >&2
  exit 2
fi
if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid --run-tag: ${RUN_TAG}" >&2
  exit 2
fi
if [[ -n "${EXPECTED_MEASUREMENTS}" && ! "${EXPECTED_MEASUREMENTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --expected-measurements: ${EXPECTED_MEASUREMENTS}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 2; }
[[ -f "${MATRIX}" ]] || { echo "Missing matrix: ${MATRIX}" >&2; exit 2; }
CONFIG="$(cd "$(dirname "${CONFIG}")" && pwd)/$(basename "${CONFIG}")"
MATRIX="$(cd "$(dirname "${MATRIX}")" && pwd)/$(basename "${MATRIX}")"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing v2 run from a dirty Git tree" >&2
  exit 2
fi

INPUT_ARGS=(
  --matrix "${MATRIX}"
  --config "${CONFIG}"
  --cell-id "${CELL_ID}"
  --seed "${SEED}"
  --input-check-only
)
if [[ -n "${EXPECTED_MEASUREMENTS}" ]]; then
  INPUT_ARGS+=(--expected-measurements "${EXPECTED_MEASUREMENTS}")
fi
set +e
INPUT_REPORT="$(
  "${FASTA2V_OVI_ENV}/bin/python" -B \
    scripts/validate_ovi_cfg_ablation_v2_run.py "${INPUT_ARGS[@]}" 2>&1
)"
INPUT_STATUS=$?
set -e
printf '%s\n' "${INPUT_REPORT}"
if [[ ${INPUT_STATUS} -ne 0 ]]; then
  echo "V2 cell input contract failed" >&2
  exit "${INPUT_STATUS}"
fi
PROMPT_FILE="$(
  printf '%s\n' "${INPUT_REPORT}" | "${FASTA2V_OVI_ENV}/bin/python" -B -c \
    'import json,sys; print(json.load(sys.stdin)["prompt_path"])'
)"
MANIFEST_FILE="$(
  printf '%s\n' "${INPUT_REPORT}" | "${FASTA2V_OVI_ENV}/bin/python" -B -c \
    'import json,sys; print(json.load(sys.stdin)["manifest_path"])'
)"
if [[ -z "${EXPECTED_MEASUREMENTS}" ]]; then
  EXPECTED_MEASUREMENTS="$(
    printf '%s\n' "${INPUT_REPORT}" | "${FASTA2V_OVI_ENV}/bin/python" -B -c \
      'import json,sys; print(json.load(sys.stdin)["expected_measurements"])'
  )"
fi
[[ -f "${PROMPT_FILE}" ]] || { echo "Missing bound prompt CSV: ${PROMPT_FILE}" >&2; exit 2; }
[[ -f "${MANIFEST_FILE}" ]] || { echo "Missing materialization manifest: ${MANIFEST_FILE}" >&2; exit 2; }

for required in \
  "${FASTA2V_CACHE_ROOT}/checkpoint_manifest.json" \
  "${FASTA2V_CACHE_ROOT}/ovi-environment.freeze.txt"; do
  [[ -f "${required}" ]] || { echo "Missing frozen evidence input: ${required}" >&2; exit 2; }
done

RUN_PARENT="${FASTA2V_CACHE_ROOT}/runs/ovi_cfg_ablation_v2"
RUN_DIR="${RUN_PARENT}/${RUN_TAG}"
mkdir -p "${RUN_PARENT}"
if ! mkdir "${RUN_DIR}"; then
  echo "Refusing to reuse existing run directory: ${RUN_DIR}" >&2
  exit 2
fi
export FASTA2V_RUN_DIR="${RUN_DIR}"

"${FASTA2V_OVI_ENV}/bin/python" -B scripts/check_pre_run_gpu.py \
  --device-index 0 \
  --output "${RUN_DIR}/pre_run_gpu.json"
cp "${FASTA2V_CACHE_ROOT}/ovi-environment.freeze.txt" \
  "${RUN_DIR}/environment.freeze.txt"
cp "${FASTA2V_CACHE_ROOT}/checkpoint_manifest.json" \
  "${RUN_DIR}/checkpoint_manifest.json"
"${FASTA2V_OVI_ENV}/bin/python" -B scripts/preflight_ovi.py \
  --attention-method dense \
  --output "${RUN_DIR}/preflight.json"

TELEMETRY_TMP="${RUN_PARENT}/.${RUN_TAG}.gpu-telemetry.$$.tmp"
if [[ -e "${TELEMETRY_TMP}" ]]; then
  echo "Refusing existing telemetry temporary file: ${TELEMETRY_TMP}" >&2
  exit 2
fi

record_gpu_telemetry() {
  local phase="$1"
  local mode="$2"
  "${FASTA2V_OVI_ENV}/bin/python" -B - "${phase}" "${mode}" "${TELEMETRY_TMP}" <<'PY'
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

phase, mode, output_name = sys.argv[1:]
fields = (
    "index",
    "uuid",
    "name",
    "pci.bus_id",
    "driver_version",
    "pstate",
    "temperature.gpu",
    "power.draw",
    "clocks.current.sm",
    "clocks.current.memory",
    "memory.used",
    "utilization.gpu",
)
command = [
    "nvidia-smi",
    "-i",
    "0",
    "--query-gpu=" + ",".join(fields),
    "--format=csv,noheader,nounits",
]
started_unix = time.time()
started_monotonic = time.monotonic()
completed = subprocess.run(
    command,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    check=False,
)
finished_monotonic = time.monotonic()
if completed.returncode != 0:
    detail = completed.stderr.strip() or "no stderr diagnostics"
    raise SystemExit(f"nvidia-smi telemetry query failed: {detail}")
rows = list(csv.reader(completed.stdout.splitlines()))
if len(rows) != 1 or len(rows[0]) != len(fields):
    raise SystemExit(f"unexpected nvidia-smi telemetry row: {rows!r}")
values = [value.strip() for value in rows[0]]
record = {
    "schema_version": 1,
    "phase": phase,
    "query_status": "ok",
    "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
    "query_started_at_unix_seconds": started_unix,
    "query_started_at_monotonic_seconds": started_monotonic,
    "query_finished_at_monotonic_seconds": finished_monotonic,
    "command": command,
    "index": int(values[0]),
    "uuid": values[1],
    "name": values[2],
    "pci_bus_id": values[3],
    "driver_version": values[4],
    "pstate": values[5],
    "temperature_c": float(values[6]),
    "power_draw_w": float(values[7]),
    "sm_clock_mhz": float(values[8]),
    "memory_clock_mhz": float(values[9]),
    "memory_used_mib": float(values[10]),
    "utilization_gpu_percent": float(values[11]),
}
path = Path(output_name)
with path.open(mode, encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()
PY
}

record_gpu_telemetry pre_inference x
set +e
"${FASTA2V_OVI_ENV}/bin/python" -B inference.py \
  --config-file "${CONFIG}" \
  2>&1 | tee "${RUN_DIR}/stdout.log"
PIPE_STATUS=("${PIPESTATUS[@]}")
set -e
record_gpu_telemetry post_inference a
mv "${TELEMETRY_TMP}" "${RUN_DIR}/gpu_telemetry.jsonl"
if [[ ${PIPE_STATUS[0]} -ne 0 ]]; then
  echo "Ovi inference failed with status ${PIPE_STATUS[0]}" >&2
  exit "${PIPE_STATUS[0]}"
fi
if [[ ${PIPE_STATUS[1]} -ne 0 ]]; then
  echo "tee failed with status ${PIPE_STATUS[1]}" >&2
  exit "${PIPE_STATUS[1]}"
fi

# Core inference permits only its fixed pre-run evidence names.  Snapshot the
# immutable cell inputs immediately after inference, before any validation.
cp "${MATRIX}" "${RUN_DIR}/matrix.csv"
cp "${CONFIG}" "${RUN_DIR}/frozen_config.yaml"
cp "${PROMPT_FILE}" "${RUN_DIR}/prompt.csv"
cp "${MANIFEST_FILE}" "${RUN_DIR}/materialization_manifest.json"

"${FASTA2V_OVI_ENV}/bin/python" -B scripts/verify_ovi_output.py \
  --media-only "${RUN_DIR}"
"${FASTA2V_OVI_ENV}/bin/python" -B scripts/hash_ovi_decoded_streams.py \
  "${RUN_DIR}" \
  --output "${RUN_DIR}/decoded_stream_hashes.json"
"${FASTA2V_OVI_ENV}/bin/python" -B \
  scripts/validate_ovi_cfg_ablation_v2_run.py \
  --matrix "${MATRIX}" \
  --config "${CONFIG}" \
  --cell-id "${CELL_ID}" \
  --seed "${SEED}" \
  --expected-measurements "${EXPECTED_MEASUREMENTS}" \
  --run-dir "${RUN_DIR}" \
  --output "${RUN_DIR}/protocol_validation.json"

echo "Validated Ovi CFG ablation v2 cell: ${RUN_DIR}"
