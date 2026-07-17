#!/usr/bin/env bash
set -Eeuo pipefail

EVALUATOR_ROOT_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_GENERATION_COMMIT="84bfb9a0ee43d32be89bd18224906ce37647e76b"

usage() {
  cat >&2 <<'EOF'
Usage: run_ovi_cfg_ablation_v2_stage3_recovery.sh \
  --stage-tag TAG --generation-repo PATH [--evaluator-repo PATH] [--plan]

Resumes only a contiguous Stage-3 prefix frozen by commit 84bfb9a. Existing
failed cells are externally reverified only when their sole failures are the
legacy p006/p007 RMS/peak/active-ratio thresholds.
EOF
}

STAGE_TAG=""
GENERATION_REPO=""
EVALUATOR_REPO="${EVALUATOR_ROOT_DEFAULT}"
PLAN_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage-tag) [[ $# -ge 2 ]] || { usage; exit 2; }; STAGE_TAG="$2"; shift 2 ;;
    --generation-repo) [[ $# -ge 2 ]] || { usage; exit 2; }; GENERATION_REPO="$2"; shift 2 ;;
    --evaluator-repo) [[ $# -ge 2 ]] || { usage; exit 2; }; EVALUATOR_REPO="$2"; shift 2 ;;
    --plan) PLAN_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${STAGE_TAG}" || ! "${STAGE_TAG}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Missing or invalid --stage-tag: ${STAGE_TAG}" >&2
  exit 2
fi
if [[ -z "${GENERATION_REPO}" ]]; then
  echo "--generation-repo is required" >&2
  exit 2
fi

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

if [[ ${PLAN_ONLY} -eq 1 ]]; then
  ordinal=0
  for item in "${RUN_PLAN[@]}"; do
    ordinal=$((ordinal + 1))
    seed="${item%%:*}"
    label="${item#*:}"
    printf 'CELL %02d seed=%s label=%s run_tag=%s-%02d-seed%s-%s\n' \
      "${ordinal}" "${seed}" "${label}" "${STAGE_TAG}" "${ordinal}" "${seed}" "${label}"
  done
  exit 0
fi

[[ -d "${GENERATION_REPO}" ]] || { echo "Missing generation repo: ${GENERATION_REPO}" >&2; exit 2; }
[[ -d "${EVALUATOR_REPO}" ]] || { echo "Missing evaluator repo: ${EVALUATOR_REPO}" >&2; exit 2; }
GENERATION_REPO="$(cd "${GENERATION_REPO}" && pwd)"
EVALUATOR_REPO="$(cd "${EVALUATOR_REPO}" && pwd)"
if [[ "${GENERATION_REPO}" == "${EVALUATOR_REPO}" ]]; then
  echo "Generation and evaluator repos must be distinct worktrees" >&2
  exit 2
fi

GENERATION_HEAD="$(git -C "${GENERATION_REPO}" rev-parse HEAD)"
if [[ "${GENERATION_HEAD}" != "${EXPECTED_GENERATION_COMMIT}" ]]; then
  echo "Generation repo must be exact commit ${EXPECTED_GENERATION_COMMIT}, found ${GENERATION_HEAD}" >&2
  exit 2
fi
if [[ -n "$(git -C "${GENERATION_REPO}" status --porcelain --untracked-files=all)" ]]; then
  echo "Generation repo is dirty" >&2
  exit 2
fi
EVALUATOR_HEAD="$(git -C "${EVALUATOR_REPO}" rev-parse HEAD)"
if [[ -n "$(git -C "${EVALUATOR_REPO}" status --porcelain --untracked-files=all)" ]]; then
  echo "Evaluator repo is dirty; deploy a committed repaired evaluator first" >&2
  exit 2
fi

GEN_VERIFY="${GENERATION_REPO}/scripts/verify_ovi_output.py"
EVAL_VERIFY="${EVALUATOR_REPO}/scripts/verify_ovi_output.py"
EVAL_HASH="${EVALUATOR_REPO}/scripts/hash_ovi_decoded_streams.py"
EVAL_VALIDATE="${EVALUATOR_REPO}/scripts/validate_ovi_cfg_ablation_v2_run.py"
RECOVERY_GUARD="${EVALUATOR_REPO}/scripts/guard_ovi_cfg_ablation_v2_stage3_recovery.py"
GEN_CELL="${GENERATION_REPO}/scripts/run_ovi_cfg_ablation_v2_cell.sh"
GEN_PACKET="${GENERATION_REPO}/scripts/prepare_ovi_cfg_ablation_v2_stage3.py"
for required in "${GEN_VERIFY}" "${EVAL_VERIFY}" "${EVAL_HASH}" "${EVAL_VALIDATE}" "${RECOVERY_GUARD}" "${GEN_CELL}" "${GEN_PACKET}"; do
  [[ -f "${required}" ]] || { echo "Missing recovery component: ${required}" >&2; exit 2; }
done
if cmp -s "${GEN_VERIFY}" "${EVAL_VERIFY}"; then
  echo "Evaluator verifier is byte-identical to the legacy generation verifier" >&2
  exit 2
fi

source "${GENERATION_REPO}/scripts/env.sh"
PYTHON="${FASTA2V_OVI_ENV}/bin/python"
[[ -x "${PYTHON}" ]] || { echo "Missing Ovi Python: ${PYTHON}" >&2; exit 2; }

STAGE_ROOT="${FASTA2V_CACHE_ROOT}/protocol_inputs/ovi_cfg_ablation_v2/${STAGE_TAG}"
FROZEN_RECEIPT="${STAGE_ROOT}/frozen_candidates.json"
MATERIALIZED_ROOT="${STAGE_ROOT}/materialized"
RUN_ROOT="${FASTA2V_CACHE_ROOT}/runs/ovi_cfg_ablation_v2"
MATRIX="${GENERATION_REPO}/configs/matrix/ovi_cfg_cache_ablation_v2_matrix.csv"
BLIND_ROOT="${STAGE_ROOT}/stage3_blind_review"
RECOVERY_MANIFEST="${STAGE_ROOT}/stage3_recovery_manifest.json"

[[ -d "${STAGE_ROOT}" ]] || { echo "Missing existing Stage-3 root: ${STAGE_ROOT}" >&2; exit 2; }
[[ -f "${FROZEN_RECEIPT}" ]] || { echo "Missing frozen receipt: ${FROZEN_RECEIPT}" >&2; exit 2; }
[[ -f "${MATERIALIZED_ROOT}/manifest.json" ]] || { echo "Missing materialization manifest" >&2; exit 2; }
[[ ! -e "${RECOVERY_MANIFEST}" ]] || { echo "Recovery manifest already exists: ${RECOVERY_MANIFEST}" >&2; exit 2; }

"${PYTHON}" -B "${RECOVERY_GUARD}" check-frozen \
  --receipt "${FROZEN_RECEIPT}" \
  --stage-tag "${STAGE_TAG}" \
  --expected-commit "${EXPECTED_GENERATION_COMMIT}" >/dev/null

read -r NEW_12_CONFIG_ID NEW_14_CONFIG_ID < <(
  "${PYTHON}" -B - "${FROZEN_RECEIPT}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))["configurations"]
print(p["new_12"]["config_id"], p["new_14"]["config_id"])
PY
)

config_id_for_label() {
  case "$1" in
    dense) printf '%s\n' dense ;;
    old_12) printf '%s\n' current_6_23_r3 ;;
    new_12) printf '%s\n' "${NEW_12_CONFIG_ID}" ;;
    old_14) printf '%s\n' current_9_26_r5_anchor ;;
    new_14) printf '%s\n' "${NEW_14_CONFIG_ID}" ;;
    *) echo "Unknown Stage-3 label: $1" >&2; exit 2 ;;
  esac
}

config_for() {
  local config_id="$1" seed="$2"
  local -a matches
  shopt -s nullglob
  matches=("${MATERIALIZED_ROOT}/configs/"*"_${config_id}_"*"_seed${seed}.yaml")
  shopt -u nullglob
  if [[ ${#matches[@]} -ne 1 || ! -f "${matches[0]}" ]]; then
    echo "Expected exactly one materialized config for ${config_id}/seed${seed}" >&2
    exit 2
  fi
  printf '%s\n' "${matches[0]}"
}

validation_passes() {
  local validation="$1" config_id="$2" seed="$3"
  [[ -f "${validation}" ]] || return 1
  "${PYTHON}" -B "${RECOVERY_GUARD}" check-validation \
    --validation "${validation}" --config-id "${config_id}" --seed "${seed}" \
    --expected-commit "${EXPECTED_GENERATION_COMMIT}" >/dev/null 2>&1
}

# Existing planned directories must form one contiguous prefix. Run01 is the
# failed generation that authorizes this recovery path; later gaps may be filled.
seen_missing=0
ordinal=0
for item in "${RUN_PLAN[@]}"; do
  ordinal=$((ordinal + 1))
  seed="${item%%:*}"; label="${item#*:}"
  run_dir="${RUN_ROOT}/${STAGE_TAG}-$(printf '%02d' "${ordinal}")-seed${seed}-${label}"
  if [[ -d "${run_dir}" ]]; then
    if [[ ${seen_missing} -eq 1 ]]; then
      echo "Existing Stage-3 run directories are not a contiguous prefix: ${run_dir}" >&2
      exit 2
    fi
  else
    seen_missing=1
  fi
done
RUN01="${RUN_ROOT}/${STAGE_TAG}-01-seed503-dense"
[[ -d "${RUN01}" ]] || { echo "Recovery requires the existing failed run01: ${RUN01}" >&2; exit 2; }

repair_old_audio_gate_failure() {
  local run_dir="$1" config_id="$2" seed="$3" config="$4"
  local verification="${run_dir}/verification.json"
  local backup="${run_dir}/verification.generation_old_audio_gate.failed.json"
  local gate="${run_dir}/stage3_external_reverification_gate.json"
  local decoded="${run_dir}/decoded_stream_hashes.json"
  local validation="${run_dir}/protocol_validation.json"

  [[ -f "${verification}" ]] || { echo "Missing verification: ${verification}" >&2; return 2; }
  verification_status="$("${PYTHON}" -B -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", ""))' "${verification}")"
  if [[ "${verification_status}" == "failed" ]]; then
    if [[ -f "${backup}" ]]; then
      "${PYTHON}" -B "${RECOVERY_GUARD}" gate \
        --verification "${backup}" --backup "${backup}" --receipt "${gate}" >/dev/null
    else
      "${PYTHON}" -B "${RECOVERY_GUARD}" gate \
        --verification "${verification}" --backup "${backup}" --receipt "${gate}" >/dev/null
    fi
    "${PYTHON}" -B "${EVAL_VERIFY}" --media-only "${run_dir}"
  elif [[ "${verification_status}" != "ok" ]]; then
    echo "Verification has an invalid status: ${verification_status}" >&2
    return 2
  fi
  if [[ ! -f "${backup}" ]]; then
    echo "Passed verification lacks the preserved legacy failure: ${run_dir}" >&2
    return 2
  fi
  "${PYTHON}" -B "${RECOVERY_GUARD}" check-reverified \
    --verification "${verification}" --backup "${backup}" >/dev/null

  if [[ ! -f "${decoded}" ]]; then
    "${PYTHON}" -B "${EVAL_HASH}" "${run_dir}" --output "${decoded}"
  fi
  if [[ ! -f "${validation}" ]]; then
    "${PYTHON}" -B "${EVAL_VALIDATE}" \
      --matrix "${MATRIX}" --config "${config}" --cell-id "${config_id}" \
      --seed "${seed}" --expected-measurements 8 --run-dir "${run_dir}" \
      --output "${validation}"
  fi
  validation_passes "${validation}" "${config_id}" "${seed}"
}

ordinal=0
for item in "${RUN_PLAN[@]}"; do
  ordinal=$((ordinal + 1))
  seed="${item%%:*}"; label="${item#*:}"
  config_id="$(config_id_for_label "${label}")"
  config="$(config_for "${config_id}" "${seed}")"
  run_tag="${STAGE_TAG}-$(printf '%02d' "${ordinal}")-seed${seed}-${label}"
  run_dir="${RUN_ROOT}/${run_tag}"
  validation="${run_dir}/protocol_validation.json"

  if validation_passes "${validation}" "${config_id}" "${seed}"; then
    continue
  fi
  if [[ -d "${run_dir}" ]]; then
    repair_old_audio_gate_failure "${run_dir}" "${config_id}" "${seed}" "${config}" || {
      echo "Existing cell is not safely recoverable: ${run_dir}" >&2
      exit 2
    }
    continue
  fi

  set +e
  (
    cd "${GENERATION_REPO}"
    bash "${GEN_CELL}" \
      --config "${config}" --matrix "${MATRIX}" --cell-id "${config_id}" \
      --seed "${seed}" --run-tag "${run_tag}" --expected-measurements 8
  )
  cell_status=$?
  set -e
  if [[ ${cell_status} -eq 0 ]]; then
    validation_passes "${validation}" "${config_id}" "${seed}" || {
      echo "Generation cell returned success without a passed receipt: ${run_dir}" >&2
      exit 2
    }
  else
    # Any inference/hash/validator failure, any speech-prompt failure, or any
    # non-whitelisted error is rejected by the guard before external revalidation.
    repair_old_audio_gate_failure "${run_dir}" "${config_id}" "${seed}" "${config}" || {
      echo "Generation cell failed outside the legacy p006/p007 audio gates: ${run_dir}" >&2
      exit "${cell_status}"
    }
  fi
done

[[ -f "${RUN01}/verification.generation_old_audio_gate.failed.json" ]] || {
  echo "Run01 completed without preserving its original failed verification" >&2
  exit 2
}

# Recheck all fifteen receipts before allowing identity randomization.
ordinal=0
for item in "${RUN_PLAN[@]}"; do
  ordinal=$((ordinal + 1))
  seed="${item%%:*}"; label="${item#*:}"
  config_id="$(config_id_for_label "${label}")"
  run_dir="${RUN_ROOT}/${STAGE_TAG}-$(printf '%02d' "${ordinal}")-seed${seed}-${label}"
  validation_passes "${run_dir}/protocol_validation.json" "${config_id}" "${seed}" || {
    echo "Final Stage-3 validation sweep failed: ${run_dir}" >&2
    exit 2
  }
done

if [[ ! -d "${BLIND_ROOT}" ]]; then
  (
    cd "${GENERATION_REPO}"
    "${PYTHON}" -B "${GEN_PACKET}" blind-packet \
      --frozen-receipt "${FROZEN_RECEIPT}" --run-root "${RUN_ROOT}" \
      --output-dir "${BLIND_ROOT}"
  )
fi
[[ -f "${BLIND_ROOT}/packet/manifest.json" && -f "${BLIND_ROOT}/private_mapping.json" ]] || {
  echo "Blind packet is incomplete: ${BLIND_ROOT}" >&2
  exit 2
}

EVAL_VERIFY_SHA="$(sha256sum "${EVAL_VERIFY}" | awk '{print $1}')"
"${PYTHON}" -B - \
  "${RECOVERY_MANIFEST}" "${STAGE_TAG}" "${EXPECTED_GENERATION_COMMIT}" \
  "${EVALUATOR_HEAD}" "${EVAL_VERIFY}" "${EVAL_VERIFY_SHA}" "${FROZEN_RECEIPT}" \
  "${RUN_ROOT}" "${BLIND_ROOT}" "${NEW_12_CONFIG_ID}" "${NEW_14_CONFIG_ID}" <<'PY'
import hashlib, json, os, re, sys
from datetime import datetime, timezone
from pathlib import Path

(output, tag, generation_commit, evaluator_commit, evaluator_verifier,
 evaluator_verifier_sha, frozen, run_root, blind_root, new12, new14) = sys.argv[1:]
output = Path(output); run_root = Path(run_root); blind_root = Path(blind_root)
order = {
  503:("dense","old_12","new_12","old_14","new_14"),
  887:("new_12","old_14","new_14","dense","old_12"),
  1291:("new_14","dense","old_12","new_12","old_14"),
}
mapping={"dense":"dense","old_12":"current_6_23_r3","new_12":new12,
         "old_14":"current_9_26_r5_anchor","new_14":new14}
def sha(path):
  h=hashlib.sha256()
  with Path(path).open('rb') as f:
    for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
  return h.hexdigest()
runs=[]; ordinal=0
for seed, labels in order.items():
  for label in labels:
    ordinal += 1
    run_tag=f"{tag}-{ordinal:02d}-seed{seed}-{label}"
    directory=run_root/run_tag
    validation=directory/'protocol_validation.json'
    data=json.loads(validation.read_text())
    if data.get('status')!='passed' or data.get('cell_id')!=mapping[label] or data.get('seed')!=seed:
      raise SystemExit(f"invalid final validation: {validation}")
    backup=directory/'verification.generation_old_audio_gate.failed.json'
    gate=directory/'stage3_external_reverification_gate.json'
    runs.append({"ordinal":ordinal,"seed":seed,"label":label,"config_id":mapping[label],
      "run_tag":run_tag,"validation":{"path":str(validation),"sha256":sha(validation)},
      "external_reverification": bool(backup.exists()),
      "preserved_failed_verification": ({"path":str(backup),"sha256":sha(backup)} if backup.exists() else None),
      "reverification_gate": ({"path":str(gate),"sha256":sha(gate)} if gate.exists() else None)})
packet=blind_root/'packet/manifest.json'; private=blind_root/'private_mapping.json'
packet_data=json.loads(packet.read_text())
if packet_data.get('pair_count') != 48 or packet_data.get('rating_row_count') != 144:
  raise SystemExit('blind packet count mismatch')
payload={"schema_version":1,"record_type":"ovi_cfg_ablation_v2_stage3_recovery_manifest",
 "status":"machine_complete_human_pending","completed_at_utc":datetime.now(timezone.utc).isoformat(),
 "stage_tag":tag,"generation_commit":generation_commit,
 "evaluator":{"commit":evaluator_commit,"verifier_path":evaluator_verifier,"verifier_sha256":evaluator_verifier_sha},
 "frozen_receipt":{"path":frozen,"sha256":sha(frozen)},"runs":runs,
 "blind_packet":{"root":str(blind_root),"manifest_sha256":sha(packet),"private_mapping_sha256":sha(private)},
 "pending_evaluations":packet_data.get('pending_evaluations')}
encoded=(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
with os.fdopen(fd,'wb') as f: f.write(encoded); f.flush(); os.fsync(f.fileno())
PY

echo "Stage 3 recovery complete: ${RECOVERY_MANIFEST}"
echo "Blind reviewer packet: ${BLIND_ROOT}/packet"
