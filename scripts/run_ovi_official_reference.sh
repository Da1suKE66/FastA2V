#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/env.sh"

OFFICIAL_COMMIT="5b69b25a4b3115216e9ea53a37a04410be6ad39a"
OFFICIAL_WORKTREE="${FASTA2V_CACHE_ROOT}/reference/ovi-official-5b69b25"
if [[ ! -e "${OFFICIAL_WORKTREE}/.git" ]]; then
  mkdir -p "$(dirname "${OFFICIAL_WORKTREE}")"
  git -C "${REPO_ROOT}" worktree add --detach \
    "${OFFICIAL_WORKTREE}" "${OFFICIAL_COMMIT}"
fi
if [[ "$(git -C "${OFFICIAL_WORKTREE}" rev-parse HEAD)" != "${OFFICIAL_COMMIT}" ]]; then
  echo "Official Ovi worktree is not pinned to ${OFFICIAL_COMMIT}." >&2
  exit 2
fi

RUN_TAG="${FASTA2V_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid FASTA2V_RUN_TAG: ${RUN_TAG}" >&2
  exit 2
fi
RUN_PARENT="${FASTA2V_CACHE_ROOT}/reference/runs/ovi_official_720ckpt_smoke_20step"
RUN_DIR="${RUN_PARENT}/${RUN_TAG}"
mkdir -p "${RUN_PARENT}"
if ! mkdir "${RUN_DIR}"; then
  echo "Refusing to reuse existing run directory: ${RUN_DIR}" >&2
  exit 2
fi

export FASTA2V_OFFICIAL_RUN_DIR="${RUN_DIR}"
export FASTA2V_PROMPT_FILE="${REPO_ROOT}/prompts/ovi_smoke.csv"
cp "${FASTA2V_CACHE_ROOT}/ovi-environment.freeze.txt" "${RUN_DIR}/environment.freeze.txt"
cp "${FASTA2V_CACHE_ROOT}/checkpoint_manifest.json" "${RUN_DIR}/checkpoint_manifest.json"
git -C "${OFFICIAL_WORKTREE}" rev-parse HEAD > "${RUN_DIR}/official_commit.txt"

cd "${OFFICIAL_WORKTREE}"
"${FASTA2V_OVI_ENV}/bin/python" inference.py \
  --config-file "${REPO_ROOT}/configs/ovi_720x720_5s_official_smoke.yaml" \
  2>&1 | tee "${RUN_DIR}/stdout.log"
# The pinned, unmodified upstream saver exposes MoviePy 1.0.3's endpoint
# rounding quirk: a 121-frame tensor is muxed with one duplicate tail frame.
"${FASTA2V_OVI_ENV}/bin/python" "${REPO_ROOT}/scripts/verify_ovi_output.py" \
  "${RUN_DIR}" --media-only --expected-video-frames 122
echo "Official reference artifacts: ${RUN_DIR}"
