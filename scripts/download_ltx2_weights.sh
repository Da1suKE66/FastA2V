#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/ltx2_env.sh"

HF_REPO="Lightricks/LTX-2.3"
HF_REVISION="76730e634e70a28f4e8d51f5e29c08e40e2d8e74"
GEMMA_REPO="google/gemma-3-12b-it-qat-q4_0-unquantized"
GEMMA_REVISION="e6bcb2d431337974ca112a1ece124601d5f9c44d"
HF_BIN="${FASTA2V_LTX2_HF_BIN:-${FASTA2V_LTX2_ENV}/bin/hf}"
MAX_ATTEMPTS="${FASTA2V_LTX2_DOWNLOAD_ATTEMPTS:-3}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ ! "${MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "FASTA2V_LTX2_DOWNLOAD_ATTEMPTS must be a positive integer." >&2
  exit 2
fi
if [[ ! -x "${HF_BIN}" ]]; then
  HF_BIN="$(command -v hf || true)"
fi
if [[ -z "${HF_BIN}" || ! -x "${HF_BIN}" ]]; then
  echo "The Hugging Face hf CLI is required; run scripts/setup_ltx2_env.sh first." >&2
  exit 2
fi

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${path}" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${path}" | awk '{print $1}'
  else
    echo "No SHA-256 utility is available." >&2
    return 127
  fi
}

file_matches() {
  local path="$1"
  local expected_bytes="$2"
  local expected_sha256="$3"
  local actual_bytes actual_sha256

  [[ -f "${path}" ]] || return 1
  actual_bytes="$(wc -c < "${path}" | tr -d '[:space:]')"
  [[ "${actual_bytes}" == "${expected_bytes}" ]] || return 1
  actual_sha256="$(sha256_file "${path}")"
  [[ "${actual_sha256}" == "${expected_sha256}" ]]
}

download_one() {
  local filename="$1"
  local expected_bytes="$2"
  local expected_sha256="$3"
  local path="${FASTA2V_LTX2_CHECKPOINT_ROOT}/${filename}"
  local attempt
  local -a force_args

  if file_matches "${path}" "${expected_bytes}" "${expected_sha256}"; then
    echo "Verified existing ${filename}."
    return 0
  fi
  if [[ -e "${path}" && ! -f "${path}" ]]; then
    echo "Checkpoint path exists but is not a regular file: ${path}" >&2
    return 1
  fi
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    force_args=()
    if [[ -f "${path}" ]]; then
      force_args=(--force-download)
    fi
    echo "Downloading ${filename} (${attempt}/${MAX_ATTEMPTS}) from ${HF_ENDPOINT}."
    if "${HF_BIN}" download \
      "${HF_REPO}" \
      "${filename}" \
      --revision "${HF_REVISION}" \
      --local-dir "${FASTA2V_LTX2_CHECKPOINT_ROOT}" \
      "${force_args[@]}" && \
      file_matches "${path}" "${expected_bytes}" "${expected_sha256}"; then
      echo "Verified ${filename}."
      return 0
    fi
    echo "Download or verification failed for ${filename}." >&2
  done

  if [[ -f "${path}" ]]; then
    echo "Expected ${expected_bytes} bytes and SHA-256 ${expected_sha256}." >&2
    echo "Found $(wc -c < "${path}" | tr -d '[:space:]') bytes and SHA-256 $(sha256_file "${path}")." >&2
  else
    echo "The hf CLI returned without creating ${path}." >&2
  fi
  return 1
}

mkdir -p "${FASTA2V_LTX2_CHECKPOINT_ROOT}"
download_one \
  "ltx-2.3-22b-distilled-1.1.safetensors" \
  "46149345334" \
  "b33b7fe4bbfe084f484be4aaf90b0f1d95dca20d403ac4c0e037eb8c4f0af7cc"
download_one \
  "ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
  "995743560" \
  "5f416311fa8172b65af67530758964708d29a317b830d689a51143b7f91913ed"
download_one \
  "ltx-2.3-22b-dev.safetensors" \
  "46149344974" \
  "7ab7225325bc403448ea84b6db2269811a880e5118cd2ee2b6282a93d585016f"

cat <<EOF
Public LTX-2.3 checkpoints are ready in ${FASTA2V_LTX2_CHECKPOINT_ROOT}.

Gemma is gated and is intentionally not downloaded by this script. Accept the
terms at https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized,
then authenticate and download the pinned revision from the official endpoint:

  HF_HOME="${HF_HOME}" HF_ENDPOINT=https://huggingface.co "${HF_BIN}" auth login
  HF_HOME="${HF_HOME}" HF_ENDPOINT=https://huggingface.co "${HF_BIN}" download "${GEMMA_REPO}" --revision "${GEMMA_REVISION}" --local-dir "${FASTA2V_LTX2_GEMMA_ROOT}"
EOF
