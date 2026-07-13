#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/env.sh"

if [[ -n "${FASTA2V_HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT="${FASTA2V_HF_ENDPOINT}"
fi

cd "${REPO_ROOT}"
MAX_ATTEMPTS="${FASTA2V_DOWNLOAD_ATTEMPTS:-10}"
if [[ ! "${MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "FASTA2V_DOWNLOAD_ATTEMPTS must be a positive integer." >&2
  exit 2
fi
download_complete=false
for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
  if "${FASTA2V_OVI_ENV}/bin/python" download_weights.py \
    --models 720x720_5s \
    --output-dir "${FASTA2V_CACHE_ROOT}/ckpts"; then
    download_complete=true
    break
  fi
  echo "Weight download attempt ${attempt}/${MAX_ATTEMPTS} failed; retrying resumably." >&2
  sleep 5
done

if [[ "${download_complete}" != true ]]; then
  echo "Weight download failed after ${MAX_ATTEMPTS} attempts." >&2
  exit 1
fi

"${FASTA2V_OVI_ENV}/bin/python" scripts/write_checkpoint_manifest.py
