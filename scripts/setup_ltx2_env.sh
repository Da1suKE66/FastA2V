#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/ltx2_env.sh"

PIN_FILE="${REPO_ROOT}/third_party/LTX-2.commit"
LTX2_GIT_URL="${FASTA2V_LTX2_GIT_URL:-ssh://git@ssh.github.com:443/Lightricks/LTX-2.git}"
GITHUB_KEY="${FASTA2V_GITHUB_KEY:-${HOME}/.ssh/id_ed25519_github}"
UV_BIN="${UV_BIN:-/cache/llc/bin/uv}"

if [[ -z "${UV_DEFAULT_INDEX:-}" && -n "${PIP_INDEX_URL:-}" ]]; then
  export UV_DEFAULT_INDEX="${PIP_INDEX_URL}"
fi

IFS= read -r EXPECTED_COMMIT < "${PIN_FILE}"
if [[ ! "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Invalid LTX-2 commit pin in ${PIN_FILE}." >&2
  exit 2
fi

if [[ "${LTX2_GIT_URL}" == ssh://* || "${LTX2_GIT_URL}" == git@* ]]; then
  if [[ ! -r "${GITHUB_KEY}" ]]; then
    echo "GitHub SSH key is not readable: ${GITHUB_KEY}" >&2
    exit 2
  fi
  export GIT_SSH_COMMAND="ssh -i ${GITHUB_KEY} -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -p 443"
fi
export GIT_TERMINAL_PROMPT=0

mkdir -p "$(dirname "${FASTA2V_LTX2_SOURCE}")"
if [[ -e "${FASTA2V_LTX2_SOURCE}" ]]; then
  if ! git -C "${FASTA2V_LTX2_SOURCE}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "LTX-2 source path exists but is not a Git checkout: ${FASTA2V_LTX2_SOURCE}" >&2
    exit 1
  fi
else
  git clone --filter=blob:none "${LTX2_GIT_URL}" "${FASTA2V_LTX2_SOURCE}"
fi

if [[ -n "$(git -C "${FASTA2V_LTX2_SOURCE}" status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing to replace a dirty LTX-2 source checkout: ${FASTA2V_LTX2_SOURCE}" >&2
  exit 1
fi

if ! git -C "${FASTA2V_LTX2_SOURCE}" cat-file -e "${EXPECTED_COMMIT}^{commit}" 2>/dev/null; then
  git -C "${FASTA2V_LTX2_SOURCE}" fetch --no-tags --depth=1 origin "${EXPECTED_COMMIT}"
fi
git -C "${FASTA2V_LTX2_SOURCE}" checkout --detach "${EXPECTED_COMMIT}"

ACTUAL_COMMIT="$(git -C "${FASTA2V_LTX2_SOURCE}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "LTX-2 commit mismatch: expected ${EXPECTED_COMMIT}, got ${ACTUAL_COMMIT}." >&2
  exit 1
fi

if [[ ! -x "${FASTA2V_LTX2_PYTHON}" ]]; then
  echo "Python 3.12 is not executable: ${FASTA2V_LTX2_PYTHON}" >&2
  exit 2
fi
PYTHON_VERSION="$("${FASTA2V_LTX2_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PYTHON_VERSION}" != "3.12" ]]; then
  echo "LTX-2 requires Python 3.12; found ${PYTHON_VERSION}." >&2
  exit 2
fi

if [[ ! -x "${UV_BIN}" ]]; then
  UV_BIN="$(command -v uv || true)"
fi
if [[ -z "${UV_BIN}" || ! -x "${UV_BIN}" ]]; then
  echo "uv is required; set UV_BIN to an executable uv binary." >&2
  exit 2
fi

(
  cd "${FASTA2V_LTX2_SOURCE}"
  "${UV_BIN}" sync \
    --frozen \
    --package ltx-pipelines \
    --no-dev \
    --python "${FASTA2V_LTX2_PYTHON}"
)

"${FASTA2V_LTX2_ENV}/bin/python" -c \
  'import ltx_core, ltx_pipelines, torch; print(f"LTX-2 ready: torch={torch.__version__}, cuda={torch.version.cuda}")'
