#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/env.sh"

UPSTREAM_URL="https://github.com/thu-ml/SpargeAttn.git"
PIN_FILE="${REPO_ROOT}/third_party/SpargeAttn.commit"
PINNED_COMMIT="$(tr -d '[:space:]' < "${PIN_FILE}")"
SOURCE_PARENT="${FASTA2V_CACHE_ROOT}/sources"
SOURCE_DIR="${SOURCE_PARENT}/SpargeAttn-${PINNED_COMMIT}"
RECEIPT_PATH="${FASTA2V_CACHE_ROOT}/spargeattn-install.json"

if [[ ! "${PINNED_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Invalid SpargeAttn pin in ${PIN_FILE}: ${PINNED_COMMIT}" >&2
  exit 2
fi
if [[ ! -x "${FASTA2V_OVI_ENV}/bin/python" ]]; then
  echo "Ovi environment is missing: ${FASTA2V_OVI_ENV}" >&2
  echo "Run bash scripts/setup_ovi_env.sh first." >&2
  exit 2
fi

mkdir -p "${SOURCE_PARENT}"
if [[ ! -e "${SOURCE_DIR}/.git" ]]; then
  if [[ -e "${SOURCE_DIR}" ]]; then
    echo "Refusing non-git source path: ${SOURCE_DIR}" >&2
    exit 2
  fi
  git clone "${UPSTREAM_URL}" "${SOURCE_DIR}"
fi

if [[ -n "$(git -C "${SOURCE_DIR}" status --porcelain)" ]]; then
  echo "Refusing to alter dirty SpargeAttn source tree: ${SOURCE_DIR}" >&2
  exit 2
fi
if [[ "$(git -C "${SOURCE_DIR}" remote get-url origin)" != "${UPSTREAM_URL}" ]]; then
  echo "Unexpected SpargeAttn origin in ${SOURCE_DIR}" >&2
  exit 2
fi

git -C "${SOURCE_DIR}" fetch --depth 1 origin "${PINNED_COMMIT}"
git -C "${SOURCE_DIR}" checkout --detach "${PINNED_COMMIT}"
ACTUAL_COMMIT="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${PINNED_COMMIT}" ]]; then
  echo "SpargeAttn checkout mismatch: ${ACTUAL_COMMIT}" >&2
  exit 2
fi

# Build only the official extension.  Setting the architecture explicitly
# makes setup.py reproducible without asking it to enumerate/initialize GPUs.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-8}"
if ! "${FASTA2V_OVI_ENV}/bin/python" -c 'import ninja, packaging'; then
  echo "Missing build prerequisites in the fixed Ovi environment." >&2
  echo "Run bash scripts/setup_ovi_env.sh before installing SpargeAttn." >&2
  exit 2
fi
"${FASTA2V_OVI_ENV}/bin/python" -m pip install \
  --no-build-isolation \
  --no-deps \
  --force-reinstall \
  "${SOURCE_DIR}"

SPARGEATTN_SOURCE_DIR="${SOURCE_DIR}" \
SPARGEATTN_RECEIPT_PATH="${RECEIPT_PATH}" \
SPARGEATTN_PINNED_COMMIT="${PINNED_COMMIT}" \
SPARGEATTN_UPSTREAM_URL="${UPSTREAM_URL}" \
"${FASTA2V_OVI_ENV}/bin/python" - <<'PY'
import importlib.metadata
import hashlib
import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import spas_sage_attn
from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda

required = {
    "dropout_p",
    "topk",
    "is_causal",
    "pvthreshd",
    "smooth_k",
    "tensor_layout",
    "return_sparsity",
}
parameters = inspect.signature(spas_sage2_attn_meansim_topk_cuda).parameters
missing = sorted(required - set(parameters))
if missing:
    raise RuntimeError(f"official SpargeAttn API is missing parameters: {missing}")

package_root = Path(spas_sage_attn.__file__).resolve().parent
installed_files = {}
for path in sorted(package_root.rglob("*")):
    if not path.is_file() or path.suffix not in {".py", ".so"}:
        continue
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    installed_files[str(path.relative_to(package_root))] = {
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }

receipt = {
    "repository": os.environ["SPARGEATTN_UPSTREAM_URL"],
    "commit": os.environ["SPARGEATTN_PINNED_COMMIT"],
    "api": "spas_sage2_attn_meansim_topk_cuda",
    "package": "spas_sage_attn",
    "package_version": importlib.metadata.version("spas_sage_attn"),
    "source_dir": os.environ["SPARGEATTN_SOURCE_DIR"],
    "installed_package_root": str(package_root),
    "installed_files": installed_files,
    "python": os.sys.version.split()[0],
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_home": os.environ.get("CUDA_HOME"),
    "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
    "installed_at_utc": datetime.now(timezone.utc).isoformat(),
}
path = Path(os.environ["SPARGEATTN_RECEIPT_PATH"])
path.write_text(
    json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
PY

"${FASTA2V_OVI_ENV}/bin/python" -m pip freeze \
  > "${FASTA2V_CACHE_ROOT}/ovi-environment.freeze.txt"
echo "Installed official SpargeAttn ${PINNED_COMMIT}"
echo "Receipt: ${RECEIPT_PATH}"
