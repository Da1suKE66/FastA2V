#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/env.sh"

UPSTREAM_URL="https://github.com/thu-ml/SpargeAttn.git"
UPSTREAM_CLONE_URL="ssh://git@ssh.github.com:443/thu-ml/SpargeAttn.git"
GITHUB_SSH_KEY="/home/ma-user/.ssh/id_ed25519_github"
GITHUB_SSH_COMMAND="ssh -i ${GITHUB_SSH_KEY} -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o BatchMode=yes"
PIN_FILE="${REPO_ROOT}/third_party/SpargeAttn.commit"
PINNED_COMMIT="$(tr -d '[:space:]' < "${PIN_FILE}")"
SOURCE_PARENT="${FASTA2V_CACHE_ROOT}/sources"
SOURCE_DIR="${SOURCE_PARENT}/SpargeAttn-${PINNED_COMMIT}"
RECEIPT_PATH="${FASTA2V_CACHE_ROOT}/spargeattn-install.json"
BUILD_LOG_PATH="${FASTA2V_CACHE_ROOT}/spargeattn-build.log"

if [[ ! "${PINNED_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Invalid SpargeAttn pin in ${PIN_FILE}: ${PINNED_COMMIT}" >&2
  exit 2
fi
if [[ ! -x "${FASTA2V_OVI_ENV}/bin/python" ]]; then
  echo "Ovi environment is missing: ${FASTA2V_OVI_ENV}" >&2
  echo "Run bash scripts/setup_ovi_env.sh first." >&2
  exit 2
fi
if [[ ! -r "${GITHUB_SSH_KEY}" ]]; then
  echo "GitHub SSH key is missing or unreadable: ${GITHUB_SSH_KEY}" >&2
  exit 2
fi

mkdir -p "${SOURCE_PARENT}"
if [[ ! -e "${SOURCE_DIR}/.git" ]]; then
  if [[ -e "${SOURCE_DIR}" ]]; then
    echo "Refusing non-git source path: ${SOURCE_DIR}" >&2
    exit 2
  fi
  git -c "core.sshCommand=${GITHUB_SSH_COMMAND}" \
    clone "${UPSTREAM_CLONE_URL}" "${SOURCE_DIR}"
fi

if [[ -n "$(git -C "${SOURCE_DIR}" status --porcelain)" ]]; then
  echo "Refusing to alter dirty SpargeAttn source tree: ${SOURCE_DIR}" >&2
  exit 2
fi
if [[ "$(git -C "${SOURCE_DIR}" remote get-url origin)" != "${UPSTREAM_CLONE_URL}" ]]; then
  echo "Unexpected SpargeAttn origin in ${SOURCE_DIR}" >&2
  exit 2
fi

git -c "core.sshCommand=${GITHUB_SSH_COMMAND}" -C "${SOURCE_DIR}" \
  fetch --depth 1 origin "${PINNED_COMMIT}"
git -C "${SOURCE_DIR}" checkout --detach "${PINNED_COMMIT}"
ACTUAL_COMMIT="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${PINNED_COMMIT}" ]]; then
  echo "SpargeAttn checkout mismatch: ${ACTUAL_COMMIT}" >&2
  exit 2
fi

# Build only the official extension.  Setting the architecture explicitly
# makes setup.py reproducible without asking it to enumerate/initialize GPUs.
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS="${MAX_JOBS:-2}"
if [[ ! "${MAX_JOBS}" =~ ^[1-4]$ ]]; then
  echo "MAX_JOBS must be an integer from 1 through 4 for this audited build." >&2
  exit 2
fi
if ! "${FASTA2V_OVI_ENV}/bin/python" -c 'import ninja, packaging'; then
  echo "Missing build prerequisites in the fixed Ovi environment." >&2
  echo "Run bash scripts/setup_ovi_env.sh before installing SpargeAttn." >&2
  exit 2
fi
"${FASTA2V_OVI_ENV}/bin/python" -m pip install \
  --no-build-isolation \
  --no-deps \
  --force-reinstall \
  "${SOURCE_DIR}" \
  2>&1 | tee "${BUILD_LOG_PATH}"

SPARGEATTN_SOURCE_DIR="${SOURCE_DIR}" \
SPARGEATTN_RECEIPT_PATH="${RECEIPT_PATH}" \
SPARGEATTN_PINNED_COMMIT="${PINNED_COMMIT}" \
SPARGEATTN_UPSTREAM_URL="${UPSTREAM_URL}" \
SPARGEATTN_CLONE_URL="${UPSTREAM_CLONE_URL}" \
SPARGEATTN_BUILD_LOG_PATH="${BUILD_LOG_PATH}" \
FASTA2V_REPO_ROOT="${REPO_ROOT}" \
"${FASTA2V_OVI_ENV}/bin/python" - <<'PY'
import importlib.metadata
import hashlib
import inspect
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import spas_sage_attn
from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda

sys.path.insert(0, os.environ["FASTA2V_REPO_ROOT"])
from scripts.sparge_attn_microtest import run_microtest

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
microtest = run_microtest(
    kernel=spas_sage2_attn_meansim_topk_cuda,
    device_index=0,
)
installed_files = {}
ldd_env = os.environ.copy()
torch_lib = Path(torch.__file__).resolve().parent / "lib"
cuda_lib = Path(os.environ["CUDA_HOME"]) / "lib64"
ldd_env["LD_LIBRARY_PATH"] = ":".join(
    [str(torch_lib), str(cuda_lib), ldd_env.get("LD_LIBRARY_PATH", "")]
).rstrip(":")
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
    if path.suffix == ".so":
        ldd_output = subprocess.check_output(
            ["ldd", str(path)],
            text=True,
            stderr=subprocess.STDOUT,
            env=ldd_env,
        )
        missing = [
            line.strip()
            for line in ldd_output.splitlines()
            if "not found" in line
        ]
        if missing:
            raise RuntimeError(f"unresolved shared-library dependency: {missing}")
        installed_files[str(path.relative_to(package_root))]["ldd_not_found"] = []

build_log_path = Path(os.environ["SPARGEATTN_BUILD_LOG_PATH"])
build_log_digest = hashlib.sha256(build_log_path.read_bytes()).hexdigest()

def command_output(command):
    return subprocess.check_output(
        command,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()

receipt = {
    "repository": os.environ["SPARGEATTN_UPSTREAM_URL"],
    "clone_url": os.environ["SPARGEATTN_CLONE_URL"],
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
    "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
    "triton": importlib.metadata.version("triton"),
    "cuda_home": os.environ.get("CUDA_HOME"),
    "nvcc_version": command_output([os.environ["CUDA_HOME"] + "/bin/nvcc", "--version"]),
    "gcc_version": command_output(["gcc", "--version"]).splitlines()[0],
    "gxx_version": command_output(["g++", "--version"]).splitlines()[0],
    "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
    "max_jobs": int(os.environ["MAX_JOBS"]),
    "build_log": {
        "path": str(build_log_path),
        "bytes": build_log_path.stat().st_size,
        "sha256": build_log_digest,
    },
    "microtest": microtest,
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
