#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/env.sh"

UPSTREAM_URL="https://github.com/mit-han-lab/radial-attention.git"
UPSTREAM_CLONE_URL="ssh://git@ssh.github.com:443/mit-han-lab/radial-attention.git"
GITHUB_SSH_KEY="/home/ma-user/.ssh/id_ed25519_github"
GITHUB_SSH_COMMAND="ssh -i ${GITHUB_SSH_KEY} -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o BatchMode=yes"
PIN_FILE="${REPO_ROOT}/third_party/radial-attention.commit"
PATCH_FILE="${REPO_ROOT}/third_party/radial-attention-optional-imports.patch"
PINNED_COMMIT="$(tr -d '[:space:]' < "${PIN_FILE}")"
SOURCE_DIR="${FASTA2V_CACHE_ROOT}/sources/radial-attention-${PINNED_COMMIT}"
DERIVED_DIR="${FASTA2V_CACHE_ROOT}/derived/radial-attention-${PINNED_COMMIT}"
SOURCE_MODULE="${SOURCE_DIR}/radial_attn/attn_mask.py"
DERIVED_MODULE="${DERIVED_DIR}/radial_attn/attn_mask.py"
RECEIPT_PATH="${FASTA2V_CACHE_ROOT}/radialattn-install.json"
FLASHINFER_MANIFEST_PATH="${FASTA2V_CACHE_ROOT}/radial-flashinfer-manifest.json"
FLASHINFER_VERSION="0.2.5+cu124torch2.6"
FLASHINFER_INDEX="https://flashinfer.ai/whl/cu124/torch2.6/"
EXPECTED_SOURCE_SHA256="663dd94c8be0b20d8ab71c56209f0d03514b2fb90d4a2dfdb2cfaf3238b529ee"
EXPECTED_PATCH_SHA256="2adf006c3a81600ecf3bc0c228372385b1c99009fc0c30be95ee45c2bd208997"
EXPECTED_DERIVED_SHA256="aafac6551f0a73a7548ed7ec987d718c17cf1269605e454af7d5089b4f9263c5"

if [[ ! "${PINNED_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Invalid Radial Attention pin: ${PINNED_COMMIT}" >&2
  exit 2
fi
if [[ ! -x "${FASTA2V_OVI_ENV}/bin/python" ]]; then
  echo "Fixed Ovi environment is missing: ${FASTA2V_OVI_ENV}" >&2
  exit 2
fi
if [[ ! -r "${GITHUB_SSH_KEY}" ]]; then
  echo "GitHub SSH key is missing or unreadable: ${GITHUB_SSH_KEY}" >&2
  exit 2
fi
if ! command -v patch >/dev/null 2>&1; then
  echo "Required patch executable was not found" >&2
  exit 2
fi
if ! command -v ldd >/dev/null 2>&1; then
  echo "Required ldd executable was not found" >&2
  exit 2
fi
if [[ "$(sha256sum "${PATCH_FILE}" | awk '{print $1}')" != "${EXPECTED_PATCH_SHA256}" ]]; then
  echo "Radial optional-imports patch hash mismatch" >&2
  exit 2
fi

mkdir -p "${FASTA2V_CACHE_ROOT}/sources" "${FASTA2V_CACHE_ROOT}/derived"
if [[ ! -e "${SOURCE_DIR}/.git" ]]; then
  if [[ -e "${SOURCE_DIR}" ]]; then
    echo "Refusing non-git Radial source path: ${SOURCE_DIR}" >&2
    exit 2
  fi
  git -c "core.sshCommand=${GITHUB_SSH_COMMAND}" \
    clone "${UPSTREAM_CLONE_URL}" "${SOURCE_DIR}"
fi
if [[ -n "$(git -C "${SOURCE_DIR}" status --porcelain)" ]]; then
  echo "Refusing dirty Radial source checkout: ${SOURCE_DIR}" >&2
  exit 2
fi
if [[ "$(git -C "${SOURCE_DIR}" remote get-url origin)" != "${UPSTREAM_CLONE_URL}" ]]; then
  echo "Unexpected Radial origin in ${SOURCE_DIR}" >&2
  exit 2
fi
git -c "core.sshCommand=${GITHUB_SSH_COMMAND}" -C "${SOURCE_DIR}" \
  fetch --depth 1 origin "${PINNED_COMMIT}"
git -C "${SOURCE_DIR}" checkout --detach "${PINNED_COMMIT}"
if [[ "$(git -C "${SOURCE_DIR}" rev-parse HEAD)" != "${PINNED_COMMIT}" ]]; then
  echo "Radial checkout does not match the fixed commit" >&2
  exit 2
fi
if [[ -n "$(git -C "${SOURCE_DIR}" status --porcelain)" ]]; then
  echo "Pinned Radial source checkout became dirty" >&2
  exit 2
fi
if [[ "$(sha256sum "${SOURCE_MODULE}" | awk '{print $1}')" != "${EXPECTED_SOURCE_SHA256}" ]]; then
  echo "Pinned Radial source module hash mismatch" >&2
  exit 2
fi

if [[ ! -e "${DERIVED_MODULE}" ]]; then
  if [[ -e "${DERIVED_DIR}" ]]; then
    echo "Refusing incomplete derived Radial path: ${DERIVED_DIR}" >&2
    exit 2
  fi
  mkdir -p "${DERIVED_DIR}/radial_attn"
  cp "${SOURCE_MODULE}" "${DERIVED_MODULE}"
  patch --batch --forward --fuzz=0 -p1 -d "${DERIVED_DIR}" \
    < "${PATCH_FILE}"
fi
if [[ "$(sha256sum "${DERIVED_MODULE}" | awk '{print $1}')" != "${EXPECTED_DERIVED_SHA256}" ]]; then
  echo "Derived Radial source differs from the audited optional-imports patch" >&2
  exit 2
fi

# Wheel installation does not execute an attention kernel.  The formal runner
# owns the physical-GPU idle guard; runtime shape/API checks remain fail-closed.
"${FASTA2V_OVI_ENV}/bin/python" -m pip install \
  --no-deps \
  --force-reinstall \
  --index-url "${FLASHINFER_INDEX}" \
  "flashinfer-python==${FLASHINFER_VERSION}"

FASTA2V_REPO_ROOT="${REPO_ROOT}" \
RADIAL_SOURCE_DIR="${SOURCE_DIR}" \
RADIAL_DERIVED_DIR="${DERIVED_DIR}" \
RADIAL_SOURCE_MODULE="${SOURCE_MODULE}" \
RADIAL_DERIVED_MODULE="${DERIVED_MODULE}" \
RADIAL_PATCH_FILE="${PATCH_FILE}" \
RADIAL_RECEIPT_PATH="${RECEIPT_PATH}" \
RADIAL_FLASHINFER_MANIFEST_PATH="${FLASHINFER_MANIFEST_PATH}" \
RADIAL_UPSTREAM_URL="${UPSTREAM_URL}" \
RADIAL_CLONE_URL="${UPSTREAM_CLONE_URL}" \
RADIAL_PINNED_COMMIT="${PINNED_COMMIT}" \
"${FASTA2V_OVI_ENV}/bin/python" - <<'PY'
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import sysconfig
from datetime import datetime, timezone
from pathlib import Path

import torch

repo_root = Path(os.environ["FASTA2V_REPO_ROOT"])
sys.path.insert(0, str(repo_root))
from ovi.modules.radial_attention_backend import audit_and_repair_radial_mask
from ovi.radial_evidence import (
    FLASHINFER_MANIFEST_SCHEMA,
    FLASHINFER_REQUIRED_APIS,
    FLASHINFER_VERSION,
    RADIAL_BLOCK_SIZE,
    RADIAL_GRID,
    RADIAL_MODEL_TYPE,
    RADIAL_PROFILE_AUDITS,
    RADIAL_SEQUENCE,
    normalize_ldd_output,
)

if Path(sys.prefix).resolve() != Path(os.environ["FASTA2V_OVI_ENV"]).resolve():
    raise RuntimeError("Radial installer did not use the fixed Ovi environment")
if importlib.metadata.version("flashinfer-python") != FLASHINFER_VERSION:
    raise RuntimeError("installed FlashInfer distribution differs from fixed candidate")
import flashinfer
missing = [name for name in FLASHINFER_REQUIRED_APIS if not callable(getattr(flashinfer, name, None))]
if missing:
    raise RuntimeError(f"fixed FlashInfer candidate lacks APIs: {missing}")

derived_module_path = Path(os.environ["RADIAL_DERIVED_MODULE"])
spec = importlib.util.spec_from_file_location(
    "fasta2v_radial_install_audit", derived_module_path
)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load derived Radial source")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
mask_generator = getattr(module, "gen_log_mask_shrinked", None)
if not callable(mask_generator):
    raise RuntimeError("derived Radial source lacks gen_log_mask_shrinked")

query_device = torch.empty(0, device="cpu")
cpu_mask_audits = {}
for profile, expected in RADIAL_PROFILE_AUDITS.items():
    mask = mask_generator(
        query_device,
        RADIAL_SEQUENCE,
        RADIAL_SEQUENCE,
        RADIAL_GRID[0],
        block_size=RADIAL_BLOCK_SIZE,
        sparse_type="radial",
        decay_factor=expected["decay_factor"],
        model_type=RADIAL_MODEL_TYPE,
    )
    _repaired, audit = audit_and_repair_radial_mask(mask, profile)
    cpu_mask_audits[profile] = audit

def fingerprint(path):
    path = Path(path).resolve()
    payload = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }

flashinfer_module_path = Path(flashinfer.__file__).resolve()
flashinfer_package_root = flashinfer_module_path.parent
allowed_site_roots = {
    Path(path).resolve()
    for path in (sysconfig.get_path("purelib"), sysconfig.get_path("platlib"))
    if path
}
if not any(
    flashinfer_package_root == root or root in flashinfer_package_root.parents
    for root in allowed_site_roots
):
    raise RuntimeError(
        "flashinfer was imported outside the fixed Ovi environment: "
        f"{flashinfer_package_root}"
    )
installed_flashinfer_files = {}
native_flashinfer_files = []
ldd_env = os.environ.copy()
torch_lib = Path(torch.__file__).resolve().parent / "lib"
cuda_lib = Path(os.environ["CUDA_HOME"]) / "lib64"
ldd_env["LD_LIBRARY_PATH"] = ":".join(
    [str(torch_lib), str(cuda_lib), ldd_env.get("LD_LIBRARY_PATH", "")]
).rstrip(":")
for installed_path in sorted(flashinfer_package_root.rglob("*")):
    if (
        not installed_path.is_file()
        or "__pycache__" in installed_path.parts
        or installed_path.suffix == ".pyc"
    ):
        continue
    relative_name = str(installed_path.relative_to(flashinfer_package_root))
    metadata_value = fingerprint(installed_path)
    metadata_value.pop("path")
    if installed_path.suffix == ".so":
        ldd_output = subprocess.check_output(
            ["ldd", str(installed_path)],
            text=True,
            stderr=subprocess.STDOUT,
            env=ldd_env,
        )
        missing_libraries = [
            line.strip()
            for line in ldd_output.splitlines()
            if "not found" in line
        ]
        if missing_libraries:
            raise RuntimeError(
                "FlashInfer native library has unresolved dependencies: "
                f"{missing_libraries}"
            )
        metadata_value["ldd_not_found"] = []
        metadata_value["ldd_output"] = ldd_output
        metadata_value["ldd_normalized_output"] = normalize_ldd_output(
            ldd_output
        )
        metadata_value["ldd_sha256"] = hashlib.sha256(
            metadata_value["ldd_normalized_output"].encode("utf-8")
        ).hexdigest()
        native_flashinfer_files.append(relative_name)
    installed_flashinfer_files[relative_name] = metadata_value
if "__init__.py" not in installed_flashinfer_files:
    raise RuntimeError("installed FlashInfer package lacks __init__.py")
if not native_flashinfer_files:
    raise RuntimeError(
        "fixed FlashInfer candidate exposes no native .so to hash and ldd-audit"
    )

flashinfer_manifest = {
    "schema": FLASHINFER_MANIFEST_SCHEMA,
    "distribution": "flashinfer-python",
    "version": importlib.metadata.version("flashinfer-python"),
    "wheel_index": "https://flashinfer.ai/whl/cu124/torch2.6/",
    "required_apis": list(FLASHINFER_REQUIRED_APIS),
    "package_root": str(flashinfer_package_root),
    "module": fingerprint(flashinfer_module_path),
    "files": installed_flashinfer_files,
    "native_files": sorted(native_flashinfer_files),
}
flashinfer_manifest_path = Path(
    os.environ["RADIAL_FLASHINFER_MANIFEST_PATH"]
)
flashinfer_manifest_path.write_text(
    json.dumps(
        flashinfer_manifest,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    + "\n",
    encoding="utf-8",
)

receipt = {
    "repository": os.environ["RADIAL_UPSTREAM_URL"],
    "clone_url": os.environ["RADIAL_CLONE_URL"],
    "commit": os.environ["RADIAL_PINNED_COMMIT"],
    "mask_api": "gen_log_mask_shrinked",
    "source_dir": os.environ["RADIAL_SOURCE_DIR"],
    "derived_dir": os.environ["RADIAL_DERIVED_DIR"],
    "source_module": fingerprint(os.environ["RADIAL_SOURCE_MODULE"]),
    "derived_module": fingerprint(os.environ["RADIAL_DERIVED_MODULE"]),
    "optional_imports_patch": fingerprint(os.environ["RADIAL_PATCH_FILE"]),
    "patch_scope": ["radial_attn/attn_mask.py"],
    "patch_purpose": "optional_imports_only",
    "flashinfer_distribution": "flashinfer-python",
    "flashinfer_version": importlib.metadata.version("flashinfer-python"),
    "flashinfer_wheel_index": "https://flashinfer.ai/whl/cu124/torch2.6/",
    "flashinfer_required_apis": list(FLASHINFER_REQUIRED_APIS),
    "installed_flashinfer_package_root": str(flashinfer_package_root),
    "flashinfer_module": fingerprint(flashinfer_module_path),
    "installed_flashinfer_files": installed_flashinfer_files,
    "flashinfer_manifest": fingerprint(flashinfer_manifest_path),
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "model_type": RADIAL_MODEL_TYPE,
    "block_size": RADIAL_BLOCK_SIZE,
    "sequence": RADIAL_SEQUENCE,
    "prefix_sequence": 14976,
    "tail_sequence": 28,
    "grid": list(RADIAL_GRID),
    "cpu_mask_audits": cpu_mask_audits,
    "cuda_kernel_launched": False,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}
path = Path(os.environ["RADIAL_RECEIPT_PATH"])
path.write_text(
    json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
print(f"Wrote audited Radial receipt: {path}")
PY

"${FASTA2V_OVI_ENV}/bin/python" -m pip freeze --all \
  | LC_ALL=C sort > "${FASTA2V_CACHE_ROOT}/ovi-environment.freeze.txt"
