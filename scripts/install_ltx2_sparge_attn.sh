#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/ltx2_env.sh"

PIN_FILE="${REPO_ROOT}/third_party/SpargeAttn.commit"
PINNED_COMMIT="$(tr -d '[:space:]' < "${PIN_FILE}")"
SOURCE_DIR="${FASTA2V_CACHE_ROOT}/sources/SpargeAttn-${PINNED_COMMIT}"
GIT_URL="${FASTA2V_SPARGE_GIT_URL:-ssh://git@ssh.github.com:443/thu-ml/SpargeAttn.git}"
GITHUB_KEY="${FASTA2V_GITHUB_KEY:-${HOME}/.ssh/id_ed25519_github}"
UV_BIN="${UV_BIN:-/cache/llc/bin/uv}"
BUILD_LOG="${FASTA2V_CACHE_ROOT}/logs/ltx2-spargeattn-build.log"
SMOKE_LOG="${FASTA2V_CACHE_ROOT}/logs/ltx2-spargeattn-smoke.log"

if [[ ! "${PINNED_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Invalid SpargeAttn commit pin in ${PIN_FILE}." >&2
  exit 2
fi
if [[ ! -x "${FASTA2V_LTX2_ENV}/bin/python" ]]; then
  echo "LTX environment is missing; run scripts/setup_ltx2_env.sh first." >&2
  exit 2
fi
if [[ ! -x "${UV_BIN}" ]]; then
  UV_BIN="$(command -v uv || true)"
fi
if [[ -z "${UV_BIN}" || ! -x "${UV_BIN}" ]]; then
  echo "uv is required; set UV_BIN to an executable uv binary." >&2
  exit 2
fi
if [[ ! -r "${GITHUB_KEY}" ]]; then
  echo "GitHub SSH key is not readable: ${GITHUB_KEY}" >&2
  exit 2
fi

export GIT_SSH_COMMAND="ssh -i ${GITHUB_KEY} -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -p 443"
export GIT_TERMINAL_PROMPT=0
mkdir -p "$(dirname "${SOURCE_DIR}")" "$(dirname "${BUILD_LOG}")"

if [[ -e "${SOURCE_DIR}" ]]; then
  if ! git -C "${SOURCE_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "SpargeAttn source path exists but is not a Git checkout: ${SOURCE_DIR}" >&2
    exit 1
  fi
else
  git clone "${GIT_URL}" "${SOURCE_DIR}"
fi
if [[ -n "$(git -C "${SOURCE_DIR}" status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing to alter dirty SpargeAttn checkout: ${SOURCE_DIR}" >&2
  exit 1
fi
if ! git -C "${SOURCE_DIR}" cat-file -e "${PINNED_COMMIT}^{commit}" 2>/dev/null; then
  git -C "${SOURCE_DIR}" fetch --no-tags --depth=1 origin "${PINNED_COMMIT}"
fi
git -C "${SOURCE_DIR}" checkout --detach "${PINNED_COMMIT}"
if [[ "$(git -C "${SOURCE_DIR}" rev-parse HEAD)" != "${PINNED_COMMIT}" ]]; then
  echo "SpargeAttn checkout does not match the pinned commit." >&2
  exit 1
fi

export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS="${MAX_JOBS:-2}"
if [[ ! "${MAX_JOBS}" =~ ^[1-4]$ ]]; then
  echo "MAX_JOBS must be an integer from 1 through 4." >&2
  exit 2
fi

if [[ -z "${UV_DEFAULT_INDEX:-}" && -n "${PIP_INDEX_URL:-}" ]]; then
  export UV_DEFAULT_INDEX="${PIP_INDEX_URL}"
fi
"${UV_BIN}" pip install \
  --python "${FASTA2V_LTX2_ENV}/bin/python" \
  --no-deps \
  "ninja==1.13.0" \
  "packaging==25.0"
"${FASTA2V_LTX2_ENV}/bin/python" - <<'PY'
import torch

if torch.__version__.split("+")[0] != "2.9.1" or torch.version.cuda != "12.8":
    raise RuntimeError(
        f"pinned LTX SpargeAttn build requires torch 2.9.1 / CUDA 12.8; "
        f"got torch={torch.__version__}, CUDA={torch.version.cuda}"
    )
import ninja  # noqa: F401, E402
import packaging  # noqa: F401, E402

print(f"Building against torch={torch.__version__}, CUDA={torch.version.cuda}")
PY
if ! "${CUDA_HOME}/bin/nvcc" --version | grep -q 'release 12\.8'; then
  echo "LTX SpargeAttn build requires nvcc 12.8 from ${CUDA_HOME}." >&2
  exit 2
fi
"${UV_BIN}" pip install \
  --python "${FASTA2V_LTX2_ENV}/bin/python" \
  --no-build-isolation \
  --no-deps \
  --reinstall \
  "${SOURCE_DIR}" \
  2>&1 | tee "${BUILD_LOG}"

"${FASTA2V_LTX2_ENV}/bin/python" - 2>&1 <<'PY' | tee "${SMOKE_LOG}"
import inspect
import json
import math

import torch
import torch.nn.functional as F
from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda

required = {
    "dropout_p",
    "is_causal",
    "topk",
    "pvthreshd",
    "smooth_k",
    "tensor_layout",
    "return_sparsity",
}
missing = required - set(inspect.signature(spas_sage2_attn_meansim_topk_cuda).parameters)
if missing:
    raise RuntimeError(f"official SpargeAttn API is missing parameters: {sorted(missing)}")

device = torch.device("cuda", torch.cuda.current_device())
if torch.cuda.get_device_capability(device) != (8, 0):
    raise RuntimeError(f"LTX SpargeAttn build targets sm80; got {torch.cuda.get_device_capability(device)}")
generator = torch.Generator(device=device).manual_seed(0)
q, k, v = [
    torch.randn((1, 132, 32, 128), generator=generator, device=device, dtype=torch.bfloat16)
    for _ in range(3)
]
common = {
    "dropout_p": 0.0,
    "is_causal": False,
    "pvthreshd": 50.0,
    "smooth_k": True,
    "tensor_layout": "NHD",
    "return_sparsity": False,
}
sparse = spas_sage2_attn_meansim_topk_cuda(q, k, v, topk=0.5, **common)
full = spas_sage2_attn_meansim_topk_cuda(q, k, v, topk=1.0, **common)
reference = F.scaled_dot_product_attention(
    q.transpose(1, 2),
    k.transpose(1, 2),
    v.transpose(1, 2),
    dropout_p=0.0,
    is_causal=False,
).transpose(1, 2)
torch.cuda.synchronize()
for name, output in (("topk_0.5", sparse), ("topk_1.0", full)):
    if (
        output.shape != q.shape
        or output.dtype != q.dtype
        or output.device != device
        or not torch.isfinite(output).all()
    ):
        raise RuntimeError(f"SpargeAttn CUDA smoke output is invalid for {name}")
cosine = float(
    F.cosine_similarity(full.float().reshape(-1), reference.float().reshape(-1), dim=0).item()
)
if not math.isfinite(cosine) or cosine < 0.90:
    raise RuntimeError(f"SpargeAttn topk=1.0 differs from SDPA: cosine={cosine}")
print(
    json.dumps(
        {
            "status": "ok",
            "shape": list(q.shape),
            "dtype": str(q.dtype),
            "device": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "tested_topk": [0.5, 1.0],
            "topk_1_cosine_vs_sdpa": cosine,
        },
        sort_keys=True,
    )
)
PY

echo "Installed official SpargeAttn ${PINNED_COMMIT} in ${FASTA2V_LTX2_ENV}."
