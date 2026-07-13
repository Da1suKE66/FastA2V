#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="${FASTA2V_CACHE_ROOT:-/cache/liluchen/FastA2V}"
ENV_PREFIX="${FASTA2V_OVI_ENV:-${CACHE_ROOT}/envs/ovi}"
CONDA_BIN="${CONDA_BIN:-/home/ma-user/miniconda3/bin/conda}"

export CONDA_PKGS_DIRS="${CACHE_ROOT}/conda_pkgs"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export HF_HOME="${CACHE_ROOT}/hf"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TMPDIR="${CACHE_ROOT}/build/tmp"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS="${MAX_JOBS:-8}"
export PYTHONNOUSERSITE=1

mkdir -p \
  "${CONDA_PKGS_DIRS}" \
  "${PIP_CACHE_DIR}" \
  "${HF_HOME}" \
  "${TORCH_HOME}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${TMPDIR}"

if [[ ! -e "${ENV_PREFIX}/conda-meta" ]]; then
  "${CONDA_BIN}" create -y -p "${ENV_PREFIX}" python=3.11 pip setuptools wheel
fi

"${ENV_PREFIX}/bin/python" -m pip install --upgrade pip ninja packaging
"${ENV_PREFIX}/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
"${ENV_PREFIX}/bin/python" -m pip install -r "${REPO_ROOT}/requirements.txt"
"${ENV_PREFIX}/bin/python" -m pip install flash-attn --no-build-isolation
"${ENV_PREFIX}/bin/python" -m pip freeze > "${CACHE_ROOT}/ovi-environment.freeze.txt"

