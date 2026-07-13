#!/usr/bin/env bash

export FASTA2V_CACHE_ROOT="${FASTA2V_CACHE_ROOT:-/cache/liluchen/FastA2V}"
export FASTA2V_OVI_ENV="${FASTA2V_OVI_ENV:-${FASTA2V_CACHE_ROOT}/envs/ovi}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"

export PATH="${FASTA2V_OVI_ENV}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/compat:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1

export PIP_CACHE_DIR="${FASTA2V_CACHE_ROOT}/pip"
export HF_HOME="${FASTA2V_CACHE_ROOT}/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${FASTA2V_CACHE_ROOT}/torch"
export TORCH_EXTENSIONS_DIR="${FASTA2V_CACHE_ROOT}/torch_extensions"
export XDG_CACHE_HOME="${FASTA2V_CACHE_ROOT}/xdg"
export TMPDIR="${FASTA2V_CACHE_ROOT}/build/tmp"

mkdir -p \
  "${PIP_CACHE_DIR}" \
  "${HF_HUB_CACHE}" \
  "${TORCH_HOME}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${TMPDIR}"

