#!/usr/bin/env bash

if [[ -z "${FASTA2V_OVI_ENV:-}" || -z "${CUDA_HOME:-}" ]]; then
  echo "Source scripts/env.sh before scripts/radial_env.sh" >&2
  return 2 2>/dev/null || exit 2
fi
if [[ "${CUDA_HOME}" != "/usr/local/cuda-12.1" ]]; then
  echo "Radial requires CUDA_HOME=/usr/local/cuda-12.1, got ${CUDA_HOME}" >&2
  return 2 2>/dev/null || exit 2
fi

while IFS= read -r variable; do
  if [[ "${variable}" == LD_* ]]; then
    unset "${variable}"
  fi
done < <(compgen -e)
unset GLIBC_TUNABLES

if ! RADIAL_TORCH_LIB="$(
  readlink -f -- "${FASTA2V_OVI_ENV}/lib/python3.11/site-packages/torch/lib"
)" || ! RADIAL_CUDA_LIB="$(readlink -f -- "${CUDA_HOME}/lib64")"; then
  echo "Radial could not canonicalize fixed loader directories" >&2
  return 2 2>/dev/null || exit 2
fi
if [[ ! -d "${RADIAL_TORCH_LIB}" || ! -d "${RADIAL_CUDA_LIB}" ]]; then
  echo "Radial fixed loader directories are missing" >&2
  return 2 2>/dev/null || exit 2
fi
export LD_LIBRARY_PATH="${RADIAL_TORCH_LIB}:${RADIAL_CUDA_LIB}"
unset RADIAL_TORCH_LIB RADIAL_CUDA_LIB
