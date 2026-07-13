#!/usr/bin/env python3
"""Launch the exact pinned Radial prefix/tail protocol on real BF16 sm80."""

import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ovi.gpu_process_monitor import query_gpu_compute_processes
from ovi.modules.radial_attention_backend import (
    RadialAttentionDependencyError,
    RadialVideoSelfAttentionBackend,
    load_flashinfer_api,
    load_official_radial_mask_module,
    verify_radial_install_receipt,
    verify_radial_runtime_loader_environment,
)
from ovi.radial_evidence import (
    RADIAL_GRID,
    RADIAL_HEAD_DIM,
    RADIAL_HEADS,
    RADIAL_MASK_API,
    RADIAL_SEQUENCE,
)


class _IdentityOviAttention:
    use_sp = False
    window_size = (-1, -1)

    def __init__(self, q, k, v):
        self.q = q
        self.k = k
        self.v = v

    def qkv_fn(self, _unused_hidden):
        return self.q, self.k, self.v

    def o(self, value):
        return value


def _current_pid_namespace_chain():
    """Return host-to-container PIDs for this process, or the local PID."""

    local_pid = os.getpid()
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("NSpid:"):
                chain = [int(value) for value in line.split()[1:]]
                if chain and chain[-1] == local_pid:
                    return chain
    except (OSError, ValueError):
        pass
    return [local_pid]


def run_microtest(device_index=0):
    if int(device_index) != 0:
        raise RadialAttentionDependencyError(
            "Radial microtest must launch on logical CUDA device 0"
        )
    receipt_path, receipt = verify_radial_install_receipt()
    verify_radial_runtime_loader_environment(receipt)
    import torch

    if not torch.cuda.is_available():
        raise RadialAttentionDependencyError(
            "CUDA is unavailable for the required Radial FlashInfer microtest"
        )
    flashinfer = load_flashinfer_api(
        receipt["installed_flashinfer_package_root"]
    )
    source_module = load_official_radial_mask_module(
        receipt["derived_module"]["path"]
    )

    device = torch.device("cuda", int(device_index))
    compute_capability = tuple(torch.cuda.get_device_capability(device))
    if compute_capability != (8, 0):
        raise RadialAttentionDependencyError(
            "The fixed Radial protocol targets A100 sm80; got "
            f"compute capability {compute_capability}"
        )
    generator = torch.Generator(device=device).manual_seed(0)
    shape = (1, RADIAL_SEQUENCE, RADIAL_HEADS, RADIAL_HEAD_DIM)
    q, k, v = (
        torch.randn(
            shape,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        for _ in range(3)
    )
    backend = RadialVideoSelfAttentionBackend(
        torch_module=torch,
        flashinfer_module=flashinfer,
        mask_generator=getattr(source_module, RADIAL_MASK_API),
        get_indptr_from_mask=source_module.get_indptr_from_mask,
        get_indices_from_mask=source_module.get_indices_from_mask,
        rope_apply_fn=lambda value, _grid, _freqs: value,
        profile="conservative",
        install_receipt={
            "path": str(receipt_path),
            "commit": receipt["commit"],
            "derived_module_sha256": receipt["derived_module"]["sha256"],
            "flashinfer_version": receipt["flashinfer_version"],
        },
    )
    output = backend(
        _IdentityOviAttention(q, k, v),
        None,
        torch.tensor([RADIAL_SEQUENCE], dtype=torch.int64),
        torch.tensor([RADIAL_GRID], dtype=torch.int64),
        None,
    )
    torch.cuda.synchronize(device)
    if tuple(output.shape) != (1, RADIAL_SEQUENCE, RADIAL_HEADS * RADIAL_HEAD_DIM):
        raise RadialAttentionDependencyError(
            f"Radial microtest returned incompatible shape {tuple(output.shape)}"
        )
    if output.dtype != torch.bfloat16 or output.device != device:
        raise RadialAttentionDependencyError(
            "Radial microtest changed BF16 dtype or CUDA device"
        )
    finite = bool(torch.isfinite(output).all().item())
    if not finite:
        raise RadialAttentionDependencyError(
            "Radial microtest output contains NaN or Inf"
        )
    output_abs_mean = float(output.float().abs().mean().item())
    output_abs_max = float(output.float().abs().max().item())

    gpu_identity = query_gpu_compute_processes(0)
    runtime_device_name = torch.cuda.get_device_name(device)
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES")
    pid_namespace_chain = _current_pid_namespace_chain()
    if (
        gpu_identity.get("available") is not True
        or gpu_identity.get("device_index") != 0
        or gpu_identity.get("device_name") != runtime_device_name
        or not isinstance(gpu_identity.get("device_uuid"), str)
        or not gpu_identity.get("device_uuid", "").startswith("GPU-")
        or gpu_identity.get("process_count") != 1
        or not isinstance(gpu_identity.get("processes"), list)
        or len(gpu_identity["processes"]) != 1
        or gpu_identity["processes"][0].get("host_pid")
        not in pid_namespace_chain
        or visible_device != gpu_identity.get("device_uuid")
    ):
        raise RadialAttentionDependencyError(
            "Radial microtest could not bind logical CUDA 0 to uncontended "
            "physical GPU 0/current process: "
            f"gpu={gpu_identity}, CUDA_VISIBLE_DEVICES={visible_device!r}, "
            f"pid_namespace_chain={pid_namespace_chain}"
        )
    metrics = backend.metrics()
    return {
        "status": "ok",
        "device": runtime_device_name,
        "device_uuid": gpu_identity["device_uuid"],
        "cuda_visible_devices": visible_device,
        "physical_device_index": gpu_identity["device_index"],
        "logical_cuda_device_index": device.index,
        "host_pid": gpu_identity["processes"][0]["host_pid"],
        "python_pid": os.getpid(),
        "pid_namespace_chain": pid_namespace_chain,
        "gpu_process_count": gpu_identity["process_count"],
        "gpu_processes": gpu_identity["processes"],
        "compute_capability": list(compute_capability),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "dtype": str(output.dtype),
        "shape": list(shape),
        "grid": list(RADIAL_GRID),
        "profile": metrics["profile"],
        "decay_factor": metrics["decay_factor"],
        "prefix_sequence": metrics["prefix_sequence"],
        "tail_sequence": metrics["tail_sequence"],
        "tail_strategy": metrics["tail_strategy"],
        "calls": metrics["calls"],
        "plan_cache_entries": metrics["plan_cache_entries"],
        "plan_cache_misses": metrics["plan_cache_misses"],
        "plan_cache_hits": metrics["plan_cache_hits"],
        "mask_audit": metrics["last_mask_audit"],
        "finite": finite,
        "output_abs_mean": output_abs_mean,
        "output_abs_max": output_abs_max,
    }


if __name__ == "__main__":
    print(json.dumps(run_microtest(), indent=2, sort_keys=True, allow_nan=False))
