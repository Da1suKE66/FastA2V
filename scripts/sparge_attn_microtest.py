#!/usr/bin/env python3
"""Launch the pinned official SpargeAttn API on real BF16 sm80 tensors."""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ovi.modules.sparge_attention_backend import (
    SPARGEATTN_MICROTEST_MIN_COSINE,
    SPARGEATTN_MICROTEST_SHAPE,
    SpargeAttentionDependencyError,
    load_official_sparge_kernel,
)


def run_microtest(kernel=None, device_index=0):
    if not torch.cuda.is_available():
        raise SpargeAttentionDependencyError(
            "CUDA is unavailable for the required SpargeAttn microtest"
        )
    if kernel is None:
        kernel = load_official_sparge_kernel()

    device = torch.device("cuda", int(device_index))
    compute_capability = tuple(torch.cuda.get_device_capability(device))
    if compute_capability != (8, 0):
        raise SpargeAttentionDependencyError(
            "The pinned SpargeAttn install protocol targets A100 sm80; got "
            f"compute capability {compute_capability}"
        )
    generator = torch.Generator(device=device).manual_seed(0)
    tensors = [
        torch.randn(
            SPARGEATTN_MICROTEST_SHAPE,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        for _ in range(3)
    ]
    q, k, v = tensors
    common = {
        "dropout_p": 0.0,
        "is_causal": False,
        "pvthreshd": 50,
        "smooth_k": True,
        "tensor_layout": "NHD",
        "return_sparsity": False,
    }
    sparse_output = kernel(q, k, v, topk=0.5, **common)
    full_output = kernel(q, k, v, topk=1.0, **common)
    reference = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2)
    torch.cuda.synchronize(device)

    outputs = {"topk_0_5": sparse_output, "topk_1_0": full_output}
    if any(
        isinstance(output, tuple) or tuple(output.shape) != tuple(q.shape)
        for output in outputs.values()
    ):
        raise SpargeAttentionDependencyError(
            "SpargeAttn CUDA microtest returned an incompatible output"
        )
    if any(
        output.dtype != torch.bfloat16 or output.device != device
        for output in outputs.values()
    ):
        raise SpargeAttentionDependencyError(
            "SpargeAttn CUDA microtest changed BF16 dtype or CUDA device"
        )
    if any(not torch.isfinite(output).all() for output in outputs.values()):
        raise SpargeAttentionDependencyError(
            "SpargeAttn CUDA microtest returned NaN or Inf"
        )
    cosine = float(
        F.cosine_similarity(
            full_output.float().reshape(-1),
            reference.float().reshape(-1),
            dim=0,
        ).item()
    )
    if cosine < SPARGEATTN_MICROTEST_MIN_COSINE:
        raise SpargeAttentionDependencyError(
            "SpargeAttn CUDA microtest differs too far from full SDPA: "
            f"cosine={cosine:.6f} < {SPARGEATTN_MICROTEST_MIN_COSINE}"
        )
    return {
        "status": "ok",
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(compute_capability),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "dtype": str(q.dtype),
        "tensor_layout": "NHD",
        "shape": list(SPARGEATTN_MICROTEST_SHAPE),
        "tested_topk": [0.5, 1.0],
        "cosine_vs_sdpa": cosine,
        "max_abs_difference_vs_sdpa": float(
            (full_output.float() - reference.float()).abs().max().item()
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run_microtest(), indent=2, sort_keys=True))
