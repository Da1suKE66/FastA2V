#!/usr/bin/env python3
"""Run a real FlashAttention kernel with Ovi's video-attention layout."""

import json
import sys
from pathlib import Path

import torch
from flash_attn import flash_attn_func

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ovi.modules.attention import flash_attention


def run_microtest():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    torch.manual_seed(0)
    device = torch.device("cuda", 0)
    shape = (1, 128, 24, 128)  # [batch, sequence, heads, head_dim]
    q = torch.randn(shape, device=device, dtype=torch.bfloat16)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16)
    v = torch.randn(shape, device=device, dtype=torch.bfloat16)

    direct = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
    wrapped = flash_attention(q, k, v, causal=False, version=2)
    torch.cuda.synchronize(device)

    if direct.shape != q.shape or wrapped.shape != q.shape:
        raise AssertionError(
            f"unexpected shapes: direct={direct.shape}, wrapped={wrapped.shape}"
        )
    if not torch.isfinite(direct).all() or not torch.isfinite(wrapped).all():
        raise AssertionError("FlashAttention returned NaN or Inf")
    torch.testing.assert_close(direct, wrapped, rtol=2e-2, atol=2e-2)

    return {
        "status": "ok",
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "dtype": str(q.dtype),
        "shape": list(shape),
        "max_abs_difference": float((direct - wrapped).abs().max().item()),
    }


if __name__ == "__main__":
    print(json.dumps(run_microtest(), indent=2, sort_keys=True))
