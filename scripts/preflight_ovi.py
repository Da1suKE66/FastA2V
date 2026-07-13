#!/usr/bin/env python3
import importlib
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path


CACHE_ROOT = Path(os.environ.get("FASTA2V_CACHE_ROOT", "/cache/liluchen/FastA2V"))
CHECKPOINT_ROOT = CACHE_ROOT / "ckpts"


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    report = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "cache_root": str(CACHE_ROOT),
        "checkpoint_root": str(CHECKPOINT_ROOT),
        "ffmpeg": shutil.which("ffmpeg"),
        "packages": {},
        "checkpoints": {},
        "errors": [],
    }

    for package in (
        "torch",
        "torchvision",
        "torchaudio",
        "flash-attn",
        "transformers",
        "diffusers",
        "omegaconf",
    ):
        report["packages"][package] = package_version(package)

    try:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            report["compute_capability"] = list(torch.cuda.get_device_capability(0))
        else:
            report["errors"].append("CUDA is not available to PyTorch")
    except Exception as exc:
        report["errors"].append(f"torch import failed: {exc!r}")

    for module in ("cv2", "diffusers", "transformers", "omegaconf", "flash_attn"):
        try:
            importlib.import_module(module)
        except Exception as exc:
            report["errors"].append(f"{module} import failed: {exc!r}")

    required_checkpoints = (
        "Ovi/model.safetensors",
        "Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
        "MMAudio/ext_weights/best_netG.pt",
        "MMAudio/ext_weights/v1-16.pth",
    )
    for relative_path in required_checkpoints:
        path = CHECKPOINT_ROOT / relative_path
        exists = path.is_file()
        report["checkpoints"][relative_path] = {
            "exists": exists,
            "bytes": path.stat().st_size if exists else None,
        }
        if not exists:
            report["errors"].append(f"missing checkpoint: {path}")

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

