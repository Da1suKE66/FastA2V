#!/usr/bin/env python3
import argparse
import importlib
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

CACHE_ROOT = Path(os.environ.get("FASTA2V_CACHE_ROOT", "/cache/liluchen/FastA2V"))
CHECKPOINT_ROOT = CACHE_ROOT / "ckpts"


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main(output_path=None):
    report = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "cache_root": str(CACHE_ROOT),
        "checkpoint_root": str(CHECKPOINT_ROOT),
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "packages": {},
        "checkpoints": {},
        "errors": [],
    }

    for executable in ("ffmpeg", "ffprobe"):
        if report[executable] is None:
            report["errors"].append(f"required executable not found: {executable}")

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

    for module in (
        "cv2",
        "diffusers",
        "einops",
        "flash_attn",
        "moviepy",
        "numpy",
        "omegaconf",
        "optimum.quanto",
        "pandas",
        "safetensors",
        "scipy",
        "transformers",
    ):
        try:
            importlib.import_module(module)
        except Exception as exc:
            report["errors"].append(f"{module} import failed: {exc!r}")

    try:
        importlib.import_module("inference")
        importlib.import_module("ovi.ovi_fusion_engine")
    except Exception as exc:
        report["errors"].append(f"Ovi inference entrypoint import failed: {exc!r}")

    required_checkpoints = (
        "Ovi/model.safetensors",
        "Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
        "MMAudio/ext_weights/best_netG.pt",
        "MMAudio/ext_weights/v1-16.pth",
        "Wan2.2-TI2V-5B/google/umt5-xxl/spiece.model",
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

    manifest_path = CACHE_ROOT / "checkpoint_manifest.json"
    report["checkpoint_manifest"] = str(manifest_path)
    if not manifest_path.is_file():
        report["errors"].append(f"missing checkpoint manifest: {manifest_path}")
    else:
        try:
            manifest = json.loads(manifest_path.read_text())
            for relative_path, metadata in manifest.get("files", {}).items():
                path = CHECKPOINT_ROOT / relative_path
                if not path.is_file() or path.stat().st_size != metadata.get("bytes"):
                    report["errors"].append(
                        f"checkpoint size differs from manifest: {path}"
                    )
        except Exception as exc:
            report["errors"].append(f"invalid checkpoint manifest: {exc!r}")

    if report.get("cuda_available") and not any(
        error.startswith("flash_attn import failed") for error in report["errors"]
    ):
        try:
            from flash_attn_smoke import run_microtest

            report["flash_attn_microtest"] = run_microtest()
        except Exception as exc:
            report["errors"].append(f"FlashAttention microtest failed: {exc!r}")

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output_path is not None:
        Path(output_path).write_text(rendered)
    print(rendered, end="")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    cli_args = parser.parse_args()
    raise SystemExit(main(cli_args.output))
