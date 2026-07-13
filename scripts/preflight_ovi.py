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


def main(output_path=None, attention_method="dense"):
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
        "attention_method": attention_method,
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

    if attention_method == "sparge":
        try:
            from ovi.gpu_process_monitor import validate_pre_run_gpu_report
            from ovi.modules.sparge_attention_backend import (
                SPARGEATTN_API,
                SPARGEATTN_COMMIT,
                load_official_sparge_kernel,
                verify_sparge_install_receipt,
            )
            from scripts.sparge_attn_microtest import run_microtest

            run_dir = os.environ.get("FASTA2V_RUN_DIR")
            pre_run_path = Path(run_dir or "") / "pre_run_gpu.json"
            if not run_dir or not pre_run_path.is_file():
                raise RuntimeError(
                    "Sparge preflight requires runner-created pre_run_gpu.json"
                )
            pre_run_gpu = json.loads(pre_run_path.read_text(encoding="utf-8"))
            pre_run_errors = validate_pre_run_gpu_report(pre_run_gpu)
            if pre_run_errors:
                raise RuntimeError(
                    "invalid Sparge pre-run GPU evidence: "
                    + "; ".join(pre_run_errors)
                )

            receipt_path, receipt = verify_sparge_install_receipt()
            kernel = load_official_sparge_kernel(
                receipt["installed_package_root"]
            )
            report["spargeattn"] = {
                "package_version": package_version("spas_sage_attn"),
                "pinned_commit": SPARGEATTN_COMMIT,
                "api": SPARGEATTN_API,
                "install_receipt": str(receipt_path),
                "install_receipt_contents": receipt,
                "installed_files_verified": True,
            }
            microtest = run_microtest(
                kernel=kernel,
                device_index=0,
            )
            if microtest.get("device_uuid") != pre_run_gpu.get("device_uuid"):
                raise RuntimeError(
                    "Sparge preflight CUDA microtest GPU UUID differs from "
                    "pre-run idle evidence"
                )
            report["spargeattn_microtest"] = microtest
        except Exception as exc:
            report["errors"].append(
                f"official SpargeAttn dependency check failed: {exc!r}"
            )

    if attention_method == "radial":
        try:
            from ovi.gpu_process_monitor import validate_pre_run_gpu_report
            from ovi.modules.radial_attention_backend import (
                load_flashinfer_api,
                load_official_radial_mask_module,
                verify_radial_install_receipt,
            )
            from ovi.radial_evidence import (
                RADIAL_COMMIT,
                RADIAL_MASK_API,
            )
            from scripts.radial_flashinfer_microtest import run_microtest

            run_dir = os.environ.get("FASTA2V_RUN_DIR")
            pre_run_path = Path(run_dir or "") / "pre_run_gpu.json"
            if not run_dir or not pre_run_path.is_file():
                raise RuntimeError(
                    "Radial preflight requires runner-created pre_run_gpu.json"
                )
            pre_run_gpu = json.loads(pre_run_path.read_text(encoding="utf-8"))
            pre_run_errors = validate_pre_run_gpu_report(pre_run_gpu)
            if pre_run_errors:
                raise RuntimeError(
                    "invalid Radial pre-run GPU evidence: "
                    + "; ".join(pre_run_errors)
                )

            receipt_path, receipt = verify_radial_install_receipt()
            cached_flashinfer_manifest = Path(
                receipt["flashinfer_manifest"]["path"]
            )
            copied_flashinfer_manifest = (
                Path(run_dir) / "radial-flashinfer-manifest.json"
            )
            if (
                not copied_flashinfer_manifest.is_file()
                or copied_flashinfer_manifest.read_bytes()
                != cached_flashinfer_manifest.read_bytes()
            ):
                raise RuntimeError(
                    "runner-copied FlashInfer manifest differs from audited "
                    "cache manifest"
                )
            flashinfer = load_flashinfer_api(
                receipt["installed_flashinfer_package_root"]
            )
            source_module = load_official_radial_mask_module(
                receipt["derived_module"]["path"]
            )
            report["radialattn"] = {
                "pinned_commit": RADIAL_COMMIT,
                "mask_api": RADIAL_MASK_API,
                "install_receipt": str(receipt_path),
                "install_receipt_contents": receipt,
                "source_files_verified": True,
                "flashinfer_files_verified": True,
                "flashinfer_manifest_verified": True,
                "cpu_mask_audits_verified": True,
                "flashinfer_version": package_version("flashinfer-python"),
                "flashinfer_apis": {
                    name: callable(getattr(flashinfer, name, None))
                    for name in (
                        "BlockSparseAttentionWrapper",
                        "single_prefill_with_kv_cache",
                        "merge_state",
                    )
                },
                "derived_mask_api_callable": callable(
                    getattr(source_module, RADIAL_MASK_API, None)
                ),
                "install_cuda_kernel_launched": False,
                "preflight_cuda_microtest_required": True,
            }
            microtest = run_microtest(device_index=0)
            if microtest.get("device_uuid") != pre_run_gpu.get("device_uuid"):
                raise RuntimeError(
                    "Radial FlashInfer microtest GPU UUID differs from pre-run "
                    "idle evidence"
                )
            report["radialattn_microtest"] = microtest
        except Exception as exc:
            report["errors"].append(
                f"official Radial Attention dependency check failed: {exc!r}"
            )

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

    rendered = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    if output_path is not None:
        Path(output_path).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--attention-method",
        choices=("dense", "sparge", "radial", "svg"),
        default="dense",
    )
    cli_args = parser.parse_args()
    raise SystemExit(main(cli_args.output, cli_args.attention_method))
