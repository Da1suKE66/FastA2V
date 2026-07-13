"""Pure-Python validators for pinned official SpargeAttn evidence."""

import math
from pathlib import Path


SPARGEATTN_REPOSITORY = "https://github.com/thu-ml/SpargeAttn.git"
SPARGEATTN_CLONE_URL = "ssh://git@ssh.github.com:443/thu-ml/SpargeAttn.git"
SPARGEATTN_COMMIT = "ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a"
SPARGEATTN_API = "spas_sage2_attn_meansim_topk_cuda"
SPARGEATTN_MICROTEST_SHAPE = (1, 132, 24, 128)
SPARGEATTN_MICROTEST_MIN_COSINE = 0.90
SPARGEATTN_EXPECTED_DEVICE = "NVIDIA A100-SXM4-80GB"
SPARGEATTN_EXPECTED_TORCH = "2.6.0+cu124"
SPARGEATTN_EXPECTED_TORCH_CUDA = "12.4"


def sparge_microtest_evidence_errors(microtest, expected_gpu_uuid=None):
    """Return structural errors in persisted real-CUDA launch evidence."""

    if not isinstance(microtest, dict):
        return ["microtest evidence must be a JSON object"]
    errors = []
    expected = {
        "status": "ok",
        "device": SPARGEATTN_EXPECTED_DEVICE,
        "compute_capability": [8, 0],
        "torch": SPARGEATTN_EXPECTED_TORCH,
        "torch_cuda": SPARGEATTN_EXPECTED_TORCH_CUDA,
        "torch_cxx11_abi": False,
        "dtype": "torch.bfloat16",
        "tensor_layout": "NHD",
        "shape": list(SPARGEATTN_MICROTEST_SHAPE),
        "tested_topk": [0.5, 1.0],
    }
    for field, expected_value in expected.items():
        if microtest.get(field) != expected_value:
            errors.append(
                f"microtest {field}={microtest.get(field)!r} != "
                f"{expected_value!r}"
            )
    device_uuid = microtest.get("device_uuid")
    if not isinstance(device_uuid, str) or not device_uuid.startswith("GPU-"):
        errors.append("microtest device_uuid is missing or invalid")
    if expected_gpu_uuid is not None and device_uuid != expected_gpu_uuid:
        errors.append(
            f"microtest device_uuid={device_uuid!r} != expected "
            f"{expected_gpu_uuid!r}"
        )
    for field, minimum, maximum in (
        ("cosine_vs_sdpa", SPARGEATTN_MICROTEST_MIN_COSINE, 1.0001),
        ("max_abs_difference_vs_sdpa", 0.0, None),
    ):
        try:
            value = float(microtest.get(field))
        except (TypeError, ValueError):
            errors.append(f"microtest {field} is not numeric")
            continue
        if not math.isfinite(value):
            errors.append(f"microtest {field} must be finite")
        elif value < minimum or (maximum is not None and value > maximum):
            errors.append(
                f"microtest {field}={value!r} is outside "
                f"[{minimum}, {maximum}]"
            )
    return errors


def sparge_receipt_evidence_errors(
    receipt,
    expected_gpu_uuid=None,
    expected_cache_root="/cache/liluchen/FastA2V",
):
    """Validate receipt contents without trusting paths on the current host."""

    if not isinstance(receipt, dict):
        return ["install receipt must be a JSON object"]
    errors = []
    expected = {
        "repository": SPARGEATTN_REPOSITORY,
        "clone_url": SPARGEATTN_CLONE_URL,
        "commit": SPARGEATTN_COMMIT,
        "api": SPARGEATTN_API,
        "package": "spas_sage_attn",
        "package_version": "0.1.0",
        "python": "3.11.15",
        "torch": SPARGEATTN_EXPECTED_TORCH,
        "torch_cuda": SPARGEATTN_EXPECTED_TORCH_CUDA,
        "torch_cxx11_abi": False,
        "triton": "3.2.0",
        "cuda_home": "/usr/local/cuda-12.1",
        "torch_cuda_arch_list": "8.0",
        "source_dir": str(
            Path(expected_cache_root)
            / "sources"
            / f"SpargeAttn-{SPARGEATTN_COMMIT}"
        ),
        "installed_package_root": str(
            Path(expected_cache_root)
            / "envs"
            / "ovi"
            / "lib"
            / "python3.11"
            / "site-packages"
            / "spas_sage_attn"
        ),
    }
    for field, expected_value in expected.items():
        if receipt.get(field) != expected_value:
            errors.append(
                f"receipt {field}={receipt.get(field)!r} != {expected_value!r}"
            )
    max_jobs = receipt.get("max_jobs")
    if (
        not isinstance(max_jobs, int)
        or isinstance(max_jobs, bool)
        or not 1 <= max_jobs <= 4
    ):
        errors.append("receipt max_jobs must be an integer from 1 through 4")

    installed_files = receipt.get("installed_files")
    if not isinstance(installed_files, dict) or not installed_files:
        errors.append("receipt installed_files is missing")
        installed_core = None
    else:
        required_artifacts = {
            "core.py": any(Path(name).name == "core.py" for name in installed_files),
            "_qattn*.so": any(
                Path(name).name.startswith("_qattn")
                and Path(name).suffix == ".so"
                for name in installed_files
            ),
            "_fused*.so": any(
                Path(name).name.startswith("_fused")
                and Path(name).suffix == ".so"
                for name in installed_files
            ),
        }
        for artifact, present in required_artifacts.items():
            if not present:
                errors.append(f"receipt installed_files lacks {artifact}")
        for name, metadata in installed_files.items():
            if not isinstance(metadata, dict):
                errors.append(f"receipt installed file metadata is invalid: {name!r}")
                continue
            if not isinstance(metadata.get("bytes"), int) or metadata.get(
                "bytes", 0
            ) <= 0:
                errors.append(f"receipt installed file byte count is invalid: {name}")
            digest = metadata.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                errors.append(f"receipt installed file SHA256 is invalid: {name}")
            if Path(name).suffix == ".so" and metadata.get("ldd_not_found") != []:
                errors.append(f"receipt shared library was not ldd-verified: {name}")
        installed_core = next(
            (
                metadata
                for name, metadata in installed_files.items()
                if Path(name).name == "core.py"
            ),
            None,
        )
    source_core = receipt.get("source_core")
    if not isinstance(source_core, dict):
        errors.append("receipt source_core fingerprint is missing")
    elif not isinstance(source_core.get("path"), str) or not source_core.get(
        "path"
    ):
        errors.append("receipt source_core path is missing")
    elif source_core.get("path") != (
        receipt.get("source_dir", "") + "/spas_sage_attn/core.py"
    ):
        errors.append("receipt source_core path differs from pinned checkout")
    elif not isinstance(installed_core, dict):
        errors.append("receipt installed core.py fingerprint is missing")
    elif (
        source_core.get("bytes") != installed_core.get("bytes")
        or source_core.get("sha256") != installed_core.get("sha256")
    ):
        errors.append("installed core.py does not match pinned source core.py")

    build_log = receipt.get("build_log")
    if not isinstance(build_log, dict):
        errors.append("receipt build_log fingerprint is missing")
    else:
        if build_log.get("path") != str(
            Path(expected_cache_root) / "spargeattn-build.log"
        ):
            errors.append("receipt build_log path differs from fixed cache path")
        if not isinstance(build_log.get("bytes"), int) or build_log.get("bytes", 0) <= 0:
            errors.append("receipt build_log byte count is invalid")
        digest = build_log.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            errors.append("receipt build_log SHA256 is invalid")

    install_gpu = receipt.get("install_pre_run_gpu")
    if not isinstance(install_gpu, dict):
        errors.append("receipt install_pre_run_gpu fingerprint is missing")
    else:
        if install_gpu.get("path") != str(
            Path(expected_cache_root) / "spargeattn-pre_run_gpu.json"
        ):
            errors.append(
                "receipt install_pre_run_gpu path differs from fixed cache path"
            )
        if not isinstance(install_gpu.get("bytes"), int) or install_gpu.get(
            "bytes", 0
        ) <= 0:
            errors.append("receipt install_pre_run_gpu byte count is invalid")
        digest = install_gpu.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            errors.append("receipt install_pre_run_gpu SHA256 is invalid")
        microtest = receipt.get("microtest")
        microtest_uuid = (
            microtest.get("device_uuid") if isinstance(microtest, dict) else None
        )
        if install_gpu.get("device_uuid") != microtest_uuid:
            errors.append("install GPU UUID differs from CUDA microtest UUID")

    errors.extend(
        sparge_microtest_evidence_errors(
            receipt.get("microtest"), expected_gpu_uuid=expected_gpu_uuid
        )
    )
    return errors
