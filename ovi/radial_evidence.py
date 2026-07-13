"""Pure-Python constants and validators for the pinned Radial Attention source.

The runtime backend imports the upstream mask generator from an audited derived
copy.  The only source change is the repository patch that makes unrelated
plotting and SageAttention imports optional.  No mask or kernel implementation
is copied into FastA2V.
"""

import hashlib
import math
from pathlib import Path
import re


RADIAL_REPOSITORY = "https://github.com/mit-han-lab/radial-attention.git"
RADIAL_CLONE_URL = (
    "ssh://git@ssh.github.com:443/mit-han-lab/radial-attention.git"
)
RADIAL_COMMIT = "72788d4f0a6d202f1ec5f1c98a6e4c8b2e34fdbc"
RADIAL_SOURCE_MODULE_SHA256 = (
    "663dd94c8be0b20d8ab71c56209f0d03514b2fb90d4a2dfdb2cfaf3238b529ee"
)
RADIAL_OPTIONAL_IMPORTS_PATCH_SHA256 = (
    "2adf006c3a81600ecf3bc0c228372385b1c99009fc0c30be95ee45c2bd208997"
)
RADIAL_DERIVED_MODULE_SHA256 = (
    "aafac6551f0a73a7548ed7ec987d718c17cf1269605e454af7d5089b4f9263c5"
)
RADIAL_MASK_API = "gen_log_mask_shrinked"

# This is an installation candidate, not a claim that the wheel has already
# passed an Ovi CUDA launch.  The installer and preflight both reject any other
# distribution/version, and the first formal run remains invalid until the
# strict output verifier sees a successful Radial dispatcher receipt.
FLASHINFER_DISTRIBUTION = "flashinfer-python"
FLASHINFER_VERSION = "0.2.5+cu124torch2.6"
FLASHINFER_WHEEL_INDEX = "https://flashinfer.ai/whl/cu124/torch2.6/"
FLASHINFER_REQUIRED_APIS = (
    "BlockSparseAttentionWrapper",
    "single_prefill_with_kv_cache",
    "merge_state",
)
FLASHINFER_MANIFEST_FILENAME = "radial-flashinfer-manifest.json"
FLASHINFER_MANIFEST_SCHEMA = "fasta2v.flashinfer-install-manifest.v1"

RADIAL_SEQUENCE = 15004
RADIAL_PREFIX_SEQUENCE = 14976
RADIAL_TAIL_SEQUENCE = 28
RADIAL_GRID = (31, 22, 22)
RADIAL_HEADS = 24
RADIAL_HEAD_DIM = 128
RADIAL_BLOCK_SIZE = 128
RADIAL_MODEL_TYPE = "wan"
RADIAL_EMPTY_ROWS = (22, 56, 90)
RADIAL_EXPECTED_DEVICE = "NVIDIA A100-SXM4-80GB"
RADIAL_EXPECTED_TORCH = "2.6.0+cu124"
RADIAL_EXPECTED_TORCH_CUDA = "12.4"

RADIAL_PROFILE_AUDITS = {
    "aggressive": {
        "decay_factor": 1.0,
        "raw_true_blocks": 4159,
        "raw_sha256": (
            "7d88d9af9fc45035c59cc21059ff6003347f702d70adf4328fae70b265971558"
        ),
        "empty_rows": list(RADIAL_EMPTY_ROWS),
        "repaired_true_blocks": 4510,
        "repaired_sha256": (
            "6df2f9c615ca97d8eea551146b179d28466a3f01eb0734e886fd8f480c6a55f2"
        ),
    },
    "conservative": {
        "decay_factor": 4.0,
        "raw_true_blocks": 5338,
        "raw_sha256": (
            "6e50584db38cca8487df7cc3db2e7e2548a7747bd7a3a1e3cb183430946b6b92"
        ),
        "empty_rows": list(RADIAL_EMPTY_ROWS),
        "repaired_true_blocks": 5689,
        "repaired_sha256": (
            "9ae15b593ab2b446abac856fcadc2f144c8da4288b9fbf2b50ee5c72814b7b44"
        ),
    },
}


def radial_profile(profile):
    """Return a defensive copy of one fixed Radial profile."""

    normalized = str(profile).strip().lower()
    if normalized not in RADIAL_PROFILE_AUDITS:
        raise ValueError(
            f"radial_profile={normalized!r} is not audited; expected one of "
            f"{sorted(RADIAL_PROFILE_AUDITS)}"
        )
    profile_copy = dict(RADIAL_PROFILE_AUDITS[normalized])
    profile_copy["empty_rows"] = list(profile_copy["empty_rows"])
    return profile_copy


def _valid_fingerprint(metadata, *, allow_empty=False):
    return (
        isinstance(metadata, dict)
        and isinstance(metadata.get("bytes"), int)
        and not isinstance(metadata.get("bytes"), bool)
        and (metadata["bytes"] >= 0 if allow_empty else metadata["bytes"] > 0)
        and isinstance(metadata.get("sha256"), str)
        and len(metadata["sha256"]) == 64
    )


def normalize_ldd_output(output):
    """Remove ASLR-only addresses while preserving resolved library paths."""

    return re.sub(r"0x[0-9A-Fa-f]+", "0xADDR", str(output))


def expected_flashinfer_manifest(receipt):
    """Return the immutable FlashInfer provenance manifest bound by a receipt."""

    if not isinstance(receipt, dict):
        return None
    return {
        "schema": FLASHINFER_MANIFEST_SCHEMA,
        "distribution": receipt.get("flashinfer_distribution"),
        "version": receipt.get("flashinfer_version"),
        "wheel_index": receipt.get("flashinfer_wheel_index"),
        "required_apis": receipt.get("flashinfer_required_apis"),
        "package_root": receipt.get("installed_flashinfer_package_root"),
        "module": receipt.get("flashinfer_module"),
        "files": receipt.get("installed_flashinfer_files"),
        "native_files": sorted(
            name
            for name in (receipt.get("installed_flashinfer_files") or {})
            if Path(name).suffix == ".so"
        ),
    }


def flashinfer_manifest_evidence_errors(manifest, receipt):
    """Cross-check a copied FlashInfer manifest against its install receipt."""

    if not isinstance(manifest, dict):
        return ["FlashInfer manifest must be a JSON object"]
    expected = expected_flashinfer_manifest(receipt)
    if manifest != expected:
        return ["FlashInfer manifest contents differ from install receipt"]
    return []


def radial_receipt_evidence_errors(
    receipt,
    expected_cache_root="/cache/liluchen/FastA2V",
):
    """Validate persisted dependency metadata without touching its paths."""

    if not isinstance(receipt, dict):
        return ["install receipt must be a JSON object"]
    errors = []
    cache_root = Path(expected_cache_root)
    source_dir = cache_root / "sources" / f"radial-attention-{RADIAL_COMMIT}"
    derived_dir = cache_root / "derived" / f"radial-attention-{RADIAL_COMMIT}"
    expected = {
        "repository": RADIAL_REPOSITORY,
        "clone_url": RADIAL_CLONE_URL,
        "commit": RADIAL_COMMIT,
        "mask_api": RADIAL_MASK_API,
        "source_dir": str(source_dir),
        "derived_dir": str(derived_dir),
        "patch_scope": ["radial_attn/attn_mask.py"],
        "patch_purpose": "optional_imports_only",
        "flashinfer_distribution": FLASHINFER_DISTRIBUTION,
        "flashinfer_version": FLASHINFER_VERSION,
        "flashinfer_wheel_index": FLASHINFER_WHEEL_INDEX,
        "flashinfer_required_apis": list(FLASHINFER_REQUIRED_APIS),
        "installed_flashinfer_package_root": str(
            cache_root
            / "envs"
            / "ovi"
            / "lib"
            / "python3.11"
            / "site-packages"
            / "flashinfer"
        ),
        "python": "3.11.15",
        "torch": "2.6.0+cu124",
        "torch_cuda": "12.4",
        "model_type": RADIAL_MODEL_TYPE,
        "block_size": RADIAL_BLOCK_SIZE,
        "sequence": RADIAL_SEQUENCE,
        "prefix_sequence": RADIAL_PREFIX_SEQUENCE,
        "tail_sequence": RADIAL_TAIL_SEQUENCE,
        "grid": list(RADIAL_GRID),
    }
    for field, expected_value in expected.items():
        if receipt.get(field) != expected_value:
            errors.append(
                f"receipt {field}={receipt.get(field)!r} != {expected_value!r}"
            )

    fingerprints = {
        "source_module": (
            source_dir / "radial_attn" / "attn_mask.py",
            RADIAL_SOURCE_MODULE_SHA256,
        ),
        "optional_imports_patch": (
            None,
            RADIAL_OPTIONAL_IMPORTS_PATCH_SHA256,
        ),
        "derived_module": (
            derived_dir / "radial_attn" / "attn_mask.py",
            RADIAL_DERIVED_MODULE_SHA256,
        ),
        "flashinfer_manifest": (
            cache_root / FLASHINFER_MANIFEST_FILENAME,
            None,
        ),
    }
    for field, (expected_path, expected_digest) in fingerprints.items():
        metadata = receipt.get(field)
        if not _valid_fingerprint(metadata):
            errors.append(f"receipt {field} fingerprint is missing or invalid")
            continue
        if expected_path is not None and metadata.get("path") != str(expected_path):
            errors.append(f"receipt {field} path differs from fixed cache path")
        if expected_digest is not None and metadata.get("sha256") != expected_digest:
            errors.append(f"receipt {field} SHA256 differs from audited content")

    flashinfer_root = receipt.get("installed_flashinfer_package_root")
    flashinfer_module = receipt.get("flashinfer_module")
    if not _valid_fingerprint(flashinfer_module):
        errors.append("receipt flashinfer_module fingerprint is missing or invalid")
    elif flashinfer_module.get("path") != str(
        Path(flashinfer_root or "") / "__init__.py"
    ):
        errors.append("receipt flashinfer_module path differs from package root")
    installed_files = receipt.get("installed_flashinfer_files")
    native_files = []
    if not isinstance(installed_files, dict) or not installed_files:
        errors.append("receipt installed_flashinfer_files is missing")
    else:
        if "__init__.py" not in installed_files:
            errors.append("receipt installed FlashInfer files lack __init__.py")
        for name, metadata in installed_files.items():
            if not _valid_fingerprint(metadata, allow_empty=True):
                errors.append(
                    f"receipt installed FlashInfer fingerprint is invalid: {name}"
                )
                continue
            if Path(name).is_absolute() or ".." in Path(name).parts:
                errors.append(
                    f"receipt installed FlashInfer path escapes package: {name}"
                )
            if Path(name).suffix == ".so" and metadata.get("ldd_not_found") != []:
                errors.append(
                    f"receipt FlashInfer native library was not ldd-verified: {name}"
                )
            if Path(name).suffix == ".so":
                native_files.append(name)
                ldd_output = metadata.get("ldd_output")
                ldd_normalized_output = metadata.get("ldd_normalized_output")
                ldd_sha256 = metadata.get("ldd_sha256")
                if not isinstance(ldd_output, str) or not ldd_output.strip():
                    errors.append(
                        f"receipt FlashInfer native ldd output is missing: {name}"
                    )
                elif ldd_normalized_output != normalize_ldd_output(ldd_output):
                    errors.append(
                        f"receipt FlashInfer normalized ldd is invalid: {name}"
                    )
                elif (
                    not isinstance(ldd_sha256, str)
                    or hashlib.sha256(
                        ldd_normalized_output.encode("utf-8")
                    ).hexdigest()
                    != ldd_sha256
                ):
                    errors.append(
                        f"receipt FlashInfer native ldd hash is invalid: {name}"
                    )
                if isinstance(ldd_output, str) and "not found" in ldd_output:
                    errors.append(
                        f"receipt FlashInfer native ldd has unresolved libraries: {name}"
                    )
        installed_init = installed_files.get("__init__.py")
        if isinstance(installed_init, dict) and isinstance(flashinfer_module, dict):
            if (
                installed_init.get("bytes") != flashinfer_module.get("bytes")
                or installed_init.get("sha256")
                != flashinfer_module.get("sha256")
            ):
                errors.append(
                    "receipt FlashInfer module differs from installed __init__.py"
                )
    if not native_files:
        errors.append("receipt installed FlashInfer files lack a native .so")

    audits = receipt.get("cpu_mask_audits")
    if not isinstance(audits, dict):
        errors.append("receipt cpu_mask_audits is missing")
    else:
        for name, expected_audit in RADIAL_PROFILE_AUDITS.items():
            if audits.get(name) != expected_audit:
                errors.append(
                    f"receipt CPU mask audit for {name!r} differs from fixed audit"
                )
    if receipt.get("cuda_kernel_launched") is not False:
        errors.append("install receipt must state cuda_kernel_launched=false")
    return errors


def radial_microtest_evidence_errors(microtest, expected_gpu_uuid=None):
    """Validate a real exact-shape FlashInfer launch performed by preflight."""

    if not isinstance(microtest, dict):
        return ["Radial microtest evidence must be a JSON object"]
    errors = []
    expected = {
        "status": "ok",
        "device": RADIAL_EXPECTED_DEVICE,
        "physical_device_index": 0,
        "logical_cuda_device_index": 0,
        "gpu_process_count": 1,
        "compute_capability": [8, 0],
        "torch": RADIAL_EXPECTED_TORCH,
        "torch_cuda": RADIAL_EXPECTED_TORCH_CUDA,
        "torch_cxx11_abi": False,
        "dtype": "torch.bfloat16",
        "shape": [1, RADIAL_SEQUENCE, RADIAL_HEADS, RADIAL_HEAD_DIM],
        "grid": list(RADIAL_GRID),
        "profile": "conservative",
        "decay_factor": 4.0,
        "prefix_sequence": RADIAL_PREFIX_SEQUENCE,
        "tail_sequence": RADIAL_TAIL_SEQUENCE,
        "tail_strategy": "dense_lse_merge_no_padding",
        "calls": 1,
        "plan_cache_entries": 1,
        "plan_cache_misses": 1,
        "plan_cache_hits": 0,
        "mask_audit": RADIAL_PROFILE_AUDITS["conservative"],
        "finite": True,
    }
    for field, expected_value in expected.items():
        if microtest.get(field) != expected_value:
            errors.append(
                f"Radial microtest {field}={microtest.get(field)!r} != "
                f"{expected_value!r}"
            )
    device_uuid = microtest.get("device_uuid")
    if not isinstance(device_uuid, str) or not device_uuid.startswith("GPU-"):
        errors.append("Radial microtest device_uuid is missing or invalid")
    if expected_gpu_uuid is not None and device_uuid != expected_gpu_uuid:
        errors.append(
            f"Radial microtest device_uuid={device_uuid!r} != expected "
            f"{expected_gpu_uuid!r}"
        )
    if microtest.get("cuda_visible_devices") != device_uuid:
        errors.append(
            "Radial microtest CUDA_VISIBLE_DEVICES does not equal physical GPU UUID"
        )
    host_pid = microtest.get("host_pid")
    if (
        not isinstance(host_pid, int)
        or isinstance(host_pid, bool)
        or host_pid <= 0
    ):
        errors.append("Radial microtest host_pid is missing or invalid")
    python_pid = microtest.get("python_pid")
    pid_chain = microtest.get("pid_namespace_chain")
    if (
        not isinstance(python_pid, int)
        or isinstance(python_pid, bool)
        or python_pid <= 0
        or not isinstance(pid_chain, list)
        or not pid_chain
        or any(
            not isinstance(pid, int)
            or isinstance(pid, bool)
            or pid <= 0
            for pid in pid_chain
        )
        or pid_chain[-1] != python_pid
        or host_pid not in pid_chain
    ):
        errors.append("Radial microtest PID namespace evidence is invalid")
    processes = microtest.get("gpu_processes")
    if (
        not isinstance(processes, list)
        or len(processes) != 1
        or not isinstance(processes[0], dict)
        or processes[0].get("host_pid") != host_pid
        or not isinstance(processes[0].get("used_memory_mib"), int)
        or isinstance(processes[0].get("used_memory_mib"), bool)
        or processes[0]["used_memory_mib"] < 0
    ):
        errors.append(
            "Radial microtest GPU process evidence is not the current unique PID"
        )
    for field, require_positive in (
        ("output_abs_mean", True),
        ("output_abs_max", True),
    ):
        try:
            value = float(microtest.get(field))
        except (TypeError, ValueError):
            errors.append(f"Radial microtest {field} is not numeric")
            continue
        if not math.isfinite(value) or (require_positive and value <= 0.0):
            errors.append(
                f"Radial microtest {field} must be finite and positive"
            )
    return errors
