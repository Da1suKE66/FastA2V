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
FLASHINFER_WHEEL_FILENAME = (
    "flashinfer_python-0.2.5+cu124torch2.6-cp38-abi3-linux_x86_64.whl"
)
FLASHINFER_WHEEL_URL = (
    "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.5/"
    + FLASHINFER_WHEEL_FILENAME
)
FLASHINFER_WHEEL_BYTES = 544230876
FLASHINFER_WHEEL_SHA256 = (
    "43d767b912c0c43a04be99595e0123eab9385fc72530a2874b5fb08e3145c0be"
)
FLASHINFER_REQUIRED_APIS = (
    "BlockSparseAttentionWrapper",
    "single_prefill_with_kv_cache",
    "merge_state",
)
FLASHINFER_MANIFEST_FILENAME = "radial-flashinfer-manifest.json"
FLASHINFER_MANIFEST_SCHEMA = "fasta2v.flashinfer-install-manifest.v3"

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
RADIAL_CUDA_HOME = "/usr/local/cuda-12.1"
RADIAL_LDD_EXECUTABLE = "/usr/bin/ldd"
RADIAL_FORBIDDEN_LOADER_PREFIXES = ("LD_",)
RADIAL_FORBIDDEN_LOADER_VARIABLES = ("GLIBC_TUNABLES",)

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


def ldd_resolved_library_paths(output):
    """Return unique absolute paths reported by one raw ``ldd`` invocation."""

    paths = []
    for line in str(output).splitlines():
        match = re.search(
            r"(?:=>\s+)?(/[^\s]+)\s+\(0x[0-9A-Fa-f]+\)", line
        )
        if match and match.group(1) not in paths:
            paths.append(match.group(1))
    return tuple(paths)


def ldd_resolved_libraries(output):
    """Return ordered ``(loader_name, absolute_path)`` entries from ldd."""

    libraries = []
    for line in str(output).splitlines():
        arrow = re.match(
            r"^\s*(\S+)\s+=>\s+(/\S+)\s+\(0x[0-9A-Fa-f]+\)", line
        )
        if arrow:
            item = (arrow.group(1), arrow.group(2))
        else:
            direct = re.match(
                r"^\s*(/\S+)\s+\(0x[0-9A-Fa-f]+\)", line
            )
            if not direct:
                continue
            item = (Path(direct.group(1)).name, direct.group(1))
        if item not in libraries:
            libraries.append(item)
    return tuple(libraries)


def loaded_shared_object_paths(maps_text=None):
    """Return canonical file-backed shared objects from Linux proc maps."""

    if maps_text is None:
        maps_path = Path("/proc/self/maps")
        try:
            maps_text = maps_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"cannot read runtime loader maps: {exc}") from exc
    paths = set()
    for line in str(maps_text).splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        mapped_path = fields[5]
        if not mapped_path.startswith("/") or ".so" not in Path(mapped_path).name:
            continue
        paths.add(str(Path(mapped_path).resolve()))
    return tuple(sorted(paths))


def library_alias_matches_path(alias, path):
    """Match an ldd loader name to its mapped versioned file name."""

    alias = str(alias)
    name = Path(path).name
    return name == alias or name.startswith(alias + ".")


def mapped_library_paths_by_alias(aliases, mapped_paths):
    """Map each dependency loader name to every matching loaded object."""

    unique_paths = tuple(sorted(set(str(Path(path).resolve()) for path in mapped_paths)))
    return {
        str(alias): [
            path
            for path in unique_paths
            if library_alias_matches_path(alias, path)
        ]
        for alias in sorted(set(str(alias) for alias in aliases))
    }


def _canonical_loader_search_paths(search_paths):
    try:
        raw_paths = tuple(search_paths)
    except TypeError as exc:
        raise ValueError("ldd search paths must be an iterable") from exc
    if not raw_paths:
        raise ValueError("ldd search paths must be non-empty")
    canonical = []
    for raw_path in raw_paths:
        if not isinstance(raw_path, (str, Path)):
            raise ValueError("ldd search path entries must be strings or Paths")
        text = str(raw_path)
        if not text:
            raise ValueError("ldd search paths cannot contain an empty entry")
        if ":" in text:
            raise ValueError("ldd search paths cannot contain ':'")
        path = Path(text)
        if not path.is_absolute():
            raise ValueError("ldd search paths must be absolute")
        resolved = str(path.resolve())
        if resolved != text:
            raise ValueError("ldd search paths must be canonical absolute paths")
        canonical.append(resolved)
    if len(set(canonical)) != len(canonical):
        raise ValueError("ldd search paths cannot contain duplicates")
    return tuple(canonical)


def radial_ldd_search_paths(
    cache_root="/cache/liluchen/FastA2V",
):
    """Return the fixed library search order used for FlashInfer ldd audits."""

    cache_root = Path(cache_root)
    return (
        str(
            (
                cache_root
                / "envs"
                / "ovi"
                / "lib"
                / "python3.11"
                / "site-packages"
                / "torch"
                / "lib"
            ).resolve()
        ),
        str((Path(RADIAL_CUDA_HOME) / "lib64").resolve()),
    )


def deterministic_ldd_environment(search_paths):
    """Build an ldd environment that cannot inherit loader settings."""

    paths = _canonical_loader_search_paths(search_paths)
    return {
        "PATH": "/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "LD_LIBRARY_PATH": ":".join(paths),
    }


def radial_runtime_loader_environment(search_paths):
    """Return the exact loader contract required by Radial Python processes."""

    paths = _canonical_loader_search_paths(search_paths)
    return {
        "LD_LIBRARY_PATH": ":".join(paths),
        "forbidden_prefixes": list(RADIAL_FORBIDDEN_LOADER_PREFIXES),
        "unset": list(RADIAL_FORBIDDEN_LOADER_VARIABLES),
    }


def expected_flashinfer_manifest(receipt):
    """Return the immutable FlashInfer provenance manifest bound by a receipt."""

    if not isinstance(receipt, dict):
        return None
    return {
        "schema": FLASHINFER_MANIFEST_SCHEMA,
        "distribution": receipt.get("flashinfer_distribution"),
        "version": receipt.get("flashinfer_version"),
        "wheel_index": receipt.get("flashinfer_wheel_index"),
        "wheel_url": receipt.get("flashinfer_wheel_url"),
        "wheel": receipt.get("flashinfer_wheel"),
        "required_apis": receipt.get("flashinfer_required_apis"),
        "cuda_home": receipt.get("cuda_home"),
        "ldd_executable": receipt.get("ldd_executable"),
        "ldd_search_paths": receipt.get("ldd_search_paths"),
        "ldd_dependencies": receipt.get("ldd_dependencies"),
        "runtime_loaded_dependencies": receipt.get(
            "runtime_loaded_dependencies"
        ),
        "runtime_loader_environment": receipt.get(
            "runtime_loader_environment"
        ),
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
        "flashinfer_wheel_url": FLASHINFER_WHEEL_URL,
        "flashinfer_required_apis": list(FLASHINFER_REQUIRED_APIS),
        "cuda_home": RADIAL_CUDA_HOME,
        "ldd_search_paths": list(radial_ldd_search_paths(cache_root)),
        "runtime_loader_environment": radial_runtime_loader_environment(
            radial_ldd_search_paths(cache_root)
        ),
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
        "flashinfer_wheel": (
            cache_root / "wheels" / FLASHINFER_WHEEL_FILENAME,
            FLASHINFER_WHEEL_SHA256,
        ),
        "ldd_executable": (
            Path(RADIAL_LDD_EXECUTABLE).resolve(),
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
        if field == "flashinfer_wheel" and metadata.get("bytes") != FLASHINFER_WHEEL_BYTES:
            errors.append("receipt FlashInfer wheel byte count differs from audited file")

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
    native_dependency_paths = set()
    native_dependency_aliases = set()
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
                ldd_dependency_paths = metadata.get("ldd_dependency_paths")
                ldd_dependency_libraries = metadata.get(
                    "ldd_dependency_libraries"
                )
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
                parsed_dependency_paths = list(
                    ldd_resolved_library_paths(ldd_output or "")
                )
                if ldd_dependency_paths != parsed_dependency_paths:
                    errors.append(
                        "receipt FlashInfer native dependency paths differ from "
                        f"ldd output: {name}"
                    )
                else:
                    native_dependency_paths.update(parsed_dependency_paths)
                parsed_dependency_libraries = [
                    {"name": alias, "path": path}
                    for alias, path in ldd_resolved_libraries(ldd_output or "")
                ]
                if ldd_dependency_libraries != parsed_dependency_libraries:
                    errors.append(
                        "receipt FlashInfer native dependency aliases differ from "
                        f"ldd output: {name}"
                    )
                else:
                    native_dependency_aliases.update(
                        item["name"] for item in parsed_dependency_libraries
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

    ldd_dependencies = receipt.get("ldd_dependencies")
    if not isinstance(ldd_dependencies, dict) or not ldd_dependencies:
        errors.append("receipt ldd_dependencies is missing")
    else:
        if set(ldd_dependencies) != native_dependency_paths:
            errors.append(
                "receipt ldd dependency inventory differs from native ldd outputs"
            )
        for reported_path, metadata in ldd_dependencies.items():
            if not Path(reported_path).is_absolute():
                errors.append(
                    f"receipt ldd dependency path is not absolute: {reported_path}"
                )
            if not _valid_fingerprint(metadata):
                errors.append(
                    "receipt ldd dependency fingerprint is missing or invalid: "
                    f"{reported_path}"
                )
            elif not Path(metadata.get("path", "")).is_absolute():
                errors.append(
                    "receipt resolved ldd dependency path is not absolute: "
                    f"{reported_path}"
                )

    runtime_dependencies = receipt.get("runtime_loaded_dependencies")
    expected_runtime_aliases = native_dependency_aliases | {
        Path(name).name for name in native_files
    }
    if not isinstance(runtime_dependencies, dict) or not runtime_dependencies:
        errors.append("receipt runtime_loaded_dependencies is missing")
    else:
        if set(runtime_dependencies) != expected_runtime_aliases:
            errors.append(
                "receipt runtime dependency aliases differ from ldd/native files"
            )
        for alias, fingerprints in runtime_dependencies.items():
            if not isinstance(fingerprints, list) or not fingerprints:
                errors.append(
                    f"receipt runtime dependency mappings are missing: {alias}"
                )
                continue
            paths = []
            for metadata in fingerprints:
                if not _valid_fingerprint(metadata):
                    errors.append(
                        "receipt runtime dependency fingerprint is invalid: "
                        f"{alias}"
                    )
                    continue
                path = metadata.get("path", "")
                if not Path(path).is_absolute():
                    errors.append(
                        "receipt runtime dependency path is not absolute: "
                        f"{alias}"
                    )
                if not library_alias_matches_path(alias, path):
                    errors.append(
                        "receipt runtime dependency path does not match alias: "
                        f"{alias} -> {path}"
                    )
                paths.append(path)
            if len(paths) != len(set(paths)):
                errors.append(
                    f"receipt runtime dependency paths are duplicated: {alias}"
                )

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
    before_cuda = microtest.get("runtime_dependencies_before_cuda")
    after_cuda = microtest.get("runtime_dependencies_after_cuda")
    if before_cuda != after_cuda:
        errors.append("Radial microtest runtime dependencies changed after CUDA")
    if not isinstance(before_cuda, dict):
        errors.append("Radial microtest lacks runtime dependency evidence")
    else:
        if before_cuda.get("status") != "ok":
            errors.append("Radial microtest runtime dependency status is not ok")
        if not isinstance(before_cuda.get("aliases"), int) or before_cuda.get(
            "aliases", 0
        ) <= 0:
            errors.append("Radial microtest runtime dependency alias count is invalid")
        if not isinstance(before_cuda.get("mapped_files"), int) or before_cuda.get(
            "mapped_files", 0
        ) <= 0:
            errors.append("Radial microtest runtime mapped file count is invalid")
        digest = before_cuda.get("inventory_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            errors.append("Radial microtest runtime dependency digest is invalid")
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
