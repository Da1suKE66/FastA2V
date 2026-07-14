"""Pure-Python constants and validators for the pinned Radial Attention source.

The runtime backend imports the upstream mask generator from an audited derived
copy.  The only source change is the repository patch that makes unrelated
plotting and SageAttention imports optional.  No mask or kernel implementation
is copied into FastA2V.
"""

import hashlib
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from ovi.gpu_process_monitor import (
    GPU_EVIDENCE_SCHEMA_VERSION,
    gpu_compute_snapshot_errors,
    gpu_compute_snapshot_sequence_errors,
    trusted_nvidia_smi_metadata_errors,
    validate_pre_run_gpu_report,
)


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
RADIAL_GPU_PROCESS_BINDING_SCHEMA_VERSION = 2
RADIAL_GPU_IDLE_GUARD_MAX_AGE_SECONDS = 600.0
RADIAL_GPU_PROCESS_BINDING_METHODS = (
    "direct_nspid",
    "sampled_temporal_association_after_idle_guard",
)
RADIAL_GPU_PROCESS_CLAIM_SCOPES = {
    "direct_nspid": "snapshot_bound_not_continuous_exclusivity",
    "sampled_temporal_association_after_idle_guard": (
        "sampled_temporal_association_not_pid_ownership_or_"
        "continuous_exclusivity"
    ),
}
RADIAL_PMON_OBSERVATION_MODES = (
    "direct_c_observed",
    "pmon_reported_all_idle_during_audited_window",
)
RADIAL_GPU_QUERY_INTERVAL_SECONDS = 0.1
RADIAL_GPU_QUERY_MAX_GAP_SECONDS = 1.0
RADIAL_GPU_QUERY_MAX_DURATION_SECONDS = 1.0
RADIAL_GPU_QUERY_MIN_BACKEND_SAMPLES = 2
RADIAL_PMON_SAMPLE_INTERVAL_SECONDS = 1.0
RADIAL_PMON_MAX_RECEIPT_GAP_SECONDS = 2.0
RADIAL_QKV_STORAGE_BYTES = (
    3 * RADIAL_SEQUENCE * RADIAL_HEADS * RADIAL_HEAD_DIM * 2
)

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
        if not _exact_json_value(receipt.get(field), expected_value):
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


def parse_nvidia_smi_pmon_output(raw_stdout):
    """Parse auditable ``nvidia-smi pmon`` text using its dynamic header.

    Idle ``-`` rows are retained rather than silently discarded.  The caller
    can therefore distinguish a direct-compute observation from a sample that
    did not expose any process.
    """

    errors = []
    rows = []
    header_columns = None
    if not isinstance(raw_stdout, str):
        return {
            "header_columns": None,
            "rows": [],
            "errors": ["pmon raw stdout must be text"],
        }
    for line_number, raw_line in enumerate(raw_stdout.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            candidate = stripped[1:].strip().lower().split()
            if {"gpu", "pid", "type", "command"}.issubset(candidate):
                if len(candidate) != len(set(candidate)):
                    errors.append(
                        f"pmon header line {line_number} has duplicate columns"
                    )
                elif candidate[-1] != "command":
                    errors.append(
                        "pmon command column must be last for lossless parsing"
                    )
                elif header_columns is None:
                    header_columns = candidate
                elif candidate != header_columns:
                    errors.append("pmon header changed within one capture")
            continue
        if header_columns is None:
            errors.append(
                f"pmon data line {line_number} appeared before a valid header"
            )
            continue
        fields = stripped.split(maxsplit=len(header_columns) - 1)
        if len(fields) != len(header_columns):
            errors.append(
                f"pmon data line {line_number} has {len(fields)} fields; "
                f"expected {len(header_columns)}"
            )
            continue
        columns = dict(zip(header_columns, fields))
        try:
            gpu_index = int(columns["gpu"])
        except (KeyError, ValueError):
            errors.append(
                f"pmon data line {line_number} has an invalid GPU index"
            )
            continue
        pid_text = columns.get("pid")
        if pid_text == "-":
            host_pid = None
        else:
            try:
                host_pid = int(pid_text)
            except (TypeError, ValueError):
                errors.append(
                    f"pmon data line {line_number} has an invalid PID"
                )
                continue
        process_type = columns.get("type")
        if process_type == "-":
            process_type = None
        source_date = columns.get("date")
        source_time = columns.get("time")
        source_timestamp = None
        if (source_date is None) != (source_time is None):
            errors.append(
                f"pmon data line {line_number} has incomplete source time"
            )
            continue
        if source_date is not None:
            try:
                source_timestamp = datetime.strptime(
                    f"{source_date} {source_time}",
                    "%Y%m%d %H:%M:%S",
                ).replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                errors.append(
                    f"pmon data line {line_number} has invalid source time"
                )
                continue
        rows.append(
            {
                "line_number": line_number,
                "gpu_index": gpu_index,
                "host_pid": host_pid,
                "process_type": process_type,
                "command": columns.get("command", ""),
                "source_date": source_date,
                "source_time": source_time,
                "source_timestamp_unix_seconds": source_timestamp,
                "columns": columns,
            }
        )
    if header_columns is None:
        errors.append("pmon output lacks a valid gpu/pid/type/command header")
    return {
        "header_columns": header_columns,
        "rows": rows,
        "errors": errors,
    }


def _finite_number(value, *, nonnegative=False):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and (not nonnegative or float(value) >= 0.0)
    )


def _positive_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    )


def _json_integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _exact_json_value(value, expected):
    """Compare canonical JSON values without bool/int aliasing."""

    if isinstance(expected, bool):
        return isinstance(value, bool) and value is expected
    if isinstance(expected, int):
        return _json_integer(value) and value == expected
    if isinstance(expected, float):
        return _finite_number(value) and float(value) == expected
    if isinstance(expected, list):
        return (
            isinstance(value, list)
            and len(value) == len(expected)
            and all(
                _exact_json_value(item, expected_item)
                for item, expected_item in zip(value, expected)
            )
        )
    if isinstance(expected, dict):
        return (
            isinstance(value, dict)
            and set(value) == set(expected)
            and all(
                _exact_json_value(value[field], expected_value)
                for field, expected_value in expected.items()
            )
        )
    return value == expected


def _gpu_snapshot_errors(
    sample,
    expected_identity,
    expected_nvidia_smi_binary,
    context,
):
    errors = []
    if not isinstance(sample, dict):
        return [f"{context} must be a JSON object"]
    required_fields = {
        "available",
        "error",
        "device_index",
        "device_uuid",
        "device_name",
        "processes",
        "process_count",
        "sampled_at_unix_seconds",
        "sampled_at_monotonic_seconds",
        "query_started_at_unix_seconds",
        "query_finished_at_unix_seconds",
        "query_started_at_monotonic_seconds",
        "query_finished_at_monotonic_seconds",
        "boot_id",
        "nvidia_smi_binary",
        "query_receipt",
    }
    missing_fields = sorted(required_fields - set(sample))
    if missing_fields:
        errors.append(f"{context} is missing fields: {missing_fields}")
    errors.extend(
        f"{context}: {error}"
        for error in gpu_compute_snapshot_errors(sample)
    )
    if sample.get("available") is not True or sample.get("error") is not None:
        errors.append(f"{context} GPU query is unavailable")
    identity = (
        sample.get("device_index"),
        sample.get("device_uuid"),
        sample.get("device_name"),
        sample.get("boot_id"),
    )
    if identity != expected_identity:
        errors.append(f"{context} GPU identity or boot ID drifted")
    if sample.get("nvidia_smi_binary") != expected_nvidia_smi_binary:
        errors.append(f"{context} trusted nvidia-smi binary drifted")
    errors.extend(
        f"{context}: {error}"
        for error in trusted_nvidia_smi_metadata_errors(
            sample.get("nvidia_smi_binary")
        )
    )
    processes = sample.get("processes")
    count = sample.get("process_count")
    if (
        not isinstance(processes, list)
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count != len(processes)
        or count not in (0, 1)
    ):
        errors.append(f"{context} process list/count is invalid")
        processes = []
    for process in processes:
        if (
            not isinstance(process, dict)
            or not _positive_integer(process.get("host_pid"))
            or not _positive_integer(process.get("used_memory_mib"))
            or not isinstance(process.get("process_name"), str)
            or not process.get("process_name", "").strip()
        ):
            errors.append(f"{context} process details are invalid")
    for suffix in ("unix_seconds", "monotonic_seconds"):
        started = sample.get(f"query_started_at_{suffix}")
        sampled = sample.get(f"sampled_at_{suffix}")
        finished = sample.get(f"query_finished_at_{suffix}")
        if not all(
            _finite_number(value, nonnegative=True)
            for value in (started, sampled, finished)
        ) or not float(started) <= float(sampled) <= float(finished):
            errors.append(f"{context} {suffix} query timestamps are invalid")
        elif (
            float(finished) - float(started)
            > RADIAL_GPU_QUERY_MAX_DURATION_SECONDS
        ):
            errors.append(
                f"{context} {suffix} query duration exceeds the fixed maximum"
            )
    return errors


def _gpu_query_receipt_digest(sample):
    """Return a canonical identity for one independently timed raw receipt."""

    if not isinstance(sample, dict) or not isinstance(
        sample.get("query_receipt"), dict
    ):
        return None
    try:
        payload = json.dumps(
            sample["query_receipt"],
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(payload).hexdigest()


def _pmon_evidence_errors(
    pmon,
    *,
    expected_host_pid,
    expected_nvidia_smi_binary,
    immediate_finished_monotonic,
    cuda_touch_started_monotonic,
    context_live_finished_monotonic,
    backend_started_wall,
    backend_started_monotonic,
    backend_returned_wall,
    backend_returned_monotonic,
    sync_finished_wall,
    sync_finished_monotonic,
):
    """Recompute strict/direct or explicitly degraded all-idle pmon evidence."""

    errors = []
    if not isinstance(pmon, dict):
        return ["Radial GPU binding pmon evidence is missing"], []
    for required_field in (
        "compute_observation_deadline_reached_at_unix_seconds",
        "compute_observation_deadline_reached_at_monotonic_seconds",
        "window_compute_ready_at_unix_seconds",
        "window_compute_ready_at_monotonic_seconds",
        "window_compute_line_number",
    ):
        if required_field not in pmon:
            errors.append(f"Radial pmon field {required_field} is missing")
    observation_mode = pmon.get("observation_mode")
    if observation_mode not in RADIAL_PMON_OBSERVATION_MODES:
        errors.append("Radial pmon observation mode is invalid")
    direct_mode = observation_mode == "direct_c_observed"
    all_idle_mode = (
        observation_mode
        == "pmon_reported_all_idle_during_audited_window"
    )
    raw_stdout = pmon.get("raw_stdout")
    parsed = parse_nvidia_smi_pmon_output(raw_stdout)
    errors.extend(
        f"Radial pmon raw parse: {error}" for error in parsed["errors"]
    )
    if pmon.get("header_columns") != parsed["header_columns"]:
        errors.append("Radial pmon stored header differs from raw stdout")
    if pmon.get("rows") != parsed["rows"]:
        errors.append("Radial pmon stored rows differ from raw stdout")
    expected_command = [
        "/usr/bin/nvidia-smi",
        "pmon",
        "-i",
        "0",
        "-s",
        "um",
        "-d",
        "1",
        "-o",
        "DT",
    ]
    if pmon.get("command") != expected_command:
        errors.append("Radial pmon command differs from fixed continuous query")
    if pmon.get("locale") != {"LC_ALL": "C", "LANG": "C", "TZ": "UTC"}:
        errors.append("Radial pmon locale/time zone is not fixed to C/UTC")
    if pmon.get("nvidia_smi_binary") != expected_nvidia_smi_binary:
        errors.append("Radial pmon trusted nvidia-smi metadata drifted")
    errors.extend(
        f"Radial pmon: {error}"
        for error in trusted_nvidia_smi_metadata_errors(
            pmon.get("nvidia_smi_binary")
        )
    )
    if pmon.get("expected_host_pid") != expected_host_pid:
        errors.append("Radial pmon expected host PID is not cross-bound")
    expected_status = "ok" if direct_mode else "degraded" if all_idle_mode else None
    if (
        pmon.get("status") != expected_status
        or pmon.get("collection_status") != "ok"
        or pmon.get("direct_compute_type_observed") is not direct_mode
        or pmon.get("host_pid_observed") is not direct_mode
        or pmon.get("continuous_exclusivity_proven") is not False
        or not _finite_number(
            pmon.get("compute_observation_timeout_seconds"),
            nonnegative=True,
        )
        or float(pmon.get("compute_observation_timeout_seconds", math.nan))
        != 20.0
        or not isinstance(raw_stdout, str)
        or not isinstance(pmon.get("raw_stderr"), str)
        or not _positive_integer(pmon.get("process_pid"))
        or not isinstance(pmon.get("resolved_executable"), str)
        or not Path(pmon.get("resolved_executable", "")).is_absolute()
        or pmon.get("resolved_executable") != "/usr/bin/nvidia-smi"
        or not _json_integer(pmon.get("exit_code"))
        or pmon.get("exit_code") not in (0, -15)
        or pmon.get("termination_method") != "terminate"
        or pmon.get("timed_out") is not False
        or pmon.get("parser_errors") != []
        or pmon.get("errors") != []
    ):
        errors.append("Radial pmon process completion evidence is invalid")
    stdout_bytes = raw_stdout.encode("utf-8") if isinstance(raw_stdout, str) else b""
    raw_stderr = pmon.get("raw_stderr")
    stderr_bytes = (
        raw_stderr.encode("utf-8") if isinstance(raw_stderr, str) else b""
    )
    if (
        not _json_integer(pmon.get("raw_stdout_bytes"))
        or pmon.get("raw_stdout_bytes") != len(stdout_bytes)
        or pmon.get("raw_stdout_sha256")
        != hashlib.sha256(stdout_bytes).hexdigest()
        or not _json_integer(pmon.get("raw_stderr_bytes"))
        or pmon.get("raw_stderr_bytes") != len(stderr_bytes)
        or pmon.get("raw_stderr_sha256")
        != hashlib.sha256(stderr_bytes).hexdigest()
    ):
        errors.append("Radial pmon raw stream fingerprints are invalid")
    stderr_records = pmon.get("stderr_line_records")
    if not isinstance(stderr_records, list):
        errors.append("Radial pmon stderr line records are missing")
        stderr_records = []
    elif "".join(
        record.get("raw_line", "") if isinstance(record, dict) else ""
        for record in stderr_records
    ) != raw_stderr:
        errors.append("Radial pmon stderr records do not reconstruct raw stderr")
    if raw_stderr:
        errors.append("Radial pmon emitted stderr during the audited window")
    line_records = pmon.get("stdout_line_records")
    if not isinstance(line_records, list) or not line_records:
        errors.append("Radial pmon stdout line records are missing")
        line_records = []
    elif "".join(
        record.get("raw_line", "") if isinstance(record, dict) else ""
        for record in line_records
    ) != raw_stdout:
        errors.append("Radial pmon line records do not reconstruct raw stdout")
    rows_by_line = {row["line_number"]: row for row in parsed["rows"]}
    previous_wall = None
    previous_monotonic = None
    for index, record in enumerate(line_records):
        wall = (
            record.get("received_at_unix_seconds")
            if isinstance(record, dict)
            else None
        )
        monotonic = (
            record.get("received_at_monotonic_seconds")
            if isinstance(record, dict)
            else None
        )
        if (
            not isinstance(record, dict)
            or not _json_integer(record.get("line_index"))
            or record.get("line_index") != index
            or not isinstance(record.get("raw_line"), str)
            or record.get("parsed_row") != rows_by_line.get(index + 1)
            or not _finite_number(wall, nonnegative=True)
            or not _finite_number(monotonic, nonnegative=True)
            or (previous_wall is not None and float(wall) < previous_wall)
            or (
                previous_monotonic is not None
                and float(monotonic) < previous_monotonic
            )
        ):
            errors.append("Radial pmon stdout line record is invalid")
            break
        previous_wall = float(wall)
        previous_monotonic = float(monotonic)
    lifecycle_fields = (
        "spawn_started_at",
        "process_started_at",
        "header_ready_at",
        "idle_baseline_ready_at",
        "host_pid_bound_at",
        "backend_window_started_at",
        (
            "window_compute_ready_at"
            if direct_mode
            else "compute_observation_deadline_reached_at"
        ),
        "final_sync_covered_at",
        "stop_requested_at",
        "process_exited_at",
    )
    lifecycle_values = {}
    for suffix in ("unix_seconds", "monotonic_seconds"):
        values = []
        for field in lifecycle_fields:
            value = pmon.get(f"{field}_{suffix}")
            lifecycle_values[(field, suffix)] = value
            values.append(value)
        if not all(_finite_number(value, nonnegative=True) for value in values):
            errors.append(f"Radial pmon {suffix} lifecycle timestamps are invalid")
        elif values != sorted(values):
            errors.append(f"Radial pmon {suffix} lifecycle order is invalid")
    pmon_started = lifecycle_values.get(
        ("process_started_at", "monotonic_seconds")
    )
    pmon_spawn_started = lifecycle_values.get(
        ("spawn_started_at", "monotonic_seconds")
    )
    idle_ready = lifecycle_values.get(
        ("idle_baseline_ready_at", "monotonic_seconds")
    )
    host_pid_bound = lifecycle_values.get(
        ("host_pid_bound_at", "monotonic_seconds")
    )
    backend_window_started = lifecycle_values.get(
        ("backend_window_started_at", "monotonic_seconds")
    )
    backend_window_started_wall = lifecycle_values.get(
        ("backend_window_started_at", "unix_seconds")
    )
    observation_ready = lifecycle_values.get(
        (
            "window_compute_ready_at"
            if direct_mode
            else "compute_observation_deadline_reached_at",
            "monotonic_seconds",
        )
    )
    final_sync_covered = lifecycle_values.get(
        ("final_sync_covered_at", "monotonic_seconds")
    )
    stop_requested = lifecycle_values.get(
        ("stop_requested_at", "monotonic_seconds")
    )
    if not all(
        _finite_number(value, nonnegative=True)
        for value in (
            pmon_started,
            pmon_spawn_started,
            immediate_finished_monotonic,
            cuda_touch_started_monotonic,
            context_live_finished_monotonic,
            idle_ready,
            host_pid_bound,
            backend_window_started,
            backend_window_started_wall,
            backend_started_wall,
            backend_started_monotonic,
            backend_returned_monotonic,
            observation_ready,
            sync_finished_monotonic,
            final_sync_covered,
            stop_requested,
        )
    ) or not (
        float(immediate_finished_monotonic) <= float(pmon_spawn_started)
        <= float(pmon_started) <= float(idle_ready)
        <= float(cuda_touch_started_monotonic)
        <= float(context_live_finished_monotonic)
        <= float(host_pid_bound)
        <= float(backend_window_started)
        == float(backend_started_monotonic)
        <= float(observation_ready)
        <= float(backend_returned_monotonic)
        <= float(sync_finished_monotonic)
        <= float(final_sync_covered)
        <= float(stop_requested)
    ):
        errors.append("Radial pmon does not span exact backend through final sync")
    if (
        not _finite_number(backend_window_started_wall, nonnegative=True)
        or not _finite_number(backend_started_wall, nonnegative=True)
        or backend_window_started_wall != backend_started_wall
    ):
        errors.append(
            "Radial pmon Unix backend-window start differs from GPU binding"
        )
    if all_idle_mode and all(
        _finite_number(value, nonnegative=True)
        for value in (observation_ready, backend_started_monotonic)
    ) and (
        float(observation_ready) - float(backend_started_monotonic)
        < float(pmon.get("compute_observation_timeout_seconds", math.inf))
    ):
        errors.append("Radial all-idle pmon observation ended before its timeout")
    rows = parsed["rows"]
    if not rows:
        errors.append("Radial pmon raw output contains no process rows")
    previous_source_time = None
    for row in rows:
        source_time = row.get("source_timestamp_unix_seconds")
        line_number = row.get("line_number")
        record = (
            line_records[line_number - 1]
            if isinstance(line_number, int)
            and 0 < line_number <= len(line_records)
            else None
        )
        received_wall = (
            record.get("received_at_unix_seconds")
            if isinstance(record, dict)
            else None
        )
        if (
            not _finite_number(source_time, nonnegative=True)
            or not _finite_number(received_wall, nonnegative=True)
            or float(source_time) > float(received_wall) + 1.0
            or float(received_wall) - float(source_time) > 5.0
            or (
                previous_source_time is not None
                and float(source_time) < previous_source_time
            )
        ):
            errors.append("Radial pmon source timestamp is invalid or delayed")
        elif previous_source_time is None or float(source_time) >= previous_source_time:
            previous_source_time = float(source_time)
        command = str(row.get("command", "")).lower()
        idle_row = row.get("host_pid") is None
        if idle_row:
            valid_row = (
                row.get("gpu_index") == 0
                and row.get("process_type") is None
                and row.get("command") == "-"
            )
        else:
            valid_row = (
                direct_mode
                and
                row.get("gpu_index") == 0
                and row.get("host_pid") == expected_host_pid
                and row.get("process_type") == "C"
                and bool(row.get("command"))
                and "mps" not in command
            )
        if not valid_row:
            errors.append(
                "Radial pmon row is neither idle nor the stable direct C client"
            )
            break
    baseline_line = pmon.get("idle_baseline_line_number")
    baseline_row = next(
        (
            row
            for row in rows
            if row.get("line_number") == baseline_line
        ),
        None,
    )
    if (
        not _positive_integer(baseline_line)
        or not isinstance(baseline_row, dict)
        or baseline_row.get("gpu_index") != 0
        or baseline_row.get("host_pid") is not None
        or baseline_row.get("process_type") is not None
        or baseline_row.get("command") != "-"
    ):
        errors.append("Radial pmon idle baseline row is invalid")
    if _positive_integer(baseline_line) and any(
        row.get("host_pid") is not None
        and isinstance(row.get("line_number"), int)
        and row["line_number"] <= baseline_line
        for row in rows
    ):
        errors.append("Radial pmon observed a process before the idle baseline")
    window_start_line = pmon.get("backend_window_start_line_number")
    window_compute_line = pmon.get("window_compute_line_number")
    matching_rows = [
        row
        for row in rows
        if _positive_integer(window_start_line)
        and row.get("line_number", 0) > window_start_line
        and row.get("gpu_index") == 0
        and row.get("host_pid") == expected_host_pid
        and row.get("process_type") == "C"
    ]
    selected_window_row = next(
        (
            row
            for row in matching_rows
            if row.get("line_number") == window_compute_line
        ),
        None,
    )
    selected_window_record = (
        line_records[window_compute_line - 1]
        if _positive_integer(window_compute_line)
        and 0 < window_compute_line <= len(line_records)
        else None
    )
    selected_source_time = (
        selected_window_row.get("source_timestamp_unix_seconds")
        if isinstance(selected_window_row, dict)
        else None
    )
    if direct_mode and (
        not _positive_integer(window_start_line)
        or not _positive_integer(baseline_line)
        or not _positive_integer(window_compute_line)
        or window_start_line < baseline_line
        or window_start_line > len(line_records)
        or not isinstance(selected_window_row, dict)
        or not isinstance(selected_window_record, dict)
    ):
        errors.append("Radial pmon backend-window C line binding is invalid")
    if direct_mode and (not all(
        _finite_number(value, nonnegative=True)
        for value in (
            selected_source_time,
            backend_started_wall,
            backend_returned_wall,
        )
    ) or not (
        float(backend_started_wall) < float(selected_source_time)
        <= float(backend_returned_wall)
    )):
        errors.append(
            "Radial pmon source-DT sample is not strictly inside the backend window"
        )
    if all_idle_mode:
        if matching_rows or any(row.get("host_pid") is not None for row in rows):
            errors.append("Radial all-idle pmon mode contains a process row")
        in_backend_idle_rows = [
            row
            for row in rows
            if _positive_integer(window_start_line)
            and _positive_integer(row.get("line_number"))
            and row["line_number"] > window_start_line
            and _finite_number(
                row.get("source_timestamp_unix_seconds"), nonnegative=True
            )
            and _finite_number(backend_started_wall, nonnegative=True)
            and _finite_number(backend_returned_wall, nonnegative=True)
            and float(backend_started_wall)
            < float(row["source_timestamp_unix_seconds"])
            <= float(backend_returned_wall)
        ]
        if not in_backend_idle_rows:
            errors.append(
                "Radial all-idle pmon evidence has no source-DT row inside the "
                "exact backend window"
            )
        if (
            pmon.get("window_compute_ready_at_unix_seconds") is not None
            or pmon.get("window_compute_ready_at_monotonic_seconds") is not None
            or pmon.get("window_compute_line_number") is not None
        ):
            errors.append("Radial all-idle pmon mode forged a direct-C readiness")
    elif direct_mode and (
        pmon.get("compute_observation_deadline_reached_at_unix_seconds")
        is not None
        or pmon.get(
            "compute_observation_deadline_reached_at_monotonic_seconds"
        )
        is not None
    ):
        errors.append("Radial direct-C pmon mode also claims an all-idle deadline")
    header_records = [
        record
        for record in line_records
        if isinstance(record, dict)
        and isinstance(record.get("raw_line"), str)
        and record["raw_line"].lstrip().startswith("#")
        and {
            "gpu",
            "pid",
            "type",
            "command",
        }.issubset(record["raw_line"].lstrip("# ").lower().split())
    ]
    baseline_record = (
        line_records[baseline_line - 1]
        if _positive_integer(baseline_line)
        and 0 < baseline_line <= len(line_records)
        else None
    )
    final_sync_line = pmon.get("final_sync_covered_line_number")
    final_sync_row = next(
        (
            row
            for row in rows
            if row.get("line_number") == final_sync_line
        ),
        None,
    )
    final_sync_record = (
        line_records[final_sync_line - 1]
        if _positive_integer(final_sync_line)
        and 0 < final_sync_line <= len(line_records)
        else None
    )
    if (
        not isinstance(final_sync_row, dict)
        or not isinstance(final_sync_record, dict)
        or not _finite_number(sync_finished_wall, nonnegative=True)
        or not _finite_number(sync_finished_monotonic, nonnegative=True)
        or not _finite_number(
            final_sync_row.get("source_timestamp_unix_seconds"),
            nonnegative=True,
        )
        or float(final_sync_row["source_timestamp_unix_seconds"])
        < float(sync_finished_wall)
        or float(final_sync_record.get(
            "received_at_monotonic_seconds", -math.inf
        )) < float(sync_finished_monotonic)
    ):
        errors.append("Radial pmon has no valid source-DT row after final sync")
    if all_idle_mode:
        valid_window_start = (
            _positive_integer(window_start_line)
            and _positive_integer(baseline_line)
            and baseline_line <= window_start_line <= len(line_records)
        )
        if not valid_window_start:
            errors.append(
                "Radial all-idle pmon backend-window line boundary is invalid"
            )
        coverage_start_candidates = []
        if valid_window_start and all(
            _finite_number(value, nonnegative=True)
            for value in (backend_started_wall, backend_started_monotonic)
        ):
            for row in rows:
                line_number = row.get("line_number")
                source_time = row.get("source_timestamp_unix_seconds")
                record = (
                    line_records[line_number - 1]
                    if isinstance(line_number, int)
                    and not isinstance(line_number, bool)
                    and 0 < line_number <= len(line_records)
                    else None
                )
                received_monotonic = (
                    record.get("received_at_monotonic_seconds")
                    if isinstance(record, dict)
                    else None
                )
                if (
                    isinstance(line_number, int)
                    and line_number <= window_start_line
                    and _finite_number(source_time, nonnegative=True)
                    and _finite_number(received_monotonic, nonnegative=True)
                    and float(source_time) <= float(backend_started_wall)
                    and float(received_monotonic)
                    <= float(backend_started_monotonic)
                ):
                    coverage_start_candidates.append((row, record))
        if not coverage_start_candidates:
            errors.append(
                "Radial all-idle pmon cadence lacks a raw row sampled before "
                "the backend window"
            )
        elif not isinstance(final_sync_row, dict):
            errors.append(
                "Radial all-idle pmon cadence lacks a final-sync endpoint"
            )
        else:
            coverage_start_row, _ = coverage_start_candidates[-1]
            coverage_start_line = coverage_start_row.get("line_number")
            cadence_rows = []
            for row in rows:
                line_number = row.get("line_number")
                if not (
                    isinstance(line_number, int)
                    and not isinstance(line_number, bool)
                    and isinstance(coverage_start_line, int)
                    and isinstance(final_sync_line, int)
                    and coverage_start_line <= line_number <= final_sync_line
                ):
                    continue
                record = (
                    line_records[line_number - 1]
                    if 0 < line_number <= len(line_records)
                    else None
                )
                cadence_rows.append((row, record))
            cadence_error = len(cadence_rows) < 2
            for (previous_row, previous_record), (row, record) in zip(
                cadence_rows,
                cadence_rows[1:],
            ):
                previous_source = previous_row.get(
                    "source_timestamp_unix_seconds"
                )
                source = row.get("source_timestamp_unix_seconds")
                previous_received = (
                    previous_record.get("received_at_monotonic_seconds")
                    if isinstance(previous_record, dict)
                    else None
                )
                received = (
                    record.get("received_at_monotonic_seconds")
                    if isinstance(record, dict)
                    else None
                )
                if not all(
                    _finite_number(value, nonnegative=True)
                    for value in (
                        previous_source,
                        source,
                        previous_received,
                        received,
                    )
                ):
                    cadence_error = True
                    break
                source_gap = float(source) - float(previous_source)
                receipt_gap = float(received) - float(previous_received)
                if not (
                    0.0 < source_gap <= RADIAL_PMON_SAMPLE_INTERVAL_SECONDS
                    and 0.0
                    < receipt_gap
                    <= RADIAL_PMON_MAX_RECEIPT_GAP_SECONDS
                ):
                    cadence_error = True
                    break
            if cadence_error:
                errors.append(
                    "Radial all-idle pmon raw rows violate the fixed cadence "
                    "or continuous backend/final-sync coverage"
                )
    if (
        not header_records
        or not isinstance(baseline_record, dict)
        or (direct_mode and not isinstance(selected_window_record, dict))
        or not isinstance(final_sync_record, dict)
    ):
        errors.append("Radial pmon ready events lack matching raw line records")
    else:
        lifecycle_record_pairs = [
            ("header_ready_at", header_records[0]),
            ("idle_baseline_ready_at", baseline_record),
            ("final_sync_covered_at", final_sync_record),
        ]
        if direct_mode:
            lifecycle_record_pairs.append(
                ("window_compute_ready_at", selected_window_record)
            )
        for event_name, record in lifecycle_record_pairs:
            for suffix in ("unix_seconds", "monotonic_seconds"):
                if pmon.get(f"{event_name}_{suffix}") != record.get(
                    f"received_at_{suffix}"
                ):
                    errors.append(
                        "Radial pmon lifecycle event differs from its raw line"
                    )
                    break
    return errors, rows


def radial_gpu_process_binding_errors(
    binding,
    *,
    expected_pre_run_gpu=None,
    expected_pre_run_gpu_sha256=None,
    expected_pre_run_gpu_path=None,
    expected_python_executable=None,
):
    """Independently validate the fail-closed Radial GPU snapshot binding."""

    if not isinstance(binding, dict):
        return ["Radial GPU process binding must be a JSON object"]
    errors = []
    if (
        not _json_integer(binding.get("schema_version"))
        or binding.get("schema_version")
        != RADIAL_GPU_PROCESS_BINDING_SCHEMA_VERSION
    ):
        errors.append("Radial GPU process binding schema is unsupported")
    method = binding.get("binding_method")
    if method not in RADIAL_GPU_PROCESS_BINDING_METHODS:
        errors.append("Radial GPU process binding method is invalid")
    if binding.get("claim_scope") != RADIAL_GPU_PROCESS_CLAIM_SCOPES.get(method):
        errors.append("Radial GPU process binding claim scope is invalid")
    pmon_observation_mode = binding.get("pmon_observation_mode")
    if pmon_observation_mode not in RADIAL_PMON_OBSERVATION_MODES:
        errors.append("Radial GPU process binding pmon mode is invalid")
    expected_ownership = (
        "proven_by_nspid"
        if method == "direct_nspid"
        else "unknown_sampled_temporal_association_only"
    )
    if binding.get("host_pid_ownership") != expected_ownership:
        errors.append("Radial GPU process ownership scope is invalid")

    pre_run_gpu = binding.get("pre_run_gpu")
    if not isinstance(pre_run_gpu, dict):
        errors.append("Radial GPU binding lacks pre-run GPU evidence")
        pre_run_gpu = {}
    if expected_pre_run_gpu is not None and pre_run_gpu != expected_pre_run_gpu:
        errors.append("Radial GPU binding pre-run evidence differs from run evidence")
    pre_sha = binding.get("pre_run_gpu_sha256")
    if (
        not isinstance(pre_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", pre_sha) is None
    ):
        errors.append("Radial GPU binding pre-run SHA256 is invalid")
    if (
        expected_pre_run_gpu_sha256 is not None
        and pre_sha != expected_pre_run_gpu_sha256
    ):
        errors.append("Radial GPU binding pre-run SHA256 differs from exact file bytes")
    pre_path = binding.get("pre_run_gpu_path")
    if not isinstance(pre_path, str) or not Path(pre_path).is_absolute():
        errors.append("Radial GPU binding pre-run path is not absolute")
    if expected_pre_run_gpu_path is not None and pre_path != expected_pre_run_gpu_path:
        errors.append("Radial GPU binding pre-run path differs from verified file")

    device_uuid = pre_run_gpu.get("device_uuid")
    device_name = pre_run_gpu.get("device_name")
    boot_id = pre_run_gpu.get("boot_id")
    nvidia_smi_binary = pre_run_gpu.get("nvidia_smi_binary")
    expected_identity = (0, device_uuid, device_name, boot_id)
    errors.extend(
        f"Radial pre-run GPU evidence: {error}"
        for error in validate_pre_run_gpu_report(
            pre_run_gpu,
            cuda_visible_devices=pre_run_gpu.get("cuda_visible_devices"),
        )
    )
    errors.extend(
        f"Radial pre-run GPU evidence: {error}"
        for error in trusted_nvidia_smi_metadata_errors(nvidia_smi_binary)
    )
    if (
        not _json_integer(pre_run_gpu.get("schema_version"))
        or pre_run_gpu.get("schema_version") != GPU_EVIDENCE_SCHEMA_VERSION
        or pre_run_gpu.get("check_type") != "pre_run_idle"
        or not _json_integer(pre_run_gpu.get("physical_device_index"))
        or pre_run_gpu.get("physical_device_index") != 0
        or not _json_integer(pre_run_gpu.get("device_index"))
        or pre_run_gpu.get("device_index") != 0
        or pre_run_gpu.get("available") is not True
        or pre_run_gpu.get("error") is not None
        or pre_run_gpu.get("processes") != []
        or pre_run_gpu.get("process_count") != 0
        or pre_run_gpu.get("idle") is not True
        or pre_run_gpu.get("valid_for_run") is not True
        or pre_run_gpu.get("errors") != []
    ):
        errors.append("Radial GPU binding pre-run idle evidence is invalid")
    boot_pattern = (
        r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
    )
    if (
        not isinstance(device_uuid, str)
        or not device_uuid.startswith("GPU-")
        or not isinstance(device_name, str)
        or not device_name
        or not isinstance(boot_id, str)
        or re.fullmatch(boot_pattern, boot_id) is None
        or not isinstance(pre_run_gpu.get("run_nonce"), str)
        or re.fullmatch(r"[0-9a-f]{32}", pre_run_gpu.get("run_nonce", ""))
        is None
    ):
        errors.append("Radial GPU binding pre-run identity/nonce is invalid")
    if (
        binding.get("current_cuda_device_uuid") != device_uuid
        or binding.get("current_cuda_device_name") != device_name
        or not _json_integer(binding.get("current_cuda_device_index"))
        or binding.get("current_cuda_device_index") != 0
        or binding.get("cuda_visible_devices") != device_uuid
        or pre_run_gpu.get("cuda_visible_devices") != device_uuid
    ):
        errors.append("Radial GPU binding CUDA identity is not cross-bound")

    python_executable = binding.get("python_executable")
    resolved_executable = binding.get("python_executable_resolved")
    if (
        not isinstance(python_executable, str)
        or not Path(python_executable).is_absolute()
        or not isinstance(resolved_executable, str)
        or not Path(resolved_executable).is_absolute()
    ):
        errors.append("Radial GPU binding Python executable evidence is invalid")
    if (
        expected_python_executable is not None
        and python_executable != expected_python_executable
    ):
        errors.append("Radial GPU binding Python executable differs from preflight")

    phase_fields = (
        "microtest_started_at",
        "cuda_touch_started_at",
        "setup_cuda_synchronize_started_at",
        "setup_cuda_synchronized_at",
        "exact_backend_started_at",
        "exact_backend_returned_at",
        "cuda_synchronize_started_at",
        "cuda_synchronized_at",
    )
    phase_times = {}
    for suffix in ("unix_seconds", "monotonic_seconds"):
        values = []
        for field in phase_fields:
            value = binding.get(f"{field}_{suffix}")
            phase_times[(field, suffix)] = value
            values.append(value)
        if not all(_finite_number(value, nonnegative=True) for value in values):
            errors.append(f"Radial GPU binding {suffix} phase timestamps are invalid")
        elif values != sorted(values):
            errors.append(f"Radial GPU binding {suffix} phase order is invalid")
    pre_wall = pre_run_gpu.get("sampled_at_unix_seconds")
    pre_mono = pre_run_gpu.get("sampled_at_monotonic_seconds")
    micro_wall = phase_times.get(("microtest_started_at", "unix_seconds"))
    micro_mono = phase_times.get(("microtest_started_at", "monotonic_seconds"))
    if not all(
        _finite_number(value, nonnegative=True)
        for value in (pre_wall, pre_mono, micro_wall, micro_mono)
    ):
        errors.append("Radial GPU binding pre-run timestamps are invalid")
    elif not (
        float(pre_wall) <= float(micro_wall)
        and float(pre_mono) <= float(micro_mono)
        and float(micro_wall) - float(pre_wall)
        <= RADIAL_GPU_IDLE_GUARD_MAX_AGE_SECONDS
    ):
        errors.append("Radial GPU binding pre-run guard is stale, future, or rebooted")

    immediate = binding.get("immediate_pre_cuda_sample")
    context_live = binding.get("context_live_sample")
    interval_samples = binding.get("interval_samples")
    post_samples = binding.get("post_cuda_samples")
    if (
        not _finite_number(binding.get("interval_seconds"), nonnegative=True)
        or float(binding.get("interval_seconds", math.nan))
        != RADIAL_GPU_QUERY_INTERVAL_SECONDS
        or not isinstance(interval_samples, list)
        or len(interval_samples) < 2
    ):
        errors.append("Radial GPU binding interval samples are missing")
        interval_samples = []
    if not isinstance(post_samples, list) or len(post_samples) != 2:
        errors.append("Radial GPU binding requires exactly two post-CUDA samples")
        post_samples = []
    if (
        not _finite_number(
            binding.get("max_query_gap_seconds"), nonnegative=True
        )
        or float(binding.get("max_query_gap_seconds", math.nan))
        != RADIAL_GPU_QUERY_MAX_GAP_SECONDS
        or not _json_integer(binding.get("minimum_backend_query_samples"))
        or binding.get("minimum_backend_query_samples")
        != RADIAL_GPU_QUERY_MIN_BACKEND_SAMPLES
    ):
        errors.append("Radial GPU process-query coverage policy is invalid")
    labelled_samples = [
        ("immediate pre-CUDA sample", immediate),
        ("context-live sample", context_live),
        *[
            (f"interval sample {index}", sample)
            for index, sample in enumerate(interval_samples)
        ],
        *[
            (f"post-CUDA sample {index}", sample)
            for index, sample in enumerate(post_samples)
        ],
    ]
    for context, sample in labelled_samples:
        errors.extend(
            _gpu_snapshot_errors(
                sample,
                expected_identity,
                nvidia_smi_binary,
                context,
            )
        )
    if len(interval_samples) >= 2:
        errors.extend(
            f"Radial interval raw receipts: {error}"
            for error in gpu_compute_snapshot_sequence_errors(
                interval_samples
            )
        )
    if isinstance(immediate, dict) and (
        immediate.get("process_count") != 0 or immediate.get("processes") != []
    ):
        errors.append("Radial immediate pre-CUDA sample is not idle")
    for context, sample in [
        ("context-live", context_live),
        *[
            (f"post-CUDA {index}", sample)
            for index, sample in enumerate(post_samples)
        ],
    ]:
        if not isinstance(sample, dict) or sample.get("process_count") != 1:
            errors.append(f"Radial {context} sample is not a singleton")

    immediate_finished = (
        immediate.get("query_finished_at_monotonic_seconds")
        if isinstance(immediate, dict)
        else None
    )
    touch_mono = phase_times.get(("cuda_touch_started_at", "monotonic_seconds"))
    context_finished = (
        context_live.get("query_finished_at_monotonic_seconds")
        if isinstance(context_live, dict)
        else None
    )
    backend_started = phase_times.get(
        ("exact_backend_started_at", "monotonic_seconds")
    )
    sync_finished = phase_times.get(
        ("cuda_synchronized_at", "monotonic_seconds")
    )
    post_starts = [
        sample.get("query_started_at_monotonic_seconds")
        for sample in post_samples
        if isinstance(sample, dict)
    ]
    post_finishes = [
        sample.get("query_finished_at_monotonic_seconds")
        for sample in post_samples
        if isinstance(sample, dict)
    ]
    ordered_phase_samples = (
        immediate_finished,
        touch_mono,
        phase_times.get(
            ("setup_cuda_synchronize_started_at", "monotonic_seconds")
        ),
        phase_times.get(
            ("setup_cuda_synchronized_at", "monotonic_seconds")
        ),
        (
            context_live.get("query_started_at_monotonic_seconds")
            if isinstance(context_live, dict)
            else None
        ),
        context_finished,
        backend_started,
        sync_finished,
        *post_starts,
        *post_finishes,
    )
    if len(post_starts) != 2 or len(post_finishes) != 2 or not all(
        _finite_number(value, nonnegative=True)
        for value in ordered_phase_samples
    ):
        errors.append("Radial GPU sample/phase timestamps are invalid")
    elif not (
        float(immediate_finished) <= float(touch_mono)
        <= float(
            phase_times[(
                "setup_cuda_synchronize_started_at",
                "monotonic_seconds",
            )]
        )
        <= float(
            phase_times[("setup_cuda_synchronized_at", "monotonic_seconds")]
        )
        <= float(context_live["query_started_at_monotonic_seconds"])
        <= float(context_finished) <= float(backend_started)
        <= float(sync_finished) <= float(post_starts[0])
        <= float(post_finishes[0]) <= float(post_starts[1])
        <= float(post_finishes[1])
        and float(post_starts[1]) - float(post_finishes[0]) >= 0.1
    ):
        errors.append("Radial GPU sample/phase timestamp binding is invalid")
    if len(post_samples) == 2 and isinstance(post_samples[-1], dict):
        final_wall = post_samples[-1].get("query_finished_at_unix_seconds")
        if not all(
            _finite_number(value, nonnegative=True)
            for value in (pre_wall, final_wall)
        ) or not (
            float(pre_wall) <= float(final_wall)
            and float(final_wall) - float(pre_wall)
            <= RADIAL_GPU_IDLE_GUARD_MAX_AGE_SECONDS
        ):
            errors.append("Radial GPU binding final snapshot exceeds guard age")

    all_samples = [
        sample
        for _, sample in labelled_samples
        if isinstance(sample, dict)
        and _finite_number(
            sample.get("query_finished_at_monotonic_seconds"),
            nonnegative=True,
        )
    ]
    ordered_samples = sorted(
        all_samples,
        key=lambda sample: float(sample["query_finished_at_monotonic_seconds"]),
    )
    backend_returned = phase_times.get(
        ("exact_backend_returned_at", "monotonic_seconds")
    )
    backend_complete_samples = [
        sample
        for sample in interval_samples
        if isinstance(sample, dict)
        if _finite_number(
            sample.get("query_started_at_monotonic_seconds"),
            nonnegative=True,
        )
        and _finite_number(
            sample.get("query_finished_at_monotonic_seconds"),
            nonnegative=True,
        )
        and _finite_number(backend_started, nonnegative=True)
        and _finite_number(backend_returned, nonnegative=True)
        and float(backend_started)
        <= float(sample["query_started_at_monotonic_seconds"])
        <= float(sample["query_finished_at_monotonic_seconds"])
        <= float(backend_returned)
    ]
    if backend_complete_samples:
        errors.extend(
            f"Radial backend raw receipts: {error}"
            for error in gpu_compute_snapshot_sequence_errors(
                backend_complete_samples
            )
        )
    unique_backend_receipts = {
        digest
        for digest in (
            _gpu_query_receipt_digest(sample)
            for sample in backend_complete_samples
        )
        if digest is not None
    }
    if any(
        _gpu_query_receipt_digest(sample) is None
        for sample in backend_complete_samples
    ):
        errors.append("Radial backend raw query receipt identity is invalid")
    if (
        len(unique_backend_receipts)
        < RADIAL_GPU_QUERY_MIN_BACKEND_SAMPLES
    ):
        errors.append(
            "Radial sampled evidence lacks two unique complete raw "
            "process-query receipts inside the backend window"
        )
    coverage_samples = []
    coverage_keys = set()
    for sample in [context_live, *all_samples]:
        if not isinstance(sample, dict):
            continue
        key = (
            sample.get("query_started_at_monotonic_seconds"),
            sample.get("query_finished_at_monotonic_seconds"),
        )
        if key not in coverage_keys:
            coverage_keys.add(key)
            coverage_samples.append(sample)
    coverage_samples.sort(
        key=lambda sample: float(
            sample.get("query_started_at_monotonic_seconds", math.inf)
        )
    )
    coverage_end = context_finished
    if _finite_number(coverage_end, nonnegative=True):
        for sample in coverage_samples:
            query_start = sample.get("query_started_at_monotonic_seconds")
            query_finish = sample.get("query_finished_at_monotonic_seconds")
            if not (
                _finite_number(query_start, nonnegative=True)
                and _finite_number(query_finish, nonnegative=True)
                and float(query_start) <= float(query_finish)
            ):
                continue
            if float(query_finish) <= float(coverage_end):
                continue
            if (
                float(query_start) - float(coverage_end)
                > RADIAL_GPU_QUERY_MAX_GAP_SECONDS
            ):
                errors.append(
                    "Radial sampled process-query coverage exceeds the maximum gap"
                )
                break
            coverage_end = max(float(coverage_end), float(query_finish))
        final_query_finish = (
            post_samples[-1].get("query_finished_at_monotonic_seconds")
            if post_samples and isinstance(post_samples[-1], dict)
            else None
        )
        if not _finite_number(final_query_finish, nonnegative=True) or float(
            coverage_end
        ) < float(final_query_finish):
            errors.append(
                "Radial sampled process-query coverage does not reach the final "
                "post-sync snapshot"
            )
    positive_seen = False
    positive_processes = []
    for sample in ordered_samples:
        if sample.get("process_count") == 1:
            positive_seen = True
            processes = sample.get("processes")
            if isinstance(processes, list) and len(processes) == 1:
                positive_processes.append(processes[0])
        elif positive_seen and sample.get("process_count") == 0:
            errors.append("Radial GPU process disappeared after first live snapshot")
            break
    if _finite_number(context_finished, nonnegative=True):
        for sample in all_samples:
            started = sample.get("query_started_at_monotonic_seconds")
            if (
                _finite_number(started, nonnegative=True)
                and float(started) >= float(context_finished)
                and sample.get("process_count") != 1
            ):
                errors.append("Radial live-phase GPU sample is not a singleton")
                break
    positive_identities = {
        (
            process.get("host_pid") if isinstance(process, dict) else None,
            process.get("process_name", "").strip()
            if isinstance(process, dict)
            else "",
        )
        for process in positive_processes
    }
    if len(positive_identities) != 1:
        errors.append("Radial GPU singleton PID/process name is not stable")
        host_pid = None
        process_name = ""
    else:
        host_pid, process_name = next(iter(positive_identities))
    normalized_name = process_name.lower()
    if (
        process_name
        and normalized_name != "[not found]"
        and "python" not in normalized_name
    ) or "mps" in normalized_name:
        errors.append("Radial GPU process name is not an allowed direct client")

    container_pid = binding.get("container_pid")
    recorded_host_pid = binding.get("nvidia_smi_host_pid")
    if (
        not _positive_integer(container_pid)
        or not _positive_integer(recorded_host_pid)
        or recorded_host_pid != host_pid
    ):
        errors.append("Radial GPU binding PID evidence is invalid")
    qkv_bytes = binding.get("qkv_storage_bytes")
    allocator_bytes = binding.get("allocator_memory_bytes")
    reserved_bytes = binding.get("reserved_memory_bytes")
    if qkv_bytes != RADIAL_QKV_STORAGE_BYTES:
        errors.append("Radial GPU binding QKV storage size differs from fixed shape")
    if (
        not _positive_integer(allocator_bytes)
        or not _positive_integer(reserved_bytes)
        or not _positive_integer(qkv_bytes)
        or allocator_bytes < qkv_bytes
        or reserved_bytes < allocator_bytes
    ):
        errors.append("Radial GPU allocator/reserve evidence is invalid")
    if len(post_samples) == 2 and isinstance(post_samples[-1], dict):
        final_processes = post_samples[-1].get("processes")
        final_used_mib = (
            final_processes[0].get("used_memory_mib")
            if isinstance(final_processes, list)
            and len(final_processes) == 1
            and isinstance(final_processes[0], dict)
            else None
        )
        if (
            not _positive_integer(final_used_mib)
            or not _positive_integer(reserved_bytes)
            or final_used_mib * 1024 * 1024 + 1024 * 1024 < reserved_bytes
        ):
            errors.append("Radial nvidia-smi memory does not cover CUDA reserve")

    mps = binding.get("mps")
    if not isinstance(mps, dict):
        errors.append("Radial GPU binding MPS evidence is missing")
        mps = {}
    if mps.get("cuda_mps_environment_variables") != []:
        errors.append("Radial GPU binding has CUDA MPS environment variables")
    pmon_errors, pmon_rows = _pmon_evidence_errors(
        mps.get("pmon"),
        expected_host_pid=host_pid,
        expected_nvidia_smi_binary=nvidia_smi_binary,
        immediate_finished_monotonic=immediate_finished,
        cuda_touch_started_monotonic=touch_mono,
        context_live_finished_monotonic=context_finished,
        backend_started_wall=phase_times.get(
            ("exact_backend_started_at", "unix_seconds")
        ),
        backend_started_monotonic=backend_started,
        backend_returned_wall=phase_times.get(
            ("exact_backend_returned_at", "unix_seconds")
        ),
        backend_returned_monotonic=phase_times.get(
            ("exact_backend_returned_at", "monotonic_seconds")
        ),
        sync_finished_wall=phase_times.get(
            ("cuda_synchronized_at", "unix_seconds")
        ),
        sync_finished_monotonic=sync_finished,
    )
    errors.extend(pmon_errors)
    recomputed_mps = bool(
        any(
            row.get("host_pid") is not None
            and (
                row.get("process_type") != "C"
                or "mps" in str(row.get("command", "")).lower()
            )
            for row in pmon_rows
        )
        or "mps" in normalized_name
    )
    if recomputed_mps:
        errors.append("Radial GPU binding detected MPS")
    expected_mps_status = (
        "not_observed"
        if pmon_observation_mode == "direct_c_observed"
        else "unknown"
    )
    if (
        mps.get("mps_status") != expected_mps_status
        or mps.get("direct_compute_type_observed")
        is not (pmon_observation_mode == "direct_c_observed")
        or mps.get("host_pid_observed_by_pmon")
        is not (pmon_observation_mode == "direct_c_observed")
        or mps.get("continuous_exclusivity_proven") is not False
        or not isinstance(mps.get("pmon"), dict)
        or mps["pmon"].get("observation_mode") != pmon_observation_mode
    ):
        errors.append("Radial pmon/MPS observation scope is inconsistent")

    nspid = binding.get("nspid")
    proc_visibility = binding.get("host_pid_proc_visibility")
    chain = nspid.get("chain") if isinstance(nspid, dict) else None
    if (
        not isinstance(proc_visibility, dict)
        or proc_visibility.get("status") not in ("visible", "not_visible")
        or proc_visibility.get("error") is not None
    ):
        errors.append("Radial host PID proc visibility evidence is invalid")
    if (
        not isinstance(nspid, dict)
        or nspid.get("status") != "ok"
        or nspid.get("error") is not None
        or not isinstance(chain, list)
        or not chain
        or any(not _positive_integer(pid) for pid in chain)
        or chain[-1] != container_pid
    ):
        errors.append("Radial NSpid evidence is unavailable or malformed")
    if method == "direct_nspid":
        if not isinstance(chain, list) or len(chain) < 2 or chain[0] != host_pid:
            errors.append("Radial direct NSpid binding lacks the outer host PID")
    elif method == "sampled_temporal_association_after_idle_guard":
        if (
            chain != [container_pid]
            or host_pid in (chain or [])
            or proc_visibility != {"status": "not_visible", "error": None}
        ):
            errors.append(
                "Radial sampled temporal association lacks strict namespace "
                "negative evidence"
            )
    if binding.get("exact_kernel_completed") is not True:
        errors.append("Radial exact backend completion is not proven")
    if not _positive_integer(binding.get("exact_backend_call_count")):
        errors.append("Radial exact backend call count is invalid")
    if binding.get("cuda_synchronize_completed") is not True:
        errors.append("Radial final CUDA synchronization is not proven")
    return errors


def radial_microtest_evidence_errors(
    microtest,
    expected_gpu_uuid=None,
    expected_pre_run_gpu=None,
    expected_pre_run_gpu_sha256=None,
    expected_pre_run_gpu_path=None,
    expected_python_executable=None,
):
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
        "cuda_synchronized": True,
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
        "mask_audit": RADIAL_PROFILE_AUDITS["conservative"],
        "finite": True,
    }
    for field, expected_value in expected.items():
        if not _exact_json_value(microtest.get(field), expected_value):
            errors.append(
                f"Radial microtest {field}={microtest.get(field)!r} != "
                f"{expected_value!r}"
            )
    call_count = microtest.get("exact_backend_call_count")
    binding = microtest.get("gpu_process_binding")
    calls = microtest.get("calls")
    plan_cache_entries = microtest.get("plan_cache_entries")
    plan_cache_misses = microtest.get("plan_cache_misses")
    plan_cache_hits = microtest.get("plan_cache_hits")
    if (
        not _positive_integer(call_count)
        or not isinstance(binding, dict)
        or binding.get("exact_backend_call_count") != call_count
        or not _positive_integer(calls)
        or calls != call_count
        or not _positive_integer(plan_cache_entries)
        or plan_cache_entries != 1
        or not _positive_integer(plan_cache_misses)
        or plan_cache_misses != 1
        or not _json_integer(plan_cache_hits)
        or plan_cache_hits < 0
        or plan_cache_hits != call_count - 1
    ):
        errors.append(
            "Radial repeated backend calls/plan-cache metrics are not cross-bound"
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
        if not _positive_integer(before_cuda.get("aliases")):
            errors.append("Radial microtest runtime dependency alias count is invalid")
        if not _positive_integer(before_cuda.get("mapped_files")):
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
    pre_run_gpu = microtest.get("pre_run_gpu")
    if not isinstance(pre_run_gpu, dict):
        errors.append("Radial microtest lacks pre-run idle GPU evidence")
        pre_run_gpu = {}
    expected_pre_run_fields = {
        "schema_version": GPU_EVIDENCE_SCHEMA_VERSION,
        "check_type": "pre_run_idle",
        "physical_device_index": 0,
        "cuda_visible_devices": device_uuid,
        "available": True,
        "error": None,
        "device_index": 0,
        "device_uuid": device_uuid,
        "device_name": microtest.get("device"),
        "processes": [],
        "process_count": 0,
        "idle": True,
        "valid_for_run": True,
        "errors": [],
    }
    for field, expected_value in expected_pre_run_fields.items():
        if not _exact_json_value(pre_run_gpu.get(field), expected_value):
            errors.append(
                f"Radial microtest pre-run {field}="
                f"{pre_run_gpu.get(field)!r} != {expected_value!r}"
            )
    if (
        expected_pre_run_gpu is not None
        and pre_run_gpu != expected_pre_run_gpu
    ):
        errors.append(
            "Radial microtest pre-run GPU evidence differs from run evidence"
        )
    errors.extend(
        radial_gpu_process_binding_errors(
            binding,
            expected_pre_run_gpu=expected_pre_run_gpu,
            expected_pre_run_gpu_sha256=expected_pre_run_gpu_sha256,
            expected_pre_run_gpu_path=expected_pre_run_gpu_path,
            expected_python_executable=expected_python_executable,
        )
    )
    if isinstance(binding, dict):
        cross_bound_fields = {
            "host_pid": binding.get("nvidia_smi_host_pid"),
            "python_pid": binding.get("container_pid"),
            "pid_namespace_chain": (
                binding.get("nspid", {}).get("chain")
                if isinstance(binding.get("nspid"), dict)
                else None
            ),
            "pid_binding_method": binding.get("binding_method"),
            "pmon_observation_mode": binding.get("pmon_observation_mode"),
            "gpu_process_claim_scope": binding.get("claim_scope"),
            "host_pid_ownership": binding.get("host_pid_ownership"),
            "mps_status": (
                binding.get("mps", {}).get("mps_status")
                if isinstance(binding.get("mps"), dict)
                else None
            ),
            "pre_run_gpu": binding.get("pre_run_gpu"),
            "pre_run_gpu_sha256": binding.get("pre_run_gpu_sha256"),
            "post_cuda_sampled_at_unix_seconds": (
                binding.get("post_cuda_samples", [{}])[-1].get(
                    "sampled_at_unix_seconds"
                )
                if isinstance(binding.get("post_cuda_samples"), list)
                and binding.get("post_cuda_samples")
                and isinstance(binding.get("post_cuda_samples")[-1], dict)
                else None
            ),
            "gpu_process_count": (
                binding.get("post_cuda_samples", [{}])[-1].get(
                    "process_count"
                )
                if isinstance(binding.get("post_cuda_samples"), list)
                and binding.get("post_cuda_samples")
                and isinstance(binding.get("post_cuda_samples")[-1], dict)
                else None
            ),
            "gpu_processes": (
                binding.get("post_cuda_samples", [{}])[-1].get("processes")
                if isinstance(binding.get("post_cuda_samples"), list)
                and binding.get("post_cuda_samples")
                and isinstance(binding.get("post_cuda_samples")[-1], dict)
                else None
            ),
        }
        for field, expected_value in cross_bound_fields.items():
            if not _exact_json_value(
                microtest.get(field), expected_value
            ):
                errors.append(
                    f"Radial microtest {field} differs from GPU process binding"
                )
        expected_visible = binding.get("binding_method") == "direct_nspid"
        if microtest.get("host_pid_namespace_visible") is not expected_visible:
            errors.append(
                "Radial microtest host PID visibility differs from binding method"
            )
    pre_run_sample = pre_run_gpu.get("sampled_at_unix_seconds")
    post_cuda_sample = microtest.get("post_cuda_sampled_at_unix_seconds")
    if (
        not isinstance(pre_run_sample, (int, float))
        or isinstance(pre_run_sample, bool)
        or not math.isfinite(float(pre_run_sample))
        or not isinstance(post_cuda_sample, (int, float))
        or isinstance(post_cuda_sample, bool)
        or not math.isfinite(float(post_cuda_sample))
        or float(post_cuda_sample) <= float(pre_run_sample)
    ):
        errors.append(
            "Radial microtest post-CUDA GPU sample does not follow idle evidence"
        )
    for field, require_positive in (
        ("output_abs_mean", True),
        ("output_abs_max", True),
    ):
        raw_value = microtest.get(field)
        if not _finite_number(raw_value):
            errors.append(f"Radial microtest {field} is not numeric")
            continue
        value = float(raw_value)
        if not math.isfinite(value) or (require_positive and value <= 0.0):
            errors.append(
                f"Radial microtest {field} must be finite and positive"
            )
    return errors
