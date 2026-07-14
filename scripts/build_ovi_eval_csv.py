#!/usr/bin/env python3
"""Build the fixed Ovi evaluation CSV from explicitly selected run folders.

This script intentionally has no "latest run" discovery.  A run can enter the
table only when the caller binds a matrix method id to one exact run directory
and the persisted verifier, environment, timings, GPU monitor, checkpoint, and
artifact hashes all agree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import stat
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ovi.gpu_process_monitor import (
    GPU_EVIDENCE_SCHEMA_VERSION,
    GPU_PROCESS_MONITOR_SCHEMA_VERSION,
    GPU_QUERY_CADENCE_TOLERANCE_SECONDS,
    gpu_compute_snapshot_maximum_gap_seconds,
    gpu_compute_snapshot_observation_span_seconds,
    gpu_compute_snapshot_sequence_errors,
    gpu_compute_snapshot_errors,
    trusted_nvidia_smi_metadata_errors,
    validate_pre_run_gpu_report,
)
from ovi.eval_protocol import prompt_sequence_sha256
from ovi.sparge_evidence import (
    SPARGEATTN_API,
    SPARGEATTN_CLONE_URL,
    SPARGEATTN_COMMIT,
    SPARGEATTN_REPOSITORY,
    sparge_microtest_evidence_errors,
    sparge_receipt_evidence_errors,
)
from ovi.radial_evidence import (
    FLASHINFER_VERSION,
    RADIAL_BLOCK_SIZE,
    RADIAL_COMMIT,
    RADIAL_EMPTY_ROWS,
    RADIAL_GRID,
    RADIAL_MASK_API,
    RADIAL_MODEL_TYPE,
    RADIAL_PREFIX_SEQUENCE,
    RADIAL_PROFILE_AUDITS,
    RADIAL_REPOSITORY,
    RADIAL_SEQUENCE,
    RADIAL_TAIL_SEQUENCE,
    flashinfer_manifest_evidence_errors,
    radial_microtest_evidence_errors,
    radial_receipt_evidence_errors,
)


DEFAULT_MANIFEST = REPO_ROOT / "configs" / "ovi_eval_matrix.json"
REQUIRED_METHOD_IDS = (
    "dense",
    "dense_cfg_cache",
    "sparge_topk50",
    "sparge_topk75",
    "radial_conservative",
    "radial_aggressive",
    "best_sparse_cfg",
)
OPTIONAL_METHOD_IDS = ("block_cache",)
BEST_SPARSE_CFG_RUN_KINDS = (
    "sparge_topk50_cfg_benchmark",
    "sparge_topk75_cfg_benchmark",
    "radial_conservative_cfg_benchmark",
    "radial_aggressive_cfg_benchmark",
)
BLOCK_CACHE_RUN_KINDS = (
    "sparge_topk50_block_cache_benchmark",
    "sparge_topk75_block_cache_benchmark",
    "radial_conservative_block_cache_benchmark",
    "radial_aggressive_block_cache_benchmark",
)
SPARSE_COMBO_RUN_KIND_CONTRACTS = {
    "sparge_topk50_cfg_benchmark": {
        "attention_method": "sparge",
        "sparge_topk": 0.5,
        "sparge_pvthreshd": 50.0,
        "sparge_smooth_k": True,
        "use_cfg_cache": True,
        "use_block_cache": False,
        "block_cache_policy": "fixed",
    },
    "sparge_topk75_cfg_benchmark": {
        "attention_method": "sparge",
        "sparge_topk": 0.75,
        "sparge_pvthreshd": 50.0,
        "sparge_smooth_k": True,
        "use_cfg_cache": True,
        "use_block_cache": False,
        "block_cache_policy": "fixed",
    },
    "radial_conservative_cfg_benchmark": {
        "attention_method": "radial",
        "radial_profile": "conservative",
        "radial_decay_factor": 4.0,
        "radial_model_type": "wan",
        "radial_block_size": 128,
        "use_cfg_cache": True,
        "use_block_cache": False,
        "block_cache_policy": "fixed",
    },
    "radial_aggressive_cfg_benchmark": {
        "attention_method": "radial",
        "radial_profile": "aggressive",
        "radial_decay_factor": 1.0,
        "radial_model_type": "wan",
        "radial_block_size": 128,
        "use_cfg_cache": True,
        "use_block_cache": False,
        "block_cache_policy": "fixed",
    },
    "sparge_topk50_block_cache_benchmark": {
        "attention_method": "sparge",
        "sparge_topk": 0.5,
        "sparge_pvthreshd": 50.0,
        "sparge_smooth_k": True,
        "use_cfg_cache": False,
        "use_block_cache": True,
        "block_cache_policy": "fixed",
    },
    "sparge_topk75_block_cache_benchmark": {
        "attention_method": "sparge",
        "sparge_topk": 0.75,
        "sparge_pvthreshd": 50.0,
        "sparge_smooth_k": True,
        "use_cfg_cache": False,
        "use_block_cache": True,
        "block_cache_policy": "fixed",
    },
    "radial_conservative_block_cache_benchmark": {
        "attention_method": "radial",
        "radial_profile": "conservative",
        "radial_decay_factor": 4.0,
        "radial_model_type": "wan",
        "radial_block_size": 128,
        "use_cfg_cache": False,
        "use_block_cache": True,
        "block_cache_policy": "fixed",
    },
    "radial_aggressive_block_cache_benchmark": {
        "attention_method": "radial",
        "radial_profile": "aggressive",
        "radial_decay_factor": 1.0,
        "radial_model_type": "wan",
        "radial_block_size": 128,
        "use_cfg_cache": False,
        "use_block_cache": True,
        "block_cache_policy": "fixed",
    },
}
SPARSE_PROFILE_BY_RUN_KIND = {
    run_kind: (
        "sparge_topk50"
        if run_kind.startswith("sparge_topk50_")
        else "sparge_topk75"
        if run_kind.startswith("sparge_topk75_")
        else "radial_conservative"
        if run_kind.startswith("radial_conservative_")
        else "radial_aggressive"
    )
    for run_kind in SPARSE_COMBO_RUN_KIND_CONTRACTS
}
COMBO_METHOD_RUN_KINDS = {
    "best_sparse_cfg": BEST_SPARSE_CFG_RUN_KINDS,
    "block_cache": BLOCK_CACHE_RUN_KINDS,
}
MEASUREMENT_COUNT = 3
GIB = 1024 ** 3
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
HEX_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
RADIAL_OPTIONAL_IMPORT_LIB64 = (
    "/cache/liluchen/FastA2V/envs/ovi/lib/python3.11/lib64"
)
RADIAL_INSTALL_RECEIPT_PATH = (
    "/cache/liluchen/FastA2V/radialattn-install.json"
)
REQUIRED_PREFLIGHT_PACKAGES = (
    "torch",
    "torchvision",
    "torchaudio",
    "flash-attn",
    "transformers",
    "diffusers",
    "omegaconf",
)
REQUIRED_PREFLIGHT_CHECKPOINTS = (
    "Ovi/model.safetensors",
    "Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
    "MMAudio/ext_weights/best_netG.pt",
    "MMAudio/ext_weights/v1-16.pth",
    "Wan2.2-TI2V-5B/google/umt5-xxl/spiece.model",
)
SPARGE_PROVENANCE = {
    "backend": "official_spargeattn",
    "repository": SPARGEATTN_REPOSITORY,
    "clone_url": SPARGEATTN_CLONE_URL,
    "pinned_commit": SPARGEATTN_COMMIT,
    "api": SPARGEATTN_API,
    "tensor_layout": "NHD",
    "return_sparsity": False,
}

CSV_FIELDS = (
    "method_id",
    "label",
    "required",
    "implementation_status",
    "status",
    "timing_status",
    "pending_reason",
    "run_dir",
    "run_id",
    "verification_sha256",
    "preflight_sha256",
    "timings_path",
    "timings_bytes",
    "timings_sha256",
    "timings_record_count",
    "warmup_timings_path",
    "warmup_timings_bytes",
    "warmup_timings_sha256",
    "warmup_record_count",
    "git_commit",
    "checkpoint_manifest_sha256",
    "checkpoint_fingerprint_sha256",
    "gpu_uuid",
    "gpu_name",
    "radial_evidence_mode",
    "radial_pmon_status",
    "radial_pid_association",
    "radial_claim_scope",
    "radial_host_pid_ownership",
    "radial_mps_status",
    "prompt_sha256",
    "prompt",
    "prompt_set_sha256",
    "prompt_count",
    "selected_sparse_profile",
    "seed",
    "seed_count",
    "seeds",
    "requested_height",
    "requested_width",
    "actual_height",
    "actual_width",
    "sample_steps",
    "measurement_count",
    "measurement_indices",
    "artifact_count",
    "denoise_seconds_median",
    "total_generation_seconds_median",
    "artifact_ready_seconds_median",
    "peak_memory_allocated_gib_median",
    "peak_memory_reserved_gib_median",
    "denoise_speedup_vs_dense",
    "total_speedup_vs_dense",
    "artifact_sha256",
    "metrics_sidecar_sha256",
    "quality_metric_name",
    "quality_score",
    "manual_review",
)


class EvaluationError(ValueError):
    """Raised when evidence cannot safely enter the comparison table."""


class _StableFileSnapshot:
    """One no-follow regular-file read, plus identity for final revalidation."""

    __slots__ = (
        "path",
        "data",
        "sha256",
        "device",
        "inode",
        "size",
        "mtime_ns",
        "ctime_ns",
    )

    def __init__(self, path: Path, data: bytes, metadata: os.stat_result):
        self.path = Path(path)
        self.data = data
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.device = metadata.st_dev
        self.inode = metadata.st_ino
        self.size = metadata.st_size
        self.mtime_ns = metadata.st_mtime_ns
        self.ctime_ns = metadata.st_ctime_ns


def _is_json_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _strict_json_equal(actual: Any, expected: Any) -> bool:
    """Compare persisted JSON without Python bool/int equivalence."""

    if expected is None or isinstance(expected, (bool, str)):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, int):
        return _is_json_int(actual) and actual == expected
    if isinstance(expected, float):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and float(actual) == expected
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _strict_json_equal(actual_value, expected_value)
                for actual_value, expected_value in zip(actual, expected)
            )
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(
                _strict_json_equal(actual[key], expected_value)
                for key, expected_value in expected.items()
            )
        )
    return type(actual) is type(expected) and actual == expected


def _fail(context: str, message: str) -> None:
    raise EvaluationError(f"{context}: {message}")


def _require(condition: bool, context: str, message: str) -> None:
    if not condition:
        _fail(context, message)


def _snapshot_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _stable_file_snapshot(path: Path, context: str) -> _StableFileSnapshot:
    """Read one immutable-by-evidence byte snapshot without following symlinks.

    The descriptor and directory entry are checked before and after the read.
    ``validate_run`` checks the same identity again before returning, closing
    the old parse-then-reopen-for-hash gap and detecting replacement while a
    table row is being assembled.
    """

    path = Path(path)
    try:
        initial_entry = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        _fail(context, f"cannot stat evidence file {path}: {exc}")
    if stat.S_ISLNK(initial_entry.st_mode):
        _fail(context, f"evidence file must not be a symlink: {path}")
    if not stat.S_ISREG(initial_entry.st_mode):
        _fail(context, f"evidence file must be a regular file: {path}")

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        _fail(context, f"cannot open no-follow evidence file {path}: {exc}")
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _fail(context, f"opened evidence is not a regular file: {path}")
        if _snapshot_identity(before) != _snapshot_identity(initial_entry):
            _fail(context, f"evidence file changed before read: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    if _snapshot_identity(after) != _snapshot_identity(before):
        _fail(context, f"evidence file changed while being read: {path}")
    if len(data) != after.st_size:
        _fail(context, f"evidence byte count changed while being read: {path}")
    try:
        final_entry = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        _fail(context, f"cannot re-stat evidence file {path}: {exc}")
    if (
        not stat.S_ISREG(final_entry.st_mode)
        or _snapshot_identity(final_entry) != _snapshot_identity(after)
    ):
        _fail(context, f"evidence file was replaced while being read: {path}")
    return _StableFileSnapshot(path, data, after)


def _revalidate_snapshot(snapshot: _StableFileSnapshot, context: str) -> None:
    try:
        current = os.stat(snapshot.path, follow_symlinks=False)
    except OSError as exc:
        _fail(context, f"cannot revalidate evidence file {snapshot.path}: {exc}")
    expected = (
        snapshot.device,
        snapshot.inode,
        snapshot.size,
        snapshot.mtime_ns,
        snapshot.ctime_ns,
    )
    if not stat.S_ISREG(current.st_mode) or _snapshot_identity(current) != expected:
        _fail(
            context,
            f"evidence file changed after its stable byte snapshot: {snapshot.path}",
        )


def _snapshot_json(
    path: Path,
    context: str,
) -> tuple[_StableFileSnapshot, Any]:
    snapshot = _stable_file_snapshot(path, context)
    try:
        payload = json.loads(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(context, f"cannot decode valid JSON from {path}: {exc}")
    return snapshot, payload


def _snapshot_jsonl(
    path: Path,
    context: str,
) -> tuple[_StableFileSnapshot, list[dict[str, Any]]]:
    snapshot = _stable_file_snapshot(path, context)
    try:
        lines = snapshot.data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        _fail(context, f"cannot decode UTF-8 JSONL from {path}: {exc}")
    records = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            _fail(context, f"blank JSONL record at {path}:{line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(context, f"invalid JSON at {path}:{line_number}: {exc}")
        if not isinstance(record, dict):
            _fail(context, f"record at {path}:{line_number} is not an object")
        records.append(record)
    return snapshot, records


def _validate_jsonl_binding(
    binding: Any,
    *,
    path: Path,
    snapshot: _StableFileSnapshot,
    record_count: int,
    context: str,
    label: str,
) -> None:
    """Cross-bind a verifier receipt to one stable canonical JSONL snapshot."""

    _require(
        isinstance(binding, dict)
        and set(binding) == {"path", "bytes", "sha256", "record_count"},
        context,
        f"verification protocol {label} has invalid fields",
    )
    _require(
        binding.get("path") == str(path),
        context,
        f"{label} path is not the canonical selected-run path",
    )
    _require(
        _is_json_int(binding.get("bytes"))
        and binding.get("bytes") == snapshot.size,
        context,
        f"{label} byte count differs from the stable snapshot",
    )
    _require(
        isinstance(binding.get("sha256"), str)
        and HEX_SHA256.fullmatch(binding["sha256"]) is not None
        and binding.get("sha256") == snapshot.sha256,
        context,
        f"{label} SHA256 differs from the stable snapshot",
    )
    _require(
        _is_json_int(binding.get("record_count"))
        and binding.get("record_count") == record_count,
        context,
        f"{label} record count differs from the stable snapshot",
    )


def _validate_file_binding(
    binding: Any,
    *,
    path: Path,
    snapshot: _StableFileSnapshot,
    context: str,
    label: str,
) -> None:
    """Cross-bind a verifier artifact receipt to one stable file snapshot."""

    _require(
        isinstance(binding, dict)
        and set(binding) == {"path", "bytes", "sha256"},
        context,
        f"verification artifact {label} has invalid fields",
    )
    _require(
        binding.get("path") == str(path),
        context,
        f"{label} path is not the canonical selected-run path",
    )
    _require(
        _is_json_int(binding.get("bytes"))
        and binding.get("bytes") == snapshot.size,
        context,
        f"{label} byte count differs from the stable snapshot",
    )
    _require(
        isinstance(binding.get("sha256"), str)
        and HEX_SHA256.fullmatch(binding["sha256"]) is not None
        and binding.get("sha256") == snapshot.sha256,
        context,
        f"{label} SHA256 differs from the stable snapshot",
    )


def _read_json(path: Path, context: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        _fail(context, f"cannot read valid JSON from {path}: {exc}")


def _read_jsonl(path: Path, context: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _fail(context, f"cannot read {path}: {exc}")
    records = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            _fail(context, f"blank JSONL record at {path}:{line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(context, f"invalid JSON at {path}:{line_number}: {exc}")
        if not isinstance(record, dict):
            _fail(context, f"record at {path}:{line_number} is not an object")
        records.append(record)
    return records


def _finite_number(
    payload: dict[str, Any],
    field: str,
    context: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(context, f"{field} must be a finite number, found {value!r}")
    value = float(value)
    if not math.isfinite(value):
        _fail(context, f"{field} must be finite, found {value!r}")
    if positive and value <= 0:
        _fail(context, f"{field} must be positive, found {value!r}")
    if nonnegative and value < 0:
        _fail(context, f"{field} must be nonnegative, found {value!r}")
    return value


def _values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(float(actual))
            and float(actual) == float(expected)
        )
    return actual == expected


def _validate_expected_fields(
    payload: dict[str, Any], expected: dict[str, Any], context: str
) -> None:
    for field, expected_value in expected.items():
        actual_value = payload.get(field)
        if not _values_equal(actual_value, expected_value):
            _fail(
                context,
                f"{field}={actual_value!r} does not match fixed value "
                f"{expected_value!r}",
            )


def _validate_combo_run_environment(
    method: Mapping[str, Any],
    environment: Mapping[str, Any],
    context: str,
) -> None:
    """Bind a selection slot to one of its four immutable sparse profiles."""

    method_id = method.get("method_id")
    allowed_run_kinds = COMBO_METHOD_RUN_KINDS.get(method_id)
    if allowed_run_kinds is None:
        return
    run_kind = environment.get("run_kind")
    _require(
        isinstance(run_kind, str) and run_kind in allowed_run_kinds,
        context,
        f"run_kind={run_kind!r} is not allowed for selected slot {method_id}",
    )
    contract = SPARSE_COMBO_RUN_KIND_CONTRACTS[run_kind]
    for field, expected in contract.items():
        _require(
            _values_equal(environment.get(field), expected),
            context,
            f"{run_kind} requires {field}={expected!r}, found "
            f"{environment.get(field)!r}",
        )


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    path = Path(path)
    manifest = _read_json(path, "evaluation manifest")
    _require(isinstance(manifest, dict), "evaluation manifest", "root must be an object")
    _require(
        _is_json_int(manifest.get("schema_version"))
        and manifest.get("schema_version") == 1,
        "evaluation manifest",
        "unsupported schema_version",
    )
    methods = manifest.get("methods")
    _require(isinstance(methods, list), "evaluation manifest", "methods must be a list")

    ids = []
    for index, method in enumerate(methods):
        context = f"evaluation manifest method[{index}]"
        _require(isinstance(method, dict), context, "method must be an object")
        method_id = method.get("method_id")
        _require(isinstance(method_id, str) and method_id, context, "method_id is required")
        _require(method_id not in ids, context, f"duplicate method_id {method_id!r}")
        ids.append(method_id)
        _require(
            isinstance(method.get("label"), str) and method.get("label"),
            context,
            "label is required",
        )
        _require(type(method.get("required")) is bool, context, "required must be boolean")
        _require(
            method.get("implementation_status") in {"ready", "pending"},
            context,
            "implementation_status must be ready or pending",
        )
        _require(
            isinstance(method.get("expected_environment"), dict),
            context,
            "expected_environment must be an object",
        )
        combo_run_kinds = COMBO_METHOD_RUN_KINDS.get(method_id)
        if combo_run_kinds is not None:
            _require(
                method.get("implementation_status") == "ready",
                context,
                "sparse combination slot implementation must be ready",
            )
            _require(
                method.get("selection_required") is True,
                context,
                "sparse combination slot must require an explicit selection",
            )
            _require(
                isinstance(method.get("allowed_run_kinds"), list)
                and tuple(method["allowed_run_kinds"]) == combo_run_kinds,
                context,
                "allowed_run_kinds differ from the four fixed sparse profiles",
            )
            expected_environment = (
                {
                    "use_block_cache": False,
                    "use_cfg_cache": True,
                }
                if method_id == "best_sparse_cfg"
                else {
                    "block_cache_policy": "fixed",
                    "use_block_cache": True,
                    "use_cfg_cache": False,
                }
            )
            _require(
                method.get("expected_environment") == expected_environment,
                context,
                "selection-slot expected_environment differs from the fixed "
                "cache contract",
            )
        else:
            _require(
                "allowed_run_kinds" not in method
                and "selection_required" not in method,
                context,
                "only sparse combination slots may declare selection metadata",
            )

    _require(
        tuple(ids) == REQUIRED_METHOD_IDS + OPTIONAL_METHOD_IDS,
        "evaluation manifest",
        "method slots or order differ from the fixed seven required plus block optional matrix",
    )
    required_ids = tuple(method["method_id"] for method in methods if method["required"])
    optional_ids = tuple(method["method_id"] for method in methods if not method["required"])
    _require(
        required_ids == REQUIRED_METHOD_IDS,
        "evaluation manifest",
        "required slots differ from the fixed matrix",
    )
    _require(
        optional_ids == OPTIONAL_METHOD_IDS,
        "evaluation manifest",
        "block_cache must be the only optional slot",
    )
    contract = manifest.get("comparison_contract")
    _require(isinstance(contract, dict), "evaluation manifest", "comparison_contract is required")
    _require(
        contract.get("allow_latest_run_discovery") is False,
        "evaluation manifest",
        "latest-run discovery must stay disabled",
    )
    _require(
        contract.get("explicit_method_run_mapping_required") is True,
        "evaluation manifest",
        "explicit method-to-run mapping must be required",
    )
    _require(
        contract.get("measurement_count") == MEASUREMENT_COUNT,
        "evaluation manifest",
        f"measurement_count must be {MEASUREMENT_COUNT}",
    )
    fixed_protocol = manifest.get("fixed_protocol")
    _require(isinstance(fixed_protocol, dict), "evaluation manifest", "fixed_protocol is required")
    return manifest


def parse_run_mappings(
    values: Iterable[str], allowed_method_ids: Iterable[str]
) -> dict[str, Path]:
    allowed = set(allowed_method_ids)
    mappings: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise EvaluationError(
                f"run mapping {value!r} must use METHOD_ID=RUN_DIR; "
                "automatic latest-run discovery is intentionally unsupported"
            )
        method_id, raw_path = value.split("=", 1)
        if not method_id or not raw_path:
            raise EvaluationError(f"invalid run mapping {value!r}; use METHOD_ID=RUN_DIR")
        if method_id not in allowed:
            raise EvaluationError(f"unknown evaluation method_id {method_id!r}")
        if method_id in mappings:
            raise EvaluationError(f"duplicate run mapping for method_id {method_id!r}")
        mappings[method_id] = Path(raw_path).expanduser()
    return mappings


def _checkpoint_fingerprint(
    manifest: dict[str, Any], context: str
) -> str:
    files = manifest.get("files")
    _require(isinstance(files, dict) and files, context, "checkpoint files are missing")
    canonical = {}
    for relative_path, metadata in sorted(files.items()):
        _require(
            isinstance(relative_path, str) and relative_path,
            context,
            "checkpoint relative path is invalid",
        )
        _require(isinstance(metadata, dict), context, f"metadata for {relative_path} is invalid")
        sha = metadata.get("sha256")
        size = metadata.get("bytes")
        _require(
            isinstance(sha, str) and HEX_SHA256.fullmatch(sha) is not None,
            context,
            f"checkpoint SHA256 is invalid for {relative_path}",
        )
        _require(
            isinstance(size, int) and not isinstance(size, bool) and size > 0,
            context,
            f"checkpoint byte count is invalid for {relative_path}",
        )
        canonical[relative_path] = {"bytes": size, "sha256": sha}
    serialized = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _validate_gpu_monitor(
    monitor: Any,
    environment: dict[str, Any],
    expected_nvidia_smi_binary: dict[str, Any],
    expected_boot_id: str,
    expected_interval_seconds: float,
    minimum_coverage_seconds: float,
    context: str,
) -> None:
    _require(isinstance(monitor, dict), context, "gpu_process_monitor must be an object")
    _require(
        _is_json_int(monitor.get("schema_version"))
        and monitor.get("schema_version")
        == GPU_PROCESS_MONITOR_SCHEMA_VERSION,
        context,
        "unsupported GPU process monitor evidence schema",
    )
    required_true = (
        "identity_consistent",
        "nvidia_smi_binary_fixed_valid",
        "nvidia_smi_binary_consistent",
        "single_distinct_host_pid",
        "exact_singleton_process_per_sample",
        "valid_for_benchmark",
    )
    for field in required_true:
        _require(monitor.get(field) is True, context, f"GPU monitor {field} must be true")
    _require(
        monitor.get("sample_validation_errors") == [],
        context,
        "GPU monitor samples contain validation errors",
    )
    _require(
        monitor.get("snapshot_validation_errors") == [],
        context,
        "GPU monitor raw snapshots contain validation errors",
    )
    interval_seconds = monitor.get("interval_seconds")
    _require(
        not isinstance(interval_seconds, bool)
        and isinstance(interval_seconds, (int, float))
        and math.isfinite(float(interval_seconds))
        and float(interval_seconds) > 0.0
        and interval_seconds == expected_interval_seconds,
        context,
        "GPU monitor interval differs from environment",
    )
    _require(
        monitor.get("boot_id_consistent") is True
        and monitor.get("boot_id") == expected_boot_id,
        context,
        "GPU monitor boot ID differs from pre-run evidence",
    )
    _require(
        monitor.get("contention_detected") is False,
        context,
        "GPU contention was detected",
    )
    _require(
        monitor.get("no_process_detected") is False,
        context,
        "GPU monitor lost the benchmark process",
    )
    _require(
        _is_json_int(monitor.get("unavailable_sample_count"))
        and monitor.get("unavailable_sample_count") == 0,
        context,
        "GPU samples were unavailable",
    )
    _require(
        monitor.get("nvidia_smi_binary") == expected_nvidia_smi_binary,
        context,
        "GPU monitor nvidia-smi binary differs from pre-run evidence",
    )
    _require(
        monitor.get("nvidia_smi_binary_validation_errors") == [],
        context,
        "GPU monitor reports nvidia-smi binary validation errors",
    )
    sample_count = monitor.get("sample_count")
    _require(
        _is_json_int(sample_count) and sample_count >= 2,
        context,
        "GPU monitor requires at least entry and exit samples",
    )
    _require(
        _is_json_int(monitor.get("available_sample_count"))
        and monitor.get("available_sample_count") == sample_count,
        context,
        "not all GPU samples were available",
    )
    _require(
        _is_json_int(monitor.get("min_process_count"))
        and monitor.get("min_process_count") == 1,
        context,
        "GPU sample process count was not one",
    )
    _require(
        _is_json_int(monitor.get("max_process_count"))
        and monitor.get("max_process_count") == 1,
        context,
        "GPU sample process count was not one",
    )
    _require(
        _is_json_int(monitor.get("device_index"))
        and monitor.get("device_index") == 0,
        context,
        "GPU monitor physical device index must be integer zero",
    )
    _require(
        monitor.get("device_uuid") == environment.get("gpu_uuid"),
        context,
        "GPU monitor UUID differs from environment",
    )
    _require(
        monitor.get("device_name") == environment.get("gpu_name"),
        context,
        "GPU monitor name differs from environment",
    )
    distinct_pids = monitor.get("distinct_host_pids")
    _require(
        isinstance(distinct_pids, list)
        and len(distinct_pids) == 1
        and _is_json_int(distinct_pids[0])
        and distinct_pids[0] > 0,
        context,
        "GPU monitor must record exactly one positive host PID",
    )
    samples = monitor.get("samples")
    _require(
        isinstance(samples, list) and len(samples) == sample_count,
        context,
        "raw GPU samples are incomplete",
    )
    maximum_gap_limit = (
        float(expected_interval_seconds)
        + GPU_QUERY_CADENCE_TOLERANCE_SECONDS
        if not isinstance(expected_interval_seconds, bool)
        and isinstance(expected_interval_seconds, (int, float))
        and math.isfinite(float(expected_interval_seconds))
        else None
    )
    sequence_errors = gpu_compute_snapshot_sequence_errors(
        samples,
        maximum_gap_limit,
    )
    _require(
        not sequence_errors
        and monitor.get("sample_sequence_validation_errors") == [],
        context,
        "GPU snapshot sequence is duplicated, overlapping, or reversed",
    )
    observation_span_seconds = (
        gpu_compute_snapshot_observation_span_seconds(samples)
    )
    maximum_sample_gap_seconds = (
        gpu_compute_snapshot_maximum_gap_seconds(samples)
    )
    _require(
        monitor.get("cadence_tolerance_seconds")
        == GPU_QUERY_CADENCE_TOLERANCE_SECONDS
        and monitor.get("maximum_sample_gap_seconds")
        == maximum_sample_gap_seconds,
        context,
        "GPU monitor cadence summary differs from raw samples",
    )
    _require(
        monitor.get("observation_span_seconds") == observation_span_seconds
        and not isinstance(minimum_coverage_seconds, bool)
        and isinstance(minimum_coverage_seconds, (int, float))
        and math.isfinite(float(minimum_coverage_seconds))
        and float(minimum_coverage_seconds) > 0.0
        and observation_span_seconds is not None
        and observation_span_seconds >= float(minimum_coverage_seconds),
        context,
        "GPU observation span does not cover total generation time",
    )
    for sample_index, sample in enumerate(samples):
        sample_context = f"{context} sample[{sample_index}]"
        _require(isinstance(sample, dict), sample_context, "sample must be an object")
        _require(sample.get("available") is True, sample_context, "sample is unavailable")
        _require(
            _is_json_int(sample.get("process_count"))
            and sample.get("process_count") == 1,
            sample_context,
            "sample must contain one process",
        )
        _require(
            _is_json_int(sample.get("device_index"))
            and sample.get("device_index") == 0,
            sample_context,
            "physical GPU index must be zero",
        )
        _require(sample.get("device_uuid") == environment.get("gpu_uuid"), sample_context, "GPU UUID differs")
        _require(sample.get("device_name") == environment.get("gpu_name"), sample_context, "GPU name differs")
        _require(
            sample.get("boot_id") == expected_boot_id,
            sample_context,
            "boot ID differs from pre-run evidence",
        )
        sample_snapshot_errors = gpu_compute_snapshot_errors(sample)
        _require(
            not sample_snapshot_errors,
            sample_context,
            "raw GPU snapshot is invalid: "
            + "; ".join(sample_snapshot_errors),
        )
        sample_binary_errors = trusted_nvidia_smi_metadata_errors(
            sample.get("nvidia_smi_binary")
        )
        _require(
            not sample_binary_errors,
            sample_context,
            "nvidia-smi binary fixed metadata is invalid: "
            + "; ".join(sample_binary_errors),
        )
        _require(
            sample.get("nvidia_smi_binary") == expected_nvidia_smi_binary,
            sample_context,
            "nvidia-smi binary metadata differs from pre-run evidence",
        )
        processes = sample.get("processes")
        _require(
            isinstance(processes, list)
            and len(processes) == 1
            and isinstance(processes[0], dict)
            and _is_json_int(processes[0].get("host_pid"))
            and processes[0].get("host_pid") == distinct_pids[0]
            and _is_json_int(processes[0].get("used_memory_mib"))
            and processes[0].get("used_memory_mib") > 0,
            sample_context,
            "sample process evidence differs from the stable benchmark PID "
            "or has invalid used memory",
        )


def _shape(value: Any, context: str, field: str, *, length: int | None = None) -> tuple[int, ...]:
    _require(isinstance(value, list), context, f"{field} must be a list")
    _require(
        all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value),
        context,
        f"{field} must contain positive integers",
    )
    if length is not None:
        _require(len(value) == length, context, f"{field} must contain {length} values")
    return tuple(value)


def _expected_radial_runtime_dependencies(receipt: Any) -> dict[str, Any] | None:
    if not isinstance(receipt, dict):
        return None
    inventory = receipt.get("runtime_loaded_dependencies")
    if not isinstance(inventory, dict) or not inventory:
        return None
    mapped_paths = set()
    for fingerprints in inventory.values():
        if not isinstance(fingerprints, list) or not fingerprints:
            return None
        for metadata in fingerprints:
            if not isinstance(metadata, dict):
                return None
            path = metadata.get("path")
            if not isinstance(path, str) or not path:
                return None
            mapped_paths.add(path)
    canonical = json.dumps(
        inventory,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "status": "ok",
        "aliases": len(inventory),
        "mapped_files": len(mapped_paths),
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _radial_runtime_dependency_errors(
    evidence: Any,
    expected: Any,
    context: str,
) -> list[str]:
    errors = []
    expected_fields = {
        "status",
        "aliases",
        "mapped_files",
        "inventory_sha256",
    }
    if not isinstance(evidence, dict):
        return [f"{context}: runtime dependency evidence is missing"]
    if set(evidence) != expected_fields:
        errors.append(f"{context}: runtime dependency fields are invalid")
    if expected is None:
        errors.append(f"{context}: copied receipt has no runtime inventory")
    elif not _strict_json_equal(evidence, expected):
        errors.append(
            f"{context}: runtime dependency evidence differs from copied receipt"
        )
    return errors


def _radial_optional_import_loader_errors(
    evidence: Any,
    expected_runtime: Any,
    context: str,
) -> list[str]:
    errors = []
    if not isinstance(evidence, dict):
        return [f"{context}: optional-import loader evidence is missing"]
    if set(evidence) != {
        "status",
        "restored",
        "removed_prepend_paths",
        "runtime_dependencies",
    }:
        errors.append(f"{context}: optional-import loader fields are invalid")
    if evidence.get("status") != "ok" or evidence.get("restored") is not True:
        errors.append(f"{context}: audited loader environment was not restored")
    errors.extend(
        _radial_runtime_dependency_errors(
            evidence.get("runtime_dependencies"),
            expected_runtime,
            f"{context}.runtime_dependencies",
        )
    )
    removed = evidence.get("removed_prepend_paths")
    if not isinstance(removed, list) or len(removed) != 1:
        errors.append(
            f"{context}: expected exactly one OpenCV loader prepend path"
        )
    elif (
        not isinstance(removed[0], str)
        or not removed[0]
        or str(Path(removed[0]).resolve())
        != str(Path(RADIAL_OPTIONAL_IMPORT_LIB64).resolve())
    ):
        errors.append(
            f"{context}: removed loader prepend path is not the fixed env lib64"
        )
    return errors


def _radial_loader_bootstrap_errors(
    evidence: Any,
    expected_runtime: Any,
) -> list[str]:
    context = "environment Radial loader bootstrap"
    if not isinstance(evidence, dict):
        return [f"{context}: evidence is missing"]
    errors = []
    if set(evidence) != {
        "status",
        "receipt_path",
        "before_optional_imports",
        "after_optional_imports",
    }:
        errors.append(f"{context}: fields are invalid")
    if evidence.get("status") != "ok":
        errors.append(f"{context}: status is not ok")
    if evidence.get("receipt_path") != RADIAL_INSTALL_RECEIPT_PATH:
        errors.append(f"{context}: receipt path is not the fixed audited path")
    errors.extend(
        _radial_runtime_dependency_errors(
            evidence.get("before_optional_imports"),
            expected_runtime,
            f"{context}.before_optional_imports",
        )
    )
    errors.extend(
        _radial_optional_import_loader_errors(
            evidence.get("after_optional_imports"),
            expected_runtime,
            f"{context}.after_optional_imports",
        )
    )
    return errors


def _sparge_static_evidence_errors(
    preflight: Any,
    receipt: Any,
    build_log_snapshot: _StableFileSnapshot,
    install_gpu: Any,
    install_gpu_snapshot: _StableFileSnapshot,
    pre_run_gpu: Mapping[str, Any],
) -> list[str]:
    """Revalidate copied Sparge originals and their receipt/preflight chain."""

    errors: list[str] = []
    expected_gpu_uuid = pre_run_gpu.get("device_uuid")
    errors.extend(
        f"copied SpargeAttn receipt: {error}"
        for error in sparge_receipt_evidence_errors(
            receipt,
            expected_gpu_uuid=expected_gpu_uuid,
        )
    )
    if not isinstance(receipt, dict):
        return errors

    build_metadata = receipt.get("build_log")
    if not isinstance(build_metadata, dict):
        errors.append("SpargeAttn receipt build_log fingerprint is missing")
    elif (
        not _is_json_int(build_metadata.get("bytes"))
        or build_metadata.get("bytes") != build_log_snapshot.size
        or build_metadata.get("sha256") != build_log_snapshot.sha256
    ):
        errors.append("copied SpargeAttn build log differs from install receipt")

    install_metadata = receipt.get("install_pre_run_gpu")
    if not isinstance(install_metadata, dict):
        errors.append("SpargeAttn receipt install GPU fingerprint is missing")
    elif (
        not _is_json_int(install_metadata.get("bytes"))
        or install_metadata.get("bytes") != install_gpu_snapshot.size
        or install_metadata.get("sha256") != install_gpu_snapshot.sha256
        or install_metadata.get("device_uuid") != expected_gpu_uuid
    ):
        errors.append(
            "copied SpargeAttn install GPU evidence differs from install receipt"
        )

    if not isinstance(install_gpu, dict):
        errors.append("SpargeAttn install GPU evidence must be a JSON object")
    else:
        install_gpu_errors = validate_pre_run_gpu_report(
            install_gpu,
            cuda_visible_devices=install_gpu.get("cuda_visible_devices"),
        )
        if install_gpu_errors:
            errors.append(
                "SpargeAttn install GPU evidence is invalid: "
                + "; ".join(install_gpu_errors)
            )
        if (
            not _is_json_int(install_gpu.get("schema_version"))
            or install_gpu.get("schema_version") != GPU_EVIDENCE_SCHEMA_VERSION
            or install_gpu.get("check_type") != "pre_run_idle"
            or install_gpu.get("valid_for_run") is not True
            or install_gpu.get("idle") is not True
            or not _is_json_int(install_gpu.get("process_count"))
            or install_gpu.get("process_count") != 0
            or install_gpu.get("processes") != []
            or install_gpu.get("errors") != []
            or install_gpu.get("device_uuid") != expected_gpu_uuid
        ):
            errors.append(
                "SpargeAttn install GPU evidence is not a schema-2 idle "
                "record for the benchmark GPU UUID"
            )

    if not isinstance(preflight, dict):
        errors.append("Sparge preflight must be a JSON object")
        return errors
    if preflight.get("errors") != []:
        errors.append("Sparge preflight errors must be an explicit empty list")
    if preflight.get("attention_method") != "sparge":
        errors.append("Sparge preflight attention_method must be sparge")
    preflight_sparge = preflight.get("spargeattn")
    if not isinstance(preflight_sparge, dict):
        errors.append("Sparge preflight is missing spargeattn evidence")
    else:
        expected_fields = {
            "package_version": receipt.get("package_version"),
            "pinned_commit": SPARGEATTN_COMMIT,
            "api": SPARGEATTN_API,
            "install_receipt": "/cache/liluchen/FastA2V/spargeattn-install.json",
            "install_receipt_contents": receipt,
            "installed_files_verified": True,
        }
        for field, expected in expected_fields.items():
            if not _strict_json_equal(preflight_sparge.get(field), expected):
                errors.append(
                    f"Sparge preflight {field} differs from fixed install evidence"
                )
    errors.extend(
        f"Sparge preflight microtest: {error}"
        for error in sparge_microtest_evidence_errors(
            preflight.get("spargeattn_microtest"),
            expected_gpu_uuid=expected_gpu_uuid,
        )
    )
    return errors


def _sparge_dispatcher_errors(
    dispatcher: Any,
    environment: Mapping[str, Any],
    copied_receipt: Any,
    expected_gpu_uuid: Any,
    context: str,
    *,
    block_cache_saved_calls: Any = None,
) -> list[str]:
    """Revalidate one formal Sparge dispatcher without trusting verifier JSON."""

    if not isinstance(dispatcher, dict):
        return [f"{context}: video_self_attention_dispatcher is missing"]
    errors: list[str] = []
    calls = dispatcher.get("calls_total")
    expected_calls = dispatcher.get("expected_calls")
    expected_without = dispatcher.get("expected_calls_without_block_cache")
    if not _is_json_int(calls) or calls <= 0:
        errors.append(f"{context}: dispatcher calls_total must be positive")
    if not _is_json_int(expected_calls) or expected_calls <= 0:
        errors.append(f"{context}: dispatcher expected_calls must be positive")
    if calls != expected_calls:
        errors.append(f"{context}: dispatcher calls do not match expected_calls")
    if not _is_json_int(expected_without) or expected_without <= 0:
        errors.append(
            f"{context}: expected_calls_without_block_cache must be a positive "
            "JSON integer"
        )
    elif environment.get("use_block_cache") is True:
        if (
            not _is_json_int(block_cache_saved_calls)
            or block_cache_saved_calls < 0
        ):
            errors.append(
                f"{context}: block-cache saved calls must be a nonnegative "
                "JSON integer"
            )
        elif (
            _is_json_int(expected_calls)
            and expected_without != expected_calls + block_cache_saved_calls
        ):
            errors.append(
                f"{context}: expected_calls_without_block_cache does not equal "
                "expected_calls plus block-cache savings"
            )
    elif _is_json_int(expected_calls) and expected_without != expected_calls:
        errors.append(
            f"{context}: non-block expected_calls_without_block_cache differs "
            "from expected_calls"
        )

    expected_top = {
        "configured_method": "sparge",
        "active_method": "sparge",
        "backend_ready": True,
        "fallback_allowed": False,
        "fallback_used": False,
        "fallback_count": 0,
        "fallback_reason": None,
        "calls_match_expected": True,
        "calls_by_method": {
            "dense": 0,
            "sparge": calls,
            "radial": 0,
            "svg": 0,
        },
        "errors_by_method": {
            "dense": 0,
            "sparge": 0,
            "radial": 0,
            "svg": 0,
        },
    }
    for field, expected in expected_top.items():
        if not _strict_json_equal(dispatcher.get(field), expected):
            errors.append(f"{context}: dispatcher {field} differs from fixed evidence")

    details = dispatcher.get("backend_details")
    if not isinstance(details, dict):
        errors.append(f"{context}: Sparge backend_details is missing")
        return errors
    for field, expected in SPARGE_PROVENANCE.items():
        if not _strict_json_equal(details.get(field), expected):
            errors.append(
                f"{context}: Sparge backend {field} differs from fixed provenance"
            )
    backend_calls = details.get("calls")
    if not _is_json_int(backend_calls) or backend_calls != calls:
        errors.append(f"{context}: Sparge backend calls differ from dispatcher")
    expected_backend = {
        "last_nhd_shape": [1, 15004, 24, 128],
        "last_dtype": "torch.bfloat16",
        "last_device": "cuda:0",
        "topk": environment.get("sparge_topk"),
        "pvthreshd": environment.get("sparge_pvthreshd"),
        "smooth_k": environment.get("sparge_smooth_k"),
    }
    for field, expected in expected_backend.items():
        if not _strict_json_equal(details.get(field), expected):
            errors.append(
                f"{context}: Sparge backend {field} differs from environment"
            )
    backend_receipt = details.get("install_receipt")
    if not _strict_json_equal(backend_receipt, copied_receipt):
        errors.append(
            f"{context}: Sparge backend receipt differs from copied run receipt"
        )
    else:
        errors.extend(
            f"{context}: Sparge backend receipt: {error}"
            for error in sparge_receipt_evidence_errors(
                backend_receipt,
                expected_gpu_uuid=expected_gpu_uuid,
            )
        )
    return errors


def _radial_dispatcher_errors(
    dispatcher: Any,
    environment: dict[str, Any],
    copied_receipt: Any,
    context: str,
    *,
    block_cache_saved_calls: Any = None,
) -> list[str]:
    """Revalidate each formal Radial measurement's live dispatcher receipt."""

    if not isinstance(dispatcher, dict):
        return [f"{context}: video_self_attention_dispatcher is missing"]
    errors = []
    calls = dispatcher.get("calls_total")
    expected_calls = dispatcher.get("expected_calls")
    if not _is_json_int(calls) or calls <= 0:
        errors.append(f"{context}: dispatcher calls_total must be positive")
    if not _is_json_int(expected_calls) or expected_calls <= 0:
        errors.append(f"{context}: dispatcher expected_calls must be positive")
    if calls != expected_calls:
        errors.append(f"{context}: dispatcher calls do not match expected_calls")
    expected_without_block_cache = dispatcher.get(
        "expected_calls_without_block_cache"
    )
    if (
        not _is_json_int(expected_without_block_cache)
        or expected_without_block_cache <= 0
    ):
        errors.append(
            f"{context}: dispatcher expected_calls_without_block_cache "
            "must be a positive JSON integer"
        )
    elif environment.get("use_block_cache") is True:
        if (
            not _is_json_int(block_cache_saved_calls)
            or block_cache_saved_calls < 0
        ):
            errors.append(
                f"{context}: block-cache saved calls must be a nonnegative "
                "JSON integer"
            )
        elif (
            _is_json_int(expected_calls)
            and expected_without_block_cache
            != expected_calls + block_cache_saved_calls
        ):
            errors.append(
                f"{context}: dispatcher expected_calls_without_block_cache "
                "does not equal expected_calls plus recorded block-cache savings"
            )
    elif (
        _is_json_int(expected_calls)
        and expected_without_block_cache != expected_calls
    ):
        errors.append(
            f"{context}: non-block dispatcher expected_calls_without_block_cache "
            "differs from expected_calls"
        )
    expected_top = {
        "configured_method": "radial",
        "active_method": "radial",
        "backend_ready": True,
        "fallback_allowed": False,
        "fallback_used": False,
        "fallback_count": 0,
        "fallback_reason": None,
        "calls_match_expected": True,
        "calls_by_method": {
            "dense": 0,
            "sparge": 0,
            "radial": calls,
            "svg": 0,
        },
        "errors_by_method": {
            "dense": 0,
            "sparge": 0,
            "radial": 0,
            "svg": 0,
        },
    }
    for field, expected in expected_top.items():
        if not _strict_json_equal(dispatcher.get(field), expected):
            errors.append(f"{context}: dispatcher {field} differs from fixed evidence")

    details = dispatcher.get("backend_details")
    if not isinstance(details, dict):
        errors.append(f"{context}: Radial backend_details is missing")
        return errors
    expected_provenance = {
        "backend": "official_radial_attention_flashinfer",
        "repository": RADIAL_REPOSITORY,
        "pinned_commit": RADIAL_COMMIT,
        "mask_api": RADIAL_MASK_API,
        "model_type": RADIAL_MODEL_TYPE,
        "block_size": RADIAL_BLOCK_SIZE,
        "sequence": RADIAL_SEQUENCE,
        "prefix_sequence": RADIAL_PREFIX_SEQUENCE,
        "tail_sequence": RADIAL_TAIL_SEQUENCE,
        "tail_strategy": "dense_lse_merge_no_padding",
        "empty_row_policy": "dense_row",
        "empty_rows": list(RADIAL_EMPTY_ROWS),
        "fallback_allowed": False,
        "last_shape": [1, RADIAL_SEQUENCE, 24, 128],
        "last_grid": list(RADIAL_GRID),
        "last_dtype": "torch.bfloat16",
        "last_device": "cuda:0",
        "plan_cache_entries": 1,
    }
    for field, expected in expected_provenance.items():
        if not _strict_json_equal(details.get(field), expected):
            errors.append(
                f"{context}: Radial backend {field} differs from fixed evidence"
            )
    backend_calls = details.get("calls")
    if not _is_json_int(backend_calls) or backend_calls != calls:
        errors.append(f"{context}: Radial backend calls differ from dispatcher")
    hits = details.get("plan_cache_hits")
    misses = details.get("plan_cache_misses")
    if (
        not _is_json_int(hits)
        or hits < 0
        or not _is_json_int(misses)
        or misses not in (0, 1)
        or not _is_json_int(calls)
        or hits + misses != calls
    ):
        errors.append(f"{context}: Radial plan-cache counters are invalid")

    profile = environment.get("radial_profile")
    expected_audit = RADIAL_PROFILE_AUDITS.get(profile)
    expected_settings = {
        "profile": profile,
        "decay_factor": environment.get("radial_decay_factor"),
        "model_type": environment.get("radial_model_type"),
        "block_size": environment.get("radial_block_size"),
    }
    for field, expected in expected_settings.items():
        if expected is None or not _strict_json_equal(details.get(field), expected):
            errors.append(
                f"{context}: Radial setting {field} differs from environment"
            )
    if expected_audit is None or not _strict_json_equal(
        details.get("last_mask_audit"), expected_audit
    ):
        errors.append(f"{context}: Radial mask audit differs from fixed profile")

    expected_runtime = _expected_radial_runtime_dependencies(copied_receipt)
    expected_receipt_summary = {
        "path": RADIAL_INSTALL_RECEIPT_PATH,
        "commit": (
            copied_receipt.get("commit")
            if isinstance(copied_receipt, dict)
            else None
        ),
        "derived_module_sha256": (
            copied_receipt.get("derived_module", {}).get("sha256")
            if isinstance(copied_receipt, dict)
            and isinstance(copied_receipt.get("derived_module"), dict)
            else None
        ),
        "flashinfer_version": (
            copied_receipt.get("flashinfer_version")
            if isinstance(copied_receipt, dict)
            else None
        ),
        "runtime_dependencies": expected_runtime,
    }
    receipt_summary = details.get("install_receipt")
    if not _strict_json_equal(receipt_summary, expected_receipt_summary):
        errors.append(
            f"{context}: Radial receipt summary differs from copied originals"
        )
    if not _strict_json_equal(
        details.get("runtime_dependencies_after_first_cuda"),
        expected_runtime,
    ):
        errors.append(
            f"{context}: Radial runtime inventory after first CUDA differs"
        )
    return errors


def _radial_preflight_static_errors(
    preflight: Any,
    pre_run_gpu: dict[str, Any],
    checkpoint_manifest: dict[str, Any],
    copied_receipt: Any,
    copied_flashinfer_manifest: Any,
    copied_artifacts: dict[str, _StableFileSnapshot],
    flashinfer_manifest_snapshot: _StableFileSnapshot,
) -> list[str]:
    """Revalidate Radial preflight originals instead of trusting verification.json."""

    errors = []
    if not isinstance(preflight, dict):
        return ["preflight.json must contain an object"]
    if preflight.get("errors") != []:
        errors.append("preflight errors must be an explicit empty list")
    if preflight.get("attention_method") != "radial":
        errors.append("preflight attention_method must be radial")
    python_executable = preflight.get("python_executable")
    if not isinstance(python_executable, str) or not Path(
        python_executable
    ).is_absolute():
        errors.append("preflight python_executable must be an absolute path")
    if preflight.get("cuda_available") is not True:
        errors.append("preflight did not confirm CUDA availability")
    if preflight.get("gpu") != pre_run_gpu.get("device_name"):
        errors.append("preflight CUDA device name differs from pre-run evidence")
    if preflight.get("compute_capability") != [8, 0]:
        errors.append("preflight compute_capability must be exactly [8, 0]")
    for executable in ("ffmpeg", "ffprobe"):
        value = preflight.get(executable)
        if not isinstance(value, str) or not Path(value).is_absolute():
            errors.append(f"preflight {executable} path is invalid")
    packages = preflight.get("packages")
    if not isinstance(packages, dict):
        errors.append("preflight package inventory is missing")
    else:
        for package in REQUIRED_PREFLIGHT_PACKAGES:
            version = packages.get(package)
            if not isinstance(version, str) or not version:
                errors.append(
                    f"preflight package inventory lacks version for {package}"
                )
    checkpoints = preflight.get("checkpoints")
    manifest_files = checkpoint_manifest.get("files")
    if not isinstance(checkpoints, dict) or set(checkpoints) != set(
        REQUIRED_PREFLIGHT_CHECKPOINTS
    ):
        errors.append("preflight checkpoint inventory is not the fixed six paths")
    elif not isinstance(manifest_files, dict):
        errors.append("checkpoint manifest file inventory is missing")
    else:
        for relative_path in REQUIRED_PREFLIGHT_CHECKPOINTS:
            metadata = checkpoints.get(relative_path)
            manifest_metadata = manifest_files.get(relative_path)
            if (
                not isinstance(metadata, dict)
                or set(metadata) != {"exists", "bytes"}
                or metadata.get("exists") is not True
                or not _is_json_int(metadata.get("bytes"))
                or metadata.get("bytes") <= 0
                or not isinstance(manifest_metadata, dict)
                or metadata.get("bytes") != manifest_metadata.get("bytes")
            ):
                errors.append(
                    "preflight checkpoint inventory differs from checkpoint "
                    f"manifest for {relative_path}"
                )
    if preflight.get("checkpoint_manifest") != (
        "/cache/liluchen/FastA2V/checkpoint_manifest.json"
    ):
        errors.append("preflight checkpoint manifest path is not fixed")
    flash_attn_microtest = preflight.get("flash_attn_microtest")
    expected_flash_attn = {
        "status": "ok",
        "device": pre_run_gpu.get("device_name"),
        "compute_capability": [8, 0],
        "torch": "2.6.0+cu124",
        "torch_cuda": "12.4",
        "torch_cxx11_abi": False,
        "dtype": "torch.bfloat16",
        "shape": [1, 128, 24, 128],
    }
    if not isinstance(flash_attn_microtest, dict):
        errors.append("preflight FlashAttention microtest evidence is invalid")
    else:
        for field, expected in expected_flash_attn.items():
            if not _strict_json_equal(flash_attn_microtest.get(field), expected):
                errors.append(
                    f"preflight FlashAttention microtest {field} is invalid"
                )
        difference = flash_attn_microtest.get("max_abs_difference")
        if (
            isinstance(difference, bool)
            or not isinstance(difference, (int, float))
            or not math.isfinite(float(difference))
            or float(difference) < 0.0
            or float(difference) > 0.1
        ):
            errors.append(
                "preflight FlashAttention max_abs_difference is invalid"
            )

    errors.extend(
        f"copied Radial receipt: {error}"
        for error in radial_receipt_evidence_errors(copied_receipt)
    )
    expected_runtime = _expected_radial_runtime_dependencies(copied_receipt)

    if isinstance(copied_receipt, dict):
        for field, snapshot in copied_artifacts.items():
            metadata = copied_receipt.get(field)
            if not isinstance(metadata, dict):
                errors.append(f"copied Radial receipt lacks {field}")
            elif (
                snapshot.size != metadata.get("bytes")
                or snapshot.sha256 != metadata.get("sha256")
            ):
                errors.append(
                    f"copied Radial {field} differs from install receipt"
                )
    if isinstance(copied_receipt, dict):
        manifest_fingerprint = copied_receipt.get("flashinfer_manifest")
        if not isinstance(manifest_fingerprint, dict):
            errors.append("copied Radial receipt lacks flashinfer_manifest")
        elif (
            flashinfer_manifest_snapshot.size != manifest_fingerprint.get("bytes")
            or flashinfer_manifest_snapshot.sha256
            != manifest_fingerprint.get("sha256")
        ):
            errors.append("copied FlashInfer manifest differs from install receipt")
    errors.extend(
        f"copied FlashInfer manifest: {error}"
        for error in flashinfer_manifest_evidence_errors(
            copied_flashinfer_manifest,
            copied_receipt,
        )
    )

    radial = preflight.get("radialattn")
    if not isinstance(radial, dict):
        errors.append("preflight is missing radialattn static evidence")
    else:
        expected_radial = {
            "pinned_commit": RADIAL_COMMIT,
            "mask_api": RADIAL_MASK_API,
            "source_files_verified": True,
            "flashinfer_files_verified": True,
            "flashinfer_manifest_verified": True,
            "runtime_loader_environment_verified": True,
            "cpu_mask_audits_verified": True,
            "flashinfer_version": FLASHINFER_VERSION,
            "flashinfer_apis": {
                "BlockSparseAttentionWrapper": True,
                "single_prefill_with_kv_cache": True,
                "merge_state": True,
            },
            "derived_mask_api_callable": True,
            "install_cuda_kernel_launched": False,
            "preflight_cuda_microtest_required": True,
        }
        for field, expected in expected_radial.items():
            if not _strict_json_equal(radial.get(field), expected):
                errors.append(
                    f"preflight radialattn {field} differs from fixed evidence"
                )
        if not _strict_json_equal(
            radial.get("install_receipt_contents"), copied_receipt
        ):
            errors.append("preflight radialattn receipt differs from copied receipt")
        errors.extend(
            _radial_runtime_dependency_errors(
                radial.get("runtime_dependencies_before_optional_imports"),
                expected_runtime,
                "preflight Radial before optional imports",
            )
        )
        errors.extend(
            _radial_optional_import_loader_errors(
                radial.get("optional_import_loader_evidence"),
                expected_runtime,
                "preflight Radial optional imports",
            )
        )

    microtest = preflight.get("radialattn_microtest")
    if isinstance(microtest, dict):
        for phase in ("before_cuda", "after_cuda"):
            errors.extend(
                _radial_runtime_dependency_errors(
                    microtest.get(f"runtime_dependencies_{phase}"),
                    expected_runtime,
                    f"Radial preflight microtest {phase}",
                )
            )
    return errors


def validate_run(
    method: dict[str, Any],
    run_dir: Path,
    fixed_protocol: dict[str, Any],
) -> dict[str, Any]:
    method_id = method["method_id"]
    context = f"{method_id} run"
    _require(
        method.get("implementation_status") == "ready",
        context,
        "method is still marked pending in the evaluation manifest; refusing to fabricate a result",
    )
    run_dir = Path(run_dir).resolve()
    _require(run_dir.is_dir(), context, f"run directory does not exist: {run_dir}")

    environment_path = run_dir / "environment.json"
    verification_path = run_dir / "verification.json"
    timings_path = run_dir / "timings.jsonl"
    warmup_timings_path = run_dir / "warmup_timings.jsonl"
    checkpoint_path = run_dir / "checkpoint_manifest.json"
    pre_run_gpu_path = run_dir / "pre_run_gpu.json"
    preflight_path = run_dir / "preflight.json"
    stable_snapshots: list[_StableFileSnapshot] = []
    environment_snapshot, environment = _snapshot_json(environment_path, context)
    verification_snapshot, verification = _snapshot_json(verification_path, context)
    timings_snapshot, timings = _snapshot_jsonl(timings_path, context)
    warmup_timings_snapshot, warmup_timings = _snapshot_jsonl(
        warmup_timings_path,
        context,
    )
    checkpoint_snapshot, checkpoint_manifest = _snapshot_json(
        checkpoint_path, context
    )
    pre_run_snapshot, pre_run_gpu = _snapshot_json(pre_run_gpu_path, context)
    stable_snapshots.extend(
        (
            environment_snapshot,
            verification_snapshot,
            timings_snapshot,
            warmup_timings_snapshot,
            checkpoint_snapshot,
            pre_run_snapshot,
        )
    )
    preflight_snapshot = None
    copied_receipt_snapshot = None
    copied_receipt = None
    flashinfer_manifest_snapshot = None
    copied_flashinfer_manifest = None
    copied_artifact_snapshots: dict[str, _StableFileSnapshot] = {}
    sparge_receipt_snapshot = None
    sparge_receipt = None
    sparge_build_log_snapshot = None
    sparge_install_gpu_snapshot = None
    sparge_install_gpu = None
    attention_method = (
        environment.get("attention_method")
        if isinstance(environment, dict)
        else None
    )
    if attention_method in {"radial", "sparge"}:
        preflight_snapshot, preflight = _snapshot_json(preflight_path, context)
        stable_snapshots.append(preflight_snapshot)
    else:
        preflight = None
    if attention_method == "radial":
        copied_receipt_snapshot, copied_receipt = _snapshot_json(
            run_dir / "radialattn-install.json", context
        )
        flashinfer_manifest_snapshot, copied_flashinfer_manifest = _snapshot_json(
            run_dir / "radial-flashinfer-manifest.json", context
        )
        for field, filename in (
            ("source_module", "radial-attention-source.py"),
            ("derived_module", "radial-attention-derived.py"),
            ("optional_imports_patch", "radial-attention-optional-imports.patch"),
        ):
            copied_artifact_snapshots[field] = _stable_file_snapshot(
                run_dir / filename,
                context,
            )
        stable_snapshots.extend(
            (
                copied_receipt_snapshot,
                flashinfer_manifest_snapshot,
                *copied_artifact_snapshots.values(),
            )
        )
    elif attention_method == "sparge":
        sparge_receipt_snapshot, sparge_receipt = _snapshot_json(
            run_dir / "spargeattn-install.json",
            context,
        )
        sparge_build_log_snapshot = _stable_file_snapshot(
            run_dir / "spargeattn-build.log",
            context,
        )
        sparge_install_gpu_snapshot, sparge_install_gpu = _snapshot_json(
            run_dir / "spargeattn-install-pre_run_gpu.json",
            context,
        )
        stable_snapshots.extend(
            (
                sparge_receipt_snapshot,
                sparge_build_log_snapshot,
                sparge_install_gpu_snapshot,
            )
        )
    required_payloads = [
        ("environment.json", environment),
        ("verification.json", verification),
        ("checkpoint_manifest.json", checkpoint_manifest),
        ("pre_run_gpu.json", pre_run_gpu),
    ]
    if environment.get("attention_method") in {"radial", "sparge"}:
        required_payloads.append(("preflight.json", preflight))
    if environment.get("attention_method") == "sparge":
        required_payloads.extend(
            (
                ("spargeattn-install.json", sparge_receipt),
                ("spargeattn-install-pre_run_gpu.json", sparge_install_gpu),
            )
        )
    for name, payload in required_payloads:
        _require(isinstance(payload, dict), context, f"{name} must contain an object")

    _require(verification.get("status") == "ok", context, "verification status is not ok")
    _require(
        verification.get("benchmark_valid") is True,
        context,
        "verification.json does not certify benchmark_valid=true",
    )
    protocol = verification.get("protocol")
    _require(isinstance(protocol, dict), context, "verification protocol is missing")
    _require(protocol.get("status") == "ok", context, "verification protocol status is not ok")
    _require(protocol.get("errors") == [], context, "verification protocol contains errors")
    _require(
        protocol.get("benchmark_candidate") is True,
        context,
        "verification protocol is not a benchmark candidate",
    )
    _require(
        protocol.get("benchmark_valid") is True,
        context,
        "verification protocol does not certify benchmark_valid=true",
    )

    _require(environment.get("git_dirty") is False, context, "git_dirty must be false")
    git_commit = environment.get("git_commit")
    _require(
        isinstance(git_commit, str) and HEX_GIT_COMMIT.fullmatch(git_commit) is not None,
        context,
        f"git_commit is not a full lowercase commit hash: {git_commit!r}",
    )
    _require(environment.get("benchmark_eligible") is True, context, "benchmark_eligible must be true")
    _require(environment.get("debug_forward") is False, context, "debug_forward must be false")
    _require(environment.get("pre_run_gpu_valid") is True, context, "pre-run GPU evidence is not valid")
    _require(
        _is_json_int(environment.get("gpu_physical_index"))
        and environment.get("gpu_physical_index") == 0,
        context,
        "physical GPU index must be integer zero",
    )
    pre_run_errors = validate_pre_run_gpu_report(
        pre_run_gpu,
        cuda_visible_devices=pre_run_gpu.get("cuda_visible_devices"),
    )
    _require(
        not pre_run_errors,
        context,
        "pre-run GPU evidence is invalid: " + "; ".join(pre_run_errors),
    )
    _require(
        pre_run_gpu.get("device_uuid") == environment.get("gpu_uuid")
        and pre_run_gpu.get("device_name") == environment.get("gpu_name"),
        context,
        "pre-run GPU identity differs from environment",
    )
    _require(
        pre_run_gpu.get("cuda_visible_devices")
        == environment.get("cuda_visible_devices"),
        context,
        "pre-run CUDA_VISIBLE_DEVICES differs from environment",
    )
    expected_nvidia_smi_binary = pre_run_gpu.get("nvidia_smi_binary")
    pre_run_binary_errors = trusted_nvidia_smi_metadata_errors(
        expected_nvidia_smi_binary
    )
    _require(
        not pre_run_binary_errors,
        context,
        "pre-run nvidia-smi binary fixed metadata is invalid: "
        + "; ".join(pre_run_binary_errors),
    )
    _require(
        environment.get("run_id") == run_dir.name,
        context,
        "environment run_id must equal the explicitly selected directory name",
    )
    for field in ("gpu_uuid", "gpu_name"):
        _require(
            isinstance(environment.get(field), str) and environment.get(field),
            context,
            f"{field} is missing",
        )

    # Prompt and sample cardinalities are run-level dimensions.  Their current
    # values remain fixed by the checked-in matrix, while the evidence model
    # below deliberately derives the full Cartesian identity set so a future
    # matrix revision can raise either cardinality without another schema
    # redesign.
    _validate_expected_fields(environment, fixed_protocol, context)
    _validate_expected_fields(environment, method["expected_environment"], context)
    _validate_combo_run_environment(method, environment, context)
    measurement_runs = environment.get("measurement_runs")
    prompt_count = environment.get("prompt_count")
    sample_count = environment.get("each_example_n_times")
    warmup_runs = environment.get("warmup_runs")
    for field, value in (
        ("measurement_runs", measurement_runs),
        ("prompt_count", prompt_count),
        ("each_example_n_times", sample_count),
        ("warmup_runs", warmup_runs),
    ):
        _require(
            _is_json_int(value) and value > 0,
            context,
            f"{field} must be a positive integer",
        )
    _require(
        measurement_runs == MEASUREMENT_COUNT,
        context,
        "measurement_runs must equal three",
    )
    expected_artifact_count = measurement_runs * prompt_count * sample_count
    _require(
        environment.get("expected_measurement_records")
        == expected_artifact_count,
        context,
        "expected_measurement_records does not equal "
        "measurement_runs * prompt_count * each_example_n_times",
    )
    _require(
        environment.get("expected_warmup_records") == warmup_runs,
        context,
        "expected_warmup_records differs from warmup_runs",
    )
    _require(
        protocol.get("expected_measurement_records") == expected_artifact_count
        and protocol.get("observed_measurement_records")
        == expected_artifact_count,
        context,
        "verification protocol does not contain the complete artifact matrix",
    )
    _require(
        protocol.get("expected_warmup_records") == warmup_runs
        and protocol.get("observed_warmup_records") == warmup_runs,
        context,
        "verification protocol does not contain the expected excluded warm-ups",
    )
    _require(
        len(timings) == expected_artifact_count,
        context,
        "timings.jsonl does not contain the complete artifact matrix",
    )
    _require(
        len(warmup_timings) == warmup_runs,
        context,
        "warmup_timings.jsonl record count differs from warmup_runs",
    )

    _validate_jsonl_binding(
        protocol.get("timings_binding"),
        path=timings_path,
        snapshot=timings_snapshot,
        record_count=len(timings),
        context=context,
        label="timings_binding",
    )
    _validate_jsonl_binding(
        protocol.get("warmup_timings_binding"),
        path=warmup_timings_path,
        snapshot=warmup_timings_snapshot,
        record_count=len(warmup_timings),
        context=context,
        label="warmup_timings_binding",
    )

    checkpoint_manifest_sha256 = checkpoint_snapshot.sha256
    pre_run_gpu_sha256 = pre_run_snapshot.sha256
    preflight_sha256 = (
        preflight_snapshot.sha256
        if environment.get("attention_method") in {"radial", "sparge"}
        else ""
    )
    evidence_hashes = environment.get("evidence_file_sha256")
    _require(isinstance(evidence_hashes, dict), context, "environment evidence hashes are missing")
    _require(
        evidence_hashes.get("checkpoint_manifest.json") == checkpoint_manifest_sha256,
        context,
        "checkpoint manifest hash differs from environment evidence",
    )
    _require(
        evidence_hashes.get("pre_run_gpu.json") == pre_run_gpu_sha256
        and environment.get("pre_run_gpu_sha256") == pre_run_gpu_sha256,
        context,
        "pre-run GPU hash differs from environment evidence",
    )
    radial_summary = {
        "radial_evidence_mode": "",
        "radial_pmon_status": "",
        "radial_pid_association": "",
        "radial_claim_scope": "",
        "radial_host_pid_ownership": "",
        "radial_mps_status": "",
    }
    if environment.get("attention_method") == "sparge":
        sparge_original_snapshots = {
            "preflight.json": preflight_snapshot,
            "spargeattn-install.json": sparge_receipt_snapshot,
            "spargeattn-build.log": sparge_build_log_snapshot,
            "spargeattn-install-pre_run_gpu.json": sparge_install_gpu_snapshot,
        }
        for filename, snapshot in sparge_original_snapshots.items():
            _require(
                isinstance(snapshot, _StableFileSnapshot)
                and evidence_hashes.get(filename) == snapshot.sha256,
                context,
                f"{filename} hash differs from environment evidence",
            )
        _require(
            environment.get("spas_sage_attn")
            == (
                sparge_receipt.get("package_version")
                if isinstance(sparge_receipt, dict)
                else None
            ),
            context,
            "environment SpargeAttn package version differs from install receipt",
        )
        sparge_static_errors = _sparge_static_evidence_errors(
            preflight,
            sparge_receipt,
            sparge_build_log_snapshot,
            sparge_install_gpu,
            sparge_install_gpu_snapshot,
            pre_run_gpu,
        )
        _require(
            not sparge_static_errors,
            context,
            "Sparge original evidence is invalid: "
            + "; ".join(sparge_static_errors),
        )
    if environment.get("attention_method") == "radial":
        radial_original_snapshots = {
            "radialattn-install.json": copied_receipt_snapshot,
            "radial-flashinfer-manifest.json": flashinfer_manifest_snapshot,
            "radial-attention-source.py": copied_artifact_snapshots.get(
                "source_module"
            ),
            "radial-attention-derived.py": copied_artifact_snapshots.get(
                "derived_module"
            ),
            "radial-attention-optional-imports.patch": (
                copied_artifact_snapshots.get("optional_imports_patch")
            ),
        }
        for filename, snapshot in radial_original_snapshots.items():
            _require(
                isinstance(snapshot, _StableFileSnapshot)
                and evidence_hashes.get(filename) == snapshot.sha256,
                context,
                f"{filename} hash differs from environment evidence",
            )
        _require(
            evidence_hashes.get("preflight.json") == preflight_sha256,
            context,
            "preflight hash differs from environment evidence",
        )
        radial_static_errors = _radial_preflight_static_errors(
            preflight,
            pre_run_gpu,
            checkpoint_manifest,
            copied_receipt,
            copied_flashinfer_manifest,
            copied_artifact_snapshots,
            flashinfer_manifest_snapshot,
        )
        _require(
            not radial_static_errors,
            context,
            "Radial preflight static evidence is invalid: "
            + "; ".join(radial_static_errors),
        )
        loader_errors = _radial_loader_bootstrap_errors(
            environment.get("radial_loader_bootstrap"),
            _expected_radial_runtime_dependencies(copied_receipt),
        )
        _require(
            not loader_errors,
            context,
            "Radial inference loader bootstrap is invalid: "
            + "; ".join(loader_errors),
        )
        radial_microtest = preflight.get("radialattn_microtest")
        radial_errors = radial_microtest_evidence_errors(
            radial_microtest,
            expected_gpu_uuid=pre_run_gpu.get("device_uuid"),
            expected_pre_run_gpu=pre_run_gpu,
            expected_pre_run_gpu_sha256=pre_run_gpu_sha256,
            expected_pre_run_gpu_path=str(pre_run_gpu_path.resolve()),
            expected_python_executable=preflight.get("python_executable"),
        )
        _require(
            not radial_errors,
            context,
            "Radial preflight microtest is invalid: "
            + "; ".join(radial_errors),
        )
        binding = radial_microtest.get("gpu_process_binding")
        _require(
            isinstance(binding, dict),
            context,
            "Radial GPU process binding is missing",
        )
        pmon = (
            binding.get("mps", {}).get("pmon")
            if isinstance(binding.get("mps"), dict)
            else None
        )
        _require(
            isinstance(pmon, dict),
            context,
            "Radial pmon evidence is missing",
        )
        radial_summary = {
            "radial_evidence_mode": binding.get("pmon_observation_mode"),
            "radial_pmon_status": pmon.get("status"),
            "radial_pid_association": binding.get("binding_method"),
            "radial_claim_scope": binding.get("claim_scope"),
            "radial_host_pid_ownership": binding.get("host_pid_ownership"),
            "radial_mps_status": binding.get("mps", {}).get("mps_status"),
        }
    checkpoint_fingerprint = _checkpoint_fingerprint(checkpoint_manifest, context)

    requested_shape = _shape(
        environment.get("video_frame_height_width"),
        context,
        "video_frame_height_width",
        length=2,
    )
    engine_load_seconds = _finite_number(
        environment,
        "engine_load_seconds",
        context,
        nonnegative=True,
    )

    identities: list[tuple[int, int, int]] = []
    for record_index, record in enumerate(timings):
        record_context = f"{context} measurement[{record_index}]"
        identity = tuple(
            record.get(field)
            for field in (
                "measurement_index",
                "prompt_index",
                "sample_index",
            )
        )
        _require(
            all(_is_json_int(value) and value >= 0 for value in identity),
            record_context,
            f"artifact identity must contain nonnegative integers, found {identity!r}",
        )
        identities.append(identity)
    expected_identities = {
        (measurement_index, prompt_index, sample_index)
        for measurement_index in range(measurement_runs)
        for prompt_index in range(prompt_count)
        for sample_index in range(sample_count)
    }
    _require(
        len(set(identities)) == len(identities),
        context,
        "artifact identities are duplicated",
    )
    _require(
        set(identities) == expected_identities,
        context,
        "artifact identities do not form the complete "
        "measurement_index,prompt_index,sample_index matrix",
    )
    verified_artifacts = verification.get("artifacts")
    _require(
        verification.get("artifact_count") == expected_artifact_count
        and isinstance(verified_artifacts, list)
        and len(verified_artifacts) == expected_artifact_count,
        context,
        "verification must contain the complete artifact matrix",
    )
    verified_by_path: dict[Path, tuple[int, int, int]] = {}
    verified_by_identity: dict[tuple[int, int, int], dict[str, Any]] = {}
    for report_index, report in enumerate(verified_artifacts):
        report_context = f"{context} verified artifact[{report_index}]"
        _require(isinstance(report, dict), report_context, "artifact report must be an object")
        _require(report.get("status") == "ok", report_context, "artifact status is not ok")
        _require(report.get("errors") == [], report_context, "artifact report contains errors")
        report_identity = tuple(
            report.get(field)
            for field in (
                "measurement_index",
                "prompt_index",
                "sample_index",
            )
        )
        _require(
            all(_is_json_int(value) and value >= 0 for value in report_identity),
            report_context,
            "artifact report identity must contain nonnegative JSON integers",
        )
        _require(
            report_identity not in verified_by_identity,
            report_context,
            "artifact report identity is duplicated",
        )
        report_prompt = report.get("prompt")
        report_seed = report.get("seed")
        _require(
            isinstance(report_prompt, str) and report_prompt,
            report_context,
            "artifact report prompt is missing",
        )
        _require(
            _is_json_int(report_seed),
            report_context,
            "artifact report seed is invalid",
        )
        report_path_value = report.get("path")
        _require(isinstance(report_path_value, str) and report_path_value, report_context, "artifact path is missing")
        report_path = Path(report_path_value)
        _require(report_path.is_absolute(), report_context, "artifact path must be absolute")
        _require(report_path.parent == run_dir, report_context, "artifact is outside the selected run directory")
        _require(
            report_path == run_dir / report_path.name,
            report_context,
            "artifact path is not canonical",
        )
        report_hash = report.get("sha256")
        _require(
            isinstance(report_hash, str) and HEX_SHA256.fullmatch(report_hash) is not None,
            report_context,
            "artifact SHA256 is invalid",
        )
        metrics_path_value = report.get("metrics_path")
        _require(
            isinstance(metrics_path_value, str) and metrics_path_value,
            report_context,
            "metrics sidecar path is missing",
        )
        metrics_path = Path(metrics_path_value)
        _require(
            metrics_path.is_absolute()
            and metrics_path.parent == run_dir
            and metrics_path == run_dir / metrics_path.name,
            report_context,
            "metrics sidecar path is outside the selected run directory or not canonical",
        )
        _require(
            metrics_path == report_path.with_suffix(".metrics.json"),
            report_context,
            "metrics sidecar path does not match the artifact path",
        )
        _require(report_path not in verified_by_path, report_context, "artifact path is duplicated")
        verified_by_path[report_path] = report_identity
        verified_by_identity[report_identity] = report

    _require(
        set(verified_by_identity) == expected_identities,
        context,
        "verification artifact identities do not form the complete matrix",
    )

    denoise_values = []
    total_values = []
    artifact_ready_values = []
    allocated_values = []
    reserved_values = []
    prompt_by_index: dict[int, str] = {}
    seed_by_sample_index: dict[int, int] = {}
    actual_shapes = set()
    generated_video_shapes = set()
    generated_audio_shapes = set()
    timing_paths = set()
    artifact_hashes: list[tuple[tuple[int, int, int], str]] = []
    metrics_sidecar_hashes: list[tuple[tuple[int, int, int], str]] = []

    for record_index, record in enumerate(timings):
        record_context = f"{context} measurement[{record_index}]"
        identity = identities[record_index]
        measurement_index, prompt_index, sample_index = identity
        _require(record.get("status") == "ok", record_context, "status is not ok")
        _require(record.get("record_type") == "measurement", record_context, "record_type is not measurement")
        _require(record.get("benchmark_candidate") is True, record_context, "record is not a benchmark candidate")
        _require(record.get("run_id") == environment.get("run_id"), record_context, "run_id differs from environment")
        for field in ("sample_steps", "attention_method", "use_cfg_cache", "use_block_cache"):
            _require(
                _values_equal(record.get(field), environment.get(field)),
                record_context,
                f"{field} differs from environment",
            )
        if environment.get("attention_method") == "sparge":
            dispatcher_errors = _sparge_dispatcher_errors(
                record.get("video_self_attention_dispatcher"),
                environment,
                sparge_receipt,
                pre_run_gpu.get("device_uuid"),
                record_context,
                block_cache_saved_calls=record.get(
                    "block_cache_saved_video_self_attention_calls"
                ),
            )
            _require(
                not dispatcher_errors,
                record_context,
                "Sparge dispatcher evidence is invalid: "
                + "; ".join(dispatcher_errors),
            )
        elif environment.get("attention_method") == "radial":
            dispatcher_errors = _radial_dispatcher_errors(
                record.get("video_self_attention_dispatcher"),
                environment,
                copied_receipt,
                record_context,
                block_cache_saved_calls=record.get(
                    "block_cache_saved_video_self_attention_calls"
                ),
            )
            _require(
                not dispatcher_errors,
                record_context,
                "Radial dispatcher evidence is invalid: "
                + "; ".join(dispatcher_errors),
            )

        denoise = _finite_number(record, "denoise_seconds", record_context, positive=True)
        total = _finite_number(record, "total_generation_seconds", record_context, positive=True)
        save = _finite_number(record, "save_video_seconds", record_context, nonnegative=True)
        artifact_ready = _finite_number(record, "artifact_ready_seconds", record_context, positive=True)
        _finite_number(record, "output_hash_seconds", record_context, nonnegative=True)
        allocated = _finite_number(record, "peak_memory_allocated_bytes", record_context, positive=True)
        reserved = _finite_number(record, "peak_memory_reserved_bytes", record_context, positive=True)
        _require(total >= denoise, record_context, "total_generation_seconds is shorter than denoise_seconds")
        _require(artifact_ready >= total, record_context, "artifact_ready_seconds is shorter than total generation")
        _require(artifact_ready >= save, record_context, "artifact_ready_seconds is shorter than save_video_seconds")
        _require(reserved >= allocated, record_context, "reserved memory is smaller than allocated memory")

        prompt = record.get("prompt")
        seed = record.get("seed")
        _require(isinstance(prompt, str) and prompt, record_context, "prompt is missing")
        _require(isinstance(seed, int) and not isinstance(seed, bool), record_context, "seed is invalid")
        expected_seed = environment.get("seed") + sample_index
        _require(
            seed == expected_seed,
            record_context,
            "seed differs from the fixed base seed plus sample_index",
        )
        previous_prompt = prompt_by_index.setdefault(prompt_index, prompt)
        _require(
            previous_prompt == prompt,
            record_context,
            "prompt text is inconsistent for prompt_index",
        )
        previous_seed = seed_by_sample_index.setdefault(sample_index, seed)
        _require(
            previous_seed == seed,
            record_context,
            "seed is inconsistent for sample_index",
        )
        record_requested = _shape(
            record.get("requested_video_frame_height_width"),
            record_context,
            "requested_video_frame_height_width",
            length=2,
        )
        _require(record_requested == requested_shape, record_context, "requested shape differs from environment")
        actual_shape = _shape(
            record.get("actual_video_frame_height_width"),
            record_context,
            "actual_video_frame_height_width",
            length=2,
        )
        generated_video_shape = _shape(
            record.get("generated_video_shape"),
            record_context,
            "generated_video_shape",
        )
        _require(len(generated_video_shape) == 4, record_context, "generated_video_shape must be C,F,H,W")
        _require(generated_video_shape[-2:] == actual_shape, record_context, "generated video shape differs from actual shape")
        generated_audio_shape = _shape(
            record.get("generated_audio_shape"),
            record_context,
            "generated_audio_shape",
        )

        _validate_gpu_monitor(
            record.get("gpu_process_monitor"),
            environment,
            expected_nvidia_smi_binary,
            pre_run_gpu.get("boot_id"),
            environment.get("gpu_process_monitor_interval_seconds"),
            total,
            record_context,
        )

        output_path_value = record.get("output_path")
        output_hash = record.get("output_sha256")
        _require(isinstance(output_path_value, str) and output_path_value, record_context, "output_path is missing")
        _require(
            isinstance(output_hash, str) and HEX_SHA256.fullmatch(output_hash) is not None,
            record_context,
            "output_sha256 is invalid",
        )
        output_path = Path(output_path_value)
        _require(output_path.is_absolute(), record_context, "output artifact path must be absolute")
        _require(output_path.parent == run_dir, record_context, "output artifact is outside the selected run directory")
        _require(
            output_path == run_dir / output_path.name,
            record_context,
            "output artifact path is not canonical",
        )
        _require(output_path not in timing_paths, record_context, "output artifact path is duplicated")
        metrics_path = output_path.with_suffix(".metrics.json")
        output_snapshot = _stable_file_snapshot(output_path, record_context)
        metrics_snapshot, metrics_sidecar = _snapshot_json(
            metrics_path, record_context
        )
        stable_snapshots.extend((output_snapshot, metrics_snapshot))
        _require(
            isinstance(metrics_sidecar, dict),
            record_context,
            "metrics sidecar must contain an object",
        )
        _require(
            metrics_sidecar == record,
            record_context,
            "timings.jsonl record differs from its metrics sidecar",
        )
        actual_hash = output_snapshot.sha256
        _require(actual_hash == output_hash, record_context, "output artifact SHA256 differs from timing record")
        verified_report = verified_by_identity.get(identity)
        _require(
            isinstance(verified_report, dict),
            record_context,
            "artifact identity is missing from the verification report",
        )
        _require(
            verified_report.get("path") == str(output_path)
            and verified_by_path.get(output_path) == identity,
            record_context,
            "output artifact path is assigned to a different verification identity",
        )
        _require(
            verified_report.get("sha256") == actual_hash,
            record_context,
            "output artifact SHA256 differs from verification report",
        )
        _require(
            verified_report.get("prompt") == prompt
            and _strict_json_equal(verified_report.get("seed"), seed),
            record_context,
            "prompt or seed differs from the verification report",
        )
        _require(
            verified_report.get("metrics_path") == str(metrics_path),
            record_context,
            "metrics sidecar path differs from the verification report",
        )
        _validate_file_binding(
            verified_report.get("artifact_binding"),
            path=output_path,
            snapshot=output_snapshot,
            context=record_context,
            label="artifact_binding",
        )
        _validate_file_binding(
            verified_report.get("metrics_binding"),
            path=metrics_path,
            snapshot=metrics_snapshot,
            context=record_context,
            label="metrics_binding",
        )

        denoise_values.append(denoise)
        total_values.append(total)
        artifact_ready_values.append(artifact_ready)
        allocated_values.append(allocated)
        reserved_values.append(reserved)
        actual_shapes.add(actual_shape)
        generated_video_shapes.add(generated_video_shape)
        generated_audio_shapes.add(generated_audio_shape)
        timing_paths.add(output_path)
        artifact_hashes.append((identity, actual_hash))
        metrics_sidecar_hashes.append((identity, metrics_snapshot.sha256))

    _require(timing_paths == set(verified_by_path), context, "timing artifacts differ from verified artifacts")
    _require(
        set(prompt_by_index) == set(range(prompt_count)),
        context,
        "prompt indices do not cover the declared prompt set",
    )
    _require(
        set(seed_by_sample_index) == set(range(sample_count)),
        context,
        "sample indices do not cover the declared seed schedule",
    )
    ordered_prompts = tuple(
        prompt_by_index[index] for index in range(prompt_count)
    )
    ordered_seeds = tuple(
        seed_by_sample_index[index] for index in range(sample_count)
    )
    prompt_set_sha256 = prompt_sequence_sha256(ordered_prompts)
    _require(
        environment.get("prompts_sha256") == prompt_set_sha256,
        context,
        "ordered prompt set hash differs from environment prompts_sha256",
    )

    for warmup_index, warmup in enumerate(warmup_timings):
        warmup_context = f"{context} warmup[{warmup_index}]"
        _require(warmup.get("status") == "ok", warmup_context, "status is not ok")
        _require(
            warmup.get("record_type") == "warmup",
            warmup_context,
            "record_type is not warmup",
        )
        _require(
            warmup.get("benchmark_candidate") is True,
            warmup_context,
            "warm-up is not a benchmark candidate",
        )
        _require(
            warmup.get("benchmark_valid") is False,
            warmup_context,
            "warm-up must be excluded from benchmark statistics",
        )
        _require(
            warmup.get("warmup_index") == warmup_index,
            warmup_context,
            "warmup_index is missing, duplicated, or out of order",
        )
        _require(
            warmup.get("run_id") == environment.get("run_id"),
            warmup_context,
            "run_id differs from environment",
        )
        _require(
            warmup.get("prompt") == ordered_prompts[0],
            warmup_context,
            "warm-up prompt differs from the first fixed prompt",
        )
        _require(
            warmup.get("seed") == environment.get("seed"),
            warmup_context,
            "warm-up seed differs from the fixed base seed",
        )
        for field in (
            "sample_steps",
            "attention_method",
            "use_cfg_cache",
            "use_block_cache",
        ):
            _require(
                _values_equal(warmup.get(field), environment.get(field)),
                warmup_context,
                f"{field} differs from environment",
            )
        warmup_total = _finite_number(
            warmup,
            "total_generation_seconds",
            warmup_context,
            positive=True,
        )
        _validate_gpu_monitor(
            warmup.get("gpu_process_monitor"),
            environment,
            expected_nvidia_smi_binary,
            pre_run_gpu.get("boot_id"),
            environment.get("gpu_process_monitor_interval_seconds"),
            warmup_total,
            warmup_context,
        )
        if environment.get("attention_method") == "sparge":
            dispatcher_errors = _sparge_dispatcher_errors(
                warmup.get("video_self_attention_dispatcher"),
                environment,
                sparge_receipt,
                pre_run_gpu.get("device_uuid"),
                warmup_context,
                block_cache_saved_calls=warmup.get(
                    "block_cache_saved_video_self_attention_calls"
                ),
            )
            _require(
                not dispatcher_errors,
                warmup_context,
                "Sparge dispatcher evidence is invalid: "
                + "; ".join(dispatcher_errors),
            )
        elif environment.get("attention_method") == "radial":
            dispatcher_errors = _radial_dispatcher_errors(
                warmup.get("video_self_attention_dispatcher"),
                environment,
                copied_receipt,
                warmup_context,
                block_cache_saved_calls=warmup.get(
                    "block_cache_saved_video_self_attention_calls"
                ),
            )
            _require(
                not dispatcher_errors,
                warmup_context,
                "Radial dispatcher evidence is invalid: "
                + "; ".join(dispatcher_errors),
            )

    _require(len(actual_shapes) == 1, context, "measurements have inconsistent actual shapes")
    _require(len(generated_video_shapes) == 1, context, "measurements have inconsistent video tensor shapes")
    _require(len(generated_audio_shapes) == 1, context, "measurements have inconsistent audio tensor shapes")

    prompt = ordered_prompts[0] if prompt_count == 1 else ""
    prompt_sha256 = (
        hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if prompt_count == 1
        else ""
    )
    seed = ordered_seeds[0] if len(ordered_seeds) == 1 else ""
    actual_shape = next(iter(actual_shapes))
    generated_video_shape = next(iter(generated_video_shapes))
    generated_audio_shape = next(iter(generated_audio_shapes))
    comparison_values = {
        "git_commit": git_commit,
        "checkpoint_fingerprint_sha256": checkpoint_fingerprint,
        "gpu_identity": (
            environment.get("gpu_physical_index"),
            environment.get("gpu_uuid"),
            environment.get("gpu_name"),
        ),
        "prompt_set_sha256": prompt_set_sha256,
        "prompt_count": prompt_count,
        "prompts": ordered_prompts,
        "base_seed": environment.get("seed"),
        "sample_count": sample_count,
        "sample_seeds": ordered_seeds,
        "requested_shape": requested_shape,
        "actual_shape": actual_shape,
        "generated_video_shape": generated_video_shape,
        "generated_audio_shape": generated_audio_shape,
        "sample_steps": environment.get("sample_steps"),
    }
    summary = {
        "run_dir": str(run_dir),
        "run_id": environment.get("run_id"),
        "verification_sha256": verification_snapshot.sha256,
        "preflight_sha256": preflight_sha256,
        "timings_path": str(timings_path),
        "timings_bytes": timings_snapshot.size,
        "timings_sha256": timings_snapshot.sha256,
        "timings_record_count": len(timings),
        "warmup_timings_path": str(warmup_timings_path),
        "warmup_timings_bytes": warmup_timings_snapshot.size,
        "warmup_timings_sha256": warmup_timings_snapshot.sha256,
        "warmup_record_count": len(warmup_timings),
        "git_commit": git_commit,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "checkpoint_fingerprint_sha256": checkpoint_fingerprint,
        "gpu_uuid": environment.get("gpu_uuid"),
        "gpu_name": environment.get("gpu_name"),
        **radial_summary,
        "prompt_sha256": prompt_sha256,
        "prompt": prompt,
        "prompt_set_sha256": prompt_set_sha256,
        "prompt_count": prompt_count,
        "selected_sparse_profile": SPARSE_PROFILE_BY_RUN_KIND.get(
            environment.get("run_kind"),
            "",
        ),
        "seed": seed,
        "seed_count": len(ordered_seeds),
        "seeds": ";".join(str(value) for value in ordered_seeds),
        "requested_height": requested_shape[0],
        "requested_width": requested_shape[1],
        "actual_height": actual_shape[0],
        "actual_width": actual_shape[1],
        "sample_steps": environment.get("sample_steps"),
        "measurement_count": measurement_runs,
        "measurement_indices": ";".join(
            str(index) for index in range(measurement_runs)
        ),
        "artifact_count": len(timings),
        "denoise_seconds_median": statistics.median(denoise_values),
        "total_generation_seconds_median": statistics.median(total_values),
        "artifact_ready_seconds_median": statistics.median(artifact_ready_values),
        "peak_memory_allocated_gib_median": statistics.median(allocated_values) / GIB,
        "peak_memory_reserved_gib_median": statistics.median(reserved_values) / GIB,
        "artifact_sha256": ";".join(
            f"{identity[0]}:{identity[1]}:{identity[2]}:{artifact_hash}"
            for identity, artifact_hash in sorted(artifact_hashes)
        ),
        "metrics_sidecar_sha256": ";".join(
            f"{identity[0]}:{identity[1]}:{identity[2]}:{metrics_hash}"
            for identity, metrics_hash in sorted(metrics_sidecar_hashes)
        ),
        "engine_load_seconds": engine_load_seconds,
        "comparison_values": comparison_values,
    }
    for snapshot in stable_snapshots:
        _revalidate_snapshot(snapshot, context)
    return summary


def _pending_row(method: dict[str, Any], reason: str) -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "method_id": method["method_id"],
            "label": method["label"],
            "required": method["required"],
            "implementation_status": method["implementation_status"],
            "status": "pending",
            "timing_status": "pending",
            "pending_reason": reason,
        }
    )
    return row


def build_rows(
    manifest: dict[str, Any], mappings: dict[str, Path]
) -> list[dict[str, Any]]:
    methods = manifest["methods"]
    method_by_id = {method["method_id"]: method for method in methods}
    unknown = sorted(set(mappings) - set(method_by_id))
    _require(not unknown, "run mappings", f"unknown method ids: {unknown}")

    summaries: dict[str, dict[str, Any]] = {}
    for method in methods:
        method_id = method["method_id"]
        if method_id in mappings:
            summaries[method_id] = validate_run(
                method,
                mappings[method_id],
                manifest["fixed_protocol"],
            )

    selected_cfg = summaries.get("best_sparse_cfg")
    selected_block = summaries.get("block_cache")
    if selected_cfg is not None and selected_block is not None:
        cfg_profile = selected_cfg.get("selected_sparse_profile")
        block_profile = selected_block.get("selected_sparse_profile")
        _require(
            isinstance(cfg_profile, str)
            and cfg_profile
            and block_profile == cfg_profile,
            "run mappings",
            "best_sparse_cfg and block_cache selected different sparse profiles: "
            f"{cfg_profile!r} != {block_profile!r}",
        )

    reference_id = "dense" if "dense" in summaries else next(iter(summaries), None)
    if reference_id is not None:
        reference = summaries[reference_id]["comparison_values"]
        for method_id, summary in summaries.items():
            for field, expected in reference.items():
                actual = summary["comparison_values"].get(field)
                if actual != expected:
                    _fail(
                        f"{method_id} run",
                        f"comparison field {field}={actual!r} differs from "
                        f"{reference_id}={expected!r}",
                    )

    dense = summaries.get("dense")
    rows = []
    for method in methods:
        method_id = method["method_id"]
        if method_id not in summaries:
            reason = method.get("pending_reason") or "No explicit run mapping was provided."
            rows.append(_pending_row(method, reason))
            continue

        summary = summaries[method_id]
        row = {field: "" for field in CSV_FIELDS}
        row.update(
            {
                "method_id": method_id,
                "label": method["label"],
                "required": method["required"],
                "implementation_status": method["implementation_status"],
                # Performance evidence is valid, but absent quality/manual
                # judgments must never be represented as numeric zero or done.
                "status": "pending",
                "timing_status": "valid",
                "pending_reason": "Quality metric and manual review are not yet provided.",
            }
        )
        for field in CSV_FIELDS:
            if field in summary:
                row[field] = summary[field]
        if dense is not None:
            row["denoise_speedup_vs_dense"] = (
                dense["denoise_seconds_median"]
                / summary["denoise_seconds_median"]
            )
            row["total_speedup_vs_dense"] = (
                dense["total_generation_seconds_median"]
                / summary["total_generation_seconds_median"]
            )
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        raise EvaluationError(f"cannot write CSV {output_path}: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the fixed Ovi evaluation CSV. Runs are accepted only as "
            "explicit METHOD_ID=RUN_DIR mappings; there is no latest-run scan."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"evaluation matrix manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument("--output", type=Path, required=True, help="CSV output path")
    parser.add_argument(
        "runs",
        nargs="*",
        metavar="METHOD_ID=RUN_DIR",
        help="explicit method id to exact run directory mapping",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        allowed_ids = [method["method_id"] for method in manifest["methods"]]
        mappings = parse_run_mappings(args.runs, allowed_ids)
        rows = build_rows(manifest, mappings)
        write_csv(rows, args.output)
    except EvaluationError as exc:
        parser.error(str(exc))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
