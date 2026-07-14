#!/usr/bin/env python3
"""Build the fail-closed final A--F Ovi baseline CSV.

The performance CSV, five candidate quality medians, and five completed manual
review receipts are all explicit inputs.  This command does not discover a
"latest" result, fill a missing judgment, or overwrite an existing output.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import statistics
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = REPO_ROOT / "configs" / "quality_protocol.json"
DEFAULT_MATRIX = REPO_ROOT / "configs" / "ovi_eval_matrix.json"

METHOD_IDS = (
    "dense",
    "dense_cfg_cache",
    "sparge_topk75",
    "sparge_topk50",
    "radial_conservative",
    "radial_aggressive",
)
CANDIDATE_METHOD_IDS = METHOD_IDS[1:]
FORMAL_MATRIX_ID = "ovi_720x720_5s_a100_bf16_formal8x3_v2"
FORMAL_PROMPT_COUNT = 8
FORMAL_SAMPLE_COUNT = 3
FORMAL_MEASUREMENT_COUNT = 3
FORMAL_ARTIFACT_COUNT = 72
FORMAL_PROMPTS_SHA256 = (
    "d98397111b1ab060a61d588f4ca388c5c929430a59ac6ab49b7c2e247bb6be91"
)
IDENTITY_FIELDS = ("measurement_index", "prompt_index", "sample_index")
METRIC_FIELDS = (
    "video_psnr_db",
    "video_ssim",
    "lpips_alex",
    "audio_rmse",
    "audio_max_abs_difference",
    "audio_snr_db",
    "audio_correlation",
)
MANUAL_FIELDS = (
    "measurement_index",
    "prompt_index",
    "sample_index",
    "dense_artifact_sha256",
    "candidate_artifact_sha256",
    "reviewer",
    "reviewed_at_utc",
    "sync_rating",
    "notes",
)
ALLOWED_RATINGS = ("pass", "fail", "uncertain")
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
HEX_COMMIT = re.compile(r"^[0-9a-f]{40}$")
UTC_TIMESTAMP = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:"
    r"[0-9]{2}(?:\.[0-9]+)?Z$"
)

# This is the exact schema emitted by scripts/build_ovi_eval_csv.py.  Keeping
# it explicit makes an upstream schema change a review gate instead of silently
# discarding or reinterpreting a column.
TIMING_FIELDS = (
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

FINAL_FIELDS = (
    "schema_version",
    "method_id",
    "label",
    "status",
    "timing_status",
    "quality_status",
    "manual_review_status",
    "timing_csv_path",
    "timing_csv_sha256",
    "quality_protocol_id",
    "quality_protocol_sha256",
    "evaluation_matrix_id",
    "evaluation_matrix_sha256",
    "evaluator_git_commit",
    "run_dir",
    "run_id",
    "git_commit",
    "verification_sha256",
    "timings_sha256",
    "checkpoint_manifest_sha256",
    "checkpoint_fingerprint_sha256",
    "gpu_uuid",
    "gpu_name",
    "prompt_set_sha256",
    "prompt_count",
    "seed_count",
    "seeds",
    "sample_steps",
    "measurement_count",
    "artifact_count",
    "denoise_seconds_median",
    "total_generation_seconds_median",
    "artifact_ready_seconds_median",
    "peak_memory_allocated_gib_median",
    "peak_memory_reserved_gib_median",
    "denoise_speedup_vs_dense",
    "total_speedup_vs_dense",
    "video_psnr_db_median",
    "video_ssim_median",
    "lpips_alex_median",
    "audio_rmse_median",
    "audio_max_abs_difference_median",
    "audio_snr_db_median",
    "audio_correlation_median",
    "quality_median_path",
    "quality_median_sha256",
    "manual_review_row_count",
    "manual_pass_count",
    "manual_fail_count",
    "manual_uncertain_count",
    "manual_validation_path",
    "manual_validation_sha256",
    "manual_reviews_csv_path",
    "manual_reviews_csv_sha256",
)

QUALITY_FIELDS = {
    "schema_version",
    "record_type",
    "quality_protocol_id",
    "quality_protocol_sha256",
    "comparison_script_sha256",
    "compare_media_script_sha256",
    "run_validator_script_sha256",
    "evaluation_matrix_sha256",
    "evaluator_source_receipt",
    "lpips_dependency_receipt",
    "media_tool_receipt",
    "dense_run",
    "candidate_run",
    "pairs",
    "pair_count",
    "metric_medians",
    "automatic_acceptance",
    "manual_review",
}
QUALITY_PAIR_BINDING_FIELDS = {
    *IDENTITY_FIELDS,
    "pair_sidecar_path",
    "pair_sidecar_sha256",
    "dense_artifact_sha256",
    "candidate_artifact_sha256",
}
PAIR_SIDECAR_FIELDS = {
    "schema_version",
    "record_type",
    "quality_protocol_id",
    "quality_protocol_sha256",
    *IDENTITY_FIELDS,
    "dense",
    "candidate",
    "metrics",
    "automatic_acceptance",
    "comparison_script_sha256",
    "compare_media_script_sha256",
    "run_validator_script_sha256",
    "evaluation_matrix_sha256",
    "evaluator_source_receipt",
    "lpips_dependency_receipt",
    "media_tool_receipt",
}
PAIR_METRIC_FIELDS = {
    "compared_video_frames",
    "reference_audio_samples",
    "candidate_audio_samples",
    "audio_sample_count_compared",
    "lpips_frame_count",
    *METRIC_FIELDS,
}
RUN_BINDING_FIELDS = {
    "method_id",
    "run_dir",
    "run_id",
    "verification_sha256",
    "timings_path",
    "timings_bytes",
    "timings_sha256",
    "timings_record_count",
    "warmup_timings_path",
    "warmup_timings_bytes",
    "warmup_timings_sha256",
    "warmup_record_count",
    "environment_sha256",
    "git_commit",
    "checkpoint_manifest_sha256",
    "checkpoint_fingerprint_sha256",
    "gpu_physical_index",
    "gpu_uuid",
    "gpu_name",
    "prompt_set_sha256",
    "prompt_count",
    "prompts",
    "base_seed",
    "sample_count",
    "sample_seeds",
    "selected_sparse_profile",
    "requested_shape",
    "actual_shape",
    "generated_video_shape",
    "generated_audio_shape",
    "sample_steps",
    "acceleration_environment",
    "evidence_bindings",
}
ARTIFACT_BINDING_FIELDS = RUN_BINDING_FIELDS | {
    *IDENTITY_FIELDS,
    "artifact_path",
    "artifact_sha256",
    "metrics_sidecar_path",
    "metrics_sidecar_sha256",
}
MANUAL_RECEIPT_FIELDS = {
    "schema_version",
    "record_type",
    "quality_protocol_id",
    "quality_protocol_sha256",
    "quality_median_path",
    "quality_median_sha256",
    "manual_reviews_csv_path",
    "manual_reviews_csv_sha256",
    "manual_review_status",
    "manual_review_row_count",
    "pairs",
}
MANUAL_PAIR_FIELDS = {
    *IDENTITY_FIELDS,
    "dense_artifact_sha256",
    "candidate_artifact_sha256",
}
SOURCE_ROLES = {
    "comparison_script": REPO_ROOT / "scripts" / "compare_ovi_quality.py",
    "compare_media_script": REPO_ROOT / "scripts" / "compare_media.py",
    "run_validator_script": REPO_ROOT / "scripts" / "build_ovi_eval_csv.py",
    "archive_url_policy": REPO_ROOT / "scripts" / "quality_archive_urls.py",
    "quality_protocol": DEFAULT_PROTOCOL,
    "evaluation_matrix": DEFAULT_MATRIX,
}
LOCAL_VALIDATOR_DEPENDENCIES = (
    REPO_ROOT / "ovi" / "__init__.py",
    REPO_ROOT / "ovi" / "gpu_process_monitor.py",
    REPO_ROOT / "ovi" / "eval_protocol.py",
    REPO_ROOT / "ovi" / "sparge_evidence.py",
    REPO_ROOT / "ovi" / "radial_evidence.py",
)
DEPENDENCY_LOCK_FIELDS = {
    "distribution",
    "version",
    "source_index",
    "archive_url",
    "archive_sha256",
}
LPIPS_INLINE_RECEIPT_FIELDS = {
    "receipt_path",
    "receipt_sha256",
    "environment_root",
    "python_executable",
    "sys_prefix",
    "python_version",
    "runtime_contract",
    "environment_lock_sha256",
    "packages",
    "weights",
}
LPIPS_RAW_RECEIPT_FIELDS = {
    "schema_version",
    "created_by",
    "environment_root",
    "python_executable",
    "sys_prefix",
    "python_version",
    "runtime_contract",
    "environment_lock_sha256",
    "installer_reports",
    "packages",
    "weights",
}

class FinalCsvError(ValueError):
    """Raised when any evidence required for the final table is incomplete."""


def _fail(context: str, message: str) -> None:
    raise FinalCsvError(f"{context}: {message}")


def _require(condition: bool, context: str, message: str) -> None:
    if not condition:
        _fail(context, message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _json_equal(left: Any, right: Any) -> bool:
    """JSON equality without Python's True == 1 coercion."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _sha(value: Any, context: str, field: str) -> str:
    _require(
        isinstance(value, str) and HEX_SHA256.fullmatch(value) is not None,
        context,
        f"{field} must be a lowercase full SHA256",
    )
    return value


def _commit(value: Any, context: str, field: str) -> str:
    _require(
        isinstance(value, str) and HEX_COMMIT.fullmatch(value) is not None,
        context,
        f"{field} must be a lowercase full Git commit",
    )
    return value


def _canonical_path(raw: str | os.PathLike[str], context: str) -> Path:
    value = os.fspath(raw)
    _require(isinstance(value, str) and bool(value), context, "path is missing")
    _require(os.path.isabs(value), context, f"path must be absolute: {value!r}")
    _require(
        os.path.normpath(value) == value,
        context,
        f"path is not lexically canonical: {value!r}",
    )
    _require(
        os.path.realpath(value) == value,
        context,
        f"path resolves through a symlink or alias: {value!r}",
    )
    return Path(value)


def _canonical_directory(raw: str | os.PathLike[str], context: str) -> Path:
    path = _canonical_path(raw, context)
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        _fail(context, f"cannot lstat directory {path}: {exc}")
    _require(stat.S_ISDIR(metadata.st_mode), context, f"not a directory: {path}")
    return path


@dataclass(frozen=True)
class StableSnapshot:
    path: Path
    data: bytes
    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    def revalidate(self, context: str) -> None:
        try:
            current = os.lstat(self.path)
        except OSError as exc:
            _fail(context, f"cannot re-stat {self.path}: {exc}")
        expected = (
            self.device,
            self.inode,
            self.size,
            self.mtime_ns,
            self.ctime_ns,
        )
        observed = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        _require(
            stat.S_ISREG(current.st_mode) and observed == expected,
            context,
            f"input changed after its stable snapshot: {self.path}",
        )


class SnapshotRegistry:
    def __init__(self) -> None:
        self._by_path: dict[Path, StableSnapshot] = {}

    @property
    def snapshots(self) -> tuple[StableSnapshot, ...]:
        return tuple(self._by_path.values())

    def file(self, raw: str | os.PathLike[str], context: str) -> StableSnapshot:
        path = _canonical_path(raw, context)
        cached = self._by_path.get(path)
        if cached is not None:
            cached.revalidate(context)
            return cached
        try:
            before = os.lstat(path)
        except OSError as exc:
            _fail(context, f"cannot lstat {path}: {exc}")
        _require(stat.S_ISREG(before.st_mode), context, f"not a regular file: {path}")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            _fail(context, f"cannot open no-follow input {path}: {exc}")
        try:
            opened = os.fstat(descriptor)
            _require(stat.S_ISREG(opened.st_mode), context, f"not a regular file: {path}")
            _require(
                (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
                context,
                f"file identity changed while opening {path}",
            )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        identity_before = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        _require(identity_before == identity_after, context, f"file changed while reading {path}")
        data = b"".join(chunks)
        _require(len(data) == after.st_size, context, f"short read from {path}")
        snapshot = StableSnapshot(
            path=path,
            data=data,
            sha256=hashlib.sha256(data).hexdigest(),
            device=after.st_dev,
            inode=after.st_ino,
            size=after.st_size,
            mtime_ns=after.st_mtime_ns,
            ctime_ns=after.st_ctime_ns,
        )
        snapshot.revalidate(context)
        self._by_path[path] = snapshot
        return snapshot


def _reject_duplicate_primary_inputs(
    snapshots: Iterable[StableSnapshot],
) -> None:
    paths: set[Path] = set()
    identities: set[tuple[int, int]] = set()
    hashes: set[str] = set()
    for snapshot in snapshots:
        context = "primary inputs"
        _require(snapshot.path not in paths, context, f"duplicate path {snapshot.path}")
        identity = (snapshot.device, snapshot.inode)
        _require(identity not in identities, context, f"duplicate file identity {snapshot.path}")
        _require(snapshot.sha256 not in hashes, context, f"duplicate file SHA256 {snapshot.path}")
        paths.add(snapshot.path)
        identities.add(identity)
        hashes.add(snapshot.sha256)


def _strict_json(snapshot: StableSnapshot, context: str) -> Any:
    try:
        text = snapshot.data.decode("utf-8")
    except UnicodeDecodeError as exc:
        _fail(context, f"JSON is not UTF-8: {exc}")

    def pairs_hook(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(context, f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        _fail(context, f"non-finite JSON constant {value!r} is forbidden")

    try:
        return json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, RecursionError) as exc:
        _fail(context, f"invalid JSON: {exc}")


def _strict_csv(snapshot: StableSnapshot, context: str) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    try:
        text = snapshot.data.decode("utf-8")
    except UnicodeDecodeError as exc:
        _fail(context, f"CSV is not UTF-8: {exc}")
    _require("\x00" not in text, context, "CSV contains a NUL byte")
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        rows = list(reader)
    except csv.Error as exc:
        _fail(context, f"invalid CSV: {exc}")
    fields = tuple(reader.fieldnames or ())
    for number, row in enumerate(rows, start=2):
        _require(None not in row, context, f"row {number} contains extra fields")
        _require(
            all(isinstance(value, str) for value in row.values()),
            context,
            f"row {number} contains a missing field",
        )
    return fields, rows


def _canonical_uint_cell(value: str, context: str, field: str, *, positive: bool = False) -> int:
    _require(
        isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value) is not None,
        context,
        f"{field} must be a canonical unsigned integer",
    )
    result = int(value)
    if positive:
        _require(result > 0, context, f"{field} must be positive")
    return result


def _positive_float_cell(value: str, context: str, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        _fail(context, f"{field} must be numeric")
    _require(math.isfinite(result) and result > 0.0, context, f"{field} must be finite and positive")
    return result


def _parse_artifact_hashes(
    value: str,
    expected_identities: set[tuple[int, int, int]],
    context: str,
) -> dict[tuple[int, int, int], str]:
    result: dict[tuple[int, int, int], str] = {}
    for item in value.split(";") if value else ():
        parts = item.split(":")
        _require(len(parts) == 4, context, f"invalid artifact hash binding {item!r}")
        identity_values = []
        for offset, raw in enumerate(parts[:3]):
            identity_values.append(
                _canonical_uint_cell(raw, context, IDENTITY_FIELDS[offset])
            )
        identity = tuple(identity_values)
        _require(identity in expected_identities, context, f"identity outside protocol: {identity!r}")
        _require(identity not in result, context, f"duplicate artifact identity {identity!r}")
        result[identity] = _sha(parts[3], context, "artifact SHA256")
    _require(set(result) == expected_identities, context, "artifact hashes do not cover the fixed protocol")
    return result


def _canonical_distribution(value: Any, context: str) -> str:
    _require(isinstance(value, str) and bool(value), context, "distribution is missing")
    normalized = re.sub(r"[-_.]+", "-", value).lower()
    _require(
        value == normalized
        and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value) is not None,
        context,
        f"distribution name is not canonical: {value!r}",
    )
    return value


def _validate_dependency_url(
    value: Any, source_index: Any, context: str
) -> str:
    _require(isinstance(value, str) and value, context, "archive URL is missing")
    parsed = urlparse(value)
    _require(
        parsed.scheme == "https"
        and parsed.username is None
        and parsed.password is None
        and parsed.port is None
        and not parsed.query
        and not parsed.fragment,
        context,
        "archive URL must be canonical HTTPS without credentials, query, or fragment",
    )
    if source_index == "https://pypi.org/simple":
        _require(
            parsed.hostname == "files.pythonhosted.org"
            and parsed.path.startswith("/packages/"),
            context,
            "PyPI archive URL escaped files.pythonhosted.org/packages",
        )
    elif source_index == "https://download.pytorch.org/whl/cpu":
        _require(
            parsed.hostname
            in {"download.pytorch.org", "download-r2.pytorch.org"}
            and parsed.path.startswith("/whl/cpu/"),
            context,
            "PyTorch archive URL escaped the CPU wheel namespace",
        )
    else:
        _fail(context, f"unapproved source index {source_index!r}")
    filename = Path(unquote(parsed.path)).name
    _require(filename.endswith(".whl"), context, "dependency archive is not a wheel")
    return filename


def _normalize_dependency_lock(
    records: Any, context: str
) -> list[dict[str, str]]:
    _require(isinstance(records, list) and bool(records), context, "dependency lock is empty")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for offset, item in enumerate(records):
        item_context = f"{context} package {offset}"
        _require(
            isinstance(item, dict) and set(item) == DEPENDENCY_LOCK_FIELDS,
            item_context,
            "dependency lock field set changed",
        )
        distribution = _canonical_distribution(item.get("distribution"), item_context)
        _require(distribution not in seen, item_context, "duplicate distribution")
        seen.add(distribution)
        version = item.get("version")
        _require(isinstance(version, str) and version, item_context, "version is missing")
        source_index = item.get("source_index")
        archive_url = item.get("archive_url")
        _validate_dependency_url(archive_url, source_index, item_context)
        normalized.append(
            {
                "distribution": distribution,
                "version": version,
                "source_index": source_index,
                "archive_url": archive_url,
                "archive_sha256": _sha(
                    item.get("archive_sha256"), item_context, "archive_sha256"
                ),
            }
        )
    normalized.sort(key=lambda item: item["distribution"])
    _require(
        _json_equal(records, normalized),
        context,
        "dependency lock must be in canonical distribution order",
    )
    return normalized


def _dependency_lock_sha256(records: Sequence[Mapping[str, str]]) -> str:
    rendered = json.dumps(
        list(records),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _validate_pinned_lpips_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    context = "quality protocol LPIPS trust"
    lpips = protocol.get("lpips")
    _require(isinstance(lpips, dict), context, "LPIPS contract is missing")
    _require(
        lpips.get("trusted_lock_status") == "pinned",
        context,
        "trusted_lock_status must be pinned",
    )
    trusted_lock = _normalize_dependency_lock(
        lpips.get("trusted_environment_packages"), context
    )
    trusted_lock_hash = _sha(
        lpips.get("trusted_environment_lock_sha256"),
        context,
        "trusted_environment_lock_sha256",
    )
    _require(
        _dependency_lock_sha256(trusted_lock) == trusted_lock_hash,
        context,
        "trusted environment lock payload differs from its SHA256",
    )
    locked_by_distribution = {
        item["distribution"]: item for item in trusted_lock
    }
    packages = lpips.get("packages")
    _require(isinstance(packages, list) and bool(packages), context, "direct package contracts are missing")
    direct_distributions: set[str] = set()
    for offset, package in enumerate(packages):
        package_context = f"{context} direct package {offset}"
        _require(isinstance(package, dict), package_context, "package contract is not an object")
        distribution = _canonical_distribution(package.get("distribution"), package_context)
        _require(distribution not in direct_distributions, package_context, "duplicate direct package")
        direct_distributions.add(distribution)
        _require(distribution in locked_by_distribution, package_context, "package is absent from full lock")
        trusted_archive = _sha(
            package.get("trusted_archive_sha256"),
            package_context,
            "trusted_archive_sha256",
        )
        locked = locked_by_distribution[distribution]
        _require(
            package.get("version") == locked["version"]
            and package.get("source_index") == locked["source_index"]
            and trusted_archive == locked["archive_sha256"],
            package_context,
            "direct package contract differs from full lock",
        )
        _require(
            isinstance(package.get("module"), str)
            and package["module"]
            and isinstance(package.get("module_path"), str)
            and package["module_path"],
            package_context,
            "direct module binding is missing",
        )
    weights = lpips.get("weights")
    _require(isinstance(weights, list) and bool(weights), context, "weight contracts are missing")
    weight_ids: set[str] = set()
    for offset, weight in enumerate(weights):
        weight_context = f"{context} weight {offset}"
        _require(isinstance(weight, dict), weight_context, "weight contract is not an object")
        weight_id = weight.get("weight_id")
        _require(isinstance(weight_id, str) and weight_id and weight_id not in weight_ids, weight_context, "weight id is invalid or duplicate")
        weight_ids.add(weight_id)
        _sha(weight.get("trusted_sha256"), weight_context, "trusted_sha256")
        for field in ("path", "source_type", "source"):
            _require(isinstance(weight.get(field), str) and weight[field], weight_context, f"{field} is missing")
    for field in ("receipt_path", "environment_root", "python_executable"):
        _require(isinstance(lpips.get(field), str) and lpips[field], context, f"{field} is missing")
    return lpips


def _validate_lpips_disk_receipt(
    inline: Any,
    lpips: Mapping[str, Any],
    registry: SnapshotRegistry,
    context: str,
) -> dict[str, Any]:
    _require(
        isinstance(inline, dict)
        and set(inline) == LPIPS_INLINE_RECEIPT_FIELDS,
        context,
        "normalized LPIPS receipt field set changed",
    )
    receipt_path = _canonical_path(lpips["receipt_path"], context)
    _require(
        inline.get("receipt_path") == str(receipt_path),
        context,
        "LPIPS receipt path differs from fixed protocol",
    )
    receipt_snapshot = registry.file(receipt_path, context)
    _require(
        receipt_snapshot.sha256
        == _sha(inline.get("receipt_sha256"), context, "receipt_sha256"),
        context,
        "fixed LPIPS receipt SHA256 drifted",
    )
    raw = _strict_json(receipt_snapshot, context)
    _require(
        isinstance(raw, dict) and set(raw) == LPIPS_RAW_RECEIPT_FIELDS,
        context,
        "fixed LPIPS receipt field set changed",
    )
    _require(raw.get("schema_version") == 2, context, "receipt schema_version must be 2")
    _require(
        raw.get("created_by") == "scripts/install_ovi_quality_env.sh",
        context,
        "receipt creator is not the fixed installer",
    )
    expected_environment = lpips["environment_root"]
    expected_python = lpips["python_executable"]
    expected_runtime = {
        "python_arguments": ["-I", "-S", "-B"],
        "python_minor": "3.11",
        "site_packages": str(
            Path(expected_environment)
            / "lib"
            / "python3.11"
            / "site-packages"
        ),
    }
    for field, expected in (
        ("environment_root", expected_environment),
        ("python_executable", expected_python),
        ("sys_prefix", expected_environment),
        ("runtime_contract", expected_runtime),
        (
            "environment_lock_sha256",
            lpips["trusted_environment_lock_sha256"],
        ),
    ):
        _require(_json_equal(raw.get(field), expected), context, f"raw receipt {field} differs from protocol")
        _require(_json_equal(inline.get(field), expected), context, f"inline receipt {field} differs from protocol")
    python_version = raw.get("python_version")
    _require(
        isinstance(python_version, str)
        and re.fullmatch(r"3\.11\.[0-9]+", python_version) is not None
        and inline.get("python_version") == python_version,
        context,
        "receipt Python version is not one exact 3.11.x value",
    )

    installer_reports = raw.get("installer_reports")
    _require(
        isinstance(installer_reports, list) and len(installer_reports) == 3,
        context,
        "receipt must bind the three fixed pip reports",
    )
    report_paths: set[Path] = set()
    for offset, report in enumerate(installer_reports):
        report_context = f"{context} installer report {offset}"
        _require(
            isinstance(report, dict) and set(report) == {"path", "sha256"},
            report_context,
            "installer report binding changed",
        )
        report_snapshot = registry.file(report.get("path"), report_context)
        _require(report_snapshot.path not in report_paths, report_context, "duplicate installer report")
        report_paths.add(report_snapshot.path)
        _require(
            report_snapshot.sha256
            == _sha(report.get("sha256"), report_context, "sha256"),
            report_context,
            "installer report SHA256 drifted",
        )

    trusted_lock = _normalize_dependency_lock(
        lpips["trusted_environment_packages"], context
    )
    raw_packages = raw.get("packages")
    _require(isinstance(raw_packages, list), context, "raw package list is missing")
    _require(
        _json_equal(inline.get("packages"), raw_packages),
        context,
        "inline package receipt differs from fixed disk receipt",
    )
    _require(
        [item.get("distribution") for item in raw_packages if isinstance(item, dict)]
        == [item["distribution"] for item in trusted_lock],
        context,
        "raw package set or order differs from trusted full lock",
    )
    lock_by_distribution = {item["distribution"]: item for item in trusted_lock}
    direct_by_distribution = {
        item["distribution"]: item for item in lpips["packages"]
    }
    environment_root = Path(expected_environment)
    wheelhouse = environment_root.parent.parent / "checkpoints" / "eval" / "wheels"
    site_packages = environment_root / "lib" / "python3.11" / "site-packages"
    archive_paths: set[Path] = set()
    record_paths: set[Path] = set()
    for offset, package in enumerate(raw_packages):
        package_context = f"{context} installed package {offset}"
        _require(isinstance(package, dict), package_context, "package record is not an object")
        distribution = _canonical_distribution(package.get("distribution"), package_context)
        expected_fields = {
            *DEPENDENCY_LOCK_FIELDS,
            "archive_path",
            "record_path",
            "record_sha256",
        }
        direct = direct_by_distribution.get(distribution)
        if direct is not None:
            expected_fields |= {"module", "module_path", "module_sha256"}
        _require(set(package) == expected_fields, package_context, "package receipt field set changed")
        locked = lock_by_distribution.get(distribution)
        _require(locked is not None, package_context, "package is absent from trusted full lock")
        for field in DEPENDENCY_LOCK_FIELDS:
            _require(_json_equal(package.get(field), locked[field]), package_context, f"{field} differs from trusted lock")
        filename = _validate_dependency_url(
            package.get("archive_url"), package.get("source_index"), package_context
        )
        archive_snapshot = registry.file(package.get("archive_path"), package_context)
        _require(
            archive_snapshot.path.parent == wheelhouse
            and archive_snapshot.path.name == filename
            and archive_snapshot.path not in archive_paths,
            package_context,
            "retained wheel path escaped fixed wheelhouse or is duplicate",
        )
        archive_paths.add(archive_snapshot.path)
        _require(archive_snapshot.sha256 == locked["archive_sha256"], package_context, "retained wheel SHA256 differs from trusted lock")
        record_snapshot = registry.file(package.get("record_path"), package_context)
        try:
            record_snapshot.path.relative_to(site_packages)
        except ValueError:
            _fail(package_context, "installed RECORD escaped fixed site-packages")
        _require(record_snapshot.path.name == "RECORD" and record_snapshot.path not in record_paths, package_context, "installed RECORD path is invalid or duplicate")
        record_paths.add(record_snapshot.path)
        _require(record_snapshot.sha256 == _sha(package.get("record_sha256"), package_context, "record_sha256"), package_context, "installed RECORD SHA256 drifted")
        if direct is not None:
            _require(package.get("module") == direct.get("module") and package.get("module_path") == direct.get("module_path"), package_context, "direct module binding differs from protocol")
            module_snapshot = registry.file(package.get("module_path"), package_context)
            _require(module_snapshot.sha256 == _sha(package.get("module_sha256"), package_context, "module_sha256"), package_context, "direct module SHA256 drifted")

    raw_weights = raw.get("weights")
    _require(isinstance(raw_weights, list), context, "raw weight list is missing")
    _require(_json_equal(inline.get("weights"), raw_weights), context, "inline weights differ from fixed disk receipt")
    expected_weights = {item["weight_id"]: item for item in lpips["weights"]}
    _require(
        [item.get("weight_id") for item in raw_weights if isinstance(item, dict)]
        == [item["weight_id"] for item in lpips["weights"]],
        context,
        "weight set or order differs from protocol",
    )
    weight_paths: set[Path] = set()
    for offset, weight in enumerate(raw_weights):
        weight_context = f"{context} weight {offset}"
        _require(isinstance(weight, dict), weight_context, "weight record is not an object")
        weight_id = weight.get("weight_id")
        expected = expected_weights[weight_id]
        expected_fields = {
            "weight_id",
            "path",
            "bytes",
            "sha256",
            "source_type",
            "source",
        }
        for optional in ("source_distribution", "source_version"):
            if optional in expected:
                expected_fields.add(optional)
        _require(set(weight) == expected_fields, weight_context, "weight receipt field set changed")
        for field in expected_fields - {"bytes", "sha256"}:
            _require(_json_equal(weight.get(field), expected.get(field)), weight_context, f"{field} differs from protocol")
        weight_snapshot = registry.file(weight.get("path"), weight_context)
        _require(weight_snapshot.path not in weight_paths, weight_context, "duplicate weight path")
        weight_paths.add(weight_snapshot.path)
        _require(
            weight_snapshot.size == _require_json_int(weight.get("bytes"), weight_context, "bytes", positive=True)
            and weight_snapshot.sha256 == _sha(weight.get("sha256"), weight_context, "sha256")
            and weight_snapshot.sha256 == expected["trusted_sha256"],
            weight_context,
            "weight bytes or SHA256 differs from pinned trust root",
        )
    return inline


def _full_validate_lpips_environment(
    lpips: Mapping[str, Any],
    inline: Mapping[str, Any],
    registry: SnapshotRegistry,
    context: str,
) -> None:
    _require_fixed_eval_runtime(lpips, context)
    _full_validate_lpips_environment_body(
        lpips,
        inline,
        registry,
        context,
    )


def _require_fixed_eval_runtime(
    lpips: Mapping[str, Any],
    context: str,
) -> None:
    expected_python = Path(lpips["python_executable"]).resolve()
    expected_prefix = Path(lpips["environment_root"]).resolve()
    _require(
        Path(sys.executable).resolve() == expected_python
        and Path(sys.prefix).resolve() == expected_prefix,
        context,
        "final merger must run under the fixed LPIPS Python executable and prefix",
    )
    for field in (
        "isolated",
        "no_site",
        "dont_write_bytecode",
        "no_user_site",
        "ignore_environment",
        "safe_path",
    ):
        _require(
            getattr(sys.flags, field, 0) == 1,
            context,
            f"full LPIPS audit requires Python -I -S -B ({field}=1)",
        )


def _full_validate_lpips_environment_body(
    lpips: Mapping[str, Any],
    inline: Mapping[str, Any],
    registry: SnapshotRegistry,
    context: str,
) -> None:
    source_path = SOURCE_ROLES["comparison_script"].resolve()
    try:
        source_snapshot = registry.file(source_path, context)
        code = compile(
            source_snapshot.data,
            str(source_path),
            "exec",
            dont_inherit=True,
        )
    except (OSError, SyntaxError) as exc:
        _fail(context, f"cannot compile fixed quality validator: {exc}")
    module_name = "_fasta2v_final_full_quality_validator"
    from types import ModuleType

    module = ModuleType(module_name)
    module.__file__ = str(source_path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
        validated = module.validate_lpips_receipt(
            lpips,
            receipt_path=Path(lpips["receipt_path"]),
        )
    except Exception as exc:
        _fail(context, f"checked-in full LPIPS environment audit failed: {exc}")
    finally:
        sys.modules.pop(module_name, None)
    _require(
        _json_equal(validated, inline),
        context,
        "full LPIPS environment validation differs from quality receipt",
    )


def _load_fixed_module(
    source_path: Path,
    registry: SnapshotRegistry,
    module_name: str,
    context: str,
) -> Any:
    """Compile one already-snapshotted checked-in validator without rereading it."""

    from types import ModuleType

    source_path = source_path.resolve()
    run_validator_path = SOURCE_ROLES["run_validator_script"].resolve()
    loading_run_validator = source_path == run_validator_path
    if loading_run_validator:
        for loaded_name in tuple(sys.modules):
            if loaded_name == "ovi" or loaded_name.startswith("ovi."):
                sys.modules.pop(loaded_name, None)
    try:
        source_snapshot = registry.file(source_path, context)
        code = compile(
            source_snapshot.data,
            str(source_path),
            "exec",
            dont_inherit=True,
        )
    except (OSError, SyntaxError) as exc:
        _fail(context, f"cannot compile fixed validator: {exc}")
    module = ModuleType(module_name)
    module.__file__ = str(source_path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
    except Exception as exc:
        _fail(context, f"cannot load fixed validator: {exc}")
    finally:
        sys.modules.pop(module_name, None)
    if loading_run_validator:
        expected_modules = {
            "ovi": LOCAL_VALIDATOR_DEPENDENCIES[0],
            "ovi.gpu_process_monitor": LOCAL_VALIDATOR_DEPENDENCIES[1],
            "ovi.eval_protocol": LOCAL_VALIDATOR_DEPENDENCIES[2],
            "ovi.sparge_evidence": LOCAL_VALIDATOR_DEPENDENCIES[3],
            "ovi.radial_evidence": LOCAL_VALIDATOR_DEPENDENCIES[4],
        }
        for loaded_name, expected_path in expected_modules.items():
            loaded = sys.modules.get(loaded_name)
            loaded_path = getattr(loaded, "__file__", None)
            _require(
                isinstance(loaded_path, str)
                and Path(loaded_path).resolve() == expected_path,
                context,
                f"{loaded_name} was not imported from the fixed source closure",
            )
            registry.file(expected_path, context)
    return module


def _audit_repository_source(
    registry: SnapshotRegistry,
    context: str,
) -> str:
    """Require the complete repository worktree and validator closure at HEAD."""

    def git(*arguments: str, binary: bool = False) -> Any:
        try:
            return subprocess.run(
                ["git", *arguments],
                cwd=REPO_ROOT,
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=not binary,
                timeout=60,
            ).stdout
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            _fail(context, f"cannot audit repository source: {exc}")

    root = git("rev-parse", "--show-toplevel").strip()
    _require(root == str(REPO_ROOT), context, "validator repository root differs")
    commit = git("rev-parse", "HEAD").strip()
    _commit(commit, context, "repository HEAD")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    _require(
        not status.strip(),
        context,
        "entire validator repository must be tracked and clean at HEAD",
    )

    for directory, directory_names, filenames in os.walk(
        REPO_ROOT, followlinks=False
    ):
        if ".git" in directory_names:
            directory_names.remove(".git")
        _require(
            "__pycache__" not in directory_names,
            context,
            f"repository contains forbidden __pycache__: {directory}",
        )
        for filename in filenames:
            _require(
                not filename.endswith((".pyc", ".pyo")),
                context,
                f"repository contains forbidden bytecode: {Path(directory) / filename}",
            )

    source_paths = {
        REPO_ROOT / "scripts" / "build_ovi_final_csv.py",
        *(Path(path) for path in SOURCE_ROLES.values()),
        *LOCAL_VALIDATOR_DEPENDENCIES,
    }
    for path in sorted(source_paths):
        source_context = f"{context} {path.name}"
        snapshot = registry.file(path, source_context)
        try:
            relative = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            _fail(source_context, "validator source escaped repository")
        tracked = git("ls-files", "--error-unmatch", "--", relative).strip()
        _require(tracked == relative, source_context, "source is not tracked exactly")
        head_bytes = git("show", f"HEAD:{relative}", binary=True)
        _require(
            snapshot.data == head_bytes,
            source_context,
            "source bytes differ from the HEAD blob",
        )
    return commit


def _csv_cell(value: Any) -> str:
    return "" if value is None else str(value)


def _full_validate_timing_runs(
    matrix: Mapping[str, Any],
    timing_rows: Mapping[str, Mapping[str, str]],
    registry: SnapshotRegistry,
    context: str,
) -> None:
    """Rerun the fixed run validator and bind every emitted CSV cell."""

    validator = _load_fixed_module(
        SOURCE_ROLES["run_validator_script"],
        registry,
        "_fasta2v_final_full_run_validator",
        context,
    )
    _require(
        tuple(getattr(validator, "CSV_FIELDS", ())) == TIMING_FIELDS,
        context,
        "fixed run validator CSV schema differs from final merger",
    )
    methods = {item["method_id"]: item for item in matrix["methods"]}
    summaries: dict[str, Mapping[str, Any]] = {}
    for method_id in METHOD_IDS:
        row = timing_rows[method_id]
        try:
            summary = validator.validate_run(
                methods[method_id],
                Path(row["run_dir"]),
                matrix["fixed_protocol"],
            )
        except Exception as exc:
            _fail(
                f"{context} {method_id}",
                f"fixed build_ovi_eval_csv.validate_run failed: {exc}",
            )
        _require(
            isinstance(summary, dict),
            f"{context} {method_id}",
            "fixed run validator did not return a summary",
        )
        summaries[method_id] = summary

    dense = summaries["dense"]
    for method_id in METHOD_IDS:
        row = timing_rows[method_id]
        method = methods[method_id]
        summary = summaries[method_id]
        expected = {field: "" for field in TIMING_FIELDS}
        expected.update(
            {
                "method_id": method_id,
                "label": method["label"],
                "required": _csv_cell(method["required"]),
                "implementation_status": method["implementation_status"],
                "status": "pending",
                "timing_status": "valid",
                "pending_reason": (
                    "Quality metric and manual review are not yet provided."
                ),
            }
        )
        for field in TIMING_FIELDS:
            if field in summary:
                expected[field] = _csv_cell(summary[field])
        expected["denoise_speedup_vs_dense"] = _csv_cell(
            dense["denoise_seconds_median"]
            / summary["denoise_seconds_median"]
        )
        expected["total_speedup_vs_dense"] = _csv_cell(
            dense["total_generation_seconds_median"]
            / summary["total_generation_seconds_median"]
        )
        for field in TIMING_FIELDS:
            _require(
                row[field] == expected[field],
                f"{context} {method_id}",
                f"timing CSV field {field} differs from fixed validate_run output",
            )


def _full_validate_quality_protocol(
    protocol: Mapping[str, Any],
    protocol_snapshot: StableSnapshot,
    registry: SnapshotRegistry,
    context: str,
) -> None:
    """Run the fixed evaluator's complete protocol validator."""

    validator = _load_fixed_module(
        SOURCE_ROLES["comparison_script"],
        registry,
        "_fasta2v_final_protocol_validator",
        context,
    )
    try:
        validated, validated_sha256 = validator.load_quality_protocol(
            Path(DEFAULT_PROTOCOL)
        )
    except Exception as exc:
        _fail(context, f"fixed quality protocol validation failed: {exc}")
    _require(
        _json_equal(validated, protocol)
        and validated_sha256 == protocol_snapshot.sha256,
        context,
        "fixed evaluator protocol result differs from the snapshotted protocol",
    )


def _recollect_media_receipt(
    receipt: Mapping[str, Any],
    registry: SnapshotRegistry,
    context: str,
) -> None:
    """Collect the active tools through the fixed evaluator and require equality."""

    validator = _load_fixed_module(
        SOURCE_ROLES["comparison_script"],
        registry,
        "_fasta2v_final_media_receipt_collector",
        context,
    )
    try:
        current = validator.collect_media_tool_receipt()
    except Exception as exc:
        _fail(context, f"fixed media receipt collection failed: {exc}")
    _require(
        _json_equal(current, receipt),
        context,
        "submitted media receipt differs from freshly collected ffmpeg/ffprobe receipt",
    )


def _recollect_evaluator_source(
    receipt: Mapping[str, Any],
    timing_rows: Mapping[str, Mapping[str, str]],
    repository_commit: str,
    registry: SnapshotRegistry,
    context: str,
) -> None:
    """Require a fresh clean-HEAD source receipt from the fixed evaluator."""

    validator = _load_fixed_module(
        SOURCE_ROLES["comparison_script"],
        registry,
        "_fasta2v_final_source_receipt_collector",
        context,
    )
    try:
        current = validator.capture_evaluator_source_receipt(
            Path(DEFAULT_PROTOCOL),
            Path(DEFAULT_MATRIX),
        )
    except Exception as exc:
        _fail(context, f"fixed evaluator source recollection failed: {exc}")
    _require(
        _json_equal(current, receipt),
        context,
        "submitted evaluator receipt differs from clean tracked HEAD sources",
    )
    _require(
        current.get("git_commit") == repository_commit,
        context,
        "evaluator receipt commit differs from audited repository HEAD",
    )
    for method_id in METHOD_IDS:
        _require(
            timing_rows[method_id]["git_commit"] == repository_commit,
            context,
            f"{method_id} timing commit differs from audited repository HEAD",
        )


def _recompute_quality_metrics(
    qualities: Mapping[str, QualityResult],
    protocol: Mapping[str, Any],
    media_receipt: Mapping[str, Any],
    registry: SnapshotRegistry,
    context: str,
) -> None:
    """Decode every actual MP4 and independently recompute all fixed metrics."""

    from types import SimpleNamespace

    _require(
        set(qualities) == set(CANDIDATE_METHOD_IDS),
        context,
        "quality set differs from exact B--F candidates",
    )
    validator = _load_fixed_module(
        SOURCE_ROLES["comparison_script"],
        registry,
        "_fasta2v_final_metric_recomputer",
        context,
    )
    try:
        tool_paths = validator.validate_media_tool_receipt(media_receipt)
        lpips_runner = validator.LpipsAlexCpu(protocol["lpips"], tool_paths)
        metric_runner = validator.make_metric_runner(lpips_runner, tool_paths)
    except Exception as exc:
        _fail(context, f"cannot construct fixed metric scorer: {exc}")
    for method_id in CANDIDATE_METHOD_IDS:
        quality = qualities[method_id]
        _require(
            len(quality.metric_pairs) == FORMAL_ARTIFACT_COUNT,
            context,
            f"{method_id} metric pair count must be 72",
        )
        for identity in sorted(quality.metric_pairs):
            dense_path, candidate_path, persisted = quality.metric_pairs[identity]
            pair_context = f"{context} {method_id} pair {identity}"
            registry.file(dense_path, pair_context)
            registry.file(candidate_path, pair_context)
            try:
                computed = metric_runner(
                    SimpleNamespace(path=dense_path),
                    SimpleNamespace(path=candidate_path),
                )
                rendered, _ = validator._normalize_metrics(
                    computed,
                    pair_context,
                )
            except Exception as exc:
                _fail(pair_context, f"fixed metric recomputation failed: {exc}")
            _require(
                _json_equal(rendered, persisted),
                pair_context,
                "persisted metrics differ from independent MP4 recomputation",
            )


def _validate_media_receipt(
    receipt: Any, registry: SnapshotRegistry, context: str
) -> dict[str, Any]:
    _require(isinstance(receipt, dict) and set(receipt) == {"tools"}, context, "media receipt field set changed")
    tools = receipt.get("tools")
    _require(isinstance(tools, list) and len(tools) == 2, context, "media receipt must contain ffmpeg and ffprobe")
    by_name = {
        item.get("name"): item
        for item in tools
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    _require(set(by_name) == {"ffmpeg", "ffprobe"}, context, "media tool set changed")
    for name in ("ffmpeg", "ffprobe"):
        record = by_name[name]
        tool_context = f"{context} {name}"
        _require(set(record) == {"name", "path", "sha256", "version_line"}, tool_context, "tool receipt field set changed")
        tool_snapshot = registry.file(record.get("path"), tool_context)
        _require(tool_snapshot.sha256 == _sha(record.get("sha256"), tool_context, "sha256"), tool_context, "tool SHA256 drifted")
        discovered = shutil.which(name)
        _require(
            discovered is not None
            and Path(discovered).resolve() == tool_snapshot.path,
            tool_context,
            "tool path differs from the active fixed system executable",
        )
        for trusted_path in (tool_snapshot.path, *tool_snapshot.path.parents):
            metadata = os.lstat(trusted_path)
            _require(
                metadata.st_uid == 0
                and not (
                    metadata.st_mode
                    & (stat.S_IWGRP | stat.S_IWOTH)
                ),
                tool_context,
                f"media tool trust path is not root-owned read-only: {trusted_path}",
            )
        try:
            process = subprocess.run(
                [str(tool_snapshot.path), "-version"],
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            _fail(tool_context, f"cannot query fixed media tool: {exc}")
        version_line = process.stdout.splitlines()[0] if process.stdout.splitlines() else ""
        _require(version_line == record.get("version_line"), tool_context, "tool version line drifted")
    return receipt


def _load_protocol_and_matrix(
    registry: SnapshotRegistry,
) -> tuple[dict[str, Any], StableSnapshot, dict[str, Any], StableSnapshot, int, set[tuple[int, int, int]]]:
    protocol_snapshot = registry.file(DEFAULT_PROTOCOL, "quality protocol")
    matrix_snapshot = registry.file(DEFAULT_MATRIX, "evaluation matrix")
    protocol = _strict_json(protocol_snapshot, "quality protocol")
    matrix = _strict_json(matrix_snapshot, "evaluation matrix")
    _require(isinstance(protocol, dict), "quality protocol", "root must be an object")
    _require(protocol.get("schema_version") == 2, "quality protocol", "schema_version must be 2")
    _require(
        protocol.get("protocol_id") == "ovi_720x720_5s_dense_pair_quality_v2",
        "quality protocol",
        "protocol id differs from the fixed Ovi protocol",
    )
    _require(protocol.get("reference_method_id") == "dense", "quality protocol", "Dense must be the reference")
    measurement_indices = protocol.get("measurement_indices")
    _require(
        isinstance(measurement_indices, list)
        and all(_is_int(item) for item in measurement_indices)
        and tuple(measurement_indices) == (0, 1, 2),
        "quality protocol",
        "measurement indices changed",
    )
    _require(tuple(protocol.get("pairing_key", ())) == IDENTITY_FIELDS, "quality protocol", "pairing key changed")
    manual = protocol.get("manual_reviews")
    _require(isinstance(manual, dict), "quality protocol", "manual review contract is missing")
    _require(tuple(manual.get("fields", ())) == MANUAL_FIELDS, "quality protocol", "manual CSV fields changed")
    _require(tuple(manual.get("allowed_sync_ratings", ())) == ALLOWED_RATINGS, "quality protocol", "manual ratings changed")
    media_metrics = protocol.get("media_metrics")
    _require(isinstance(media_metrics, dict), "quality protocol", "media metric contract is missing")
    _require(
        media_metrics.get("frame_policy") == "exact_all_decoded_frames"
        and media_metrics.get("audio_decode")
        == {
            "channels": 1,
            "sample_rate_hz": 16000,
            "sample_format": "f32le",
            "sample_count_policy": "exact",
        },
        "quality protocol",
        "decoded media count contract changed",
    )
    _validate_pinned_lpips_protocol(protocol)

    _require(isinstance(matrix, dict), "evaluation matrix", "root must be an object")
    _require(matrix.get("schema_version") == 1, "evaluation matrix", "schema_version must be 1")
    _require(
        matrix.get("matrix_id") == FORMAL_MATRIX_ID,
        "evaluation matrix",
        "matrix_id differs from the fixed formal8x3 matrix",
    )
    methods = matrix.get("methods")
    _require(
        isinstance(methods, list)
        and all(
            isinstance(item, dict)
            and isinstance(item.get("method_id"), str)
            and item["method_id"]
            for item in methods
        ),
        "evaluation matrix",
        "methods must be objects with string ids",
    )
    method_ids = [item.get("method_id") for item in methods if isinstance(item, dict)]
    _require(len(method_ids) == len(set(method_ids)), "evaluation matrix", "duplicate method id")
    formal_slots = tuple(
        (item.get("formal_slot"), item.get("method_id"))
        for item in methods
        if isinstance(item, dict) and item.get("formal_slot") is not None
    )
    _require(
        formal_slots == tuple(zip("ABCDEF", METHOD_IDS)),
        "evaluation matrix",
        "A--F formal slot mapping changed",
    )
    by_method = {item["method_id"]: item for item in methods if isinstance(item, dict)}
    for method_id in METHOD_IDS:
        method = by_method[method_id]
        _require(method.get("implementation_status") == "ready", "evaluation matrix", f"{method_id} is not ready")
        _require(method.get("required") is True, "evaluation matrix", f"{method_id} is not required")

    fixed = matrix.get("fixed_protocol")
    _require(isinstance(fixed, dict), "evaluation matrix", "fixed protocol is missing")
    measurement_count = fixed.get("measurement_runs")
    prompt_count = fixed.get("prompt_count")
    sample_count = fixed.get("each_example_n_times")
    _require(
        measurement_count == FORMAL_MEASUREMENT_COUNT,
        "evaluation matrix",
        "measurement count must be three",
    )
    _require(
        prompt_count == FORMAL_PROMPT_COUNT,
        "evaluation matrix",
        "prompt count must be the fixed formal8 count",
    )
    _require(
        sample_count == FORMAL_SAMPLE_COUNT,
        "evaluation matrix",
        "sample count must be the fixed three-seed schedule",
    )
    _require(
        fixed.get("prompts_sha256") == FORMAL_PROMPTS_SHA256,
        "evaluation matrix",
        "formal prompt SHA256 differs",
    )
    artifact_count = measurement_count * prompt_count * sample_count
    _require(
        artifact_count == FORMAL_ARTIFACT_COUNT,
        "evaluation matrix",
        "formal artifact count must be 72",
    )
    identities = {
        (measurement_index, prompt_index, sample_index)
        for measurement_index in protocol["measurement_indices"]
        for prompt_index in range(prompt_count)
        for sample_index in range(sample_count)
    }
    _require(len(identities) == artifact_count, "fixed protocol", "artifact cardinality is inconsistent")
    return protocol, protocol_snapshot, matrix, matrix_snapshot, artifact_count, identities


def _load_timing_rows(
    snapshot: StableSnapshot,
    matrix: Mapping[str, Any],
    artifact_count: int,
    identities: set[tuple[int, int, int]],
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, dict[tuple[int, int, int], str]],
    dict[str, dict[tuple[int, int, int], str]],
]:
    fields, rows = _strict_csv(snapshot, "timing CSV")
    _require(fields == TIMING_FIELDS, "timing CSV", "header differs from build_ovi_eval_csv.py")
    matrix_method_ids = tuple(item["method_id"] for item in matrix["methods"])
    _require(
        tuple(row["method_id"] for row in rows) == matrix_method_ids,
        "timing CSV",
        "row order or method set differs from build_ovi_eval_csv.py",
    )
    _require(
        tuple(row["method_id"] for row in rows[: len(METHOD_IDS)])
        == METHOD_IDS,
        "timing CSV",
        "A--F row order or method set differs",
    )
    methods = {item["method_id"]: item for item in matrix["methods"]}
    fixed = matrix["fixed_protocol"]
    by_method: dict[str, dict[str, str]] = {}
    artifact_hashes: dict[str, dict[tuple[int, int, int], str]] = {}
    metrics_sidecar_hashes: dict[
        str, dict[tuple[int, int, int], str]
    ] = {}
    for row in rows[len(METHOD_IDS) :]:
        method_id = row["method_id"]
        context = f"timing CSV excluded slot {method_id}"
        method = methods[method_id]
        _require(method.get("formal_slot") is None, context, "unexpected formal slot outside A--F")
        _require(row["label"] == method["label"], context, "label differs from matrix")
        _require(
            row["required"] == ("True" if method["required"] else "False"),
            context,
            "required flag differs from matrix",
        )
        _require(
            row["implementation_status"] == method["implementation_status"],
            context,
            "implementation status differs from matrix",
        )
        _require(
            row["status"] == "pending"
            and row["timing_status"] == "pending"
            and bool(row["pending_reason"]),
            context,
            "non-A--F slot must remain explicitly pending",
        )
        for field in TIMING_FIELDS:
            if field not in {
                "method_id",
                "label",
                "required",
                "implementation_status",
                "status",
                "timing_status",
                "pending_reason",
            }:
                _require(
                    row[field] == "",
                    context,
                    f"excluded slot unexpectedly contains {field}",
                )
    for row in rows[: len(METHOD_IDS)]:
        method_id = row["method_id"]
        context = f"timing CSV {method_id}"
        _require(method_id not in by_method, context, "duplicate method row")
        _require(row["label"] == methods[method_id]["label"], context, "label differs from matrix")
        _require(row["required"] == "True", context, "required flag must be True")
        _require(row["implementation_status"] == "ready", context, "implementation status must be ready")
        _require(row["status"] == "pending", context, "pre-quality row status must remain pending")
        _require(row["timing_status"] == "valid", context, "timing status is not valid")
        _require(bool(row["pending_reason"]), context, "pending reason is missing")
        for field in ("quality_metric_name", "quality_score", "manual_review"):
            _require(row[field] == "", context, f"upstream {field} must remain blank")
        for field in ("run_dir", "timings_path", "warmup_timings_path"):
            _canonical_path(row[field], f"{context} {field}")
        _require(bool(row["run_id"]), context, "run_id is missing")
        for field in (
            "verification_sha256",
            "timings_sha256",
            "warmup_timings_sha256",
            "git_commit",
            "checkpoint_manifest_sha256",
            "checkpoint_fingerprint_sha256",
            "prompt_set_sha256",
        ):
            if field == "git_commit":
                _commit(row[field], context, field)
            else:
                _sha(row[field], context, field)
        _require(bool(row["gpu_uuid"]) and bool(row["gpu_name"]), context, "GPU identity is missing")
        _require(
            _canonical_uint_cell(row["measurement_count"], context, "measurement_count", positive=True)
            == fixed["measurement_runs"],
            context,
            "measurement count differs from protocol",
        )
        _require(row["measurement_indices"] == "0;1;2", context, "measurement indices changed")
        _require(
            _canonical_uint_cell(row["prompt_count"], context, "prompt_count", positive=True)
            == fixed["prompt_count"],
            context,
            "prompt count differs from protocol",
        )
        _require(
            _canonical_uint_cell(row["artifact_count"], context, "artifact_count", positive=True)
            == artifact_count,
            context,
            "artifact count differs from protocol",
        )
        _require(
            _canonical_uint_cell(row["timings_record_count"], context, "timings_record_count", positive=True)
            == artifact_count,
            context,
            "timing record count differs from artifact count",
        )
        _require(
            _canonical_uint_cell(row["warmup_record_count"], context, "warmup_record_count", positive=True)
            == fixed["warmup_runs"],
            context,
            "warmup count differs from protocol",
        )
        _require(
            _canonical_uint_cell(row["sample_steps"], context, "sample_steps", positive=True)
            == fixed["sample_steps"],
            context,
            "sample steps differ from protocol",
        )
        seed_count = _canonical_uint_cell(row["seed_count"], context, "seed_count", positive=True)
        seed_values = tuple(row["seeds"].split(";"))
        _require(len(seed_values) == seed_count, context, "seed list length differs from seed_count")
        for seed in seed_values:
            _canonical_uint_cell(seed, context, "seed")
        for field in (
            "timings_bytes",
            "warmup_timings_bytes",
            "requested_height",
            "requested_width",
            "actual_height",
            "actual_width",
        ):
            _canonical_uint_cell(row[field], context, field, positive=True)
        for field in (
            "denoise_seconds_median",
            "total_generation_seconds_median",
            "artifact_ready_seconds_median",
            "peak_memory_allocated_gib_median",
            "peak_memory_reserved_gib_median",
            "denoise_speedup_vs_dense",
            "total_speedup_vs_dense",
        ):
            _positive_float_cell(row[field], context, field)
        if method_id == "dense":
            _require(float(row["denoise_speedup_vs_dense"]) == 1.0, context, "Dense denoise speedup must be one")
            _require(float(row["total_speedup_vs_dense"]) == 1.0, context, "Dense total speedup must be one")
        artifact_hashes[method_id] = _parse_artifact_hashes(
            row["artifact_sha256"], identities, context
        )
        metrics_sidecar_hashes[method_id] = _parse_artifact_hashes(
            row["metrics_sidecar_sha256"], identities, context
        )
        by_method[method_id] = row
    for field in ("run_dir", "run_id", "timings_path", "verification_sha256", "timings_sha256"):
        values = [by_method[method_id][field] for method_id in METHOD_IDS]
        _require(
            len(values) == len(set(values)),
            "timing CSV",
            f"{field} contains duplicate A--F bindings",
        )
    return by_method, artifact_hashes, metrics_sidecar_hashes


def _require_json_int(value: Any, context: str, field: str, *, positive: bool = False) -> int:
    _require(_is_int(value), context, f"{field} must be an integer")
    if positive:
        _require(value > 0, context, f"{field} must be positive")
    else:
        _require(value >= 0, context, f"{field} must be nonnegative")
    return value


def _bind_run(
    binding: Any,
    method_id: str,
    row: Mapping[str, str],
    matrix_method: Mapping[str, Any],
    artifact_count: int,
    registry: SnapshotRegistry,
    context: str,
) -> dict[str, Any]:
    _require(isinstance(binding, dict), context, "run binding must be an object")
    _require(set(binding) == RUN_BINDING_FIELDS, context, "run binding field set changed")
    _require(binding.get("method_id") == method_id, context, "method_id differs from assigned slot")
    string_fields = (
        "run_dir",
        "run_id",
        "verification_sha256",
        "timings_path",
        "timings_sha256",
        "warmup_timings_path",
        "warmup_timings_sha256",
        "git_commit",
        "checkpoint_manifest_sha256",
        "checkpoint_fingerprint_sha256",
        "gpu_uuid",
        "gpu_name",
        "prompt_set_sha256",
        "selected_sparse_profile",
    )
    for field in string_fields:
        _require(binding.get(field) == row[field], context, f"{field} differs from timing CSV")
    run_dir = _canonical_directory(binding["run_dir"], f"{context} run_dir")
    for field in ("timings_path", "warmup_timings_path"):
        _canonical_path(binding[field], f"{context} {field}")
    for field in (
        "verification_sha256",
        "timings_sha256",
        "warmup_timings_sha256",
        "environment_sha256",
        "checkpoint_manifest_sha256",
        "checkpoint_fingerprint_sha256",
        "prompt_set_sha256",
    ):
        _sha(binding.get(field), context, field)
    _commit(binding.get("git_commit"), context, "git_commit")
    integer_csv_fields = (
        "timings_bytes",
        "timings_record_count",
        "warmup_timings_bytes",
        "warmup_record_count",
        "prompt_count",
        "sample_steps",
    )
    for field in integer_csv_fields:
        value = _require_json_int(binding.get(field), context, field, positive=True)
        _require(value == int(row[field]), context, f"{field} differs from timing CSV")
    _require(binding["timings_record_count"] == artifact_count, context, "timing count differs from protocol")
    _require_json_int(binding.get("gpu_physical_index"), context, "gpu_physical_index")
    prompts = binding.get("prompts")
    _require(
        isinstance(prompts, list)
        and len(prompts) == binding["prompt_count"]
        and all(isinstance(item, str) and item for item in prompts),
        context,
        "prompt list is invalid",
    )
    rendered_prompts = json.dumps(
        prompts,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    _require(
        hashlib.sha256(rendered_prompts).hexdigest()
        == binding["prompt_set_sha256"],
        context,
        "ordered prompt text differs from prompt_set_sha256",
    )
    base_seed = _require_json_int(binding.get("base_seed"), context, "base_seed")
    sample_count = _require_json_int(binding.get("sample_count"), context, "sample_count", positive=True)
    sample_seeds = binding.get("sample_seeds")
    _require(
        isinstance(sample_seeds, list)
        and len(sample_seeds) == sample_count
        and all(_is_int(item) and item >= 0 for item in sample_seeds),
        context,
        "sample seed schedule is invalid",
    )
    _require(
        sample_seeds == [base_seed + index for index in range(sample_count)],
        context,
        "sample seed schedule is not base_seed + sample_index",
    )
    _require(row["seed_count"] == str(sample_count), context, "sample_count differs from timing seed_count")
    _require(row["seeds"] == ";".join(str(item) for item in sample_seeds), context, "sample seeds differ from timing CSV")
    for field, height_field, width_field in (
        ("requested_shape", "requested_height", "requested_width"),
        ("actual_shape", "actual_height", "actual_width"),
    ):
        shape = binding.get(field)
        _require(
            isinstance(shape, list)
            and len(shape) == 2
            and all(_is_int(item) and item > 0 for item in shape),
            context,
            f"{field} is invalid",
        )
        _require(shape == [int(row[height_field]), int(row[width_field])], context, f"{field} differs from timing CSV")
    generated_video_shape = binding.get("generated_video_shape")
    _require(
        isinstance(generated_video_shape, list)
        and len(generated_video_shape) == 4
        and generated_video_shape[0] == 3
        and all(_is_int(item) and item > 0 for item in generated_video_shape)
        and generated_video_shape[2:] == binding["actual_shape"],
        context,
        "generated_video_shape is not [3,frames,height,width]",
    )
    generated_audio_shape = binding.get("generated_audio_shape")
    _require(
        isinstance(generated_audio_shape, list)
        and bool(generated_audio_shape)
        and all(_is_int(item) and item > 0 for item in generated_audio_shape),
        context,
        "generated_audio_shape is invalid",
    )
    acceleration = binding.get("acceleration_environment")
    _require(isinstance(acceleration, dict), context, "acceleration environment is missing")
    expected_environment = matrix_method.get("expected_environment")
    _require(isinstance(expected_environment, dict), context, "matrix method environment is missing")
    for field, expected in expected_environment.items():
        _require(
            field in acceleration and _json_equal(acceleration[field], expected),
            context,
            f"acceleration field {field} differs from matrix",
        )
    evidence = binding.get("evidence_bindings")
    _require(isinstance(evidence, dict) and bool(evidence), context, "evidence bindings are missing")
    required_evidence = {
        "environment.json": binding["environment_sha256"],
        "verification.json": binding["verification_sha256"],
        "timings.jsonl": binding["timings_sha256"],
        "warmup_timings.jsonl": binding["warmup_timings_sha256"],
        "checkpoint_manifest.json": binding["checkpoint_manifest_sha256"],
    }
    _require(
        set(required_evidence).issubset(evidence),
        context,
        "core run evidence bindings are incomplete",
    )
    for name, file_binding in evidence.items():
        file_context = f"{context} evidence {name!r}"
        _require(
            isinstance(name, str)
            and name
            and Path(name).name == name
            and name not in {".", ".."},
            file_context,
            "evidence name is not a direct leaf",
        )
        _require(
            isinstance(file_binding, dict)
            and set(file_binding) == {"path", "bytes", "sha256"},
            file_context,
            "evidence binding field set changed",
        )
        expected_path = run_dir / name
        _require(
            file_binding.get("path") == str(expected_path),
            file_context,
            "evidence path escaped selected run",
        )
        file_snapshot = registry.file(expected_path, file_context)
        _require(
            file_snapshot.size
            == _require_json_int(
                file_binding.get("bytes"), file_context, "bytes"
            )
            and file_snapshot.sha256
            == _sha(file_binding.get("sha256"), file_context, "sha256"),
            file_context,
            "evidence bytes or SHA256 drifted",
        )
        if name in required_evidence:
            _require(
                file_snapshot.sha256 == required_evidence[name],
                file_context,
                "core evidence SHA256 differs from run binding",
            )
    _require(
        binding["timings_path"] == str(run_dir / "timings.jsonl")
        and binding["warmup_timings_path"]
        == str(run_dir / "warmup_timings.jsonl"),
        context,
        "timing evidence paths differ from selected run",
    )
    return binding


def _validate_evaluator_source(
    receipt: Any,
    registry: SnapshotRegistry,
    protocol_snapshot: StableSnapshot,
    matrix_snapshot: StableSnapshot,
    context: str,
) -> dict[str, Any]:
    _require(isinstance(receipt, dict), context, "evaluator source receipt is missing")
    _commit(receipt.get("git_commit"), context, "git_commit")
    files = receipt.get("files")
    _require(isinstance(files, dict) and set(files) == set(SOURCE_ROLES), context, "evaluator source file set changed")
    for role, expected_path in SOURCE_ROLES.items():
        record = files[role]
        role_context = f"{context} {role}"
        _require(isinstance(record, dict) and set(record) == {"path", "sha256"}, role_context, "source binding field set changed")
        expected_path = Path(expected_path).resolve()
        _require(record.get("path") == str(expected_path), role_context, "source path differs from fixed repository path")
        snapshot = registry.file(record["path"], role_context)
        _require(snapshot.sha256 == _sha(record.get("sha256"), role_context, "sha256"), role_context, "source SHA256 drifted")
    _require(files["quality_protocol"]["sha256"] == protocol_snapshot.sha256, context, "quality protocol source hash differs")
    _require(files["evaluation_matrix"]["sha256"] == matrix_snapshot.sha256, context, "evaluation matrix source hash differs")
    return receipt


def _metric_number(value: Any, context: str, field: str) -> float:
    if field == "video_psnr_db" and value == "inf":
        return math.inf
    _require(_is_number(value), context, f"{field} must be numeric")
    result = float(value)
    _require(math.isfinite(result), context, f"{field} must be finite")
    if field in {"lpips_alex", "audio_rmse", "audio_max_abs_difference"}:
        _require(result >= 0.0, context, f"{field} must be nonnegative")
    if field in {"video_ssim", "audio_correlation"}:
        _require(-1.0 <= result <= 1.0, context, f"{field} is outside [-1,1]")
    return result


def _validate_pair_metrics(
    value: Any,
    context: str,
    *,
    expected_video_frames: int,
    generated_audio_samples: int,
    audio_sample_rate: int,
) -> dict[str, float]:
    _require(isinstance(value, dict) and set(value) == PAIR_METRIC_FIELDS, context, "metric field set changed")
    for field in (
        "compared_video_frames",
        "reference_audio_samples",
        "candidate_audio_samples",
        "audio_sample_count_compared",
        "lpips_frame_count",
    ):
        _require_json_int(value.get(field), context, field, positive=True)
    _require(value["compared_video_frames"] == value["lpips_frame_count"], context, "video and LPIPS frame counts differ")
    _require(
        value["compared_video_frames"] == expected_video_frames,
        context,
        "decoded frame count differs from generated video shape",
    )
    _require(
        value["reference_audio_samples"]
        == value["candidate_audio_samples"]
        == value["audio_sample_count_compared"],
        context,
        "audio sample counts differ",
    )
    _require(
        abs(value["audio_sample_count_compared"] - generated_audio_samples)
        <= audio_sample_rate,
        context,
        "decoded audio sample count is inconsistent with generated audio shape",
    )
    return {field: _metric_number(value[field], context, field) for field in METRIC_FIELDS}


def _validate_artifact_binding(
    artifact: Any,
    run: Mapping[str, Any],
    identity: tuple[int, int, int],
    expected_hash: str,
    expected_metrics_hash: str,
    registry: SnapshotRegistry,
    context: str,
) -> None:
    _require(isinstance(artifact, dict) and set(artifact) == ARTIFACT_BINDING_FIELDS, context, "artifact binding field set changed")
    for field in RUN_BINDING_FIELDS:
        _require(field in artifact and _json_equal(artifact[field], run[field]), context, f"run field {field} differs")
    for field, expected in zip(IDENTITY_FIELDS, identity):
        _require(artifact.get(field) == expected, context, f"{field} differs")
    _require(artifact.get("artifact_sha256") == expected_hash, context, "artifact SHA256 differs")
    _require(
        artifact.get("metrics_sidecar_sha256") == expected_metrics_hash,
        context,
        "metrics sidecar SHA256 differs from timing CSV",
    )
    artifact_path = _canonical_path(artifact.get("artifact_path"), f"{context} artifact_path")
    metrics_path = _canonical_path(
        artifact.get("metrics_sidecar_path"), f"{context} metrics_sidecar_path"
    )
    run_dir = Path(run["run_dir"])
    _require(
        artifact_path.parent == run_dir
        and metrics_path.parent == run_dir
        and metrics_path == artifact_path.with_suffix(".metrics.json"),
        context,
        "artifact or metrics sidecar escaped selected run",
    )
    artifact_snapshot = registry.file(artifact_path, context)
    metrics_snapshot = registry.file(metrics_path, context)
    _require(
        artifact_snapshot.sha256 == expected_hash,
        context,
        "artifact file SHA256 drifted",
    )
    _require(
        metrics_snapshot.sha256 == expected_metrics_hash,
        context,
        "metrics sidecar file SHA256 drifted",
    )
    metrics = _strict_json(metrics_snapshot, f"{context} metrics sidecar")
    _require(isinstance(metrics, dict), context, "metrics sidecar root is not an object")
    required_metrics = {
        "status": "ok",
        "record_type": "measurement",
        "benchmark_candidate": True,
        "benchmark_valid": False,
        "run_id": run["run_id"],
        "measurement_index": identity[0],
        "prompt_index": identity[1],
        "sample_index": identity[2],
        "prompt": run["prompts"][identity[1]],
        "seed": run["base_seed"] + identity[2],
        "sample_steps": run["sample_steps"],
        "requested_video_frame_height_width": run["requested_shape"],
        "actual_video_frame_height_width": run["actual_shape"],
        "generated_video_shape": run["generated_video_shape"],
        "generated_audio_shape": run["generated_audio_shape"],
        "output_path": str(artifact_path),
        "output_sha256": expected_hash,
    }
    acceleration = run["acceleration_environment"]
    for field in ("attention_method", "use_cfg_cache", "use_block_cache"):
        if field in acceleration:
            required_metrics[field] = acceleration[field]
    for field, expected in required_metrics.items():
        _require(
            field in metrics and _json_equal(metrics[field], expected),
            context,
            f"metrics sidecar field {field} differs from bound run",
        )


@dataclass(frozen=True)
class QualityResult:
    method_id: str
    path: Path
    sha256: str
    medians: dict[str, Any]
    pairs: dict[tuple[int, int, int], tuple[str, str]]
    dense_run: dict[str, Any]
    candidate_run: dict[str, Any]
    evaluator_source: dict[str, Any]
    lpips_receipt: dict[str, Any]
    media_receipt: dict[str, Any]
    metric_pairs: dict[
        tuple[int, int, int], tuple[Path, Path, dict[str, Any]]
    ]


def _load_quality(
    method_id: str,
    snapshot: StableSnapshot,
    registry: SnapshotRegistry,
    timing_rows: Mapping[str, Mapping[str, str]],
    artifact_hashes: Mapping[str, Mapping[tuple[int, int, int], str]],
    metrics_sidecar_hashes: Mapping[
        str, Mapping[tuple[int, int, int], str]
    ],
    protocol: Mapping[str, Any],
    protocol_snapshot: StableSnapshot,
    matrix: Mapping[str, Any],
    matrix_snapshot: StableSnapshot,
    artifact_count: int,
    identities: set[tuple[int, int, int]],
) -> QualityResult:
    context = f"quality {method_id}"
    report = _strict_json(snapshot, context)
    _require(isinstance(report, dict) and set(report) == QUALITY_FIELDS, context, "median field set changed")
    _require(report.get("schema_version") == 2, context, "schema_version must be 2")
    _require(report.get("record_type") == "ovi_quality_median", context, "record_type is not ovi_quality_median")
    _require(report.get("quality_protocol_id") == protocol["protocol_id"], context, "quality protocol id differs")
    _require(report.get("quality_protocol_sha256") == protocol_snapshot.sha256, context, "quality protocol hash differs")
    _require(report.get("evaluation_matrix_sha256") == matrix_snapshot.sha256, context, "evaluation matrix hash differs")
    _require(report.get("automatic_acceptance") is None, context, "automatic acceptance must be null")
    _require(
        _json_equal(
            report.get("manual_review"),
            {
                "status": "not_provided",
                "row_count": 0,
                "csv_path": None,
                "csv_sha256": None,
            },
        ),
        context,
        "manual review must remain separate from quality median",
    )
    evaluator = _validate_evaluator_source(
        report.get("evaluator_source_receipt"),
        registry,
        protocol_snapshot,
        matrix_snapshot,
        f"{context} evaluator",
    )
    _require(evaluator["git_commit"] == timing_rows[method_id]["git_commit"], context, "evaluator commit differs from timing run")
    _require(evaluator["git_commit"] == timing_rows["dense"]["git_commit"], context, "evaluator commit differs from Dense run")
    source_hash_bindings = {
        "comparison_script_sha256": "comparison_script",
        "compare_media_script_sha256": "compare_media_script",
        "run_validator_script_sha256": "run_validator_script",
        "evaluation_matrix_sha256": "evaluation_matrix",
    }
    for field, role in source_hash_bindings.items():
        _require(report.get(field) == evaluator["files"][role]["sha256"], context, f"{field} differs from evaluator receipt")
    by_method = {item["method_id"]: item for item in matrix["methods"]}
    dense_run = _bind_run(
        report.get("dense_run"),
        "dense",
        timing_rows["dense"],
        by_method["dense"],
        artifact_count,
        registry,
        f"{context} Dense run",
    )
    candidate_run = _bind_run(
        report.get("candidate_run"),
        method_id,
        timing_rows[method_id],
        by_method[method_id],
        artifact_count,
        registry,
        f"{context} candidate run",
    )
    _require(
        candidate_run["git_commit"] == dense_run["git_commit"],
        context,
        "candidate and Dense commits differ",
    )
    for field in protocol.get("required_same_across_runs", ()):
        mapping = {
            "git_commit": "git_commit",
            "checkpoint_fingerprint_sha256": "checkpoint_fingerprint_sha256",
            "prompt_set_sha256": "prompt_set_sha256",
            "prompt_count": "prompt_count",
            "prompts": "prompts",
            "base_seed": "base_seed",
            "sample_count": "sample_count",
            "sample_seeds": "sample_seeds",
            "requested_shape": "requested_shape",
            "actual_shape": "actual_shape",
            "generated_video_shape": "generated_video_shape",
            "generated_audio_shape": "generated_audio_shape",
            "sample_steps": "sample_steps",
        }
        if field == "gpu_identity":
            left = (dense_run["gpu_physical_index"], dense_run["gpu_uuid"], dense_run["gpu_name"])
            right = (candidate_run["gpu_physical_index"], candidate_run["gpu_uuid"], candidate_run["gpu_name"])
        else:
            _require(field in mapping, context, f"unknown required-same field {field!r}")
            key = mapping[field]
            left, right = dense_run[key], candidate_run[key]
        _require(_json_equal(left, right), context, f"required-same field {field} differs")

    pair_count = _require_json_int(report.get("pair_count"), context, "pair_count", positive=True)
    _require(pair_count == artifact_count, context, "pair_count differs from fixed protocol artifact_count")
    pairs = report.get("pairs")
    _require(isinstance(pairs, list) and len(pairs) == artifact_count, context, "pair bindings are incomplete")
    expected_order = sorted(identities)
    bound_pairs: dict[tuple[int, int, int], tuple[str, str]] = {}
    metric_values: dict[str, list[float]] = {field: [] for field in METRIC_FIELDS}
    metric_pairs: dict[
        tuple[int, int, int], tuple[Path, Path, dict[str, Any]]
    ] = {}
    pair_snapshots: list[StableSnapshot] = []
    for offset, pair in enumerate(pairs):
        pair_context = f"{context} pair {offset}"
        _require(isinstance(pair, dict) and set(pair) == QUALITY_PAIR_BINDING_FIELDS, pair_context, "pair binding field set changed")
        identity = tuple(pair.get(field) for field in IDENTITY_FIELDS)
        _require(
            all(_is_int(item) and item >= 0 for item in identity),
            pair_context,
            "pair identity must contain nonnegative integers",
        )
        _require(identity == expected_order[offset], pair_context, "pair identity order or coverage differs")
        dense_hash = _sha(pair.get("dense_artifact_sha256"), pair_context, "Dense artifact SHA256")
        candidate_hash = _sha(pair.get("candidate_artifact_sha256"), pair_context, "candidate artifact SHA256")
        _require(dense_hash == artifact_hashes["dense"][identity], pair_context, "Dense artifact hash differs from timing CSV")
        _require(candidate_hash == artifact_hashes[method_id][identity], pair_context, "candidate artifact hash differs from timing CSV")
        pair_path = _canonical_path(pair.get("pair_sidecar_path"), pair_context)
        expected_name = (
            f"measurement_{identity[0]:02d}_prompt_{identity[1]:03d}_"
            f"sample_{identity[2]:03d}.quality.json"
        )
        _require(pair_path.parent == snapshot.path.parent and pair_path.name == expected_name, pair_context, "pair sidecar path is not canonical beside median")
        pair_snapshot = registry.file(pair_path, pair_context)
        pair_snapshots.append(pair_snapshot)
        _require(pair_snapshot.sha256 == _sha(pair.get("pair_sidecar_sha256"), pair_context, "pair sidecar SHA256"), pair_context, "pair sidecar SHA256 drifted")
        sidecar = _strict_json(pair_snapshot, pair_context)
        _require(isinstance(sidecar, dict) and set(sidecar) == PAIR_SIDECAR_FIELDS, pair_context, "pair sidecar field set changed")
        _require(sidecar.get("schema_version") == 2 and sidecar.get("record_type") == "ovi_quality_pair", pair_context, "pair sidecar schema is invalid")
        _require(sidecar.get("quality_protocol_id") == protocol["protocol_id"] and sidecar.get("quality_protocol_sha256") == protocol_snapshot.sha256, pair_context, "pair protocol binding differs")
        _require(sidecar.get("automatic_acceptance") is None, pair_context, "automatic acceptance must be null")
        sidecar_identity = tuple(sidecar.get(field) for field in IDENTITY_FIELDS)
        _require(
            all(_is_int(item) and item >= 0 for item in sidecar_identity)
            and sidecar_identity == identity,
            pair_context,
            "sidecar identity differs",
        )
        for field in source_hash_bindings:
            _require(sidecar.get(field) == report[field], pair_context, f"{field} differs from median")
        for field in ("evaluator_source_receipt", "lpips_dependency_receipt", "media_tool_receipt"):
            _require(_json_equal(sidecar.get(field), report[field]), pair_context, f"{field} differs from median")
        _validate_artifact_binding(
            sidecar.get("dense"),
            dense_run,
            identity,
            dense_hash,
            metrics_sidecar_hashes["dense"][identity],
            registry,
            f"{pair_context} Dense",
        )
        _validate_artifact_binding(
            sidecar.get("candidate"),
            candidate_run,
            identity,
            candidate_hash,
            metrics_sidecar_hashes[method_id][identity],
            registry,
            f"{pair_context} candidate",
        )
        numeric = _validate_pair_metrics(
            sidecar.get("metrics"),
            pair_context,
            expected_video_frames=dense_run["generated_video_shape"][1],
            generated_audio_samples=dense_run["generated_audio_shape"][-1],
            audio_sample_rate=protocol["media_metrics"]["audio_decode"][
                "sample_rate_hz"
            ],
        )
        for field, value in numeric.items():
            metric_values[field].append(value)
        metric_pairs[identity] = (
            Path(sidecar["dense"]["artifact_path"]),
            Path(sidecar["candidate"]["artifact_path"]),
            dict(sidecar["metrics"]),
        )
        bound_pairs[identity] = (dense_hash, candidate_hash)
    _reject_duplicate_primary_inputs(pair_snapshots)
    _require(set(bound_pairs) == identities, context, "pair identities are incomplete")
    medians: dict[str, Any] = {}
    for field in METRIC_FIELDS:
        median = float(statistics.median(metric_values[field]))
        medians[field] = "inf" if math.isinf(median) else median
    _require(
        isinstance(report.get("metric_medians"), dict)
        and set(report["metric_medians"]) == set(METRIC_FIELDS)
        and _json_equal(report["metric_medians"], medians),
        context,
        "metric medians differ from hash-bound pair sidecars",
    )
    lpips_receipt = report.get("lpips_dependency_receipt")
    media_receipt = report.get("media_tool_receipt")
    _require(isinstance(lpips_receipt, dict) and bool(lpips_receipt), context, "LPIPS receipt is missing")
    _require(isinstance(media_receipt, dict) and bool(media_receipt), context, "media tool receipt is missing")
    return QualityResult(
        method_id=method_id,
        path=snapshot.path,
        sha256=snapshot.sha256,
        medians=medians,
        pairs=bound_pairs,
        dense_run=dense_run,
        candidate_run=candidate_run,
        evaluator_source=evaluator,
        lpips_receipt=lpips_receipt,
        media_receipt=media_receipt,
        metric_pairs=metric_pairs,
    )


@dataclass(frozen=True)
class ManualResult:
    method_id: str
    receipt_path: Path
    receipt_sha256: str
    csv_path: Path
    csv_sha256: str
    row_count: int
    counts: dict[str, int]


def _load_manual(
    method_id: str,
    snapshot: StableSnapshot,
    registry: SnapshotRegistry,
    quality: QualityResult,
    protocol: Mapping[str, Any],
    protocol_snapshot: StableSnapshot,
    artifact_count: int,
) -> ManualResult:
    context = f"manual {method_id}"
    receipt = _strict_json(snapshot, context)
    _require(isinstance(receipt, dict) and set(receipt) == MANUAL_RECEIPT_FIELDS, context, "receipt field set changed")
    _require(receipt.get("schema_version") == 2, context, "schema_version must be 2")
    _require(receipt.get("record_type") == "ovi_manual_sync_review_validation", context, "record_type is invalid")
    _require(receipt.get("quality_protocol_id") == protocol["protocol_id"], context, "protocol id differs")
    _require(receipt.get("quality_protocol_sha256") == protocol_snapshot.sha256, context, "protocol SHA256 differs")
    _require(receipt.get("manual_review_status") == "complete", context, "manual review status is not complete")
    _require(snapshot.path.parent == quality.path.parent and snapshot.path.name == "manual-review.validation.json", context, "manual receipt is not the canonical file beside median")
    _require(receipt.get("quality_median_path") == str(quality.path), context, "quality median path differs")
    _require(receipt.get("quality_median_sha256") == quality.sha256, context, "quality median SHA256 differs")
    row_count = _require_json_int(receipt.get("manual_review_row_count"), context, "manual_review_row_count", positive=True)
    _require(row_count == artifact_count, context, "manual row count differs from protocol artifact_count")
    pairs = receipt.get("pairs")
    _require(isinstance(pairs, list) and len(pairs) == artifact_count, context, "manual pair list is incomplete")
    for offset, identity in enumerate(sorted(quality.pairs)):
        pair = pairs[offset]
        pair_context = f"{context} pair {offset}"
        _require(isinstance(pair, dict) and set(pair) == MANUAL_PAIR_FIELDS, pair_context, "pair field set changed")
        supplied_identity = tuple(pair.get(field) for field in IDENTITY_FIELDS)
        _require(
            all(_is_int(item) and item >= 0 for item in supplied_identity)
            and supplied_identity == identity,
            pair_context,
            "pair identity order or coverage differs",
        )
        expected_dense, expected_candidate = quality.pairs[identity]
        _require(pair.get("dense_artifact_sha256") == expected_dense, pair_context, "Dense artifact SHA256 differs")
        _require(pair.get("candidate_artifact_sha256") == expected_candidate, pair_context, "candidate artifact SHA256 differs")

    csv_path = _canonical_path(receipt.get("manual_reviews_csv_path"), context)
    csv_snapshot = registry.file(csv_path, f"{context} CSV")
    _require(csv_snapshot.sha256 == _sha(receipt.get("manual_reviews_csv_sha256"), context, "manual CSV SHA256"), context, "manual CSV SHA256 drifted")
    fields, rows = _strict_csv(csv_snapshot, f"{context} CSV")
    _require(fields == MANUAL_FIELDS, context, "manual CSV header changed")
    _require(len(rows) == artifact_count, context, "manual CSV row count differs from protocol")
    expected_order = sorted(quality.pairs)
    counts = {rating: 0 for rating in ALLOWED_RATINGS}
    for offset, row in enumerate(rows):
        row_context = f"{context} CSV row {offset + 2}"
        identity = tuple(
            _canonical_uint_cell(row[field], row_context, field)
            for field in IDENTITY_FIELDS
        )
        _require(identity == expected_order[offset], row_context, "identity order or coverage differs")
        dense_hash, candidate_hash = quality.pairs[identity]
        _require(row["dense_artifact_sha256"] == dense_hash, row_context, "Dense artifact SHA256 differs")
        _require(row["candidate_artifact_sha256"] == candidate_hash, row_context, "candidate artifact SHA256 differs")
        _require(bool(row["reviewer"].strip()), row_context, "reviewer is blank")
        _require(UTC_TIMESTAMP.fullmatch(row["reviewed_at_utc"]) is not None, row_context, "review timestamp is not canonical UTC")
        try:
            parsed = datetime.fromisoformat(row["reviewed_at_utc"].replace("Z", "+00:00"))
        except ValueError:
            _fail(row_context, "review timestamp is not a real calendar time")
        _require(parsed.utcoffset() == timedelta(0), row_context, "review timestamp is not UTC")
        rating = row["sync_rating"]
        _require(rating in ALLOWED_RATINGS, row_context, "sync rating is invalid")
        counts[rating] += 1
    _require(sum(counts.values()) == row_count, context, "manual rating counts differ from row_count")
    return ManualResult(
        method_id=method_id,
        receipt_path=snapshot.path,
        receipt_sha256=snapshot.sha256,
        csv_path=csv_snapshot.path,
        csv_sha256=csv_snapshot.sha256,
        row_count=row_count,
        counts=counts,
    )


@dataclass(frozen=True)
class OutputTarget:
    path: Path
    directory_fd: int
    parent_device: int
    parent_inode: int


def _output_path(raw: str | os.PathLike[str]) -> OutputTarget:
    value = os.fspath(raw)
    context = "final CSV output"
    _require(isinstance(value, str) and os.path.isabs(value), context, "path must be absolute")
    _require(os.path.normpath(value) == value, context, "path is not lexically canonical")
    path = Path(value)
    parent = path.parent
    _require(os.path.realpath(parent) == str(parent), context, "parent resolves through a symlink or alias")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(parent, directory_flags)
    except OSError as exc:
        _fail(context, f"cannot open output parent {parent}: {exc}")
    try:
        opened = os.fstat(directory_fd)
        visible = os.lstat(parent)
        _require(
            stat.S_ISDIR(opened.st_mode)
            and (opened.st_dev, opened.st_ino)
            == (visible.st_dev, visible.st_ino),
            context,
            "output parent identity changed during preflight",
        )
        try:
            os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            _fail(context, f"refusing to overwrite {path}")
        return OutputTarget(
            path=path,
            directory_fd=directory_fd,
            parent_device=opened.st_dev,
            parent_inode=opened.st_ino,
        )
    except Exception:
        os.close(directory_fd)
        raise


def _render_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=FINAL_FIELDS,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _write_atomic_exclusive(
    target: OutputTarget,
    payload: bytes,
    snapshots: Iterable[StableSnapshot],
) -> None:
    context = "final CSV output"
    output = target.path
    temp_name = f".{output.name}.{secrets.token_hex(16)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    directory_fd = target.directory_fd
    linked = False
    published_identity: tuple[int, int] | None = None
    try:
        opened_parent = os.fstat(directory_fd)
        current_parent = os.lstat(output.parent)
        _require(
            stat.S_ISDIR(opened_parent.st_mode)
            and (opened_parent.st_dev, opened_parent.st_ino)
            == (target.parent_device, target.parent_inode)
            and (opened_parent.st_dev, opened_parent.st_ino)
            == (current_parent.st_dev, current_parent.st_ino)
            and os.path.realpath(output.parent) == str(output.parent),
            context,
            "output parent changed before publication",
        )
        descriptor = os.open(temp_name, flags, 0o600, dir_fd=directory_fd)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            _require(written > 0, context, "short zero-byte output write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        for snapshot in snapshots:
            snapshot.revalidate(context)
        try:
            temporary = os.stat(
                temp_name, dir_fd=directory_fd, follow_symlinks=False
            )
            os.link(
                temp_name,
                output.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            published_identity = (temporary.st_dev, temporary.st_ino)
        except FileExistsError:
            _fail(context, f"refusing to overwrite {output}")
        published = os.stat(
            output.name, dir_fd=directory_fd, follow_symlinks=False
        )
        current_parent = os.lstat(output.parent)
        if (
            (current_parent.st_dev, current_parent.st_ino)
            != (opened_parent.st_dev, opened_parent.st_ino)
            or os.path.realpath(output.parent) != str(output.parent)
        ):
            os.unlink(output.name, dir_fd=directory_fd)
            _fail(context, "output parent changed during publication")
        visible = os.lstat(output)
        if (
            not stat.S_ISREG(published.st_mode)
            or (visible.st_dev, visible.st_ino)
            != (published.st_dev, published.st_ino)
        ):
            os.unlink(output.name, dir_fd=directory_fd)
            _fail(context, "published output identity is not stable")
        linked = True
        os.fsync(directory_fd)
    except OSError as exc:
        _fail(context, f"cannot publish {output}: {exc}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if directory_fd >= 0:
            if not linked and published_identity is not None:
                try:
                    abandoned = os.stat(
                        output.name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    if (abandoned.st_dev, abandoned.st_ino) == published_identity:
                        os.unlink(output.name, dir_fd=directory_fd)
                except (FileNotFoundError, OSError):
                    pass
            try:
                os.unlink(temp_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            except OSError:
                pass
            os.close(directory_fd)


def build_final_csv(
    *,
    timing_csv: str | os.PathLike[str],
    quality_paths: Mapping[str, str | os.PathLike[str]],
    manual_paths: Mapping[str, str | os.PathLike[str]],
    output: str | os.PathLike[str],
) -> Path:
    """Validate every A--F receipt and exclusively publish the final CSV."""

    _require(set(quality_paths) == set(CANDIDATE_METHOD_IDS), "quality inputs", "exactly one B--F quality median is required")
    _require(set(manual_paths) == set(CANDIDATE_METHOD_IDS), "manual inputs", "exactly one B--F manual receipt is required")
    registry = SnapshotRegistry()
    timing_snapshot = registry.file(timing_csv, "timing CSV")
    quality_snapshots = {
        method_id: registry.file(quality_paths[method_id], f"quality {method_id}")
        for method_id in CANDIDATE_METHOD_IDS
    }
    manual_snapshots = {
        method_id: registry.file(manual_paths[method_id], f"manual {method_id}")
        for method_id in CANDIDATE_METHOD_IDS
    }
    _reject_duplicate_primary_inputs(
        [timing_snapshot, *quality_snapshots.values(), *manual_snapshots.values()]
    )
    protocol, protocol_snapshot, matrix, matrix_snapshot, artifact_count, identities = _load_protocol_and_matrix(registry)
    _require_fixed_eval_runtime(
        protocol["lpips"],
        "final fixed evaluation runtime",
    )
    repository_commit = _audit_repository_source(
        registry,
        "final repository source audit",
    )
    _full_validate_quality_protocol(
        protocol,
        protocol_snapshot,
        registry,
        "final quality protocol audit",
    )
    timing_rows, artifact_hashes, metrics_sidecar_hashes = _load_timing_rows(
        timing_snapshot, matrix, artifact_count, identities
    )
    _full_validate_timing_runs(
        matrix,
        timing_rows,
        registry,
        "final timing run audit",
    )
    qualities = {
        method_id: _load_quality(
            method_id,
            quality_snapshots[method_id],
            registry,
            timing_rows,
            artifact_hashes,
            metrics_sidecar_hashes,
            protocol,
            protocol_snapshot,
            matrix,
            matrix_snapshot,
            artifact_count,
            identities,
        )
        for method_id in CANDIDATE_METHOD_IDS
    }
    reference = qualities[CANDIDATE_METHOD_IDS[0]]
    for method_id, quality in qualities.items():
        context = f"quality {method_id} cross-candidate binding"
        _require(_json_equal(quality.dense_run, reference.dense_run), context, "Dense run binding differs")
        _require(_json_equal(quality.evaluator_source, reference.evaluator_source), context, "evaluator receipt differs")
        _require(_json_equal(quality.lpips_receipt, reference.lpips_receipt), context, "LPIPS receipt differs")
        _require(_json_equal(quality.media_receipt, reference.media_receipt), context, "media tool receipt differs")
    _recollect_evaluator_source(
        reference.evaluator_source,
        timing_rows,
        repository_commit,
        registry,
        "final evaluator source receipt",
    )
    _validate_lpips_disk_receipt(
        reference.lpips_receipt,
        protocol["lpips"],
        registry,
        "final LPIPS dependency receipt",
    )
    _full_validate_lpips_environment(
        protocol["lpips"],
        reference.lpips_receipt,
        registry,
        "final LPIPS dependency receipt",
    )
    _validate_media_receipt(
        reference.media_receipt,
        registry,
        "final media tool receipt",
    )
    _recollect_media_receipt(
        reference.media_receipt,
        registry,
        "final media tool receipt",
    )
    _recompute_quality_metrics(
        qualities,
        protocol,
        reference.media_receipt,
        registry,
        "final quality metric audit",
    )
    manuals = {
        method_id: _load_manual(
            method_id,
            manual_snapshots[method_id],
            registry,
            qualities[method_id],
            protocol,
            protocol_snapshot,
            artifact_count,
        )
        for method_id in CANDIDATE_METHOD_IDS
    }
    matrix_id = matrix.get("matrix_id")
    _require(isinstance(matrix_id, str) and matrix_id, "evaluation matrix", "matrix_id is missing")
    evaluator_commit = reference.evaluator_source["git_commit"]
    rows: list[dict[str, Any]] = []
    copied_timing_fields = (
        "run_dir",
        "run_id",
        "git_commit",
        "verification_sha256",
        "timings_sha256",
        "checkpoint_manifest_sha256",
        "checkpoint_fingerprint_sha256",
        "gpu_uuid",
        "gpu_name",
        "prompt_set_sha256",
        "prompt_count",
        "seed_count",
        "seeds",
        "sample_steps",
        "measurement_count",
        "artifact_count",
        "denoise_seconds_median",
        "total_generation_seconds_median",
        "artifact_ready_seconds_median",
        "peak_memory_allocated_gib_median",
        "peak_memory_reserved_gib_median",
        "denoise_speedup_vs_dense",
        "total_speedup_vs_dense",
    )
    for method_id in METHOD_IDS:
        timing = timing_rows[method_id]
        row: dict[str, Any] = {field: "" for field in FINAL_FIELDS}
        row.update(
            {
                "schema_version": 1,
                "method_id": method_id,
                "label": timing["label"],
                "status": "complete",
                "timing_status": "valid",
                "timing_csv_path": str(timing_snapshot.path),
                "timing_csv_sha256": timing_snapshot.sha256,
                "quality_protocol_id": protocol["protocol_id"],
                "quality_protocol_sha256": protocol_snapshot.sha256,
                "evaluation_matrix_id": matrix_id,
                "evaluation_matrix_sha256": matrix_snapshot.sha256,
                "evaluator_git_commit": evaluator_commit,
            }
        )
        for field in copied_timing_fields:
            row[field] = timing[field]
        if method_id == "dense":
            row["quality_status"] = "reference"
            row["manual_review_status"] = "reference"
        else:
            quality = qualities[method_id]
            manual = manuals[method_id]
            row.update(
                {
                    "quality_status": "complete",
                    "manual_review_status": "complete",
                    "quality_median_path": str(quality.path),
                    "quality_median_sha256": quality.sha256,
                    "manual_review_row_count": manual.row_count,
                    "manual_pass_count": manual.counts["pass"],
                    "manual_fail_count": manual.counts["fail"],
                    "manual_uncertain_count": manual.counts["uncertain"],
                    "manual_validation_path": str(manual.receipt_path),
                    "manual_validation_sha256": manual.receipt_sha256,
                    "manual_reviews_csv_path": str(manual.csv_path),
                    "manual_reviews_csv_sha256": manual.csv_sha256,
                }
            )
            for field in METRIC_FIELDS:
                row[f"{field}_median"] = quality.medians[field]
        rows.append(row)
    payload = _render_csv(rows)
    output_target = _output_path(output)
    _write_atomic_exclusive(output_target, payload, registry.snapshots)
    return output_target.path


def _parse_assignments(
    values: Sequence[str], expected: Sequence[str], option: str
) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        _require(value.count("=") == 1, option, f"expected METHOD=ABSOLUTE_PATH, found {value!r}")
        method_id, path = value.split("=", 1)
        _require(method_id in expected, option, f"method {method_id!r} is outside B--F")
        _require(method_id not in result, option, f"duplicate method {method_id!r}")
        _require(bool(path), option, f"path is missing for {method_id}")
        result[method_id] = path
    _require(set(result) == set(expected), option, f"exact B--F method set is required: {list(expected)}")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge the exact A--F timing, quality, and human-review evidence into one final CSV."
    )
    parser.add_argument("--timing-csv", required=True)
    parser.add_argument(
        "--quality",
        action="append",
        default=[],
        metavar="METHOD=MEDIAN.QUALITY.JSON",
        help="repeat exactly once for each B--F candidate",
    )
    parser.add_argument(
        "--manual",
        action="append",
        default=[],
        metavar="METHOD=MANUAL-REVIEW.VALIDATION.JSON",
        help="repeat exactly once for each B--F candidate",
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        quality_paths = _parse_assignments(args.quality, CANDIDATE_METHOD_IDS, "--quality")
        manual_paths = _parse_assignments(args.manual, CANDIDATE_METHOD_IDS, "--manual")
        output = build_final_csv(
            timing_csv=args.timing_csv,
            quality_paths=quality_paths,
            manual_paths=manual_paths,
            output=args.output,
        )
    except FinalCsvError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
