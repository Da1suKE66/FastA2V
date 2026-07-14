#!/usr/bin/env python3
"""Select the audited C--F sparse winner from one complete final Ovi CSV.

The command has one evidence input. It never discovers a latest table, follows
secondary evidence paths, repairs incomplete rows, or overwrites a receipt.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "quality_protocol.json"
MATRIX_PATH = REPO_ROOT / "configs" / "ovi_eval_matrix.json"
SELECTOR_PATH = Path(__file__).resolve()

METHOD_IDS = (
    "dense",
    "dense_cfg_cache",
    "sparge_topk75",
    "sparge_topk50",
    "radial_conservative",
    "radial_aggressive",
)
CANDIDATE_METHOD_IDS = METHOD_IDS[2:]
FORMAL_SLOTS = dict(zip(METHOD_IDS, ("A", "B", "C", "D", "E", "F")))
FORMAL_MATRIX_ID = "ovi_720x720_5s_a100_bf16_formal8x3_v2"
FORMAL_PROMPTS_SHA256 = (
    "d98397111b1ab060a61d588f4ca388c5c929430a59ac6ab49b7c2e247bb6be91"
)
FORMAL_MEASUREMENT_COUNT = 3
FORMAL_PROMPT_COUNT = 8
FORMAL_SEED_COUNT = 3
FORMAL_ARTIFACT_COUNT = 72
FORMAL_SAMPLE_STEPS = 50
GIT_BINARY = Path("/usr/bin/git")

METHOD_ENVIRONMENT_CONTRACTS: dict[str, dict[str, Any]] = {
    "dense": {
        "run_kind": "dense_baseline", "attention_method": "dense",
        "use_cfg_cache": False, "use_block_cache": False,
    },
    "dense_cfg_cache": {
        "run_kind": "cfg_cache_benchmark", "attention_method": "dense",
        "use_cfg_cache": True, "use_block_cache": False,
    },
    "sparge_topk75": {
        "run_kind": "sparge_topk75_baseline", "attention_method": "sparge",
        "sparge_topk": 0.75, "sparge_pvthreshd": 50.0,
        "sparge_smooth_k": True, "use_cfg_cache": False,
        "use_block_cache": False,
    },
    "sparge_topk50": {
        "run_kind": "sparge_baseline", "attention_method": "sparge",
        "sparge_topk": 0.5, "sparge_pvthreshd": 50.0,
        "sparge_smooth_k": True, "use_cfg_cache": False,
        "use_block_cache": False,
    },
    "radial_conservative": {
        "run_kind": "radial_conservative_baseline", "attention_method": "radial",
        "radial_profile": "conservative", "use_cfg_cache": False,
        "use_block_cache": False,
    },
    "radial_aggressive": {
        "run_kind": "radial_aggressive_baseline", "attention_method": "radial",
        "radial_profile": "aggressive", "use_cfg_cache": False,
        "use_block_cache": False,
    },
}

# Current producer schema used by fixtures. Selection enforces REQUIRED_FIELDS;
# any added columns remain cryptographically bound by the whole-file SHA256.
FINAL_FIELDS = (
    "schema_version", "method_id", "label", "status", "timing_status",
    "quality_status", "manual_review_status", "timing_csv_path",
    "timing_csv_sha256", "quality_protocol_id", "quality_protocol_sha256",
    "evaluation_matrix_id", "evaluation_matrix_sha256", "evaluator_git_commit",
    "run_dir", "run_id", "git_commit", "verification_sha256",
    "timings_sha256", "checkpoint_manifest_sha256",
    "checkpoint_fingerprint_sha256", "gpu_uuid", "gpu_name",
    "prompt_set_sha256", "prompt_count", "seed_count", "seeds",
    "sample_steps", "measurement_count", "artifact_count",
    "denoise_seconds_median", "total_generation_seconds_median",
    "artifact_ready_seconds_median", "peak_memory_allocated_gib_median",
    "peak_memory_reserved_gib_median", "denoise_speedup_vs_dense",
    "total_speedup_vs_dense", "video_psnr_db_median", "video_ssim_median",
    "lpips_alex_median", "audio_rmse_median",
    "audio_max_abs_difference_median", "audio_snr_db_median",
    "audio_correlation_median", "quality_median_path",
    "quality_median_sha256", "manual_review_row_count", "manual_pass_count",
    "manual_fail_count", "manual_uncertain_count", "manual_validation_path",
    "manual_validation_sha256", "manual_reviews_csv_path",
    "manual_reviews_csv_sha256",
)
REQUIRED_FIELDS = (
    "schema_version", "method_id", "label", "status", "timing_status",
    "quality_status", "manual_review_status", "timing_csv_path",
    "timing_csv_sha256", "quality_protocol_id", "quality_protocol_sha256",
    "evaluation_matrix_id", "evaluation_matrix_sha256", "evaluator_git_commit",
    "run_dir", "run_id", "git_commit", "verification_sha256",
    "timings_sha256", "checkpoint_manifest_sha256",
    "checkpoint_fingerprint_sha256", "gpu_uuid", "gpu_name",
    "prompt_set_sha256", "prompt_count", "seed_count", "seeds",
    "sample_steps", "measurement_count", "artifact_count",
    "total_generation_seconds_median", "quality_median_path",
    "quality_median_sha256", "manual_review_row_count", "manual_pass_count",
    "manual_fail_count", "manual_uncertain_count", "manual_validation_path",
    "manual_validation_sha256", "manual_reviews_csv_path",
    "manual_reviews_csv_sha256",
)

HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
HEX_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
CANONICAL_UINT = re.compile(r"0|[1-9][0-9]*\Z")
DECIMAL_NUMBER = re.compile(
    r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?(?:0|[1-9][0-9]*))?\Z"
)
MAX_INPUT_BYTES = 16 * 1024 * 1024


class SelectionError(ValueError):
    """The table cannot support a fail-closed sparse selection."""


def _fail(context: str, message: str) -> None:
    raise SelectionError(f"{context}: {message}")


def _require(condition: bool, context: str, message: str) -> None:
    if not condition:
        _fail(context, message)


def _json_equal(left: Any, right: Any) -> bool:
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


def _canonical_existing_path(raw: str | os.PathLike[str], context: str) -> Path:
    try:
        value = os.fspath(raw)
    except TypeError as exc:
        _fail(context, f"path is invalid: {exc}")
    _require(isinstance(value, str) and bool(value), context, "path is missing")
    _require(os.path.isabs(value), context, f"path must be absolute: {value!r}")
    _require(os.path.normpath(value) == value, context, "path is not lexically canonical")
    _require(os.path.realpath(value) == value, context, "path resolves through a symlink or alias")
    return Path(value)


def _run_git(arguments: Sequence[str], context: str) -> bytes:
    _require(GIT_BINARY.is_file() and not GIT_BINARY.is_symlink(), context,
             f"fixed Git binary is unavailable: {GIT_BINARY}")
    command = [
        str(GIT_BINARY), "-c", "core.fsmonitor=false", "-c",
        "core.untrackedCache=false", "-C", str(REPO_ROOT), *arguments,
    ]
    environment = {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }
    try:
        completed = subprocess.run(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=30, check=False, env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _fail(context, f"cannot execute fixed Git audit: {exc}")
    _require(completed.returncode == 0, context,
             f"Git audit failed: {completed.stderr[:500]!r}")
    return completed.stdout


def _scan_repository_tree(root: Path) -> None:
    context = "repository source audit"
    _require(root.is_absolute() and os.path.realpath(root) == str(root), context,
             "repository root is not canonical")
    try:
        walker = os.walk(root, topdown=True, followlinks=False)
        for directory, directory_names, file_names in walker:
            base = Path(directory)
            if base == root and ".git" in directory_names:
                directory_names.remove(".git")
            for name in list(directory_names):
                path = base / name
                metadata = os.lstat(path)
                _require(not stat.S_ISLNK(metadata.st_mode), context,
                         f"repository directory symlink is forbidden: {path}")
                _require(stat.S_ISDIR(metadata.st_mode), context,
                         f"repository directory entry is not a directory: {path}")
                _require(name != "__pycache__", context,
                         f"repository bytecode directory is forbidden: {path}")
            for name in file_names:
                if base == root and name == ".git":
                    continue
                path = base / name
                metadata = os.lstat(path)
                _require(not stat.S_ISLNK(metadata.st_mode), context,
                         f"repository source symlink is forbidden: {path}")
                _require(stat.S_ISREG(metadata.st_mode), context,
                         f"repository entry is not a regular file: {path}")
                _require(path.suffix not in {".pyc", ".pyo"}, context,
                         f"repository bytecode file is forbidden: {path}")
    except OSError as exc:
        _fail(context, f"cannot scan repository source tree: {exc}")


def _audit_repository(expected_head: str | None = None) -> str:
    context = "repository audit"
    top = _run_git(("rev-parse", "--show-toplevel"), context)
    try:
        top_text = top.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        _fail(context, f"Git top-level output is not UTF-8: {exc}")
    _require(top_text == str(REPO_ROOT), context,
             f"Git top-level differs from fixed repository: {top_text!r}")
    head_bytes = _run_git(("rev-parse", "--verify", "HEAD^{commit}"), context)
    try:
        head = head_bytes.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        _fail(context, f"Git HEAD is not ASCII: {exc}")
    _commit(head, context, "HEAD")
    if expected_head is not None:
        _require(head == expected_head, context,
                 "tracked HEAD changed during selection")
    status = _run_git(("status", "--porcelain=v1", "--untracked-files=no"),
                      context)
    _require(status == b"", context,
             "tracked repository state is not clean")
    critical = (
        str(SELECTOR_PATH.relative_to(REPO_ROOT)),
        str(PROTOCOL_PATH.relative_to(REPO_ROOT)),
        str(MATRIX_PATH.relative_to(REPO_ROOT)),
    )
    tracked = _run_git(("ls-files", "--error-unmatch", "--", *critical),
                       context)
    try:
        tracked_paths = tuple(
            line for line in tracked.decode("utf-8", errors="strict").splitlines()
            if line
        )
    except UnicodeDecodeError as exc:
        _fail(context, f"tracked source list is not UTF-8: {exc}")
    _require(len(tracked_paths) == len(critical)
             and set(tracked_paths) == set(critical), context,
             "selector, protocol, or matrix is not tracked exactly once")
    _scan_repository_tree(REPO_ROOT)
    return head


@dataclass(frozen=True)
class StableSnapshot:
    path: Path
    data: bytes
    sha256: str
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @property
    def identity(self) -> tuple[int, int, int, int, int, int]:
        return (self.device, self.inode, self.mode, self.size,
                self.mtime_ns, self.ctime_ns)

    def revalidate(self, context: str) -> None:
        try:
            current = os.lstat(self.path)
        except OSError as exc:
            _fail(context, f"cannot re-stat {self.path}: {exc}")
        observed = (current.st_dev, current.st_ino, current.st_mode,
                    current.st_size, current.st_mtime_ns, current.st_ctime_ns)
        _require(
            stat.S_ISREG(current.st_mode) and observed == self.identity
            and os.path.realpath(self.path) == str(self.path),
            context, f"input changed after its stable snapshot: {self.path}",
        )


def _snapshot_file(raw: str | os.PathLike[str], context: str) -> StableSnapshot:
    path = _canonical_existing_path(raw, context)
    try:
        before = os.lstat(path)
    except OSError as exc:
        _fail(context, f"cannot lstat {path}: {exc}")
    _require(stat.S_ISREG(before.st_mode), context, f"not a regular file: {path}")
    _require(before.st_size <= MAX_INPUT_BYTES, context, "file is too large")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        _require(stat.S_ISREG(opened.st_mode), context, "opened input is not regular")
        _require((opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
                 context, "file identity changed while opening")
        _require(opened.st_size <= MAX_INPUT_BYTES, context, "opened file is too large")
        chunks: list[bytes] = []
        total = 0
        while True:
            remaining = MAX_INPUT_BYTES + 1 - total
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            total += len(chunk)
            _require(total <= MAX_INPUT_BYTES, context, "file grew beyond the read limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        _fail(context, f"cannot read {path}: {exc}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    opened_identity = (opened.st_dev, opened.st_ino, opened.st_mode,
                       opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_mode,
                      after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    _require(opened_identity == after_identity, context, "file changed while reading")
    data = b"".join(chunks)
    _require(len(data) == after.st_size, context, "short read")
    result = StableSnapshot(
        path, data, hashlib.sha256(data).hexdigest(), after.st_dev, after.st_ino,
        after.st_mode, after.st_size, after.st_mtime_ns, after.st_ctime_ns,
    )
    result.revalidate(context)
    return result


def _reject_nonfinite_json(value: Any, context: str) -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), context, "JSON contains a non-finite number")
    elif isinstance(value, list):
        for item in value:
            _reject_nonfinite_json(item, context)
    elif isinstance(value, dict):
        for item in value.values():
            _reject_nonfinite_json(item, context)


def _strict_json(snapshot: StableSnapshot, context: str) -> dict[str, Any]:
    try:
        text = snapshot.data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        _fail(context, f"JSON is not UTF-8: {exc}")
    _require(not text.startswith("\ufeff"), context, "UTF-8 BOM is forbidden")

    def pairs_hook(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(context, f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        _fail(context, f"non-finite JSON constant {token!r} is forbidden")

    try:
        value = json.loads(text, object_pairs_hook=pairs_hook,
                           parse_constant=reject_constant)
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        _fail(context, f"invalid JSON: {exc}")
    _require(isinstance(value, dict), context, "JSON root must be an object")
    _reject_nonfinite_json(value, context)
    return value


def _strict_csv(snapshot: StableSnapshot) -> list[dict[str, str]]:
    context = "final CSV"
    try:
        text = snapshot.data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        _fail(context, f"CSV is not UTF-8: {exc}")
    _require(not text.startswith("\ufeff"), context, "UTF-8 BOM is forbidden")
    _require("\x00" not in text, context, "CSV contains a NUL byte")
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        rows = list(reader)
    except csv.Error as exc:
        _fail(context, f"invalid CSV: {exc}")
    fields = tuple(reader.fieldnames or ())
    _require(all(isinstance(field, str) and bool(field) for field in fields),
             context, "header contains an empty field")
    _require(len(fields) == len(set(fields)), context,
             "header contains a duplicate field")
    missing = sorted(set(REQUIRED_FIELDS) - set(fields))
    _require(not missing, context,
             f"header omits required selection fields: {missing}")
    _require(len(rows) == len(METHOD_IDS), context, "exactly the A--F rows are required")
    for number, row in enumerate(rows, start=2):
        _require(None not in row, context, f"row {number} contains extra fields")
        _require(set(row) == set(fields)
                 and all(isinstance(value, str) for value in row.values()),
                 context, f"row {number} contains a missing field")
    return rows


def _full_sha(value: Any, context: str, field: str) -> str:
    _require(isinstance(value, str) and HEX_SHA256.fullmatch(value) is not None, context,
             f"{field} is not a lowercase full SHA256")
    return value


def _commit(value: Any, context: str, field: str) -> str:
    _require(isinstance(value, str) and HEX_COMMIT.fullmatch(value) is not None, context,
             f"{field} is not a lowercase full Git commit")
    return value


def _uint(value: str, context: str, field: str, *, positive: bool = False) -> int:
    _require(CANONICAL_UINT.fullmatch(value) is not None, context,
             f"{field} is not a canonical unsigned integer")
    result = int(value)
    if positive:
        _require(result > 0, context, f"{field} must be positive")
    return result


def _positive_decimal(value: str, context: str, field: str) -> Decimal:
    _require(DECIMAL_NUMBER.fullmatch(value) is not None, context,
             f"{field} is not a canonical decimal")
    try:
        result = Decimal(value)
    except InvalidOperation:
        _fail(context, f"{field} is not numeric")
    _require(result.is_finite() and result > 0, context,
             f"{field} must be finite and positive")
    return result


def _bound_absolute_path(value: str, context: str, field: str) -> str:
    _require(bool(value) and os.path.isabs(value), context,
             f"{field} must be absolute")
    _require(os.path.normpath(value) == value, context,
             f"{field} is not lexically canonical")
    return value


def _load_protocol_and_matrix(
    protocol_snapshot: StableSnapshot,
    matrix_snapshot: StableSnapshot,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], int]:
    protocol = _strict_json(protocol_snapshot, "quality protocol")
    matrix = _strict_json(matrix_snapshot, "evaluation matrix")
    _require(protocol.get("schema_version") == 2, "quality protocol",
             "schema_version must be 2")
    _require(protocol.get("protocol_id") == "ovi_720x720_5s_dense_pair_quality_v2",
             "quality protocol", "protocol_id differs from the fixed Ovi protocol")
    _require(protocol.get("reference_method_id") == "dense", "quality protocol",
             "reference method must be Dense")
    lpips = protocol.get("lpips")
    _require(isinstance(lpips, dict), "quality protocol",
             "LPIPS dependency contract is missing")
    _require(lpips.get("trusted_lock_status") == "pinned", "quality protocol",
             "LPIPS trusted_lock_status must be pinned")
    _full_sha(lpips.get("trusted_environment_lock_sha256"), "quality protocol",
              "trusted_environment_lock_sha256")
    locked_packages = lpips.get("trusted_environment_packages")
    _require(isinstance(locked_packages, list) and bool(locked_packages),
             "quality protocol", "trusted environment package lock is empty")
    locked_names: set[str] = set()
    for index, package in enumerate(locked_packages):
        context = f"quality protocol locked package {index}"
        _require(isinstance(package, dict), context, "package is not an object")
        name = package.get("distribution")
        _require(isinstance(name, str) and bool(name) and name not in locked_names,
                 context, "distribution is missing or duplicated")
        locked_names.add(name)
        _require(isinstance(package.get("version"), str) and bool(package["version"]),
                 context, "version is missing")
        _require(isinstance(package.get("archive_url"), str)
                 and package["archive_url"].startswith("https://"),
                 context, "archive URL is not HTTPS")
        _full_sha(package.get("archive_sha256"), context, "archive_sha256")
    direct_packages = lpips.get("packages")
    _require(isinstance(direct_packages, list) and bool(direct_packages),
             "quality protocol", "direct package contract is empty")
    for index, package in enumerate(direct_packages):
        context = f"quality protocol direct package {index}"
        _require(isinstance(package, dict), context, "package is not an object")
        _full_sha(package.get("trusted_archive_sha256"), context,
                  "trusted_archive_sha256")
    weights = lpips.get("weights")
    _require(isinstance(weights, list) and bool(weights), "quality protocol",
             "weight contract is empty")
    for index, weight in enumerate(weights):
        context = f"quality protocol weight {index}"
        _require(isinstance(weight, dict), context, "weight is not an object")
        _full_sha(weight.get("trusted_sha256"), context, "trusted_sha256")
    measurements = protocol.get("measurement_indices")
    _require(
        isinstance(measurements, list) and bool(measurements)
        and all(type(item) is int for item in measurements)
        and measurements == list(range(len(measurements))),
        "quality protocol", "measurement indices must be contiguous integers",
    )
    manual = protocol.get("manual_reviews")
    _require(isinstance(manual, dict), "quality protocol",
             "manual review contract is missing")
    _require(
        manual.get("allowed_sync_ratings") == ["pass", "fail", "uncertain"]
        and manual.get("automatic_population_forbidden") is True,
        "quality protocol", "manual ratings or human-only policy drifted",
    )

    _require(matrix.get("schema_version") == 1, "evaluation matrix",
             "schema_version must be 1")
    _require(matrix.get("matrix_id") == FORMAL_MATRIX_ID,
             "evaluation matrix", "matrix_id differs from formal8 v2")
    fixed = matrix.get("fixed_protocol")
    _require(isinstance(fixed, dict), "evaluation matrix",
             "fixed_protocol is missing")
    fixed_values = {
        "measurement_runs": FORMAL_MEASUREMENT_COUNT,
        "prompt_count": FORMAL_PROMPT_COUNT,
        "each_example_n_times": FORMAL_SEED_COUNT,
        "sample_steps": FORMAL_SAMPLE_STEPS,
        "seed": 103,
        "prompts_sha256": FORMAL_PROMPTS_SHA256,
    }
    for field, expected in fixed_values.items():
        _require(fixed.get(field) == expected, "evaluation matrix",
                 f"{field} differs from the fixed formal8 contract")
    _require(measurements == list(range(FORMAL_MEASUREMENT_COUNT)),
             "quality protocol", "measurement indices differ from formal 0,1,2")
    artifact_count = (fixed["measurement_runs"] * fixed["prompt_count"]
                      * fixed["each_example_n_times"])
    _require(artifact_count == FORMAL_ARTIFACT_COUNT, "evaluation matrix",
             "formal artifact count is not 72")
    methods = matrix.get("methods")
    _require(isinstance(methods, list) and len(methods) >= len(METHOD_IDS),
             "evaluation matrix", "method list is incomplete")
    method_map: dict[str, dict[str, Any]] = {}
    for index, method_id in enumerate(METHOD_IDS):
        context = f"evaluation matrix method {method_id}"
        method = methods[index]
        _require(isinstance(method, dict), context, "method is not an object")
        _require(method.get("method_id") == method_id, context,
                 "A--F method order changed")
        _require(method.get("formal_slot") == FORMAL_SLOTS[method_id], context,
                 "formal slot changed")
        _require(method.get("required") is True, context,
                 "formal method is no longer required")
        _require(method.get("implementation_status") == "ready", context,
                 "formal method is not ready")
        _require(isinstance(method.get("label"), str) and bool(method["label"]),
                 context, "label is missing")
        _require(_json_equal(method.get("expected_environment"),
                             METHOD_ENVIRONMENT_CONTRACTS[method_id]),
                 context, "immutable method profile changed")
        method_map[method_id] = method
    return protocol, matrix, method_map, artifact_count


def _validate_final_rows(
    rows: Sequence[dict[str, str]],
    protocol: Mapping[str, Any],
    protocol_snapshot: StableSnapshot,
    matrix: Mapping[str, Any],
    matrix_snapshot: StableSnapshot,
    method_map: Mapping[str, Mapping[str, Any]],
    artifact_count: int,
    repository_head: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    methods: list[dict[str, Any]] = []
    # These values must describe one fixed formal experiment across all rows.
    shared_fields = (
        "timing_csv_path", "timing_csv_sha256", "quality_protocol_id",
        "quality_protocol_sha256", "evaluation_matrix_id",
        "evaluation_matrix_sha256", "evaluator_git_commit", "git_commit",
        "checkpoint_fingerprint_sha256", "gpu_uuid", "gpu_name",
        "prompt_set_sha256", "prompt_count", "seed_count", "seeds",
        "sample_steps", "measurement_count", "artifact_count",
    )
    shared_reference: dict[str, str] | None = None
    for index, method_id in enumerate(METHOD_IDS):
        row = rows[index]
        context = f"final CSV row {FORMAL_SLOTS[method_id]} ({method_id})"
        _require(row["schema_version"] == "1", context,
                 "schema_version must be 1")
        _require(row["method_id"] == method_id, context,
                 "A--F method order changed")
        _require(row["label"] == method_map[method_id]["label"], context,
                 "label differs from the matrix")
        _require(row["status"] == "complete", context,
                 "row status is not complete")
        _require(row["timing_status"] == "valid", context,
                 "timing status is not valid")
        _require(row["quality_protocol_id"] == protocol["protocol_id"], context,
                 "quality protocol id differs")
        _require(row["quality_protocol_sha256"] == protocol_snapshot.sha256,
                 context, "quality protocol SHA256 differs")
        _require(row["evaluation_matrix_id"] == matrix["matrix_id"], context,
                 "evaluation matrix id differs")
        _require(row["evaluation_matrix_sha256"] == matrix_snapshot.sha256,
                 context, "evaluation matrix SHA256 differs")
        _bound_absolute_path(row["timing_csv_path"], context, "timing_csv_path")
        for field in (
            "timing_csv_sha256", "quality_protocol_sha256",
            "evaluation_matrix_sha256", "verification_sha256",
            "timings_sha256", "checkpoint_manifest_sha256",
            "checkpoint_fingerprint_sha256", "prompt_set_sha256",
        ):
            _full_sha(row[field], context, field)
        _require(_commit(row["evaluator_git_commit"], context,
                         "evaluator_git_commit") == repository_head,
                 context, "evaluator_git_commit differs from current clean HEAD")
        _require(_commit(row["git_commit"], context, "git_commit")
                 == repository_head, context,
                 "run git_commit differs from current clean HEAD")
        _require(row["prompt_set_sha256"] == FORMAL_PROMPTS_SHA256, context,
                 "prompt_set_sha256 differs from formal8")
        _require(_uint(row["prompt_count"], context, "prompt_count",
                       positive=True) == FORMAL_PROMPT_COUNT,
                 context, "prompt_count differs from formal8")
        _require(_uint(row["seed_count"], context, "seed_count",
                       positive=True) == FORMAL_SEED_COUNT,
                 context, "seed_count differs from formal 3-seed schedule")
        _require(row["seeds"] == "103;104;105", context,
                 "seeds differ from the fixed formal schedule")
        _require(_uint(row["sample_steps"], context, "sample_steps",
                       positive=True) == FORMAL_SAMPLE_STEPS,
                 context, "sample_steps differs from formal 50-step protocol")
        _require(bool(row["run_dir"]) and bool(row["run_id"]), context,
                 "run binding is incomplete")
        total_seconds = _positive_decimal(
            row["total_generation_seconds_median"], context,
            "total_generation_seconds_median",
        )
        _require(_uint(row["measurement_count"], context, "measurement_count",
                       positive=True) == FORMAL_MEASUREMENT_COUNT,
                 context, "measurement_count differs")
        _require(_uint(row["artifact_count"], context, "artifact_count",
                       positive=True) == FORMAL_ARTIFACT_COUNT == artifact_count,
                 context, "artifact_count differs")
        current_shared = {field: row[field] for field in shared_fields}
        if shared_reference is None:
            shared_reference = current_shared
        else:
            _require(current_shared == shared_reference, context,
                     "cross-method fixed protocol binding differs")

        if method_id == "dense":
            _require(row["quality_status"] == "reference", context,
                     "Dense quality status is not reference")
            _require(row["manual_review_status"] == "reference", context,
                     "Dense manual status is not reference")
        else:
            _require(row["quality_status"] == "complete", context,
                     "quality status is not complete")
            _require(row["manual_review_status"] == "complete", context,
                     "manual review status is not complete")
            for field in ("quality_median_path", "manual_validation_path",
                          "manual_reviews_csv_path"):
                _bound_absolute_path(row[field], context, field)
            for field in ("quality_median_sha256", "manual_validation_sha256",
                          "manual_reviews_csv_sha256"):
                _full_sha(row[field], context, field)

        binding: dict[str, Any] = {
            "formal_slot": FORMAL_SLOTS[method_id],
            "method_id": method_id,
            "label": row["label"],
            "status": row["status"],
            "timing_status": row["timing_status"],
            "quality_status": row["quality_status"],
            "manual_review_status": row["manual_review_status"],
            "total_generation_seconds_median": row["total_generation_seconds_median"],
            "run_id": row["run_id"],
            "git_commit": row["git_commit"],
            "timing_csv_sha256": row["timing_csv_sha256"],
            "verification_sha256": row["verification_sha256"],
            "timings_sha256": row["timings_sha256"],
            "checkpoint_manifest_sha256": row["checkpoint_manifest_sha256"],
            "checkpoint_fingerprint_sha256": row[
                "checkpoint_fingerprint_sha256"
            ],
            "profile": dict(method_map[method_id]["expected_environment"]),
        }
        if method_id != "dense":
            binding.update({
                "quality_median_sha256": row["quality_median_sha256"],
                "manual_validation_sha256": row["manual_validation_sha256"],
                "manual_reviews_csv_sha256": row["manual_reviews_csv_sha256"],
            })
        if method_id in CANDIDATE_METHOD_IDS:
            row_count = _uint(row["manual_review_row_count"], context,
                              "manual_review_row_count", positive=True)
            pass_count = _uint(row["manual_pass_count"], context,
                               "manual_pass_count")
            fail_count = _uint(row["manual_fail_count"], context,
                               "manual_fail_count")
            uncertain_count = _uint(row["manual_uncertain_count"], context,
                                    "manual_uncertain_count")
            _require(row_count == artifact_count, context,
                     "manual review row count differs from artifact count")
            _require(pass_count + fail_count + uncertain_count == row_count,
                     context, "manual rating counts differ from row_count")
            eligible = (pass_count == row_count and fail_count == 0
                        and uncertain_count == 0)
            binding.update({
                "manual_review_row_count": row_count,
                "manual_pass_count": pass_count,
                "manual_fail_count": fail_count,
                "manual_uncertain_count": uncertain_count,
                "eligible": eligible,
                "ineligibility_reason": (
                    "" if eligible else "one_or_more_manual_reviews_not_pass"
                ),
            })
            candidate = dict(binding)
            candidate.update({
                "ranking_value": total_seconds,
            })
            candidates.append(candidate)
        methods.append(binding)
    _require([item["method_id"] for item in candidates]
             == list(CANDIDATE_METHOD_IDS), "final CSV",
             "candidate set differs from C--F")
    eligible_candidates = [item for item in candidates if item["eligible"]]
    _require(bool(eligible_candidates), "final CSV",
             "no C--F candidate has all manual reviews pass")
    winner = min(eligible_candidates,
                 key=lambda item: (item["ranking_value"], item["method_id"]))
    return methods, candidates, winner


@dataclass(frozen=True)
class OutputTarget:
    path: Path
    directory_fd: int
    parent_device: int
    parent_inode: int


def _output_path(raw: str | os.PathLike[str]) -> OutputTarget:
    context = "selection output"
    try:
        value = os.fspath(raw)
    except TypeError as exc:
        _fail(context, f"path is invalid: {exc}")
    _require(isinstance(value, str) and bool(value), context, "path is missing")
    _require(os.path.isabs(value), context, "path must be absolute")
    _require(os.path.normpath(value) == value, context,
             "path is not lexically canonical")
    path = Path(value)
    _require(os.path.realpath(path.parent) == str(path.parent), context,
             "parent resolves through a symlink or alias")
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(path.parent, directory_flags)
    except OSError as exc:
        _fail(context, f"cannot open output parent {path.parent}: {exc}")
    try:
        opened_parent = os.fstat(directory_fd)
        visible_parent = os.lstat(path.parent)
        _require(
            stat.S_ISDIR(opened_parent.st_mode)
            and (opened_parent.st_dev, opened_parent.st_ino)
            == (visible_parent.st_dev, visible_parent.st_ino)
            and os.path.realpath(path.parent) == str(path.parent),
            context, "output parent identity changed during preflight",
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
            parent_device=opened_parent.st_dev,
            parent_inode=opened_parent.st_ino,
        )
    except SelectionError:
        os.close(directory_fd)
        raise
    except OSError as exc:
        os.close(directory_fd)
        _fail(context, f"cannot validate output target {path}: {exc}")


def _without_ranking_value(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "ranking_value"}


def _render_receipt(
    final_snapshot: StableSnapshot,
    protocol_snapshot: StableSnapshot,
    protocol: Mapping[str, Any],
    matrix_snapshot: StableSnapshot,
    matrix: Mapping[str, Any],
    selector_snapshot: StableSnapshot,
    repository_head: str,
    methods: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    winner: Mapping[str, Any],
) -> bytes:
    winner_payload = {
        "formal_slot": winner["formal_slot"],
        "method_id": winner["method_id"],
        "label": winner["label"],
        "selected_sparse_profile": winner["method_id"],
        "total_generation_seconds_median": winner[
            "total_generation_seconds_median"
        ],
        "profile": winner["profile"],
        "verification_sha256": winner["verification_sha256"],
        "timings_sha256": winner["timings_sha256"],
        "quality_median_sha256": winner["quality_median_sha256"],
        "manual_validation_sha256": winner["manual_validation_sha256"],
        "manual_reviews_csv_sha256": winner["manual_reviews_csv_sha256"],
    }
    receipt = {
        "schema_version": 1,
        "record_type": "ovi_sparse_winner_selection",
        "status": "complete",
        "created_by": "scripts/select_ovi_sparse_winner.py",
        "selector": {
            "path": str(selector_snapshot.path),
            "sha256": selector_snapshot.sha256,
        },
        "repository": {
            "path": str(REPO_ROOT),
            "head_commit": repository_head,
            "tracked_status": "clean",
            "bytecode_status": "absent",
            "source_symlink_status": "absent",
        },
        "final_csv": {
            "path": str(final_snapshot.path),
            "bytes": final_snapshot.size,
            "sha256": final_snapshot.sha256,
        },
        "quality_protocol": {
            "path": str(protocol_snapshot.path),
            "schema_version": protocol["schema_version"],
            "protocol_id": protocol["protocol_id"],
            "sha256": protocol_snapshot.sha256,
        },
        "evaluation_matrix": {
            "path": str(matrix_snapshot.path),
            "schema_version": matrix["schema_version"],
            "matrix_id": matrix["matrix_id"],
            "sha256": matrix_snapshot.sha256,
        },
        "selection_rule": {
            "candidate_formal_slots": ["C", "D", "E", "F"],
            "candidate_method_ids": list(CANDIDATE_METHOD_IDS),
            "completeness_requirements": [
                "all_A_to_F_status_complete",
                "all_A_to_F_timing_status_valid",
                "all_B_to_F_quality_and_manual_status_complete",
                "all_C_to_F_manual_rating_counts_complete",
            ],
            "candidate_eligibility": "all_manual_reviews_pass",
            "ranking_metric": "total_generation_seconds_median",
            "direction": "minimum",
            "tie_breaker": "method_id_lexicographic_ascending",
        },
        "trust_model": {
            "runtime_evidence_root": "final_csv",
            "runtime_evidence_input_count": 1,
            "binding": "whole_file_sha256",
            "checked_in_policy_inputs": [
                "quality_protocol", "evaluation_matrix", "selector"
            ],
        },
        "method_bindings": list(methods),
        "candidate_bindings": [
            _without_ranking_value(candidate) for candidate in candidates
        ],
        "winner": winner_payload,
    }
    return (
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True,
                   allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_atomic_exclusive(
    target: OutputTarget,
    payload: bytes,
    snapshots: Iterable[StableSnapshot],
    repository_head: str,
) -> None:
    context = "selection output"
    output = target.path
    temp_name = f".{output.name}.{secrets.token_hex(16)}.tmp"
    directory_fd = target.directory_fd
    descriptor = -1
    temp_created = False
    publication_complete = False
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
            context, "output parent changed before publication",
        )
        flags = (
            os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(temp_name, flags, 0o600, dir_fd=directory_fd)
        temp_created = True
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            _require(written > 0, context, "zero-byte output write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        _audit_repository(expected_head=repository_head)
        for snapshot in snapshots:
            snapshot.revalidate(context)
        try:
            temporary = os.stat(temp_name, dir_fd=directory_fd,
                                follow_symlinks=False)
            os.link(temp_name, output.name, src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd, follow_symlinks=False)
            published_identity = (temporary.st_dev, temporary.st_ino)
        except FileExistsError:
            _fail(context, f"refusing to overwrite {output}")
        published = os.stat(output.name, dir_fd=directory_fd,
                            follow_symlinks=False)
        _require(stat.S_ISREG(published.st_mode), context,
                 "published output is not regular")
        _require(
            (published.st_dev, published.st_ino) == published_identity,
            context, "published output differs from the staged receipt",
        )
        _require(published.st_size == len(payload), context,
                 "published output size differs")
        current_parent = os.lstat(output.parent)
        _require(
            (current_parent.st_dev, current_parent.st_ino)
            == (opened_parent.st_dev, opened_parent.st_ino)
            and os.path.realpath(output.parent) == str(output.parent),
            context, "output parent changed during publication",
        )
        os.unlink(temp_name, dir_fd=directory_fd)
        temp_created = False
        os.fsync(directory_fd)
        visible = os.lstat(output)
        _require(
            stat.S_ISREG(visible.st_mode)
            and (visible.st_dev, visible.st_ino)
            == (published.st_dev, published.st_ino),
            context, "published output identity is not stable",
        )
        current_parent = os.lstat(output.parent)
        _require(
            (current_parent.st_dev, current_parent.st_ino)
            == (target.parent_device, target.parent_inode)
            and os.path.realpath(output.parent) == str(output.parent),
            context, "output parent changed after publication",
        )
        publication_complete = True
    except SelectionError:
        raise
    except OSError as exc:
        _fail(context, f"cannot publish {output}: {exc}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not publication_complete and published_identity is not None:
            try:
                abandoned = os.stat(output.name, dir_fd=directory_fd,
                                    follow_symlinks=False)
                if (abandoned.st_dev, abandoned.st_ino) == published_identity:
                    os.unlink(output.name, dir_fd=directory_fd)
            except OSError:
                pass
        if temp_created:
            try:
                os.unlink(temp_name, dir_fd=directory_fd)
            except OSError:
                pass
        os.close(directory_fd)


def select_sparse_winner(
    *, final_csv: str | os.PathLike[str], output: str | os.PathLike[str]
) -> Path:
    repository_head = _audit_repository()
    final_snapshot = _snapshot_file(final_csv, "final CSV")
    protocol_snapshot = _snapshot_file(PROTOCOL_PATH, "quality protocol")
    matrix_snapshot = _snapshot_file(MATRIX_PATH, "evaluation matrix")
    selector_snapshot = _snapshot_file(SELECTOR_PATH, "winner selector")
    snapshots = (final_snapshot, protocol_snapshot, matrix_snapshot,
                 selector_snapshot)
    identities = {(item.device, item.inode) for item in snapshots}
    _require(len(identities) == len(snapshots), "selection inputs",
             "two inputs share one file identity")
    rows = _strict_csv(final_snapshot)
    protocol, matrix, method_map, artifact_count = _load_protocol_and_matrix(
        protocol_snapshot, matrix_snapshot
    )
    methods, candidates, winner = _validate_final_rows(
        rows, protocol, protocol_snapshot, matrix, matrix_snapshot,
        method_map, artifact_count, repository_head,
    )
    payload = _render_receipt(
        final_snapshot, protocol_snapshot, protocol, matrix_snapshot, matrix,
        selector_snapshot, repository_head, methods, candidates, winner,
    )
    output_target = _output_path(output)
    _write_atomic_exclusive(output_target, payload, snapshots, repository_head)
    return output_target.path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select the fastest fully accepted C--F sparse profile."
    )
    parser.add_argument("--final-csv", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        output = select_sparse_winner(final_csv=args.final_csv,
                                      output=args.output)
    except SelectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
