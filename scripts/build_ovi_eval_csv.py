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
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
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
MEASUREMENT_COUNT = 3
GIB = 1024 ** 3
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
HEX_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")

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
    "timings_sha256",
    "git_commit",
    "checkpoint_manifest_sha256",
    "checkpoint_fingerprint_sha256",
    "gpu_uuid",
    "gpu_name",
    "prompt_sha256",
    "prompt",
    "seed",
    "requested_height",
    "requested_width",
    "actual_height",
    "actual_width",
    "sample_steps",
    "measurement_count",
    "measurement_indices",
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


def _fail(context: str, message: str) -> None:
    raise EvaluationError(f"{context}: {message}")


def _require(condition: bool, context: str, message: str) -> None:
    if not condition:
        _fail(context, message)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EvaluationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


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


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    path = Path(path)
    manifest = _read_json(path, "evaluation manifest")
    _require(isinstance(manifest, dict), "evaluation manifest", "root must be an object")
    _require(
        manifest.get("schema_version") == 1,
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
    monitor: Any, environment: dict[str, Any], context: str
) -> None:
    _require(isinstance(monitor, dict), context, "gpu_process_monitor must be an object")
    required_true = (
        "identity_consistent",
        "single_distinct_host_pid",
        "exact_singleton_process_per_sample",
        "valid_for_benchmark",
    )
    for field in required_true:
        _require(monitor.get(field) is True, context, f"GPU monitor {field} must be true")
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
    _require(monitor.get("unavailable_sample_count") == 0, context, "GPU samples were unavailable")
    sample_count = monitor.get("sample_count")
    _require(
        isinstance(sample_count, int) and sample_count >= 2,
        context,
        "GPU monitor requires at least entry and exit samples",
    )
    _require(
        monitor.get("available_sample_count") == sample_count,
        context,
        "not all GPU samples were available",
    )
    _require(monitor.get("min_process_count") == 1, context, "GPU sample process count was not one")
    _require(monitor.get("max_process_count") == 1, context, "GPU sample process count was not one")
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
        and isinstance(distinct_pids[0], int)
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
    for sample_index, sample in enumerate(samples):
        sample_context = f"{context} sample[{sample_index}]"
        _require(isinstance(sample, dict), sample_context, "sample must be an object")
        _require(sample.get("available") is True, sample_context, "sample is unavailable")
        _require(sample.get("process_count") == 1, sample_context, "sample must contain one process")
        _require(sample.get("device_index") == 0, sample_context, "physical GPU index must be zero")
        _require(sample.get("device_uuid") == environment.get("gpu_uuid"), sample_context, "GPU UUID differs")
        _require(sample.get("device_name") == environment.get("gpu_name"), sample_context, "GPU name differs")
        processes = sample.get("processes")
        _require(
            isinstance(processes, list)
            and len(processes) == 1
            and isinstance(processes[0], dict)
            and processes[0].get("host_pid") == distinct_pids[0],
            sample_context,
            "sample process evidence differs from the stable benchmark PID",
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
    checkpoint_path = run_dir / "checkpoint_manifest.json"
    environment = _read_json(environment_path, context)
    verification = _read_json(verification_path, context)
    timings = _read_jsonl(timings_path, context)
    checkpoint_manifest = _read_json(checkpoint_path, context)
    for name, payload in (
        ("environment.json", environment),
        ("verification.json", verification),
        ("checkpoint_manifest.json", checkpoint_manifest),
    ):
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
    _require(environment.get("gpu_physical_index") == 0, context, "physical GPU index must be zero")
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

    _validate_expected_fields(environment, fixed_protocol, context)
    _validate_expected_fields(environment, method["expected_environment"], context)
    _require(environment.get("measurement_runs") == MEASUREMENT_COUNT, context, "measurement_runs must equal three")
    _require(environment.get("expected_measurement_records") == MEASUREMENT_COUNT, context, "expected_measurement_records must equal three")
    _require(
        protocol.get("expected_measurement_records") == MEASUREMENT_COUNT
        and protocol.get("observed_measurement_records") == MEASUREMENT_COUNT,
        context,
        "verification protocol does not contain exactly three measurements",
    )
    _require(
        protocol.get("expected_warmup_records") == 1
        and protocol.get("observed_warmup_records") == 1,
        context,
        "verification protocol does not contain exactly one excluded warm-up",
    )
    _require(len(timings) == MEASUREMENT_COUNT, context, "timings.jsonl must contain exactly three measurements")

    checkpoint_manifest_sha256 = _sha256(checkpoint_path)
    evidence_hashes = environment.get("evidence_file_sha256")
    _require(isinstance(evidence_hashes, dict), context, "environment evidence hashes are missing")
    _require(
        evidence_hashes.get("checkpoint_manifest.json") == checkpoint_manifest_sha256,
        context,
        "checkpoint manifest hash differs from environment evidence",
    )
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

    indices = [record.get("measurement_index") for record in timings]
    _require(
        all(isinstance(index, int) and not isinstance(index, bool) for index in indices),
        context,
        f"measurement indices must be integers, found {indices!r}",
    )
    _require(
        len(set(indices)) == MEASUREMENT_COUNT,
        context,
        f"measurement indices must not be duplicated, found {indices!r}",
    )
    _require(
        set(indices) == set(range(MEASUREMENT_COUNT)),
        context,
        f"measurement indices must be exactly 0,1,2, found {indices!r}",
    )

    verified_artifacts = verification.get("artifacts")
    _require(
        verification.get("artifact_count") == MEASUREMENT_COUNT
        and isinstance(verified_artifacts, list)
        and len(verified_artifacts) == MEASUREMENT_COUNT,
        context,
        "verification must contain exactly three artifacts",
    )
    verified_by_path: dict[Path, str] = {}
    for report_index, report in enumerate(verified_artifacts):
        report_context = f"{context} verified artifact[{report_index}]"
        _require(isinstance(report, dict), report_context, "artifact report must be an object")
        _require(report.get("status") == "ok", report_context, "artifact status is not ok")
        _require(report.get("errors") == [], report_context, "artifact report contains errors")
        report_path_value = report.get("path")
        _require(isinstance(report_path_value, str) and report_path_value, report_context, "artifact path is missing")
        report_path = Path(report_path_value).resolve()
        _require(report_path.parent == run_dir, report_context, "artifact is outside the selected run directory")
        report_hash = report.get("sha256")
        _require(
            isinstance(report_hash, str) and HEX_SHA256.fullmatch(report_hash) is not None,
            report_context,
            "artifact SHA256 is invalid",
        )
        _require(report_path not in verified_by_path, report_context, "artifact path is duplicated")
        verified_by_path[report_path] = report_hash

    denoise_values = []
    total_values = []
    artifact_ready_values = []
    allocated_values = []
    reserved_values = []
    prompts = set()
    seeds = set()
    actual_shapes = set()
    generated_video_shapes = set()
    generated_audio_shapes = set()
    timing_paths = set()
    artifact_hashes = []
    metrics_sidecar_hashes = []

    for record_index, record in enumerate(timings):
        record_context = f"{context} measurement[{record_index}]"
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
        _require(seed == environment.get("seed"), record_context, "seed differs from environment")
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

        _validate_gpu_monitor(record.get("gpu_process_monitor"), environment, record_context)

        output_path_value = record.get("output_path")
        output_hash = record.get("output_sha256")
        _require(isinstance(output_path_value, str) and output_path_value, record_context, "output_path is missing")
        _require(
            isinstance(output_hash, str) and HEX_SHA256.fullmatch(output_hash) is not None,
            record_context,
            "output_sha256 is invalid",
        )
        output_path = Path(output_path_value).resolve()
        _require(output_path.parent == run_dir, record_context, "output artifact is outside the selected run directory")
        _require(output_path not in timing_paths, record_context, "output artifact path is duplicated")
        _require(output_path.is_file(), record_context, f"output artifact is missing: {output_path}")
        metrics_path = output_path.with_suffix(".metrics.json")
        metrics_sidecar = _read_json(metrics_path, record_context)
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
        actual_hash = _sha256(output_path)
        _require(actual_hash == output_hash, record_context, "output artifact SHA256 differs from timing record")
        _require(
            verified_by_path.get(output_path) == actual_hash,
            record_context,
            "output artifact SHA256 differs from verification report",
        )

        denoise_values.append(denoise)
        total_values.append(total)
        artifact_ready_values.append(artifact_ready)
        allocated_values.append(allocated)
        reserved_values.append(reserved)
        prompts.add(prompt)
        seeds.add(seed)
        actual_shapes.add(actual_shape)
        generated_video_shapes.add(generated_video_shape)
        generated_audio_shapes.add(generated_audio_shape)
        timing_paths.add(output_path)
        artifact_hashes.append(actual_hash)
        metrics_sidecar_hashes.append(_sha256(metrics_path))

    _require(timing_paths == set(verified_by_path), context, "timing artifacts differ from verified artifacts")
    _require(len(prompts) == 1, context, "measurements do not use exactly one prompt")
    _require(len(seeds) == 1, context, "measurements do not use exactly one seed")
    _require(len(actual_shapes) == 1, context, "measurements have inconsistent actual shapes")
    _require(len(generated_video_shapes) == 1, context, "measurements have inconsistent video tensor shapes")
    _require(len(generated_audio_shapes) == 1, context, "measurements have inconsistent audio tensor shapes")

    prompt = next(iter(prompts))
    seed = next(iter(seeds))
    actual_shape = next(iter(actual_shapes))
    generated_video_shape = next(iter(generated_video_shapes))
    generated_audio_shape = next(iter(generated_audio_shapes))
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    comparison_values = {
        "git_commit": git_commit,
        "checkpoint_fingerprint_sha256": checkpoint_fingerprint,
        "gpu_identity": (
            environment.get("gpu_physical_index"),
            environment.get("gpu_uuid"),
            environment.get("gpu_name"),
        ),
        "prompt": prompt,
        "seed": seed,
        "requested_shape": requested_shape,
        "actual_shape": actual_shape,
        "generated_video_shape": generated_video_shape,
        "generated_audio_shape": generated_audio_shape,
        "sample_steps": environment.get("sample_steps"),
    }
    return {
        "run_dir": str(run_dir),
        "run_id": environment.get("run_id"),
        "verification_sha256": _sha256(verification_path),
        "timings_sha256": _sha256(timings_path),
        "git_commit": git_commit,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "checkpoint_fingerprint_sha256": checkpoint_fingerprint,
        "gpu_uuid": environment.get("gpu_uuid"),
        "gpu_name": environment.get("gpu_name"),
        "prompt_sha256": prompt_sha256,
        "prompt": prompt,
        "seed": seed,
        "requested_height": requested_shape[0],
        "requested_width": requested_shape[1],
        "actual_height": actual_shape[0],
        "actual_width": actual_shape[1],
        "sample_steps": environment.get("sample_steps"),
        "measurement_count": len(timings),
        "measurement_indices": ";".join(str(index) for index in sorted(indices)),
        "denoise_seconds_median": statistics.median(denoise_values),
        "total_generation_seconds_median": statistics.median(total_values),
        "artifact_ready_seconds_median": statistics.median(artifact_ready_values),
        "peak_memory_allocated_gib_median": statistics.median(allocated_values) / GIB,
        "peak_memory_reserved_gib_median": statistics.median(reserved_values) / GIB,
        "artifact_sha256": ";".join(
            f"{index}:{artifact_hash}"
            for index, artifact_hash in sorted(zip(indices, artifact_hashes))
        ),
        "metrics_sidecar_sha256": ";".join(
            f"{index}:{metrics_hash}"
            for index, metrics_hash in sorted(
                zip(indices, metrics_sidecar_hashes)
            )
        ),
        "engine_load_seconds": engine_load_seconds,
        "comparison_values": comparison_values,
    }


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
