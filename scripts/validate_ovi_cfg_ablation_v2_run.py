#!/usr/bin/env python3
"""Fail-closed validator for one Ovi CFG-cache ablation v2 cell."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ovi.cfg_ablation_v2_protocol import (  # noqa: E402
    FROZEN_MEDIA_CONTRACT,
    PROTOCOL_ID,
    ProtocolError,
    STAGE_SEEDS,
    load_and_validate_matrix,
    validate_frozen_base_config,
)
from ovi.eval_protocol import prompt_sequence_sha256  # noqa: E402


SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
STANDARD_EVIDENCE = (
    "pre_run_gpu.json",
    "preflight.json",
    "environment.freeze.txt",
    "checkpoint_manifest.json",
)
SNAPSHOT_EVIDENCE = (
    "matrix.csv",
    "frozen_config.yaml",
    "materialization_manifest.json",
    "prompt.csv",
    "gpu_telemetry.jsonl",
    "verification.json",
    "decoded_stream_hashes.json",
)


class RunValidationError(RuntimeError):
    """One strict protocol condition failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RunValidationError(message)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_json(path: Path, context: str) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RunValidationError(f"cannot read {context} {path}: {exc}") from exc


def _load_jsonl(path: Path, context: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RunValidationError(f"cannot read {context} {path}: {exc}") from exc
    _require(bool(lines), f"{context} is empty: {path}")
    records = []
    for index, line in enumerate(lines, start=1):
        _require(bool(line.strip()), f"{context} line {index} is blank")
        try:
            record = json.loads(line, parse_constant=_reject_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise RunValidationError(
                f"{context} line {index} is invalid JSON: {exc}"
            ) from exc
        _require(
            isinstance(record, dict),
            f"{context} line {index} must be a JSON object",
        )
        records.append(record)
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RunValidationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_file(path: Path, context: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RunValidationError(f"missing {context}: {path}: {exc}") from exc
    _require(resolved.is_file(), f"{context} is not a file: {resolved}")
    return resolved


def _parse_scalar(value: str) -> Any:
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none", "~"}:
        return None
    try:
        return ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        pass
    try:
        if any(character in stripped.lower() for character in (".", "e")):
            parsed_float = float(stripped)
            if math.isfinite(parsed_float):
                return parsed_float
        return int(stripped)
    except ValueError:
        return stripped


def load_flat_yaml(path: Path) -> dict[str, Any]:
    """Load the generated flat config without requiring PyYAML/OmegaConf."""

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RunValidationError(f"cannot read YAML {path}: {exc}") from exc
    try:
        parsed_json = json.loads(text, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError):
        parsed_json = None
    if isinstance(parsed_json, dict):
        return parsed_json

    result: dict[str, Any] = {}
    active_list_key: str | None = None
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith((" ", "\t")):
            raise RunValidationError(
                f"unsupported nested YAML at {path}:{line_number}"
            )
        if line.startswith("-"):
            _require(
                active_list_key is not None,
                f"orphan YAML list item at {path}:{line_number}",
            )
            result[active_list_key].append(_parse_scalar(line[1:]))
            continue
        _require(":" in line, f"invalid YAML line at {path}:{line_number}")
        key, value = line.split(":", 1)
        key = key.strip()
        _require(bool(key), f"empty YAML key at {path}:{line_number}")
        _require(key not in result, f"duplicate YAML key {key!r} at {path}:{line_number}")
        if not value.strip():
            result[key] = []
            active_list_key = key
        else:
            result[key] = _parse_scalar(value)
            active_list_key = None
    _require(bool(result), f"YAML contains no top-level fields: {path}")
    return result


def _load_prompts(path: Path) -> list[str]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            _require(
                reader.fieldnames is not None and "text_prompt" in reader.fieldnames,
                f"prompt CSV lacks text_prompt column: {path}",
            )
            prompts = [str(row.get("text_prompt") or "") for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise RunValidationError(f"cannot read prompt CSV {path}: {exc}") from exc
    _require(bool(prompts), f"prompt CSV contains no rows: {path}")
    _require(all(prompt for prompt in prompts), f"prompt CSV contains an empty prompt: {path}")
    return prompts


def _strict_int(value: Any, context: str, *, minimum: int | None = None) -> int:
    _require(type(value) is int, f"{context} must be a JSON integer")
    if minimum is not None:
        _require(value >= minimum, f"{context} must be >= {minimum}")
    return value


def _finite_number(value: Any, context: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{context} must be numeric",
    )
    converted = float(value)
    _require(math.isfinite(converted), f"{context} must be finite")
    return converted


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and float(actual) == expected
        )
    return type(actual) is type(expected) and actual == expected


def _resolve_prompt_path(config_path: Path, config: Mapping[str, Any]) -> Path:
    configured = config.get("text_prompt")
    _require(isinstance(configured, str) and configured, "config text_prompt is missing")
    candidate = Path(configured)
    if not candidate.is_absolute():
        # Generated configs are repository-relative even when stored below configs/.
        candidate = REPO_ROOT / candidate
    return _canonical_file(candidate, "configured prompt CSV")


def validate_launch_inputs(
    matrix_path: Path,
    config_path: Path,
    cell_id: str,
    seed: int,
    expected_measurements: int | None = None,
) -> dict[str, Any]:
    matrix_path = _canonical_file(matrix_path, "matrix CSV")
    config_path = _canonical_file(config_path, "frozen config YAML")
    cells = load_and_validate_matrix(matrix_path)
    matches = [cell for cell in cells if cell.config_id == cell_id]
    _require(len(matches) == 1, f"matrix config_id {cell_id!r} is not unique")
    cell = matches[0]
    config = load_flat_yaml(config_path)
    validate_frozen_base_config(config)

    execution_stage_value = config.get("cfg_ablation_execution_stage", cell.stage)
    execution_stage = str(execution_stage_value)
    _require(
        execution_stage in {"0", "1", "2", "3", "4"},
        f"invalid cfg_ablation_execution_stage {execution_stage_value!r}",
    )
    source_stage = str(config.get("cfg_ablation_source_matrix_stage", cell.stage))
    _require(
        source_stage == cell.stage,
        f"config source matrix stage {source_stage!r} != row stage {cell.stage!r}",
    )
    configured_id = config.get("cfg_ablation_config_id", cell_id)
    _require(
        configured_id == cell_id,
        f"config cfg_ablation_config_id={configured_id!r} != {cell_id!r}",
    )
    _require(
        config.get("cfg_ablation_protocol_id") == PROTOCOL_ID,
        "config cfg_ablation_protocol_id is missing or incorrect",
    )
    _require(
        config.get("cfg_ablation_stage") == int(execution_stage),
        "config cfg_ablation_stage differs from execution stage",
    )
    _require(
        config.get("cfg_cache_window_indexing") == "zero_based_inclusive",
        "config CFG-cache window indexing is not zero_based_inclusive",
    )
    if execution_stage == "3":
        allowed_seeds = set(STAGE_SEEDS["3"])
        _require(
            cell.stage in {"0", "2"},
            "stage 3 may only reuse source matrix cells from stage 0 or 2",
        )
    elif execution_stage == "4":
        allowed_seeds = {seed}
    else:
        _require(
            execution_stage == cell.stage,
            f"execution stage {execution_stage} may not relabel matrix stage {cell.stage}",
        )
        allowed_seeds = set(STAGE_SEEDS[execution_stage])
    _require(
        type(seed) is int and seed >= 0 and seed in allowed_seeds,
        f"seed {seed!r} is not allowed for execution stage {execution_stage}",
    )

    _require(
        config.get("use_cfg_cache") is cell.use_cfg_cache,
        f"config use_cfg_cache does not match matrix row {cell_id}",
    )
    if cell.use_cfg_cache:
        for field, expected in (
            ("cfg_cache_start_step", cell.start_step),
            ("cfg_cache_end_step", cell.end_step),
            ("cfg_cache_refresh_interval", cell.refresh_interval),
        ):
            _require(
                type(config.get(field)) is int and config.get(field) == expected,
                f"config {field}={config.get(field)!r} != matrix {expected!r}",
            )
    else:
        for field, expected in (
            ("cfg_cache_start_step", 0),
            ("cfg_cache_end_step", 0),
            ("cfg_cache_refresh_interval", 1),
        ):
            _require(
                type(config.get(field)) is int and config.get(field) == expected,
                f"dense config {field} must equal generator sentinel {expected}",
            )
    _require(type(config.get("seed")) is int, "config seed must be an integer")
    _require(config.get("seed") == seed, f"config seed={config.get('seed')} != --seed {seed}")
    prompt_path = _resolve_prompt_path(config_path, config)
    prompts = _load_prompts(prompt_path)
    measurement_runs = _strict_int(
        config.get("measurement_runs"), "config measurement_runs", minimum=1
    )
    each_example_n_times = _strict_int(
        config.get("each_example_n_times"),
        "config each_example_n_times",
        minimum=1,
    )
    _require(
        each_example_n_times == 1,
        "v2 cells require batch/each_example_n_times=1; seeds are separate cells",
    )
    inferred_measurements = measurement_runs * len(prompts) * each_example_n_times
    if expected_measurements is not None:
        _require(
            type(expected_measurements) is int and expected_measurements >= 1,
            "--expected-measurements must be a positive integer",
        )
        _require(
            expected_measurements == inferred_measurements,
            f"--expected-measurements={expected_measurements} != config-derived "
            f"{inferred_measurements}",
        )
    manifest_path = _canonical_file(
        config_path.parent.parent / "manifest.json",
        "materialization manifest",
    )
    manifest = _load_json(manifest_path, "materialization manifest")
    _require(
        isinstance(manifest, dict)
        and manifest.get("record_type")
        == "ovi_cfg_ablation_v2_materialization_manifest"
        and manifest.get("status") == "ok",
        "materialization manifest did not pass generator validation",
    )
    input_files = manifest.get("input_files")
    _require(isinstance(input_files, dict), "manifest input_files map is missing")
    matrix_binding = input_files.get("matrix")
    _require(
        isinstance(matrix_binding, dict)
        and matrix_binding.get("sha256") == _sha256(matrix_path),
        "materialization manifest matrix binding mismatch",
    )
    copied_inputs = manifest.get("copied_inputs")
    prompt_binding = (
        copied_inputs.get("prompt_csv")
        if isinstance(copied_inputs, dict)
        else None
    )
    _require(
        isinstance(prompt_binding, dict)
        and prompt_binding.get("sha256") == _sha256(prompt_path),
        "materialization manifest prompt binding mismatch",
    )
    materializations = manifest.get("materializations")
    _require(isinstance(materializations, list), "manifest materializations are missing")
    materialization_matches = [
        item
        for item in materializations
        if isinstance(item, dict)
        and item.get("config_id") == cell_id
        and item.get("seed") == seed
        and Path(str(item.get("config_path"))).resolve() == config_path
    ]
    _require(
        len(materialization_matches) == 1,
        "config is not uniquely bound by the materialization manifest",
    )
    materialization = materialization_matches[0]
    for field, expected in (
        ("source_matrix_stage", int(cell.stage)),
        ("execution_stage", int(execution_stage)),
        ("use_cfg_cache", cell.use_cfg_cache),
        ("start_step", cell.start_step),
        ("end_step", cell.end_step),
        ("refresh_interval", cell.refresh_interval),
        ("refreshes", cell.refreshes),
        ("cache_hits", cell.cache_hits),
        ("negative_forwards", cell.negative_forwards),
        (
            "expected_video_self_attention_calls",
            cell.expected_video_self_attention_calls,
        ),
    ):
        _require(
            materialization.get(field) == expected,
            f"manifest materialization {field} differs from matrix row",
        )
    _require(
        materialization.get("config_sha256") == _sha256(config_path),
        "manifest config SHA256 mismatch",
    )
    _require(
        manifest.get("warmup_runs") == config.get("warmup_runs")
        and manifest.get("measurement_runs") == config.get("measurement_runs"),
        "manifest run counts differ from config",
    )
    return {
        "cell": cell,
        "config": config,
        "config_path": config_path,
        "config_sha256": _sha256(config_path),
        "matrix_path": matrix_path,
        "matrix_sha256": _sha256(matrix_path),
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "prompt_path": prompt_path,
        "prompt_file_sha256": _sha256(prompt_path),
        "prompts": prompts,
        "prompts_sha256": prompt_sequence_sha256(prompts),
        "expected_measurements": inferred_measurements,
        "expected_warmups": _strict_int(
            config.get("warmup_runs"), "config warmup_runs", minimum=0
        ),
        "execution_stage": execution_stage,
        "seed": seed,
    }


def _validate_gpu_monitor(monitor: Any, context: str, expected_uuid: str) -> None:
    _require(isinstance(monitor, dict), f"{context} gpu_process_monitor is missing")
    _require(
        monitor.get("device_uuid") == expected_uuid,
        f"{context} GPU UUID differs from pre-run evidence",
    )
    _require(
        monitor.get("contention_detected") is False,
        f"{context} GPU contention was detected",
    )
    _require(
        monitor.get("valid_for_benchmark") is True,
        f"{context} GPU monitor is not valid_for_benchmark",
    )
    _require(
        monitor.get("exact_singleton_process_per_sample") is True,
        f"{context} does not have exactly one compute process per sample",
    )
    _require(
        monitor.get("single_distinct_host_pid") is True,
        f"{context} changed compute-process PID",
    )
    _require(monitor.get("collection_errors") == [], f"{context} monitor has errors")
    _require(monitor.get("min_process_count") == 1, f"{context} min process count != 1")
    _require(monitor.get("max_process_count") == 1, f"{context} max process count != 1")
    _require(monitor.get("no_process_detected") is False, f"{context} missed process samples")
    samples = monitor.get("samples")
    _require(
        isinstance(samples, list) and len(samples) >= 2,
        f"{context} needs at least entry and exit GPU samples",
    )


def _validate_dispatch_and_workload(
    metrics: Mapping[str, Any],
    *,
    cell: Any,
    expected_uuid: str,
    context: str,
) -> None:
    expected = {
        "cfg_cache_hits": cell.cache_hits,
        "cfg_cache_refreshes": cell.refreshes,
        "cfg_negative_forwards": cell.negative_forwards,
    }
    for field, value in expected.items():
        _require(metrics.get(field) == value, f"{context} {field} != {value}")
    _require(
        metrics.get("expected_cfg_cache_metrics") == expected,
        f"{context} analytical CFG counters differ",
    )
    _require(metrics.get("attention_method") == "dense", f"{context} attention is not dense")
    _require(metrics.get("use_block_cache") is False, f"{context} block cache is enabled")
    dispatcher = metrics.get("video_self_attention_dispatcher")
    _require(isinstance(dispatcher, dict), f"{context} dispatcher evidence is missing")
    for field in ("configured_method", "active_method"):
        _require(dispatcher.get(field) == "dense", f"{context} dispatcher {field} != dense")
    _require(dispatcher.get("fallback_allowed") is False, f"{context} fallback is allowed")
    _require(dispatcher.get("fallback_used") is False, f"{context} fallback was used")
    _require(dispatcher.get("fallback_count") == 0, f"{context} fallback_count != 0")
    _require(
        dispatcher.get("calls_total") == cell.expected_video_self_attention_calls,
        f"{context} attention calls != {cell.expected_video_self_attention_calls}",
    )
    _require(
        dispatcher.get("expected_calls") == cell.expected_video_self_attention_calls,
        f"{context} expected attention calls differ from matrix",
    )
    _require(dispatcher.get("calls_match_expected") is True, f"{context} call match is false")
    errors_by_method = dispatcher.get("errors_by_method")
    _require(isinstance(errors_by_method, dict), f"{context} backend error map is missing")
    _require(
        all(type(value) is int and value == 0 for value in errors_by_method.values()),
        f"{context} dispatcher recorded backend errors",
    )
    calls_by_method = dispatcher.get("calls_by_method")
    _require(isinstance(calls_by_method, dict), f"{context} method call map is missing")
    _require(
        calls_by_method.get("dense") == cell.expected_video_self_attention_calls,
        f"{context} dense call count differs",
    )
    _require(
        all(
            method == "dense" or (type(value) is int and value == 0)
            for method, value in calls_by_method.items()
        ),
        f"{context} non-dense dispatcher calls were observed",
    )
    _validate_gpu_monitor(metrics.get("gpu_process_monitor"), context, expected_uuid)


def _validate_telemetry(path: Path, expected_uuid: str) -> list[dict[str, Any]]:
    records = _load_jsonl(path, "GPU telemetry")
    _require(
        [record.get("phase") for record in records] == ["pre_inference", "post_inference"],
        "GPU telemetry must contain exactly pre_inference then post_inference",
    )
    for index, record in enumerate(records):
        context = f"gpu_telemetry[{index}]"
        _require(record.get("uuid") == expected_uuid, f"{context} UUID mismatch")
        _require(record.get("query_status") == "ok", f"{context} query failed")
        _strict_int(record.get("index"), f"{context} index", minimum=0)
        for field in (
            "temperature_c",
            "power_draw_w",
            "sm_clock_mhz",
            "memory_clock_mhz",
            "memory_used_mib",
            "utilization_gpu_percent",
        ):
            _finite_number(record.get(field), f"{context} {field}")
    return records


def _artifact_identity(payload: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        _strict_int(payload.get("measurement_index"), "measurement_index", minimum=0),
        _strict_int(payload.get("prompt_index"), "prompt_index", minimum=0),
        _strict_int(payload.get("sample_index"), "sample_index", minimum=0),
    )


def _validate_run_or_raise(contract: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve(strict=True)
    _require(run_dir.is_dir(), f"run dir is not a directory: {run_dir}")
    cell = contract["cell"]
    config = contract["config"]
    prompts = contract["prompts"]
    seed = contract["seed"]

    for filename in (*STANDARD_EVIDENCE, *SNAPSHOT_EVIDENCE, "environment.json", "run_config.yaml", "timings.jsonl"):
        _canonical_file(run_dir / filename, filename)
    _require((run_dir / "environment.freeze.txt").stat().st_size > 0, "environment freeze is empty")
    _require(_sha256(run_dir / "matrix.csv") == contract["matrix_sha256"], "matrix snapshot hash mismatch")
    _require(
        _sha256(run_dir / "frozen_config.yaml") == contract["config_sha256"],
        "frozen config snapshot hash mismatch",
    )
    _require(
        _sha256(run_dir / "materialization_manifest.json")
        == contract["manifest_sha256"],
        "materialization manifest snapshot hash mismatch",
    )
    _require(
        _sha256(run_dir / "prompt.csv") == contract["prompt_file_sha256"],
        "prompt snapshot hash mismatch",
    )

    environment = _load_json(run_dir / "environment.json", "environment")
    _require(isinstance(environment, dict), "environment JSON root must be an object")
    run_config = load_flat_yaml(run_dir / "run_config.yaml")
    for field, expected in config.items():
        if field in {"output_dir", "text_prompt"}:
            continue
        _require(
            field in run_config and _same(run_config[field], expected),
            f"resolved run config {field}={run_config.get(field)!r} != frozen {expected!r}",
        )
    _require(
        Path(str(run_config.get("output_dir"))).resolve() == run_dir,
        "resolved output_dir does not equal run dir",
    )
    frozen_environment_fields = (
        "model_name",
        "mode",
        "video_frame_height_width",
        "sample_steps",
        "solver_name",
        "shift",
        "sp_size",
        "audio_guidance_scale",
        "video_guidance_scale",
        "slg_layer",
        "video_negative_prompt",
        "audio_negative_prompt",
        "attention_method",
        "fp8",
        "qint8",
        "cpu_offload",
        "use_block_cache",
        "debug_forward",
        "debug_forward_step",
        "use_cfg_cache",
    )
    for field in frozen_environment_fields:
        _require(
            _same(environment.get(field), config.get(field)),
            f"environment {field} differs from frozen config",
        )
    _require(environment.get("git_dirty") is False, "run used a dirty Git tree")
    _require(
        isinstance(environment.get("git_commit"), str)
        and COMMIT_RE.fullmatch(environment["git_commit"]) is not None,
        "environment git_commit is not a full commit hash",
    )
    _require(environment.get("seed") == seed, "environment seed mismatch")
    _require(environment.get("prompt_count") == len(prompts), "environment prompt count mismatch")
    _require(
        environment.get("prompts_sha256") == contract["prompts_sha256"],
        "environment prompt sequence hash mismatch",
    )
    _require(
        environment.get("expected_measurement_records") == contract["expected_measurements"],
        "environment expected measurement count mismatch",
    )
    _require(
        environment.get("expected_warmup_records") == contract["expected_warmups"],
        "environment expected warmup count mismatch",
    )

    evidence_hashes = environment.get("evidence_file_sha256")
    _require(isinstance(evidence_hashes, dict), "environment evidence hash map is missing")
    for filename in STANDARD_EVIDENCE:
        _require(
            evidence_hashes.get(filename) == _sha256(run_dir / filename),
            f"environment evidence hash mismatch for {filename}",
        )
    preflight = _load_json(run_dir / "preflight.json", "preflight")
    _require(isinstance(preflight, dict) and preflight.get("errors") == [], "preflight did not pass")
    pre_run = _load_json(run_dir / "pre_run_gpu.json", "pre-run GPU evidence")
    _require(isinstance(pre_run, dict), "pre-run GPU evidence root must be an object")
    _require(pre_run.get("valid_for_run") is True, "pre-run GPU evidence is not valid")
    expected_uuid = pre_run.get("device_uuid")
    _require(
        isinstance(expected_uuid, str) and expected_uuid.startswith("GPU-"),
        "pre-run GPU UUID is invalid",
    )
    _require(environment.get("gpu_uuid") == expected_uuid, "environment GPU UUID mismatch")
    _require(environment.get("pre_run_gpu_valid") is True, "environment rejects pre-run GPU evidence")

    checkpoint = _load_json(run_dir / "checkpoint_manifest.json", "checkpoint manifest")
    _require(isinstance(checkpoint, dict), "checkpoint manifest root must be an object")
    files = checkpoint.get("files")
    _require(isinstance(files, dict), "checkpoint manifest files map is missing")
    model = files.get("Ovi/model.safetensors")
    _require(isinstance(model, dict), "Ovi model is absent from checkpoint manifest")
    _strict_int(model.get("bytes"), "Ovi checkpoint bytes", minimum=1)
    _require(
        isinstance(model.get("sha256"), str) and SHA256_RE.fullmatch(model["sha256"]) is not None,
        "Ovi checkpoint SHA256 is invalid",
    )
    telemetry = _validate_telemetry(run_dir / "gpu_telemetry.jsonl", expected_uuid)

    measurements = _load_jsonl(run_dir / "timings.jsonl", "measurement timings")
    _require(
        len(measurements) == contract["expected_measurements"],
        f"measurement count {len(measurements)} != {contract['expected_measurements']}",
    )
    metric_sidecars = sorted(run_dir.glob("*.metrics.json"))
    _require(
        len(metric_sidecars) == len(measurements),
        "measurement sidecar count differs from timings.jsonl",
    )
    identities: dict[tuple[int, int, int], Mapping[str, Any]] = {}
    for index, metrics in enumerate(measurements):
        context = f"measurement[{index}]"
        _require(metrics.get("status") == "ok", f"{context} status is not ok")
        _require(metrics.get("record_type") == "measurement", f"{context} record_type mismatch")
        identity = _artifact_identity(metrics)
        _require(identity not in identities, f"duplicate measurement identity {identity}")
        identities[identity] = metrics
        measurement_index, prompt_index, sample_index = identity
        _require(prompt_index < len(prompts), f"{context} prompt index is out of range")
        _require(metrics.get("prompt") == prompts[prompt_index], f"{context} prompt mismatch")
        _require(metrics.get("seed") == seed + sample_index, f"{context} seed mismatch")
        _require(measurement_index < int(config["measurement_runs"]), f"{context} repeat index out of range")
        _validate_dispatch_and_workload(
            metrics,
            cell=cell,
            expected_uuid=expected_uuid,
            context=context,
        )
        output_path = Path(str(metrics.get("output_path"))).resolve(strict=True)
        _require(output_path.parent == run_dir, f"{context} output escapes run dir")
        _require(_sha256(output_path) == metrics.get("output_sha256"), f"{context} MP4 hash mismatch")

    sidecar_by_identity = {}
    for path in metric_sidecars:
        payload = _load_json(path, "measurement sidecar")
        _require(isinstance(payload, dict), f"sidecar root is not an object: {path}")
        identity = _artifact_identity(payload)
        _require(identity not in sidecar_by_identity, f"duplicate sidecar identity {identity}")
        sidecar_by_identity[identity] = payload
    _require(sidecar_by_identity == identities, "sidecars differ from timings.jsonl")

    warmup_path = run_dir / "warmup_timings.jsonl"
    if contract["expected_warmups"]:
        warmups = _load_jsonl(warmup_path, "warmup timings")
        _require(len(warmups) == contract["expected_warmups"], "warmup record count mismatch")
        for index, metrics in enumerate(warmups):
            context = f"warmup[{index}]"
            _require(metrics.get("status") == "ok", f"{context} status is not ok")
            _require(metrics.get("record_type") == "warmup", f"{context} record_type mismatch")
            _require(metrics.get("seed") == seed, f"{context} seed mismatch")
            _validate_dispatch_and_workload(
                metrics,
                cell=cell,
                expected_uuid=expected_uuid,
                context=context,
            )
    else:
        _require(not warmup_path.exists(), "unexpected warmup_timings.jsonl")

    verification = _load_json(run_dir / "verification.json", "media verification")
    _require(isinstance(verification, dict), "verification root must be an object")
    _require(verification.get("status") == "ok", "media-only verification did not pass")
    artifacts = verification.get("artifacts")
    _require(
        isinstance(artifacts, list) and len(artifacts) == len(measurements),
        "media verification artifact count mismatch",
    )
    verified_by_path: dict[str, Mapping[str, Any]] = {}
    for artifact in artifacts:
        _require(isinstance(artifact, dict), "verification artifact must be an object")
        _require(artifact.get("status") == "ok", "verification artifact failed")
        video = artifact.get("video")
        audio = artifact.get("audio")
        _require(isinstance(video, dict) and isinstance(audio, dict), "media stream evidence is missing")
        _require(video.get("decoded_frames") == FROZEN_MEDIA_CONTRACT["decoded_video_frames"], "decoded frame count mismatch")
        _require(video.get("width") == FROZEN_MEDIA_CONTRACT["decoded_width"], "decoded width mismatch")
        _require(video.get("height") == FROZEN_MEDIA_CONTRACT["decoded_height"], "decoded height mismatch")
        _require(video.get("codec") == FROZEN_MEDIA_CONTRACT["video_codec"], "video codec mismatch")
        _require(audio.get("codec") == FROZEN_MEDIA_CONTRACT["audio_codec"], "audio codec mismatch")
        _require(_finite_number(audio.get("rms"), "audio RMS") > 1e-6, "audio is silent")
        canonical_path = str(Path(str(artifact.get("path"))).resolve(strict=True))
        _require(canonical_path not in verified_by_path, "duplicate verified MP4 path")
        verified_by_path[canonical_path] = artifact

    decoded = _load_json(run_dir / "decoded_stream_hashes.json", "decoded stream receipt")
    _require(isinstance(decoded, dict) and decoded.get("status") == "ok", "decoded stream hashing failed")
    _require(decoded.get("schema_version") == 1, "decoded stream receipt schema mismatch")
    decoded_artifacts = decoded.get("artifacts")
    _require(
        isinstance(decoded_artifacts, list) and len(decoded_artifacts) == len(measurements),
        "decoded stream artifact count mismatch",
    )
    decoded_receipts = {}
    expected_rgb_bytes = (
        FROZEN_MEDIA_CONTRACT["decoded_video_frames"]
        * FROZEN_MEDIA_CONTRACT["decoded_width"]
        * FROZEN_MEDIA_CONTRACT["decoded_height"]
        * 3
    )
    for artifact in decoded_artifacts:
        _require(isinstance(artifact, dict), "decoded artifact must be an object")
        path = Path(str(artifact.get("path"))).resolve(strict=True)
        canonical = str(path)
        _require(canonical in verified_by_path, "decoded artifact is not media-verified")
        container = artifact.get("container")
        video = artifact.get("video")
        audio = artifact.get("audio")
        probe = artifact.get("ffprobe")
        _require(all(isinstance(item, dict) for item in (container, video, audio, probe)), "decoded receipt fields are malformed")
        assert isinstance(container, dict) and isinstance(video, dict)
        assert isinstance(audio, dict) and isinstance(probe, dict)
        actual_container_sha = _sha256(path)
        _require(container.get("sha256") == actual_container_sha, "decoded receipt container hash mismatch")
        _require(container.get("bytes") == path.stat().st_size, "decoded receipt container size mismatch")
        video_decode = video.get("decode")
        audio_decode = audio.get("decode")
        _require(isinstance(video_decode, dict) and isinstance(audio_decode, dict), "decoded stream digests are missing")
        _require(video.get("codec_name") == "h264", "decoded receipt video codec mismatch")
        _require(audio.get("codec_name") == "aac", "decoded receipt audio codec mismatch")
        _require(video_decode.get("pixel_format") == "rgb24", "video decode is not RGB24")
        _require(video_decode.get("bytes") == expected_rgb_bytes, "full RGB24 byte count mismatch")
        _require(audio_decode.get("sample_format") == "f32le", "audio decode is not f32le")
        _require(audio_decode.get("channels") == 1, "audio decode is not mono")
        _require(audio_decode.get("sample_rate") == 16000, "audio decode is not 16 kHz")
        _require(_strict_int(audio_decode.get("bytes"), "decoded audio bytes", minimum=4) % 4 == 0, "decoded audio byte count is not float32-aligned")
        _require(_finite_number(audio_decode.get("rms"), "decoded audio RMS") > 1e-6, "decoded audio is silent")
        for label, value in (
            ("RGB24", video_decode.get("sha256")),
            ("mono16k f32le", audio_decode.get("sha256")),
            ("ffprobe stdout", probe.get("stdout_sha256")),
        ):
            _require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None, f"{label} SHA256 is invalid")
        decoded_receipts[canonical] = {
            "container_sha256": actual_container_sha,
            "rgb24_sha256": video_decode["sha256"],
            "rgb24_bytes": video_decode["bytes"],
            "mono16k_f32le_sha256": audio_decode["sha256"],
            "mono16k_f32le_bytes": audio_decode["bytes"],
        }
    _require(set(decoded_receipts) == set(verified_by_path), "decoded/media artifact sets differ")

    return {
        "cell": cell.as_json(),
        "seed": seed,
        "git_commit": environment["git_commit"],
        "gpu_uuid": expected_uuid,
        "checkpoint": {
            "manifest_sha256": _sha256(run_dir / "checkpoint_manifest.json"),
            "model_sha256": model["sha256"],
            "model_bytes": model["bytes"],
        },
        "record_counts": {
            "measurements": len(measurements),
            "warmups": contract["expected_warmups"],
        },
        "decoded_streams": decoded_receipts,
        "gpu_telemetry": telemetry,
    }


def validate_run(
    matrix_path: Path,
    config_path: Path,
    cell_id: str,
    seed: int,
    run_dir: Path,
    expected_measurements: int | None = None,
) -> dict[str, Any]:
    base = {
        "schema_version": SCHEMA_VERSION,
        "validator": str(Path(__file__).resolve()),
        "run_dir": str(run_dir.resolve()),
        "cell_id": cell_id,
        "seed": seed,
        "status": "failed",
        "errors": [],
    }
    try:
        contract = validate_launch_inputs(
            matrix_path,
            config_path,
            cell_id,
            seed,
            expected_measurements,
        )
        details = _validate_run_or_raise(contract, run_dir)
        base.update(
            {
                "status": "passed",
                "inputs": {
                    "matrix": {
                        "path": str(contract["matrix_path"]),
                        "sha256": contract["matrix_sha256"],
                    },
                    "config": {
                        "path": str(contract["config_path"]),
                        "sha256": contract["config_sha256"],
                    },
                    "materialization_manifest": {
                        "path": str(contract["manifest_path"]),
                        "sha256": contract["manifest_sha256"],
                    },
                    "prompt_csv": {
                        "path": str(contract["prompt_path"]),
                        "sha256": contract["prompt_file_sha256"],
                        "prompts_sha256": contract["prompts_sha256"],
                    },
                },
                "validation": details,
            }
        )
    except (ProtocolError, RunValidationError, OSError, ValueError) as exc:
        base["errors"] = [str(exc)]
    return base


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--expected-measurements", type=int)
    parser.add_argument("--input-check-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.input_check_only:
        try:
            contract = validate_launch_inputs(
                args.matrix,
                args.config,
                args.cell_id,
                args.seed,
                args.expected_measurements,
            )
            report = {
                "status": "passed",
                "cell": contract["cell"].as_json(),
                "seed": args.seed,
                "expected_measurements": contract["expected_measurements"],
                "expected_warmups": contract["expected_warmups"],
                "execution_stage": contract["execution_stage"],
                "matrix_sha256": contract["matrix_sha256"],
                "config_sha256": contract["config_sha256"],
                "manifest_path": str(contract["manifest_path"]),
                "manifest_sha256": contract["manifest_sha256"],
                "prompt_path": str(contract["prompt_path"]),
                "prompt_file_sha256": contract["prompt_file_sha256"],
                "prompts_sha256": contract["prompts_sha256"],
            }
        except (ProtocolError, RunValidationError, OSError, ValueError) as exc:
            report = {"status": "failed", "errors": [str(exc)]}
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0 if report["status"] == "passed" else 1

    if args.run_dir is None:
        parser.error("--run-dir is required unless --input-check-only is used")
    output = args.output or args.run_dir / "protocol_validation.json"
    report = validate_run(
        args.matrix,
        args.config,
        args.cell_id,
        args.seed,
        args.run_dir,
        args.expected_measurements,
    )
    _atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
