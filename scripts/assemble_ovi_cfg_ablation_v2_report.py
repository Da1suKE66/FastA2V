#!/usr/bin/env python3
"""Plan Stage 4 and assemble a fail-closed Ovi CFG-cache v2 machine report.

The Stage 4 plan uses a three-position cyclic order so that Dense, the frozen
12-hit policy, and the frozen 14-hit policy occupy every launch position the
same number of times.  The report keeps development and held-out evidence in
separate sections and never turns unavailable ASR, SyncNet, or human ratings
into synthetic scores.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(REPO_ROOT))

from ovi.cfg_ablation_v2_protocol import (  # noqa: E402
    CANDIDATE_FREEZE_RULE,
    PROTOCOL_ID,
    STAGE4_FIXED,
)


WORKLOADS = ("dense", "frozen_new_12", "frozen_new_14")
ORDER_CYCLE = (
    ("dense", "frozen_new_12", "frozen_new_14"),
    ("frozen_new_12", "frozen_new_14", "dense"),
    ("frozen_new_14", "dense", "frozen_new_12"),
)
LATENCY_FIELDS = (
    "pre_denoise_seconds",
    "denoise_seconds",
    "audio_decode_seconds",
    "video_decode_seconds",
    "total_generation_seconds",
    "save_video_seconds",
    "artifact_ready_seconds",
)
STAGE_TAG_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ReportError(RuntimeError):
    """Raised when a plan or report input is incomplete or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    _require(resolved.is_file(), f"input is not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read {label} {path}: {exc}") from exc


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ReportError(f"cannot read {label} {path}: {exc}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReportError(
                f"cannot parse {label} {path}:{line_number}: {exc}"
            ) from exc
        _require(isinstance(record, dict), f"{label} {path}:{line_number} is not an object")
        records.append(record)
    return records


def _finite_nonnegative(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result) and result >= 0.0, f"{label} must be finite and nonnegative")
    return result


def _atomic_write(path: Path, text: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_stage4_plan(
    *,
    stage_tag: str,
    new12_id: str,
    new14_id: str,
    blocks: int = 3,
    warmup_runs: int = 3,
    measurement_runs: int = 5,
    seed: int = 103,
) -> dict[str, Any]:
    """Build and validate the immutable Stage 4 launch order."""

    _require(STAGE_TAG_RE.fullmatch(stage_tag) is not None, f"invalid stage tag: {stage_tag!r}")
    allowed12 = tuple(CANDIDATE_FREEZE_RULE["conservative_12_hit_allowed"])
    allowed14 = tuple(CANDIDATE_FREEZE_RULE["aggressive_14_hit_allowed"])
    _require(new12_id in allowed12, f"12-hit candidate {new12_id!r} is not in {allowed12}")
    _require(new14_id in allowed14, f"14-hit candidate {new14_id!r} is not in {allowed14}")
    _require(new12_id != new14_id, "frozen 12-hit and 14-hit candidates must differ")
    _require(isinstance(blocks, int) and not isinstance(blocks, bool), "blocks must be an integer")
    _require(blocks >= 3, "Stage 4 requires at least three blocks")
    _require(blocks % 3 == 0, "balanced positional order requires a multiple of three blocks")
    _require(warmup_runs >= int(STAGE4_FIXED["minimum_warmup_runs"]), "Stage 4 requires at least three warmups per workload run")
    _require(measurement_runs >= int(STAGE4_FIXED["minimum_measurement_runs"]), "Stage 4 requires at least five measurements per workload run")
    _require(seed == 103, "Stage 4 is frozen to the Stage 0 prompt and seed 103")

    workload_ids = {
        "dense": "dense",
        "frozen_new_12": new12_id,
        "frozen_new_14": new14_id,
    }
    execution: list[dict[str, Any]] = []
    ordinal = 0
    for block_index in range(1, blocks + 1):
        order = ORDER_CYCLE[(block_index - 1) % len(ORDER_CYCLE)]
        for position, workload in enumerate(order, start=1):
            ordinal += 1
            config_id = workload_ids[workload]
            execution.append(
                {
                    "ordinal": ordinal,
                    "block_index": block_index,
                    "position": position,
                    "workload": workload,
                    "config_id": config_id,
                    "run_tag": (
                        f"{stage_tag}-b{block_index:02d}-p{position:02d}-{workload}"
                    ),
                }
            )

    position_counts = {
        workload: {
            str(position): sum(
                1
                for entry in execution
                if entry["workload"] == workload and entry["position"] == position
            )
            for position in (1, 2, 3)
        }
        for workload in WORKLOADS
    }
    expected_per_position = blocks // 3
    _require(
        all(
            count == expected_per_position
            for workload_counts in position_counts.values()
            for count in workload_counts.values()
        ),
        "internal error: Stage 4 plan is not position-balanced",
    )
    return {
        "schema_version": 1,
        "record_type": "ovi_cfg_ablation_v2_stage4_plan",
        "protocol_id": PROTOCOL_ID,
        "status": "ready",
        "stage_tag": stage_tag,
        "execution_stage": 4,
        "prompt_set": "stage0",
        "prompt_csv": "prompts/ovi_cfg_ablation_v2_stage0.csv",
        "seed": seed,
        "blocks": blocks,
        "warmup_runs_per_workload_run": warmup_runs,
        "measurement_runs_per_workload_run": measurement_runs,
        "workload_ids": workload_ids,
        "order_rule": "three-position cyclic Latin rotation",
        "balanced_configuration_order": True,
        "position_counts": position_counts,
        "execution_cost": {
            "cell_invocations": len(execution),
            "warmup_generations": len(execution) * warmup_runs,
            "measurement_generations": len(execution) * measurement_runs,
            "total_generations": len(execution) * (warmup_runs + measurement_runs),
            "note": "single-config cell runner requires the full 3+5 budget in every rotated block launch",
        },
        "execution": execution,
    }


def validate_stage4_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    _require(plan.get("record_type") == "ovi_cfg_ablation_v2_stage4_plan", "invalid Stage 4 plan record_type")
    _require(plan.get("protocol_id") == PROTOCOL_ID, "Stage 4 plan protocol_id mismatch")
    rebuilt = build_stage4_plan(
        stage_tag=str(plan.get("stage_tag", "")),
        new12_id=str((plan.get("workload_ids") or {}).get("frozen_new_12", "")),
        new14_id=str((plan.get("workload_ids") or {}).get("frozen_new_14", "")),
        blocks=int(plan.get("blocks", 0)),
        warmup_runs=int(plan.get("warmup_runs_per_workload_run", 0)),
        measurement_runs=int(plan.get("measurement_runs_per_workload_run", 0)),
        seed=int(plan.get("seed", -1)),
    )
    for field in (
        "execution_stage",
        "prompt_set",
        "prompt_csv",
        "seed",
        "blocks",
        "warmup_runs_per_workload_run",
        "measurement_runs_per_workload_run",
        "workload_ids",
        "order_rule",
        "balanced_configuration_order",
        "position_counts",
        "execution_cost",
        "execution",
    ):
        _require(plan.get(field) == rebuilt[field], f"Stage 4 plan field was altered: {field}")
    return rebuilt


def _parse_flat_scalar(raw: str) -> Any:
    value = raw.strip()
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _load_flat_yaml(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ReportError(f"cannot read frozen config {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        _require(not line[:1].isspace() and ":" in line, f"unsupported flat YAML at {path}:{line_number}")
        key, raw = line.split(":", 1)
        key = key.strip()
        _require(key and key not in result, f"invalid/duplicate key at {path}:{line_number}")
        result[key] = _parse_flat_scalar(raw)
    return result


def _stats(values: Sequence[float]) -> dict[str, Any]:
    _require(bool(values), "cannot summarize an empty latency sample")
    ordered = sorted(float(value) for value in values)

    def percentile(percent: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * percent / 100.0
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p10": percentile(10.0),
        "p90": percentile(90.0),
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def _same_machine_receipt(report: Mapping[str, Any]) -> tuple[str, str, str, str]:
    validation = report.get("validation")
    _require(isinstance(validation, Mapping), "protocol validation details are missing")
    checkpoint = validation.get("checkpoint")
    _require(isinstance(checkpoint, Mapping), "checkpoint validation details are missing")
    commit = validation.get("git_commit")
    gpu_uuid = validation.get("gpu_uuid")
    checkpoint_sha = checkpoint.get("model_sha256")
    _require(isinstance(commit, str) and len(commit) == 40, "invalid validated git commit")
    _require(isinstance(gpu_uuid, str) and gpu_uuid.startswith("GPU-"), "invalid validated GPU UUID")
    _require(isinstance(checkpoint_sha, str) and SHA256_RE.fullmatch(checkpoint_sha) is not None, "invalid validated checkpoint SHA256")
    run_dir = Path(str(report.get("run_dir")))
    freeze_sha = _sha256(run_dir / "environment.freeze.txt")
    return commit, gpu_uuid, checkpoint_sha, freeze_sha


def summarize_stage4(plan: Mapping[str, Any], run_root: Path) -> dict[str, Any]:
    canonical_plan = validate_stage4_plan(plan)
    run_root = run_root.resolve(strict=True)
    _require(run_root.is_dir(), f"Stage 4 run root is not a directory: {run_root}")
    warmup_count = int(canonical_plan["warmup_runs_per_workload_run"])
    measurement_count = int(canonical_plan["measurement_runs_per_workload_run"])
    by_workload: dict[str, dict[str, Any]] = {
        workload: {"runs": [], "samples": []} for workload in WORKLOADS
    }
    machine_receipts: set[tuple[str, str, str, str]] = set()

    for launch in canonical_plan["execution"]:
        run_dir = (run_root / launch["run_tag"]).resolve(strict=True)
        _require(run_dir.parent == run_root, f"Stage 4 run escapes run root: {run_dir}")
        validation_path = run_dir / "protocol_validation.json"
        validation = _load_json(validation_path, "protocol validation")
        _require(isinstance(validation, dict), f"protocol validation root is invalid: {validation_path}")
        _require(validation.get("status") == "passed", f"Stage 4 run did not pass protocol validation: {run_dir}")
        _require(validation.get("cell_id") == launch["config_id"], f"validated cell ID differs from plan: {run_dir}")
        _require(validation.get("seed") == canonical_plan["seed"], f"validated seed differs from plan: {run_dir}")
        counts = (validation.get("validation") or {}).get("record_counts")
        _require(isinstance(counts, Mapping), f"validated record counts are missing: {run_dir}")
        _require(counts.get("warmups") == warmup_count, f"validated warmup count differs: {run_dir}")
        _require(counts.get("measurements") == measurement_count, f"validated measurement count differs: {run_dir}")

        config = _load_flat_yaml(run_dir / "frozen_config.yaml")
        _require(config.get("cfg_ablation_execution_stage") == 4, f"run is not execution Stage 4: {run_dir}")
        _require(config.get("cfg_ablation_config_id") == launch["config_id"], f"frozen config ID differs: {run_dir}")
        _require(config.get("seed") == 103, f"Stage 4 seed differs: {run_dir}")
        _require(config.get("warmup_runs") == warmup_count, f"frozen warmup count differs: {run_dir}")
        _require(config.get("measurement_runs") == measurement_count, f"frozen measurement count differs: {run_dir}")
        _require(config.get("benchmark_eligible") is True, f"run is not benchmark eligible: {run_dir}")
        _require(config.get("profiling_enabled") is False, f"profiling is enabled: {run_dir}")
        _require(config.get("debug_forward") is False, f"debug forward is enabled: {run_dir}")
        for field in ("extra_tensor_export", "export_latents", "save_latents"):
            _require(config.get(field) in (None, False), f"{field} is enabled: {run_dir}")
        prompt_path = run_dir / "prompt.csv"
        _require(prompt_path.read_bytes().count(b"\n") == 2, f"Stage 4 must use exactly one prompt: {run_dir}")

        environment = _load_json(run_dir / "environment.json", "environment")
        _require(isinstance(environment, Mapping), f"environment root is invalid: {run_dir}")
        _require(environment.get("git_dirty") is False, f"Stage 4 run used a dirty tree: {run_dir}")
        _require(environment.get("benchmark_eligible") is True, f"environment is not benchmark eligible: {run_dir}")
        cold_load = _finite_nonnegative(environment.get("engine_load_seconds"), f"{run_dir} engine_load_seconds")

        measurements = _load_jsonl(run_dir / "timings.jsonl", "measurement timings")
        warmups = _load_jsonl(run_dir / "warmup_timings.jsonl", "warmup timings")
        _require(len(measurements) == measurement_count, f"measurement count differs: {run_dir}")
        _require(len(warmups) == warmup_count, f"warmup count differs: {run_dir}")
        for index, warmup in enumerate(warmups):
            _require(warmup.get("status") == "ok" and warmup.get("record_type") == "warmup", f"invalid warmup {index}: {run_dir}")
            _require(warmup.get("warmup_index") == index, f"warmup order differs: {run_dir}")

        run_samples: list[dict[str, Any]] = []
        for index, record in enumerate(measurements):
            _require(record.get("status") == "ok" and record.get("record_type") == "measurement", f"invalid measurement {index}: {run_dir}")
            _require(record.get("measurement_index") == index, f"measurement order differs: {run_dir}")
            _require(record.get("prompt_index") == 0 and record.get("sample_index") == 0, f"Stage 4 measurement is not the fixed single-prompt unit: {run_dir}")
            sample = {
                "block_index": launch["block_index"],
                "position": launch["position"],
                "measurement_index": index,
                **{
                    field: _finite_nonnegative(record.get(field), f"{run_dir} measurement[{index}] {field}")
                    for field in LATENCY_FIELDS
                },
            }
            _require(
                sample["artifact_ready_seconds"] >= sample["total_generation_seconds"],
                f"artifact-ready latency is shorter than generation latency: {run_dir}",
            )
            run_samples.append(sample)

        machine_receipts.add(_same_machine_receipt(validation))
        workload = launch["workload"]
        by_workload[workload]["runs"].append(
            {
                **launch,
                "run_dir": str(run_dir),
                "protocol_validation": _binding(validation_path),
                "timings": _binding(run_dir / "timings.jsonl"),
                "warmup_timings": _binding(run_dir / "warmup_timings.jsonl"),
                "cold_model_load_seconds": cold_load,
            }
        )
        by_workload[workload]["samples"].extend(run_samples)

    _require(len(machine_receipts) == 1, "Stage 4 runs differ in commit, GPU, checkpoint, or environment freeze")
    commit, gpu_uuid, checkpoint_sha, environment_freeze_sha = next(iter(machine_receipts))
    workload_summaries: dict[str, Any] = {}
    for workload in WORKLOADS:
        payload = by_workload[workload]
        samples = payload["samples"]
        _require(len(payload["runs"]) == canonical_plan["blocks"], f"workload {workload} is missing blocks")
        _require(len(samples) == canonical_plan["blocks"] * measurement_count, f"workload {workload} has an incomplete sample set")
        workload_summaries[workload] = {
            "config_id": canonical_plan["workload_ids"][workload],
            "block_count": len(payload["runs"]),
            "measurement_count": len(samples),
            "warmup_count": len(payload["runs"]) * warmup_count,
            "latency_seconds": {
                field: _stats([sample[field] for sample in samples])
                for field in LATENCY_FIELDS
            },
            "cold_model_load_seconds": _stats(
                [run["cold_model_load_seconds"] for run in payload["runs"]]
            ),
            "runs": payload["runs"],
            "samples": samples,
        }

    dense = workload_summaries["dense"]
    speedups: dict[str, Any] = {}
    for workload in ("frozen_new_12", "frozen_new_14"):
        speedups[workload] = {}
        for field in ("denoise_seconds", "total_generation_seconds", "artifact_ready_seconds"):
            dense_median = dense["latency_seconds"][field]["median"]
            candidate_median = workload_summaries[workload]["latency_seconds"][field]["median"]
            speedups[workload][field] = {
                "dense_median_seconds": dense_median,
                "candidate_median_seconds": candidate_median,
                "dense_over_candidate_speedup": dense_median / candidate_median,
                "candidate_minus_dense_seconds": candidate_median - dense_median,
            }

    return {
        "status": "complete",
        "same_machine_receipt": {
            "git_commit": commit,
            "gpu_uuid": gpu_uuid,
            "checkpoint_model_sha256": checkpoint_sha,
            "environment_freeze_sha256": environment_freeze_sha,
        },
        "order_balance": {
            "rule": canonical_plan["order_rule"],
            "position_counts": canonical_plan["position_counts"],
            "passed": True,
        },
        "workloads": workload_summaries,
        "median_speedups": speedups,
    }


def _load_split_analyses(paths: Iterable[Path], expected_split: str) -> dict[str, Any]:
    paths = list(paths)
    if not paths:
        return {"status": "pending", "analyses": []}
    opposite = "heldout" if expected_split == "development" else "development"
    analyses = []
    for path in paths:
        payload = _load_json(path, f"{expected_split} analysis")
        _require(isinstance(payload, Mapping), f"analysis root is invalid: {path}")
        _require(payload.get("record_type") == "ovi_cfg_ablation_v2_clustered_analysis", f"unexpected analysis type: {path}")
        _require(payload.get("cross_split_aggregation") == "forbidden", f"analysis does not forbid cross-split aggregation: {path}")
        splits = payload.get("splits")
        _require(isinstance(splits, Mapping), f"analysis split map is missing: {path}")
        expected = splits.get(expected_split)
        other = splits.get(opposite)
        _require(isinstance(expected, Mapping) and expected.get("status") == "ok", f"{path} has no {expected_split} records")
        _require(isinstance(other, Mapping) and other.get("status") == "no_records", f"{path} mixes {expected_split} and {opposite} records")
        analyses.append({"binding": _binding(path), "summary": expected})
    return {"status": "complete", "analyses": analyses}


def _optional_json_evidence(path: Path | None, label: str) -> dict[str, Any]:
    if path is None:
        return {"status": "pending"}
    payload = _load_json(path, label)
    _require(isinstance(payload, Mapping), f"{label} root must be an object")
    return {"status": "provided", "binding": _binding(path), "payload": payload}


def assemble_report(
    *,
    plan_path: Path,
    run_root: Path,
    development_analyses: Sequence[Path] = (),
    heldout_analyses: Sequence[Path] = (),
    stage0_gate: Path | None = None,
    candidate_freeze: Path | None = None,
) -> dict[str, Any]:
    plan = _load_json(plan_path, "Stage 4 plan")
    _require(isinstance(plan, Mapping), "Stage 4 plan root must be an object")
    canonical_plan = validate_stage4_plan(plan)
    development = _load_split_analyses(development_analyses, "development")
    heldout = _load_split_analyses(heldout_analyses, "heldout")
    stage4 = summarize_stage4(canonical_plan, run_root)
    external = {
        "asr_wer_cer": {
            "status": "pending",
            "scope": "speech prompts only; transcript inside <S>...<E>",
        },
        "syncnet_lip_sync": {
            "status": "pending",
            "requirement": "pinned SyncNet-style evaluator receipt",
        },
        "blind_human_review": {
            "status": "pending",
            "minimum_independent_ratings_per_pair": 3,
            "identity_blinding_required": True,
        },
    }
    quality_complete = development["status"] == "complete" and heldout["status"] == "complete"
    return {
        "schema_version": 1,
        "record_type": "ovi_cfg_ablation_v2_machine_report",
        "protocol_id": PROTOCOL_ID,
        "status": (
            "provisional_machine_report_external_review_pending"
            if quality_complete
            else "machine_latency_complete_quality_pending"
        ),
        "candidate_ids": canonical_plan["workload_ids"],
        "plan_binding": _binding(plan_path),
        "comparability_gate": _optional_json_evidence(stage0_gate, "Stage 0 gate"),
        "candidate_freeze": _optional_json_evidence(candidate_freeze, "candidate freeze"),
        "quality": {
            "development": development,
            "heldout": heldout,
            "cross_split_aggregation": "forbidden",
            "selection_rule": "development may select candidates; heldout may only confirm frozen candidates",
        },
        "stage4_latency": stage4,
        "external_evaluations": external,
        "final_conclusion": {
            "status": "pending",
            "reason": "ASR, SyncNet, and blind human review are not yet provided",
            "allowed_formats": [
                "late placement generalizes",
                "late placement helps only at one tier",
                "cache age dominates",
                "prompt-dependent sensitivity",
            ],
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    candidates = report["candidate_ids"]
    lines = [
        "# Ovi CFG-cache ablation v2 machine report",
        "",
        f"Status: `{report['status']}`",
        "",
        "This report is machine-only and provisional. ASR, SyncNet, and blind human review remain pending.",
        "",
        "## Frozen candidates",
        "",
        f"- 12-hit: `{candidates['frozen_new_12']}`",
        f"- 14-hit: `{candidates['frozen_new_14']}`",
        "",
        "## Evidence separation",
        "",
        f"- Development: `{report['quality']['development']['status']}`",
        f"- Held-out: `{report['quality']['heldout']['status']}`",
        "- Development and held-out aggregates are kept separate; cross-split aggregation is forbidden.",
        "",
        "## Stage 4 formal latency",
        "",
        "| Workload | Measurements | Denoise median (s) | Total warm-service median (s) | Save/mux median (s) | Artifact-ready median (s) | Cold load median (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for workload in WORKLOADS:
        summary = report["stage4_latency"]["workloads"][workload]
        latency = summary["latency_seconds"]
        lines.append(
            "| {name} | {count} | {denoise:.6f} | {total:.6f} | {save:.6f} | {ready:.6f} | {cold:.6f} |".format(
                name=workload,
                count=summary["measurement_count"],
                denoise=latency["denoise_seconds"]["median"],
                total=latency["total_generation_seconds"]["median"],
                save=latency["save_video_seconds"]["median"],
                ready=latency["artifact_ready_seconds"]["median"],
                cold=summary["cold_model_load_seconds"]["median"],
            )
        )
    lines.extend(
        [
            "",
            "Pre-denoise, audio-decode, and video-decode distributions, every measurement, run bindings, and balanced-position counts are retained in the JSON report.",
            "",
            "## Pending evaluations",
            "",
            "- ASR WER/CER for speech prompts",
            "- Pinned SyncNet-style lip-sync evaluation",
            "- At least three independent blind human ratings per pair",
            "",
            "No final adoption conclusion is issued until those evaluations are supplied.",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="emit a CPU-only balanced Stage 4 plan")
    plan.add_argument("--stage-tag", required=True)
    plan.add_argument("--new12-id", required=True)
    plan.add_argument("--new14-id", required=True)
    plan.add_argument("--blocks", type=int, default=3)
    plan.add_argument("--warmup-runs", type=int, default=3)
    plan.add_argument("--measurement-runs", type=int, default=5)
    plan.add_argument("--seed", type=int, default=103)
    plan.add_argument("--output", type=Path)

    report = subparsers.add_parser("report", help="assemble validated Stage 4 latency and split quality evidence")
    report.add_argument("--stage4-plan", type=Path, required=True)
    report.add_argument("--run-root", type=Path, required=True)
    report.add_argument("--development-analysis", type=Path, action="append", default=[])
    report.add_argument("--heldout-analysis", type=Path, action="append", default=[])
    report.add_argument("--stage0-gate", type=Path)
    report.add_argument("--candidate-freeze", type=Path)
    report.add_argument("--output-json", type=Path, required=True)
    report.add_argument("--output-markdown", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "plan":
            payload = build_stage4_plan(
                stage_tag=args.stage_tag,
                new12_id=args.new12_id,
                new14_id=args.new14_id,
                blocks=args.blocks,
                warmup_runs=args.warmup_runs,
                measurement_runs=args.measurement_runs,
                seed=args.seed,
            )
            rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
            if args.output is not None:
                _atomic_write(args.output, rendered)
            print(rendered, end="")
        elif args.command == "report":
            payload = assemble_report(
                plan_path=args.stage4_plan,
                run_root=args.run_root,
                development_analyses=args.development_analysis,
                heldout_analyses=args.heldout_analysis,
                stage0_gate=args.stage0_gate,
                candidate_freeze=args.candidate_freeze,
            )
            _atomic_write(
                args.output_json,
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            )
            _atomic_write(args.output_markdown, render_markdown(payload))
            print(json.dumps({"status": payload["status"], "output_json": str(args.output_json.resolve()), "output_markdown": str(args.output_markdown.resolve())}, indent=2, sort_keys=True))
        else:
            raise AssertionError(f"unexpected command {args.command}")
    except (ReportError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
