#!/usr/bin/env python3
"""Fail-closed Stage 0 gate for the Ovi CFG-cache ablation v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


class GateError(RuntimeError):
    pass


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GateError(f"JSON root is not an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def one_artifact_hashes(report: dict, context: str) -> dict:
    if report.get("status") != "ok" or report.get("artifact_count") != 1:
        raise GateError(f"{context} decoded-stream report is not one successful artifact")
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise GateError(f"{context} decoded-stream artifacts are malformed")
    item = artifacts[0]
    try:
        return {
            "container_sha256": item["container"]["sha256"],
            "decoded_rgb24_sha256": item["video"]["decode"]["sha256"],
            "decoded_rgb24_bytes": item["video"]["decode"]["bytes"],
            "decoded_pcm_sha256": item["audio"]["decode"]["sha256"],
            "decoded_pcm_bytes": item["audio"]["decode"]["bytes"],
        }
    except (KeyError, TypeError) as exc:
        raise GateError(f"{context} decoded-stream hashes are incomplete") from exc


def load_run(path: Path, label: str, expected_calls: int) -> dict:
    path = path.resolve(strict=True)
    protocol = load_json(path / "protocol_validation.json")
    if protocol.get("status") not in {"ok", "passed"}:
        raise GateError(f"{label} protocol validation did not pass")
    environment = load_json(path / "environment.json")
    timing_path = path / "timings.jsonl"
    try:
        lines = [line for line in timing_path.read_text(encoding="utf-8").splitlines() if line]
        timings = [json.loads(line) for line in lines]
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read {label} timings: {exc}") from exc
    if len(timings) != 1 or not isinstance(timings[0], dict):
        raise GateError(f"{label} must contain exactly one measurement")
    timing = timings[0]
    dispatcher = timing.get("video_self_attention_dispatcher")
    if not isinstance(dispatcher, dict):
        raise GateError(f"{label} dispatcher evidence is missing")
    if dispatcher.get("calls_total") != expected_calls:
        raise GateError(
            f"{label} attention calls {dispatcher.get('calls_total')} != {expected_calls}"
        )
    hashes = one_artifact_hashes(
        load_json(path / "decoded_stream_hashes.json"), label
    )
    return {
        "label": label,
        "path": str(path),
        "environment": environment,
        "timing": timing,
        "hashes": hashes,
        "evidence": {
            name: sha256_file(path / name)
            for name in (
                "preflight.json",
                "environment.freeze.txt",
                "checkpoint_manifest.json",
            )
        },
    }


def exact_stream_match(left: dict, right: dict) -> tuple[bool, dict]:
    # Exact container identity is the strongest available equality proof and
    # avoids false failures when two ffmpeg builds convert the same H.264/AAC
    # bytes to RGB/f32 with different rounding.  Only when containers differ
    # do we fall back to the protocol's decoded-stream hashes.
    container_match = (
        isinstance(left.get("container_sha256"), str)
        and left.get("container_sha256") == right.get("container_sha256")
    )
    fields = (
        "decoded_rgb24_sha256",
        "decoded_rgb24_bytes",
        "decoded_pcm_sha256",
        "decoded_pcm_bytes",
    )
    comparison = {field: left.get(field) == right.get(field) for field in fields}
    decoded_match = all(comparison.values())
    return container_match or decoded_match, {
        "criterion": (
            "exact_container_sha256"
            if container_match
            else "decoded_rgb24_and_pcm"
        ),
        "container_sha256": container_match,
        "decoded_streams": comparison,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("d0", "null", "anchor", "repeat1", "repeat2", "d1"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--old-anchor-hashes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    investigations: list[str] = []
    try:
        runs = {
            "d0": load_run(args.d0, "D0", 1770),
            "null": load_run(args.null, "12-29/r1", 1770),
            "anchor": load_run(args.anchor, "9-26/r5", 1364),
            "repeat1": load_run(args.repeat1, "12-29/r5 repetition 1", 1364),
            "repeat2": load_run(args.repeat2, "12-29/r5 repetition 2", 1364),
            "d1": load_run(args.d1, "D1", 1770),
        }
        old_anchor = one_artifact_hashes(
            load_json(args.old_anchor_hashes), "old 9-26/r5 anchor"
        )
        old_anchor_receipt = {
            "path": str(args.old_anchor_hashes.resolve(strict=True)),
            "sha256": sha256_file(args.old_anchor_hashes),
        }

        identities = {
            (
                run["environment"].get("git_commit"),
                run["environment"].get("gpu_uuid"),
                run["environment"].get("torch"),
                run["environment"].get("torch_cuda"),
            )
            for run in runs.values()
        }
        if len(identities) != 1:
            errors.append(f"Stage 0 commit/GPU/software identities differ: {sorted(identities)!r}")
        checkpoint_hashes = {run["evidence"]["checkpoint_manifest.json"] for run in runs.values()}
        freeze_hashes = {run["evidence"]["environment.freeze.txt"] for run in runs.values()}
        if len(checkpoint_hashes) != 1:
            errors.append("Stage 0 checkpoint manifests differ")
        if len(freeze_hashes) != 1:
            errors.append("Stage 0 environment freezes differ")

        comparisons = {}
        for name, left, right in (
            ("D0_equals_D1", runs["d0"]["hashes"], runs["d1"]["hashes"]),
            ("r1_equals_D0", runs["null"]["hashes"], runs["d0"]["hashes"]),
            (
                "late_r5_repetitions_equal",
                runs["repeat1"]["hashes"],
                runs["repeat2"]["hashes"],
            ),
            ("new_anchor_equals_old_anchor", runs["anchor"]["hashes"], old_anchor),
        ):
            matched, detail = exact_stream_match(left, right)
            comparisons[name] = {"matched": matched, "fields": detail}
            if not matched:
                errors.append(f"determinism/comparability gate failed: {name}")

        equal_compute_denoise = {}
        for name, left, right in (
            ("anchor_vs_late_repeat1", runs["anchor"], runs["repeat1"]),
            ("anchor_vs_late_repeat2", runs["anchor"], runs["repeat2"]),
        ):
            left_seconds = float(left["timing"]["denoise_seconds"])
            right_seconds = float(right["timing"]["denoise_seconds"])
            relative = (right_seconds - left_seconds) / left_seconds
            equal_compute_denoise[name] = {
                "left_seconds": left_seconds,
                "right_seconds": right_seconds,
                "relative_difference": relative,
                "within_one_percent": math.fabs(relative) <= 0.01,
            }
            if math.fabs(relative) > 0.01:
                investigations.append(f"{name} denoise difference exceeds 1%")

        report = {
            "schema_version": 1,
            "stage": 0,
            "status": "failed" if errors else (
                "needs_investigation" if investigations else "ok"
            ),
            "errors": errors,
            "investigations": investigations,
            "runs": {
                name: {
                    "path": run["path"],
                    "git_commit": run["environment"].get("git_commit"),
                    "gpu_uuid": run["environment"].get("gpu_uuid"),
                    "hashes": run["hashes"],
                }
                for name, run in runs.items()
            },
            "old_anchor_hashes": old_anchor,
            "old_anchor_receipt": old_anchor_receipt,
            "comparisons": comparisons,
            "equal_compute_denoise": equal_compute_denoise,
        }
    except (GateError, KeyError, TypeError, ValueError) as exc:
        report = {
            "schema_version": 1,
            "stage": 0,
            "status": "failed",
            "errors": [str(exc)],
            "investigations": [],
        }
    atomic_write(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
