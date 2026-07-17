#!/usr/bin/env python3
"""Fail-closed guards for resuming a Stage-3 run after the legacy audio gate.

This helper never evaluates or generates media.  It only proves that an old
``verification.json`` failed exclusively on the legacy RMS/peak/activity gates
for the two declared non-speech prompts (p006/p007), preserves that receipt,
and checks that an external evaluator reverified the exact same MP4 bytes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


PROTOCOL_ID = "ovi_cfg_cache_ablation_v2"
GENERATION_COMMIT = "84bfb9a0ee43d32be89bd18224906ce37647e76b"
STAGE3_ORDER = {
    503: ("dense", "old_12", "new_12", "old_14", "new_14"),
    887: ("new_12", "old_14", "new_14", "dense", "old_12"),
    1291: ("new_14", "dense", "old_12", "new_12", "old_14"),
}
ALLOWED_12 = {"late_12_29_r3", "late_15_29_r5"}
ALLOWED_14 = {"late_12_29_r5", "late_14_29_r8", "late_15_29_r15"}
ALLOWED_FAILURES = (
    (re.compile(r"^audio RMS is silent/invalid: (.+)$"), "rms", 1e-3),
    (re.compile(r"^audio peak is silent/invalid: (.+)$"), "peak", 1e-2),
    (
        re.compile(r"^audio active-sample ratio is too low: (.+)$"),
        "active_sample_ratio_abs_gt_1e-3",
        1e-2,
    ),
)
MP4_RE = re.compile(r"^p(?P<index>[0-9]{3})_")


class RecoveryError(ValueError):
    """The failed run is not eligible for external revalidation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryError(message)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"cannot read {label} {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} must be a JSON object: {path}")
    return payload


def _write_exclusive(path: Path, data: bytes, mode: int = 0o644) -> None:
    path = path.resolve()
    _require(path.parent.is_dir(), f"output parent is missing: {path.parent}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    _write_exclusive(path, data)


def expected_plan(stage_tag: str, mapping: Mapping[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordinal = 0
    for seed, labels in STAGE3_ORDER.items():
        for label in labels:
            ordinal += 1
            rows.append(
                {
                    "ordinal": ordinal,
                    "seed": seed,
                    "label": label,
                    "config_id": mapping[label],
                    "run_tag": f"{stage_tag}-{ordinal:02d}-seed{seed}-{label}",
                }
            )
    return rows


def validate_frozen_receipt(
    path: Path, stage_tag: str, expected_commit: str
) -> dict[str, Any]:
    payload = _load_json(path, "frozen candidate receipt")
    _require(payload.get("record_type") == "ovi_cfg_ablation_v2_frozen_stage3_candidates", "wrong frozen receipt type")
    _require(payload.get("protocol_id") == PROTOCOL_ID, "wrong frozen protocol ID")
    _require(payload.get("status") == "frozen", "candidate receipt is not frozen")
    _require(payload.get("frozen_before_first_heldout_run") is True, "receipt was not frozen before held-out runs")
    _require(payload.get("stage_tag") == stage_tag, "frozen receipt stage tag mismatch")
    git = payload.get("git")
    _require(isinstance(git, dict) and git.get("clean") is True, "frozen receipt Git state is not clean")
    _require(git.get("commit") == expected_commit, "frozen receipt generation commit mismatch")
    configurations = payload.get("configurations")
    _require(isinstance(configurations, dict), "frozen receipt configurations are missing")
    mapping = {
        label: (configurations.get(label) or {}).get("config_id")
        for label in ("dense", "old_12", "new_12", "old_14", "new_14")
    }
    _require(mapping["dense"] == "dense", "Dense mapping changed")
    _require(mapping["old_12"] == "current_6_23_r3", "old-12 mapping changed")
    _require(mapping["old_14"] == "current_9_26_r5_anchor", "old-14 mapping changed")
    _require(mapping["new_12"] in ALLOWED_12, "new-12 candidate is not protocol-allowed")
    _require(mapping["new_14"] in ALLOWED_14, "new-14 candidate is not protocol-allowed")
    _require(len(set(mapping.values())) == 5, "Stage-3 configurations are not distinct")
    plan = expected_plan(stage_tag, mapping)
    _require(payload.get("planned_runs") == plan, "frozen balanced plan changed")
    _require(payload.get("seeds") == [503, 887, 1291], "frozen seeds changed")
    return {"mapping": mapping, "planned_runs": plan}


def _artifact_index(artifact: Mapping[str, Any], context: str) -> int:
    index = artifact.get("prompt_index")
    _require(type(index) is int and 0 <= index < 8, f"{context} prompt_index is invalid")
    path = artifact.get("path")
    _require(isinstance(path, str), f"{context} artifact path is missing")
    match = MP4_RE.match(Path(path).name)
    _require(match is not None and int(match.group("index")) == index, f"{context} path/prompt mismatch")
    return index


def validate_legacy_failure(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    _require(payload.get("status") == "failed", "legacy verification is not failed")
    _require(payload.get("protocol") is None, "legacy receipt was not media-only verification")
    _require(payload.get("benchmark_valid") is False, "failed legacy receipt claims benchmark validity")
    _require("publication_errors" not in payload, "legacy failure includes publication errors")
    artifacts = payload.get("artifacts")
    _require(isinstance(artifacts, list) and len(artifacts) == 8, "legacy receipt must contain 8 artifacts")
    _require(payload.get("artifact_count") == 8, "legacy artifact_count mismatch")
    seen: set[int] = set()
    findings: list[dict[str, Any]] = []
    for position, artifact in enumerate(artifacts):
        _require(isinstance(artifact, dict), f"artifact[{position}] is not an object")
        index = _artifact_index(artifact, f"artifact[{position}]")
        _require(index not in seen, f"duplicate prompt index p{index:03d}")
        seen.add(index)
        errors = artifact.get("errors")
        _require(isinstance(errors, list) and all(isinstance(item, str) for item in errors), f"p{index:03d} errors are invalid")
        expected_status = "failed" if errors else "ok"
        _require(artifact.get("status") == expected_status, f"p{index:03d} status/errors disagree")
        if not errors:
            continue
        _require(index in {6, 7}, f"legacy failure touches speech prompt p{index:03d}")
        audio = artifact.get("audio")
        _require(isinstance(audio, dict), f"p{index:03d} audio evidence is missing")
        for error in errors:
            matched = False
            for pattern, field, old_threshold in ALLOWED_FAILURES:
                result = pattern.fullmatch(error)
                if result is None:
                    continue
                try:
                    value = float(result.group(1))
                except ValueError as exc:
                    raise RecoveryError(f"p{index:03d} old-gate value is invalid: {error}") from exc
                _require(math.isfinite(value) and 0.0 <= value <= old_threshold, f"p{index:03d} is not a finite old-threshold failure: {error}")
                observed = audio.get(field)
                _require(isinstance(observed, (int, float)) and not isinstance(observed, bool), f"p{index:03d} missing audio field {field}")
                _require(math.isclose(float(observed), value, rel_tol=1e-12, abs_tol=1e-15), f"p{index:03d} error/audio evidence mismatch for {field}")
                findings.append({"prompt_index": index, "field": field, "value": value, "old_threshold": old_threshold, "error": error})
                matched = True
                break
            _require(matched, f"p{index:03d} has a non-legacy failure: {error}")
    _require(seen == set(range(8)), "legacy receipt does not cover p000..p007")
    _require(bool(findings), "legacy receipt contains no eligible old audio-gate failure")
    return findings


def gate_command(args: argparse.Namespace) -> int:
    source_bytes = args.verification.resolve(strict=True).read_bytes()
    payload = _load_json(args.verification, "legacy verification")
    findings = validate_legacy_failure(payload)
    source_sha = _sha256_bytes(source_bytes)
    if args.backup.exists():
        _require(args.backup.read_bytes() == source_bytes, "existing legacy backup differs from verification")
    else:
        _write_exclusive(args.backup, source_bytes)
    receipt = {
        "schema_version": 1,
        "record_type": "ovi_cfg_ablation_v2_stage3_external_reverification_gate",
        "protocol_id": PROTOCOL_ID,
        "status": "eligible",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "legacy_verification": {"path": str(args.verification.resolve()), "bytes": len(source_bytes), "sha256": source_sha},
        "preserved_backup": {"path": str(args.backup.resolve()), "bytes": len(source_bytes), "sha256": source_sha},
        "allowed_prompt_indices": [6, 7],
        "findings": findings,
    }
    if args.receipt.exists():
        existing = _load_json(args.receipt, "recovery gate receipt")
        _require(existing.get("status") == "eligible", "existing recovery gate is not eligible")
        _require(existing.get("legacy_verification", {}).get("sha256") == source_sha, "existing recovery gate binds another verification")
    else:
        _write_json_exclusive(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def check_reverified_command(args: argparse.Namespace) -> int:
    legacy = _load_json(args.backup, "preserved legacy verification")
    validate_legacy_failure(legacy)
    current = _load_json(args.verification, "external verification")
    _require(current.get("status") == "ok", "external evaluator verification did not pass")
    _require(current.get("protocol") is None, "external verification is not media-only")
    _require(current.get("artifact_count") == 8, "external verification artifact_count mismatch")
    artifacts = current.get("artifacts")
    _require(isinstance(artifacts, list) and len(artifacts) == 8, "external verification must contain 8 artifacts")
    old_by_index = {_artifact_index(item, "legacy artifact"): item for item in legacy["artifacts"]}
    seen: set[int] = set()
    for artifact in artifacts:
        _require(isinstance(artifact, dict), "external artifact is not an object")
        index = _artifact_index(artifact, "external artifact")
        _require(index not in seen, f"duplicate external prompt p{index:03d}")
        seen.add(index)
        _require(artifact.get("status") == "ok" and artifact.get("errors") == [], f"external p{index:03d} did not pass")
        old = old_by_index[index]
        for field in ("path", "sha256", "measurement_index", "prompt_index", "sample_index", "seed"):
            _require(artifact.get(field) == old.get(field), f"external p{index:03d} changed {field}")
        old_binding = old.get("artifact_binding")
        new_binding = artifact.get("artifact_binding")
        _require(isinstance(old_binding, dict) and isinstance(new_binding, dict), f"p{index:03d} artifact binding missing")
        for field in ("path", "bytes", "sha256"):
            _require(new_binding.get(field) == old_binding.get(field), f"external p{index:03d} changed artifact {field}")
        audio = artifact.get("audio")
        _require(isinstance(audio, dict), f"external p{index:03d} audio evidence missing")
        rms = audio.get("rms")
        _require(isinstance(rms, (int, float)) and not isinstance(rms, bool) and math.isfinite(float(rms)) and float(rms) > 1e-6, f"external p{index:03d} audio remains silent")
    _require(seen == set(range(8)), "external verification does not cover p000..p007")
    print(json.dumps({"status": "passed", "artifact_count": 8, "unchanged_artifact_bytes": True}, indent=2))
    return 0


def check_validation_command(args: argparse.Namespace) -> int:
    payload = _load_json(args.validation, "protocol validation")
    _require(payload.get("status") == "passed", "protocol validation did not pass")
    _require(payload.get("cell_id") == args.config_id, "protocol validation config mismatch")
    _require(payload.get("seed") == args.seed, "protocol validation seed mismatch")
    validation = payload.get("validation")
    _require(isinstance(validation, dict), "protocol validation details missing")
    _require(validation.get("git_commit") == args.expected_commit, "validated generation commit mismatch")
    _require(validation.get("record_counts", {}).get("measurements") == 8, "validated measurement count is not 8")
    print(json.dumps({"status": "passed", "config_id": args.config_id, "seed": args.seed}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    frozen = subparsers.add_parser("check-frozen")
    frozen.add_argument("--receipt", required=True, type=Path)
    frozen.add_argument("--stage-tag", required=True)
    frozen.add_argument("--expected-commit", default=GENERATION_COMMIT)
    frozen.set_defaults(handler=lambda args: print(json.dumps(validate_frozen_receipt(args.receipt, args.stage_tag, args.expected_commit), indent=2)) or 0)
    gate = subparsers.add_parser("gate")
    gate.add_argument("--verification", required=True, type=Path)
    gate.add_argument("--backup", required=True, type=Path)
    gate.add_argument("--receipt", required=True, type=Path)
    gate.set_defaults(handler=gate_command)
    checked = subparsers.add_parser("check-reverified")
    checked.add_argument("--verification", required=True, type=Path)
    checked.add_argument("--backup", required=True, type=Path)
    checked.set_defaults(handler=check_reverified_command)
    validation = subparsers.add_parser("check-validation")
    validation.add_argument("--validation", required=True, type=Path)
    validation.add_argument("--config-id", required=True)
    validation.add_argument("--seed", required=True, type=int)
    validation.add_argument("--expected-commit", default=GENERATION_COMMIT)
    validation.set_defaults(handler=check_validation_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (RecoveryError, OSError, ValueError) as exc:
        parser.exit(2, f"recovery guard error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
