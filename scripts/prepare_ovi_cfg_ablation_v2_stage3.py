#!/usr/bin/env python3
"""Freeze Stage-3 candidates and build a blinded held-out review packet.

This helper is deliberately CPU-only.  The shell orchestrator calls ``freeze``
before the first held-out launch, then calls ``blind-packet`` only after all 15
GPU cells have passed the v2 run validator.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ovi.cfg_ablation_v2_protocol import (  # noqa: E402
    CANDIDATE_FREEZE_RULE,
    PROTOCOL_ID,
    STAGE3_BALANCED_ORDER,
    STAGE3_FIXED,
    STAGE_SEEDS,
    load_and_validate_matrix,
)


DEFAULT_MATRIX = REPO_ROOT / "configs/matrix/ovi_cfg_cache_ablation_v2_matrix.csv"
DEFAULT_PROMPTS = REPO_ROOT / "prompts/ovi_cfg_cache_heldout_prompts.csv"
DEFAULT_PROMPT_MANIFEST = (
    REPO_ROOT / "prompts/ovi_cfg_cache_heldout_prompt_manifest.csv"
)
RUN_TAG_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
MP4_PROMPT_RE = re.compile(r"^p(?P<index>[0-9]{3})_")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OLD_12_CONFIG_ID = STAGE3_FIXED["old_12_config_id"]
OLD_14_CONFIG_ID = STAGE3_FIXED["old_14_config_id"]
LABEL_TO_FIXED_CONFIG = {
    "dense": "dense",
    "old_12": OLD_12_CONFIG_ID,
    "old_14": OLD_14_CONFIG_ID,
}


class Stage3Error(ValueError):
    """Fail-closed Stage-3 preparation error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage3Error(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_binding(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    _require(path.is_file(), f"required input is not a regular file: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage3Error(f"cannot read {label} {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} must be a JSON object: {path}")
    return payload


def _write_new_json(path: Path, payload: Mapping[str, Any], mode: int = 0o644) -> None:
    path = path.resolve()
    _require(path.parent.is_dir(), f"output parent does not exist: {path.parent}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _read_prompt_contract(
    prompt_csv: Path, prompt_manifest: Path
) -> tuple[list[str], list[dict[str, str]]]:
    prompt_csv = prompt_csv.resolve(strict=True)
    prompt_manifest = prompt_manifest.resolve(strict=True)
    _require(
        prompt_csv == DEFAULT_PROMPTS.resolve(strict=True),
        "Stage 3 requires the authoritative held-out8 prompt CSV",
    )
    _require(
        prompt_manifest == DEFAULT_PROMPT_MANIFEST.resolve(strict=True),
        "Stage 3 requires the authoritative held-out8 prompt manifest",
    )
    with prompt_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require(tuple(reader.fieldnames or ()) == ("text_prompt",), "invalid held-out CSV header")
        prompts = [(row.get("text_prompt") or "").strip() for row in reader]
    _require(len(prompts) == 8 and all(prompts), "held-out CSV must contain exactly 8 prompts")
    with prompt_manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require(
            tuple(reader.fieldnames or ()) == ("prompt_id", "category", "primary_stress"),
            "invalid held-out prompt-manifest header",
        )
        manifest = [dict(row) for row in reader]
    _require(len(manifest) == 8, "held-out prompt manifest must contain exactly 8 rows")
    _require(
        [row["prompt_id"] for row in manifest] == [f"H{i:02d}" for i in range(1, 9)],
        "held-out prompt IDs must be H01..H08 in order",
    )
    return prompts, manifest


def config_map(new_12: str, new_14: str) -> dict[str, str]:
    _require(
        new_12 in CANDIDATE_FREEZE_RULE["conservative_12_hit_allowed"],
        f"invalid frozen 12-hit candidate: {new_12!r}",
    )
    _require(
        new_14 in CANDIDATE_FREEZE_RULE["aggressive_14_hit_allowed"],
        f"invalid frozen 14-hit candidate: {new_14!r}",
    )
    mapping = {**LABEL_TO_FIXED_CONFIG, "new_12": new_12, "new_14": new_14}
    _require(len(set(mapping.values())) == 5, "the five Stage-3 configurations must be distinct")
    return mapping


def expected_runs(stage_tag: str, mapping: Mapping[str, str]) -> list[dict[str, Any]]:
    _require(RUN_TAG_RE.fullmatch(stage_tag) is not None, f"invalid stage tag: {stage_tag!r}")
    rows: list[dict[str, Any]] = []
    ordinal = 0
    for seed in STAGE_SEEDS["3"]:
        for label in STAGE3_BALANCED_ORDER[seed]:
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
    _require(len(rows) == 15, "internal Stage-3 order did not produce exactly 15 cells")
    return rows


def _git_receipt() -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise Stage3Error(f"git {' '.join(args)} failed: {detail}")
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    _require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "invalid Git commit")
    dirty = run("status", "--porcelain", "--untracked-files=all")
    _require(dirty == "", "refusing to freeze Stage-3 candidates from a dirty Git tree")
    return {"commit": commit, "clean": True}


def build_frozen_receipt(
    *,
    matrix: Path,
    prompt_csv: Path,
    prompt_manifest: Path,
    new_12: str,
    new_14: str,
    stage_tag: str,
    run_root: Path,
    git_receipt: Mapping[str, Any],
    selection_evidence: Path | None = None,
) -> dict[str, Any]:
    mapping = config_map(new_12, new_14)
    matrix = matrix.resolve(strict=True)
    _require(matrix == DEFAULT_MATRIX.resolve(strict=True), "Stage 3 requires the authoritative v2 matrix")
    cells = {cell.config_id: cell for cell in load_and_validate_matrix(matrix)}
    _read_prompt_contract(prompt_csv, prompt_manifest)
    _require(cells[new_12].cache_hits == 12, "frozen new-12 row is not a 12-hit cell")
    _require(cells[new_14].cache_hits == 14, "frozen new-14 row is not a 14-hit cell")
    runs = expected_runs(stage_tag, mapping)
    run_root = run_root.resolve()
    collisions = [str(run_root / row["run_tag"]) for row in runs if (run_root / row["run_tag"]).exists()]
    _require(not collisions, f"refusing Stage-3 run reuse: {collisions}")

    evidence: dict[str, Any]
    if selection_evidence is None:
        evidence = {
            "status": "not_supplied",
            "note": "candidate IDs were supplied explicitly; no Stage-2 evidence claim is inferred",
        }
    else:
        evidence = {"status": "bound", **_file_binding(selection_evidence)}
    return {
        "schema_version": 1,
        "record_type": "ovi_cfg_ablation_v2_frozen_stage3_candidates",
        "protocol_id": PROTOCOL_ID,
        "status": "frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_before_first_heldout_run": True,
        "stage_tag": stage_tag,
        "git": dict(git_receipt),
        "inputs": {
            "matrix": _file_binding(matrix),
            "heldout_prompt_csv": _file_binding(prompt_csv),
            "heldout_prompt_manifest": _file_binding(prompt_manifest),
        },
        "selection_evidence": evidence,
        "configurations": {
            label: {"config_id": config_id, **cells[config_id].as_json()}
            for label, config_id in mapping.items()
        },
        "seeds": list(STAGE_SEEDS["3"]),
        "balanced_order": {
            str(seed): list(STAGE3_BALANCED_ORDER[seed]) for seed in STAGE_SEEDS["3"]
        },
        "planned_runs": runs,
        "heldout_runs_present_at_freeze": [],
        "pending_evaluations": {
            "asr": {"status": "pending", "reason": "no pinned ASR evaluator supplied"},
            "syncnet": {"status": "pending", "reason": "no pinned SyncNet evaluator supplied"},
            "human_blind_review": {
                "status": "pending",
                "reason": "three independent ratings per pair have not yet been collected",
            },
        },
    }


def freeze_command(args: argparse.Namespace) -> int:
    receipt = build_frozen_receipt(
        matrix=args.matrix,
        prompt_csv=args.prompt_csv,
        prompt_manifest=args.prompt_manifest,
        new_12=args.new_12_config_id,
        new_14=args.new_14_config_id,
        stage_tag=args.stage_tag,
        run_root=args.run_root,
        git_receipt=_git_receipt(),
        selection_evidence=args.selection_evidence,
    )
    _write_new_json(args.output, receipt)
    print(f"Frozen Stage-3 candidates: {args.output.resolve()}")
    return 0


def _validated_run_artifacts(
    run_dir: Path,
    expected: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[int, Path]:
    _require(run_dir.is_dir() and not run_dir.is_symlink(), f"missing or linked run directory: {run_dir}")
    report = _load_json(run_dir / "protocol_validation.json", "run protocol validation")
    _require(report.get("status") == "passed", f"run did not pass validation: {run_dir}")
    _require(report.get("cell_id") == expected["config_id"], f"cell ID mismatch: {run_dir}")
    _require(report.get("seed") == expected["seed"], f"seed mismatch: {run_dir}")
    validation = report.get("validation")
    _require(isinstance(validation, dict), f"run validation details missing: {run_dir}")
    _require(
        validation.get("record_counts", {}).get("measurements") == 8,
        f"Stage-3 run must contain exactly 8 measurements: {run_dir}",
    )
    _require(
        validation.get("git_commit") == receipt.get("git", {}).get("commit"),
        f"run Git commit differs from frozen receipt: {run_dir}",
    )
    prompt_binding = report.get("inputs", {}).get("prompt_csv", {})
    expected_prompt_sha = receipt.get("inputs", {}).get("heldout_prompt_csv", {}).get("sha256")
    _require(
        prompt_binding.get("sha256") == expected_prompt_sha,
        f"run held-out prompt SHA differs from frozen receipt: {run_dir}",
    )
    artifacts: dict[int, Path] = {}
    for path in sorted(run_dir.rglob("*.mp4")):
        match = MP4_PROMPT_RE.match(path.name)
        _require(match is not None, f"cannot bind MP4 to held-out prompt index: {path}")
        index = int(match.group("index"))
        _require(index not in artifacts, f"duplicate MP4 for prompt index {index}: {run_dir}")
        artifacts[index] = path.resolve(strict=True)
    _require(set(artifacts) == set(range(8)), f"run MP4 indices must be exactly p000..p007: {run_dir}")
    return artifacts


def _blind_side(nonce: bytes, tier: str, seed: int, prompt_id: str) -> bool:
    message = f"{PROTOCOL_ID}|{tier}|{seed}|{prompt_id}".encode("utf-8")
    digest = hmac.new(nonce, message, hashlib.sha256).digest()
    return bool(digest[0] & 1)


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def build_blind_packet(
    *,
    frozen_receipt: Path,
    run_root: Path,
    output_dir: Path,
    nonce_hex: str | None = None,
) -> dict[str, Any]:
    frozen_receipt = frozen_receipt.resolve(strict=True)
    receipt = _load_json(frozen_receipt, "frozen candidate receipt")
    _require(receipt.get("protocol_id") == PROTOCOL_ID and receipt.get("status") == "frozen", "invalid frozen candidate receipt")
    mapping = {
        label: receipt.get("configurations", {}).get(label, {}).get("config_id")
        for label in ("dense", "old_12", "new_12", "old_14", "new_14")
    }
    expected_mapping = config_map(str(mapping["new_12"]), str(mapping["new_14"]))
    _require(mapping == expected_mapping, "frozen receipt configuration map is invalid")
    stage_tag = receipt.get("stage_tag")
    _require(isinstance(stage_tag, str), "frozen receipt stage tag is missing")
    planned = expected_runs(stage_tag, expected_mapping)
    _require(receipt.get("planned_runs") == planned, "frozen receipt balanced run plan changed")
    prompts, prompt_manifest = _read_prompt_contract(DEFAULT_PROMPTS, DEFAULT_PROMPT_MANIFEST)
    run_root = run_root.resolve(strict=True)
    artifacts: dict[tuple[int, str], dict[int, Path]] = {}
    for expected in planned:
        key = (expected["seed"], expected["label"])
        artifacts[key] = _validated_run_artifacts(
            run_root / expected["run_tag"], expected, receipt
        )

    output_dir = output_dir.resolve()
    _require(not os.path.lexists(output_dir), f"refusing to reuse blind-review output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if nonce_hex is None:
        nonce_hex = secrets.token_hex(32)
    _require(SHA256_RE.fullmatch(nonce_hex) is not None, "blind nonce must be 32 bytes of lowercase hex")
    nonce = bytes.fromhex(nonce_hex)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        packet_dir = temporary / "packet"
        media_dir = packet_dir / "media"
        media_dir.mkdir(parents=True)
        review_rows: list[dict[str, Any]] = []
        private_pairs: list[dict[str, Any]] = []
        public_media: list[dict[str, Any]] = []
        pair_number = 0

        # Dense references are copied once per prompt-seed unit and contain no method name.
        reference_files: dict[tuple[int, int], str] = {}
        reference_number = 0
        for seed in STAGE_SEEDS["3"]:
            for prompt_index, manifest_row in enumerate(prompt_manifest):
                reference_number += 1
                relative = f"media/R{reference_number:03d}.mp4"
                source = artifacts[(seed, "dense")][prompt_index]
                destination = temporary / "packet" / relative
                shutil.copyfile(source, destination)
                binding = _file_binding(destination)
                public_media.append({"path": relative, "bytes": binding["bytes"], "sha256": binding["sha256"]})
                reference_files[(seed, prompt_index)] = relative

        for tier, old_label, new_label in (
            ("12-hit", "old_12", "new_12"),
            ("14-hit", "old_14", "new_14"),
        ):
            for seed in STAGE_SEEDS["3"]:
                for prompt_index, manifest_row in enumerate(prompt_manifest):
                    pair_number += 1
                    pair_id = f"P{pair_number:03d}"
                    new_on_a = _blind_side(nonce, tier, seed, manifest_row["prompt_id"])
                    side_labels = {
                        "A": new_label if new_on_a else old_label,
                        "B": old_label if new_on_a else new_label,
                    }
                    private_sides: dict[str, Any] = {}
                    public_paths: dict[str, str] = {}
                    for side in ("A", "B"):
                        label = side_labels[side]
                        source = artifacts[(seed, label)][prompt_index]
                        relative = f"media/{pair_id}_{side}.mp4"
                        destination = packet_dir / relative
                        shutil.copyfile(source, destination)
                        binding = _file_binding(destination)
                        public_media.append({"path": relative, "bytes": binding["bytes"], "sha256": binding["sha256"]})
                        public_paths[side] = relative
                        private_sides[side] = {
                            "label": label,
                            "config_id": expected_mapping[label],
                            "source_run": str(source.parent),
                            "source_artifact": str(source),
                            "packet_artifact": relative,
                            "sha256": binding["sha256"],
                        }
                    private_pairs.append(
                        {
                            "pair_id": pair_id,
                            "tier": tier,
                            "prompt_id": manifest_row["prompt_id"],
                            "seed": seed,
                            "sides": private_sides,
                        }
                    )
                    for reviewer_slot in range(1, 4):
                        review_rows.append(
                            {
                                "pair_id": pair_id,
                                "tier": tier,
                                "prompt_id": manifest_row["prompt_id"],
                                "category": manifest_row["category"],
                                "seed": seed,
                                "reviewer_slot": reviewer_slot,
                                "media_A": public_paths["A"],
                                "media_B": public_paths["B"],
                                "reference_media": reference_files[(seed, prompt_index)],
                                "human_review_status": "pending",
                                "speech_intelligibility_A": "",
                                "speech_intelligibility_B": "",
                                "lip_synchronization_A": "",
                                "lip_synchronization_B": "",
                                "visual_artifacts_temporal_stability_A": "",
                                "visual_artifacts_temporal_stability_B": "",
                                "prompt_adherence_A": "",
                                "prompt_adherence_B": "",
                                "overall_preference": "",
                                "reviewer_notes": "",
                            }
                        )

        rating_fields = list(review_rows[0])
        _write_csv(packet_dir / "ratings.csv", rating_fields, review_rows)
        readme = (
            "# Ovi CFG-cache v2 Stage 3 blind review\n\n"
            "Rate A and B without attempting to identify the method. Use three independent "
            "reviewers per pair (one reviewer per slot). The R file is an optional Dense "
            "reference; latency and method names are intentionally absent. Keep every rating "
            "blank until a reviewer supplies it. Do not distribute ../private_mapping.json.\n"
        )
        (packet_dir / "README.md").write_text(readme, encoding="utf-8")

        private_mapping = {
            "schema_version": 1,
            "record_type": "ovi_cfg_ablation_v2_private_blind_mapping",
            "protocol_id": PROTOCOL_ID,
            "status": "sealed_pending_human_review",
            "stage_tag": stage_tag,
            "randomization": {
                "algorithm": "HMAC-SHA256(nonce, protocol|tier|seed|prompt_id), low bit",
                "nonce_hex": nonce_hex,
            },
            "frozen_candidate_receipt": _file_binding(frozen_receipt),
            "pairs": private_pairs,
        }
        mapping_path = temporary / "private_mapping.json"
        _write_new_json(mapping_path, private_mapping, mode=0o600)
        mapping_sha = _sha256(mapping_path)
        packet_manifest = {
            "schema_version": 1,
            "record_type": "ovi_cfg_ablation_v2_blind_review_packet",
            "protocol_id": PROTOCOL_ID,
            "status": "pending",
            "stage_tag": stage_tag,
            "pair_count": pair_number,
            "reviewers_required_per_pair": 3,
            "rating_row_count": len(review_rows),
            "dense_reference_count": len(reference_files),
            "private_mapping_sha256": mapping_sha,
            "media": sorted(public_media, key=lambda row: row["path"]),
            "pending_evaluations": {
                "asr": {"status": "pending", "reason": "no pinned ASR results supplied"},
                "syncnet": {"status": "pending", "reason": "no pinned SyncNet results supplied"},
                "human_blind_review": {
                    "status": "pending",
                    "required_independent_ratings_per_pair": 3,
                },
            },
        }
        _write_new_json(packet_dir / "manifest.json", packet_manifest)
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "output_dir": str(output_dir),
        "packet_dir": str(output_dir / "packet"),
        "private_mapping": str(output_dir / "private_mapping.json"),
        "pair_count": 48,
        "rating_row_count": 144,
        "status": "pending",
    }


def blind_packet_command(args: argparse.Namespace) -> int:
    summary = build_blind_packet(
        frozen_receipt=args.frozen_receipt,
        run_root=args.run_root,
        output_dir=args.output_dir,
        nonce_hex=args.nonce_hex,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="freeze candidates before held-out runs")
    freeze.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    freeze.add_argument("--prompt-csv", type=Path, default=DEFAULT_PROMPTS)
    freeze.add_argument("--prompt-manifest", type=Path, default=DEFAULT_PROMPT_MANIFEST)
    freeze.add_argument("--new-12-config-id", required=True)
    freeze.add_argument("--new-14-config-id", required=True)
    freeze.add_argument("--stage-tag", required=True)
    freeze.add_argument("--run-root", type=Path, required=True)
    freeze.add_argument("--selection-evidence", type=Path)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.set_defaults(handler=freeze_command)

    packet = subparsers.add_parser("blind-packet", help="build a randomized held-out packet")
    packet.add_argument("--frozen-receipt", type=Path, required=True)
    packet.add_argument("--run-root", type=Path, required=True)
    packet.add_argument("--output-dir", type=Path, required=True)
    packet.add_argument("--nonce-hex", help=argparse.SUPPRESS)
    packet.set_defaults(handler=blind_packet_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (Stage3Error, OSError, ValueError) as exc:
        parser.exit(2, f"stage3 error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
