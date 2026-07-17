#!/usr/bin/env python3
"""Fail-closed Stage 1 early-vs-late SSIM analysis for Ovi CFG-cache v2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARE_SCRIPT = REPO_ROOT / "scripts/compare_ovi_cfg_ablation_v2.py"
PROTOCOL_ID = "ovi_cfg_cache_ablation_v2"
SEEDS = (103, 211)
EARLY_BINS = ("bin_00_04_r5", "bin_05_09_r5", "bin_10_14_r5")
LATE_BINS = ("bin_15_19_r5", "bin_20_24_r5", "bin_25_29_r5")
ALL_BINS = EARLY_BINS + LATE_BINS
CONFIG_ORDER = ("dense",) + ALL_BINS
TAG_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class Stage1AnalysisError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage1AnalysisError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_runs(stage_tag: str, run_root: Path) -> dict[tuple[int, str], Path]:
    runs: dict[tuple[int, str], Path] = {}
    ordinal = 0
    for seed in SEEDS:
        for config_id in CONFIG_ORDER:
            ordinal += 1
            run_tag = f"{stage_tag}-{ordinal:02d}-s{seed}-{config_id}"
            runs[(seed, config_id)] = run_root / run_tag
    return runs


def comparison_filename(seed: int, config_id: str) -> str:
    return f"seed{seed}__{config_id}__vs_dense.json"


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage1AnalysisError(f"cannot read {label} {path}: {exc}") from exc
    _require(isinstance(payload, Mapping), f"{label} root must be an object: {path}")
    return payload


def _validate_receipts(
    stage_tag: str, run_root: Path
) -> tuple[dict[tuple[int, str], Path], dict[str, Any]]:
    runs = expected_runs(stage_tag, run_root)
    actual = {path.resolve() for path in run_root.glob(f"{stage_tag}-*") if path.is_dir()}
    expected = {path.resolve() for path in runs.values()}
    _require(
        actual == expected,
        f"stage tag must resolve to exactly the 14 canonical run directories; "
        f"missing={sorted(str(path) for path in expected - actual)}, "
        f"extra={sorted(str(path) for path in actual - expected)}",
    )

    provenance: set[tuple[str, str, str, str]] = set()
    receipt_paths: list[str] = []
    for (seed, config_id), run_dir in runs.items():
        receipt_path = run_dir / "protocol_validation.json"
        receipt = _load_json(receipt_path, "protocol-validation receipt")
        validation = receipt.get("validation")
        _require(isinstance(validation, Mapping), f"missing validation details: {receipt_path}")
        cell = validation.get("cell")
        counts = validation.get("record_counts")
        checkpoint = validation.get("checkpoint")
        inputs = receipt.get("inputs")
        _require(isinstance(cell, Mapping), f"missing validated cell: {receipt_path}")
        _require(isinstance(counts, Mapping), f"missing record counts: {receipt_path}")
        _require(isinstance(checkpoint, Mapping), f"missing checkpoint binding: {receipt_path}")
        _require(isinstance(inputs, Mapping), f"missing input bindings: {receipt_path}")
        prompt_input = inputs.get("prompt_csv")
        _require(isinstance(prompt_input, Mapping), f"missing prompt binding: {receipt_path}")
        expected_source_stage = "0" if config_id == "dense" else "1"
        checks = (
            (receipt.get("status") == "passed", "status is not passed"),
            (receipt.get("cell_id") == config_id, "cell_id mismatch"),
            (receipt.get("seed") == seed, "seed mismatch"),
            (cell.get("config_id") == config_id, "validated config_id mismatch"),
            (cell.get("stage") == expected_source_stage, "source matrix stage mismatch"),
            (counts.get("measurements") == 3, "measurement count is not 3"),
            (
                isinstance(validation.get("decoded_streams"), Mapping)
                and len(validation["decoded_streams"]) == 3,
                "decoded stream count is not 3",
            ),
            (Path(str(receipt.get("run_dir", ""))).resolve() == run_dir.resolve(), "run_dir mismatch"),
        )
        failed = [message for ok, message in checks if not ok]
        _require(not failed, f"invalid receipt {receipt_path}: {', '.join(failed)}")
        values = (
            validation.get("git_commit"),
            validation.get("gpu_uuid"),
            checkpoint.get("model_sha256"),
            prompt_input.get("sha256"),
        )
        _require(all(isinstance(value, str) and value for value in values), f"incomplete provenance: {receipt_path}")
        provenance.add(values)  # type: ignore[arg-type]
        receipt_paths.append(str(receipt_path.resolve()))
    _require(len(provenance) == 1, "the 14 runs do not share commit/GPU/checkpoint/prompt provenance")
    git_commit, gpu_uuid, model_sha256, prompt_sha256 = provenance.pop()
    return runs, {
        "git_commit": git_commit,
        "gpu_uuid": gpu_uuid,
        "model_sha256": model_sha256,
        "prompt_csv_sha256": prompt_sha256,
        "protocol_validation_receipts": receipt_paths,
    }


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _run_comparison(
    *,
    python: str,
    dense_run: Path,
    candidate_run: Path,
    seed: int,
    config_id: str,
    output: Path,
    ffmpeg: str,
    ffprobe: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        python,
        str(COMPARE_SCRIPT),
        "compare",
        "--dense-run",
        str(dense_run),
        "--candidate-run",
        str(candidate_run),
        "--split",
        "development",
        "--seed",
        str(seed),
        "--candidate-id",
        config_id,
        "--comparison-id",
        f"stage1-seed{seed}-{config_id}-vs-dense",
        "--ffmpeg",
        ffmpeg,
        "--ffprobe",
        ffprobe,
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, check=False)
    _require(completed.returncode == 0, f"comparison command failed ({completed.returncode}): {' '.join(command)}")
    _require(output.is_file(), f"comparison command did not create {output}")


def _media_binding(record: Mapping[str, Any], key: str, run_dir: Path, context: str) -> None:
    binding = record.get(key)
    _require(isinstance(binding, Mapping), f"{context} missing {key} binding")
    path = Path(str(binding.get("path", "")))
    _require(path.is_file(), f"{context} bound media is missing: {path}")
    _require(_is_within(path, run_dir), f"{context} {key} media escapes expected run: {path}")
    _require(binding.get("sha256") == _sha256(path), f"{context} stale {key} media hash")


def _load_comparison(
    path: Path, *, dense_run: Path, candidate_run: Path, seed: int, config_id: str
) -> dict[tuple[str, str], float]:
    payload = _load_json(path, "comparison JSON")
    pairs = payload.get("pairs")
    _require(payload.get("record_type") == "ovi_cfg_ablation_v2_media_comparison", f"wrong comparison type: {path}")
    _require(payload.get("pair_count") == 3, f"comparison pair_count is not 3: {path}")
    _require(isinstance(pairs, list) and len(pairs) == 3, f"comparison must contain 3 pairs: {path}")
    values: dict[tuple[str, str], float] = {}
    for index, record in enumerate(pairs):
        context = f"{path} pair {index}"
        _require(isinstance(record, Mapping), f"{context} is not an object")
        _require(record.get("split") == "development", f"{context} split mismatch")
        _require(record.get("seed") == seed, f"{context} seed mismatch")
        _require(record.get("candidate_id") == config_id, f"{context} candidate_id mismatch")
        prompt_id = str(record.get("prompt_id") or "").strip()
        prompt = str(record.get("prompt") or "").strip()
        _require(prompt_id and prompt, f"{context} prompt identity is missing")
        key = (prompt_id, prompt)
        _require(key not in values, f"{context} duplicate prompt identity")
        metrics = record.get("metrics")
        video = metrics.get("video") if isinstance(metrics, Mapping) else None
        ssim = video.get("ssim") if isinstance(video, Mapping) else None
        _require(
            isinstance(ssim, (int, float)) and not isinstance(ssim, bool) and math.isfinite(float(ssim)),
            f"{context} SSIM is not finite numeric",
        )
        numeric_ssim = float(ssim)
        _require(-1.0 <= numeric_ssim <= 1.0, f"{context} SSIM is outside [-1,1]")
        _media_binding(record, "dense", dense_run, context)
        _media_binding(record, "candidate", candidate_run, context)
        values[key] = numeric_ssim
    return values


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    _require(TAG_RE.fullmatch(args.stage_tag) is not None, "invalid --stage-tag")
    run_root = args.run_root.resolve()
    _require(run_root.is_dir(), f"run root does not exist: {run_root}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.stage_tag}_stage1_position_analysis.json"
    csv_path = output_dir / f"{args.stage_tag}_stage1_position_units.csv"
    _require(not json_path.exists() and not csv_path.exists(), "refusing to overwrite Stage 1 analysis outputs")
    comparison_dir = (args.comparison_dir or output_dir / "comparisons").resolve()

    runs, provenance = _validate_receipts(args.stage_tag, run_root)
    scores: dict[tuple[int, str, str, str], float] = {}
    comparison_paths: list[str] = []
    prompts_by_seed: dict[int, set[tuple[str, str]]] = {}
    for seed in SEEDS:
        dense_run = runs[(seed, "dense")]
        for config_id in ALL_BINS:
            comparison_path = comparison_dir / comparison_filename(seed, config_id)
            if not comparison_path.is_file():
                _require(not args.reuse_comparisons_only, f"missing required comparison JSON: {comparison_path}")
                _run_comparison(
                    python=args.comparison_python,
                    dense_run=dense_run,
                    candidate_run=runs[(seed, config_id)],
                    seed=seed,
                    config_id=config_id,
                    output=comparison_path,
                    ffmpeg=args.ffmpeg,
                    ffprobe=args.ffprobe,
                )
            values = _load_comparison(
                comparison_path,
                dense_run=dense_run,
                candidate_run=runs[(seed, config_id)],
                seed=seed,
                config_id=config_id,
            )
            if seed not in prompts_by_seed:
                prompts_by_seed[seed] = set(values)
            _require(set(values) == prompts_by_seed[seed], f"prompt set differs in {comparison_path}")
            for (prompt_id, prompt), ssim in values.items():
                key = (seed, prompt_id, prompt, config_id)
                _require(key not in scores, f"duplicate SSIM observation: {key}")
                scores[key] = ssim
            comparison_paths.append(str(comparison_path.resolve()))
    _require(prompts_by_seed[103] == prompts_by_seed[211], "prompt identities differ between seeds")
    _require(len(prompts_by_seed[103]) == 3, "Stage 1 must contain exactly 3 prompts")

    units: list[dict[str, Any]] = []
    for seed in SEEDS:
        for prompt_id, prompt in sorted(prompts_by_seed[seed]):
            per_bin = {
                config_id: scores[(seed, prompt_id, prompt, config_id)]
                for config_id in ALL_BINS
            }
            early_mean = sum(per_bin[item] for item in EARLY_BINS) / len(EARLY_BINS)
            late_mean = sum(per_bin[item] for item in LATE_BINS) / len(LATE_BINS)
            units.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "seed": seed,
                    "ssim_by_bin": per_bin,
                    "early_mean_ssim": early_mean,
                    "late_mean_ssim": late_mean,
                    "late_minus_early_ssim": late_mean - early_mean,
                    "late_less_damaging": late_mean > early_mean,
                }
            )
    passing = sum(bool(unit["late_less_damaging"]) for unit in units)
    supported = passing >= 5
    report = {
        "schema_version": 1,
        "record_type": "ovi_cfg_ablation_v2_stage1_position_analysis",
        "protocol_id": PROTOCOL_ID,
        "status": "passed",
        "stage_tag": args.stage_tag,
        "primary_metric": "video_ssim",
        "criterion": "mean SSIM of bins 15-29 is strictly greater than mean SSIM of bins 0-14 in at least 5 of 6 prompt-seed units",
        "early_bins": list(EARLY_BINS),
        "late_bins": list(LATE_BINS),
        "unit_count": len(units),
        "late_less_damaging_unit_count": passing,
        "required_unit_count": 5,
        "position_claim_supported": supported,
        "units": units,
        "provenance": {**provenance, "comparison_jsons": comparison_paths},
        "outputs": {"json": str(json_path), "csv": str(csv_path)},
    }

    fieldnames = [
        "prompt_id", "prompt", "seed", *[f"ssim_{item}" for item in ALL_BINS],
        "early_mean_ssim", "late_mean_ssim", "late_minus_early_ssim", "late_less_damaging",
    ]
    temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    temporary_json = json_path.with_suffix(json_path.suffix + ".tmp")
    _require(not temporary_csv.exists() and not temporary_json.exists(), "refusing existing temporary output")
    with temporary_csv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for unit in units:
            row = {key: unit[key] for key in ("prompt_id", "prompt", "seed")}
            row.update({f"ssim_{key}": value for key, value in unit["ssim_by_bin"].items()})
            row.update({key: unit[key] for key in ("early_mean_ssim", "late_mean_ssim", "late_minus_early_ssim", "late_less_damaging")})
            writer.writerow(row)
    temporary_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary_csv, csv_path)
    os.replace(temporary_json, json_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-tag", required=True)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(os.environ.get("FASTA2V_CACHE_ROOT", "/cache/liluchen/FastA2V")) / "runs/ovi_cfg_ablation_v2",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--comparison-dir", type=Path)
    parser.add_argument("--reuse-comparisons-only", action="store_true")
    parser.add_argument("--comparison-python", default=sys.executable)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = analyze(args)
    except (Stage1AnalysisError, OSError, ValueError) as exc:
        print(f"stage1 analysis error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
