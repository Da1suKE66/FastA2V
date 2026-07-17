#!/usr/bin/env python3
"""Summarize Ovi CFG-cache v2 Stage-2 comparisons without freezing candidates.

Inputs may be existing JSON outputs from ``compare_ovi_cfg_ablation_v2.py`` or
explicit Dense/candidate run pairs.  Explicit run pairs are compared by calling
that existing tool in a temporary directory; this script itself never launches
generation or touches a GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARE_TOOL = REPO_ROOT / "scripts/compare_ovi_cfg_ablation_v2.py"
MATRIX = REPO_ROOT / "configs/matrix/ovi_cfg_cache_ablation_v2_matrix.csv"
R15_ID = "late_15_29_r15"
STAGE2_IDS = {
    "current_6_23_r3",
    "late_12_29_r2",
    "late_12_29_r3",
    "late_12_29_r4",
    "late_12_29_r5",
    "late_15_29_r5",
    "late_14_29_r8",
    R15_ID,
}


class AnalysisError(ValueError):
    """Fail-closed Stage-2 analysis error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def _number(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    result = float(value)
    _require(math.isfinite(result), f"{label} is not finite")
    return result


def _load_matrix() -> dict[str, dict[str, Any]]:
    with MATRIX.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        config_id = row["config_id"]
        if config_id not in STAGE2_IDS:
            continue
        _require(row["stage"] == "2", f"{config_id} is not a Stage-2 matrix row")
        result[config_id] = {
            "cache_hits": int(row["cache_hits"]),
            "attention_calls": int(row["expected_video_self_attention_calls"]),
            "max_cache_age": int(row["max_cache_age"]),
        }
    _require(set(result) == STAGE2_IDS, "Stage-2 matrix rows are incomplete")
    return result


def _load_comparison(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read comparison JSON {path}: {exc}") from exc
    _require(isinstance(payload, Mapping), f"comparison JSON root must be an object: {path}")
    _require(
        payload.get("record_type") == "ovi_cfg_ablation_v2_media_comparison",
        f"wrong comparison record_type: {path}",
    )
    pairs = payload.get("pairs")
    _require(isinstance(pairs, list) and pairs, f"comparison contains no pairs: {path}")
    _require(payload.get("pair_count") == len(pairs), f"comparison pair_count mismatch: {path}")
    _require(all(isinstance(pair, dict) for pair in pairs), f"invalid comparison pair: {path}")
    return list(pairs)


def _compare_run_pair(
    spec: Sequence[str], *, ffmpeg: str, ffprobe: str, output: Path
) -> None:
    config_id, seed_text, dense_text, candidate_text = spec
    _require(config_id in STAGE2_IDS, f"non-Stage-2 candidate ID: {config_id}")
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise AnalysisError(f"run-pair seed is not an integer: {seed_text!r}") from exc
    dense = Path(dense_text).resolve()
    candidate = Path(candidate_text).resolve()
    _require(dense.is_dir(), f"Dense run is not a directory: {dense}")
    _require(candidate.is_dir(), f"candidate run is not a directory: {candidate}")
    command = [
        sys.executable,
        str(COMPARE_TOOL),
        "compare",
        "--dense-run",
        str(dense),
        "--candidate-run",
        str(candidate),
        "--candidate-id",
        config_id,
        "--comparison-id",
        f"stage2-dense-vs-{config_id}",
        "--split",
        "development",
        "--seed",
        str(seed),
        "--ffmpeg",
        ffmpeg,
        "--ffprobe",
        ffprobe,
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    _require(completed.returncode == 0, f"comparison tool failed for {config_id}/seed{seed}")


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    _require(bool(ordered), "cannot summarize an empty metric")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _metric_summary(values: Sequence[float], *, higher_is_better: bool) -> dict[str, Any]:
    _require(bool(values), "cannot summarize an empty metric")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p10": _percentile(values, 10.0),
        "worst": min(values) if higher_is_better else max(values),
        "higher_is_better": higher_is_better,
    }


def _extract(pair: Mapping[str, Any], source: str) -> dict[str, Any]:
    config_id = pair.get("candidate_id")
    _require(config_id in STAGE2_IDS, f"{source}: missing or invalid candidate_id {config_id!r}")
    metrics = pair.get("metrics")
    _require(isinstance(metrics, Mapping), f"{source}: metrics object is missing")
    video = metrics.get("video")
    audio = metrics.get("audio")
    _require(isinstance(video, Mapping) and isinstance(audio, Mapping), f"{source}: video/audio metrics are missing")
    return {
        "config_id": config_id,
        "prompt_id": str(pair.get("prompt_id") or "unknown"),
        "seed": pair.get("seed"),
        "ssim": _number(video.get("ssim"), f"{source} SSIM"),
        "psnr_db": _number(video.get("psnr_db"), f"{source} PSNR"),
        "temporal_rmse": _number(
            video.get("temporal_frame_difference_rmse"), f"{source} temporal RMSE"
        ),
        "audio_aligned_correlation": _number(
            audio.get("aligned_correlation"), f"{source} aligned audio correlation"
        ),
        "audio_si_sdr_db": _number(audio.get("si_sdr_db"), f"{source} SI-SDR"),
        "audio_aligned_rmse": _number(audio.get("aligned_rmse"), f"{source} audio RMSE"),
        "audio_log_mel_l1": _number(
            audio.get("log_mel_l1_distance"), f"{source} log-mel distance"
        ),
        "audio_abs_lag_ms": abs(
            _number(audio.get("selected_lag_ms"), f"{source} selected audio lag")
        ),
        "audio_abs_activity_coverage_delta": abs(
            _number(
                audio.get("speech_activity_coverage_difference"),
                f"{source} activity coverage difference",
            )
        ),
        "audio_abs_silence_ratio_delta": abs(
            _number(
                audio.get("silence_ratio_difference"),
                f"{source} silence ratio difference",
            )
        ),
    }


def analyze_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    matrix = _load_matrix()
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str, Any]] = set()
    for index, pair in enumerate(records, start=1):
        record = _extract(pair, f"pair {index}")
        key = (record["config_id"], record["prompt_id"], record["seed"])
        _require(key not in seen, f"duplicate candidate/prompt/seed comparison: {key}")
        seen.add(key)
        grouped.setdefault(record["config_id"], []).append(record)
    _require(grouped, "no Stage-2 comparison records supplied")

    metric_directions = {
        "ssim": True,
        "psnr_db": True,
        "temporal_rmse": False,
        "audio_aligned_correlation": True,
        "audio_si_sdr_db": True,
        "audio_aligned_rmse": False,
        "audio_log_mel_l1": False,
        "audio_abs_lag_ms": False,
        "audio_abs_activity_coverage_delta": False,
        "audio_abs_silence_ratio_delta": False,
    }
    candidates: list[dict[str, Any]] = []
    for config_id, rows in grouped.items():
        summaries = {
            metric: _metric_summary(
                [float(row[metric]) for row in rows], higher_is_better=higher
            )
            for metric, higher in metric_directions.items()
        }
        candidates.append(
            {
                "config_id": config_id,
                **matrix[config_id],
                "comparison_units": len(rows),
                "primary_metric": {"name": "ssim", **summaries.pop("ssim")},
                "video_diagnostics": {
                    key: summaries.pop(key) for key in ("psnr_db", "temporal_rmse")
                },
                "audio_diagnostics": summaries,
                "manual_visual_check": {
                    "required": config_id == R15_ID,
                    "status": "pending" if config_id == R15_ID else "not_protocol_mandated",
                    "reason": (
                        "r15 stress cell must be inspected for severe semantic, speech, sync, or reconstruction failure"
                        if config_id == R15_ID
                        else None
                    ),
                },
            }
        )

    # Pareto frontier uses predeclared primary quality (mean SSIM, maximize) and
    # analytical attention calls (minimize). It is descriptive, never a freeze decision.
    for candidate in candidates:
        dominators = []
        quality = candidate["primary_metric"]["mean"]
        calls = candidate["attention_calls"]
        for other in candidates:
            if other is candidate:
                continue
            other_quality = other["primary_metric"]["mean"]
            other_calls = other["attention_calls"]
            if (
                other_quality >= quality
                and other_calls <= calls
                and (other_quality > quality or other_calls < calls)
            ):
                dominators.append(other["config_id"])
        candidate["pareto"] = {
            "on_frontier": not dominators,
            "dominated_by": sorted(dominators),
            "axes": ["mean_ssim_maximize", "attention_calls_minimize"],
        }
    candidates.sort(
        key=lambda item: (
            -float(item["primary_metric"]["mean"]),
            int(item["attention_calls"]),
            str(item["config_id"]),
        )
    )
    for rank, candidate in enumerate(candidates, start=1):
        candidate["primary_ssim_rank"] = rank
    return {
        "schema_version": 1,
        "record_type": "ovi_cfg_ablation_v2_stage2_descriptive_analysis",
        "status": "descriptive_only",
        "primary_metric": "decoded-aligned SSIM versus Dense",
        "ranking_rule": "mean SSIM descending, then attention calls ascending",
        "pareto_rule": "maximize mean SSIM and minimize analytical attention calls",
        "candidates": candidates,
        "freeze_decision": {
            "status": "not_made",
            "automatic_selection_forbidden": True,
            "reason": "candidate freezing requires Stage-2 evidence review and the protocol tier rules",
        },
        "pending_evaluations": {
            "r15_manual_visual_check": {
                "status": "pending" if R15_ID in grouped else "not_present",
                "config_id": R15_ID,
            },
            "asr": {"status": "pending"},
            "syncnet": {"status": "pending"},
            "human_review": {"status": "pending"},
        },
    }


def _csv_rows(payload: Mapping[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    fields = [
        "primary_ssim_rank",
        "config_id",
        "cache_hits",
        "attention_calls",
        "max_cache_age",
        "comparison_units",
        "ssim_mean",
        "ssim_median",
        "ssim_p10",
        "ssim_worst",
        "audio_aligned_correlation_mean",
        "audio_si_sdr_db_mean",
        "audio_aligned_rmse_mean",
        "audio_log_mel_l1_mean",
        "audio_abs_lag_ms_mean",
        "audio_abs_activity_coverage_delta_mean",
        "audio_abs_silence_ratio_delta_mean",
        "pareto_frontier",
        "dominated_by",
        "r15_manual_visual_check_required",
        "r15_manual_visual_check_status",
        "freeze_decision",
    ]
    rows = []
    for candidate in payload["candidates"]:
        audio = candidate["audio_diagnostics"]
        primary = candidate["primary_metric"]
        rows.append(
            {
                "primary_ssim_rank": candidate["primary_ssim_rank"],
                "config_id": candidate["config_id"],
                "cache_hits": candidate["cache_hits"],
                "attention_calls": candidate["attention_calls"],
                "max_cache_age": candidate["max_cache_age"],
                "comparison_units": candidate["comparison_units"],
                "ssim_mean": primary["mean"],
                "ssim_median": primary["median"],
                "ssim_p10": primary["p10"],
                "ssim_worst": primary["worst"],
                "audio_aligned_correlation_mean": audio["audio_aligned_correlation"]["mean"],
                "audio_si_sdr_db_mean": audio["audio_si_sdr_db"]["mean"],
                "audio_aligned_rmse_mean": audio["audio_aligned_rmse"]["mean"],
                "audio_log_mel_l1_mean": audio["audio_log_mel_l1"]["mean"],
                "audio_abs_lag_ms_mean": audio["audio_abs_lag_ms"]["mean"],
                "audio_abs_activity_coverage_delta_mean": audio["audio_abs_activity_coverage_delta"]["mean"],
                "audio_abs_silence_ratio_delta_mean": audio["audio_abs_silence_ratio_delta"]["mean"],
                "pareto_frontier": candidate["pareto"]["on_frontier"],
                "dominated_by": ";".join(candidate["pareto"]["dominated_by"]),
                "r15_manual_visual_check_required": candidate["manual_visual_check"]["required"],
                "r15_manual_visual_check_status": candidate["manual_visual_check"]["status"],
                "freeze_decision": "not_made",
            }
        )
    return fields, rows


def _write_outputs(json_path: Path, csv_path: Path, payload: Mapping[str, Any]) -> None:
    json_path = json_path.resolve()
    csv_path = csv_path.resolve()
    _require(json_path != csv_path, "JSON and CSV outputs must differ")
    _require(not os.path.lexists(json_path), f"refusing to overwrite output: {json_path}")
    _require(not os.path.lexists(csv_path), f"refusing to overwrite output: {csv_path}")
    _require(json_path.parent.is_dir(), f"JSON output parent is missing: {json_path.parent}")
    _require(csv_path.parent.is_dir(), f"CSV output parent is missing: {csv_path.parent}")
    fields, rows = _csv_rows(payload)
    try:
        with json_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        with csv_path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    except BaseException:
        json_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-json", type=Path, action="append", default=[],
        help="existing compare_ovi_cfg_ablation_v2.py JSON; repeatable",
    )
    parser.add_argument(
        "--run-pair", nargs=4, action="append", default=[],
        metavar=("CONFIG_ID", "SEED", "DENSE_RUN", "CANDIDATE_RUN"),
        help="invoke the existing comparison tool for one validated run pair; repeatable",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _require(args.comparison_json or args.run_pair, "provide comparison JSON or at least one run pair")
        records: list[dict[str, Any]] = []
        for path in args.comparison_json:
            records.extend(_load_comparison(path))
        with tempfile.TemporaryDirectory(prefix="ovi-cfg-v2-stage2-analysis-") as directory:
            temporary = Path(directory)
            for index, spec in enumerate(args.run_pair, start=1):
                comparison_path = temporary / f"comparison-{index:03d}.json"
                _compare_run_pair(spec, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe, output=comparison_path)
                records.extend(_load_comparison(comparison_path))
        payload = analyze_records(records)
        _write_outputs(args.output_json, args.output_csv, payload)
    except (AnalysisError, OSError, subprocess.SubprocessError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
