#!/usr/bin/env python3
"""Summarize held-out Stage-3 Dense-reference comparisons by equal-compute tier.

Inputs are the twelve JSON files produced by ``compare_ovi_cfg_ablation_v2.py``:
three seeds for each of old_12, new_12, old_14, and new_14, always with Dense
as the reconstruction reference.  The tool joins new and old scores per
prompt/seed, writes long-form rows, runs the existing prompt-cluster bootstrap
separately for the 12-hit and 14-hit tiers, and checks workload/time
equivalence from the candidate MP4 metric sidecars.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ovi.cfg_ablation_v2_protocol import PROTOCOL_ID, STAGE_SEEDS  # noqa: E402


DEFAULT_PROMPTS = REPO_ROOT / "prompts/ovi_cfg_cache_heldout_prompts.csv"
DEFAULT_PROMPT_MANIFEST = (
    REPO_ROOT / "prompts/ovi_cfg_cache_heldout_prompt_manifest.csv"
)
PROMPT_FILE_RE = re.compile(r"^p(?P<index>[0-9]{3})_")
LABELS = ("old_12", "new_12", "old_14", "new_14")
TIERS = {
    "12_hit": ("old_12", "new_12"),
    "14_hit": ("old_14", "new_14"),
}
MetricGetter = Callable[[Mapping[str, Any]], Any]


class SummaryError(RuntimeError):
    """Raised when held-out comparison evidence is incomplete or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SummaryError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SummaryError(f"cannot read {label} {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} must be a JSON object: {path}")
    return payload


def _finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _nested(record: Mapping[str, Any], *keys: str) -> Any:
    value: Any = record
    for key in keys:
        _require(isinstance(value, Mapping) and key in value, f"missing comparison metric: {'.'.join(keys)}")
        value = value[key]
    return value


def _abs_nested(record: Mapping[str, Any], *keys: str) -> float:
    return abs(_finite(_nested(record, *keys), ".".join(keys)))


METRICS: tuple[tuple[str, bool, MetricGetter], ...] = (
    ("video_psnr_db", True, lambda row: _nested(row, "metrics", "video", "psnr_db")),
    ("video_ssim", True, lambda row: _nested(row, "metrics", "video", "ssim")),
    (
        "temporal_frame_difference_rmse",
        False,
        lambda row: _nested(row, "metrics", "video", "temporal_frame_difference_rmse"),
    ),
    (
        "audio_aligned_correlation",
        True,
        lambda row: _nested(row, "metrics", "audio", "aligned_correlation"),
    ),
    ("audio_si_sdr_db", True, lambda row: _nested(row, "metrics", "audio", "si_sdr_db")),
    (
        "audio_aligned_rmse",
        False,
        lambda row: _nested(row, "metrics", "audio", "aligned_rmse"),
    ),
    (
        "audio_log_mel_l1_distance",
        False,
        lambda row: _nested(row, "metrics", "audio", "log_mel_l1_distance"),
    ),
    (
        "audio_activity_coverage_abs_error",
        False,
        lambda row: _abs_nested(
            row, "metrics", "audio", "speech_activity_coverage_difference"
        ),
    ),
    (
        "audio_silence_ratio_abs_error",
        False,
        lambda row: _abs_nested(row, "metrics", "audio", "silence_ratio_difference"),
    ),
)


def _load_compare_module() -> ModuleType:
    path = REPO_ROOT / "scripts/compare_ovi_cfg_ablation_v2.py"
    spec = importlib.util.spec_from_file_location("ovi_cfg_v2_compare_for_stage3", path)
    _require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prompt_contract(
    prompt_csv: Path, prompt_manifest: Path
) -> tuple[list[str], list[dict[str, str]]]:
    with prompt_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require(tuple(reader.fieldnames or ()) == ("text_prompt",), "invalid held-out prompt CSV")
        prompts = [(row.get("text_prompt") or "").strip() for row in reader]
    with prompt_manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require(
            tuple(reader.fieldnames or ())
            == ("prompt_id", "category", "primary_stress"),
            "invalid held-out prompt manifest",
        )
        manifest = [dict(row) for row in reader]
    _require(len(prompts) == len(manifest) == 8, "held-out prompt contract must contain 8 rows")
    _require(
        [row["prompt_id"] for row in manifest] == [f"H{index:02d}" for index in range(1, 9)],
        "held-out prompt IDs must be H01..H08",
    )
    return prompts, manifest


def _receipt_contract(
    receipt: Mapping[str, Any], prompt_csv: Path, prompt_manifest: Path
) -> tuple[dict[str, str], dict[tuple[int, str], str]]:
    _require(receipt.get("record_type") == "ovi_cfg_ablation_v2_frozen_stage3_candidates", "invalid frozen receipt type")
    _require(receipt.get("protocol_id") == PROTOCOL_ID and receipt.get("status") == "frozen", "invalid frozen receipt status")
    inputs = receipt.get("inputs")
    _require(isinstance(inputs, Mapping), "frozen receipt inputs are missing")
    _require(
        (inputs.get("heldout_prompt_csv") or {}).get("sha256") == _sha256(prompt_csv),
        "held-out prompt CSV differs from frozen receipt",
    )
    _require(
        (inputs.get("heldout_prompt_manifest") or {}).get("sha256")
        == _sha256(prompt_manifest),
        "held-out prompt manifest differs from frozen receipt",
    )
    configurations = receipt.get("configurations")
    _require(isinstance(configurations, Mapping), "frozen configuration map is missing")
    label_to_config = {
        label: str((configurations.get(label) or {}).get("config_id", ""))
        for label in ("dense", *LABELS)
    }
    _require(all(label_to_config.values()), "frozen configuration IDs are incomplete")
    _require(len(set(label_to_config.values())) == 5, "frozen configuration IDs are not unique")
    planned = receipt.get("planned_runs")
    _require(isinstance(planned, list) and len(planned) == 15, "frozen run plan must contain 15 cells")
    run_tags: dict[tuple[int, str], str] = {}
    for item in planned:
        _require(isinstance(item, Mapping), "frozen planned run is malformed")
        key = (int(item.get("seed", -1)), str(item.get("label", "")))
        _require(key not in run_tags, f"duplicate frozen run key: {key}")
        run_tags[key] = str(item.get("run_tag", ""))
    expected = {
        (seed, label) for seed in STAGE_SEEDS["3"] for label in ("dense", *LABELS)
    }
    _require(set(run_tags) == expected, "frozen run plan does not cover 3 seeds x 5 labels")
    return label_to_config, run_tags


def _prompt_index(path: Path) -> int:
    match = PROMPT_FILE_RE.match(path.name)
    _require(match is not None, f"cannot derive prompt index from {path.name}")
    index = int(match.group("index"))
    _require(0 <= index < 8, f"held-out prompt index is out of range: {path}")
    return index


def _sidecar(path: Path, expected_seed: int, expected_prompt: int) -> dict[str, Any]:
    sidecar = path.with_suffix(".metrics.json")
    payload = _load_json(sidecar, "measurement sidecar")
    _require(payload.get("status") == "ok" and payload.get("record_type") == "measurement", f"invalid sidecar: {sidecar}")
    _require(payload.get("seed") == expected_seed, f"sidecar seed mismatch: {sidecar}")
    _require(payload.get("prompt_index") == expected_prompt, f"sidecar prompt mismatch: {sidecar}")
    return payload


def load_reconstruction_scores(
    comparison_paths: Sequence[Path],
    *,
    receipt: Mapping[str, Any],
    prompt_csv: Path = DEFAULT_PROMPTS,
    prompt_manifest: Path = DEFAULT_PROMPT_MANIFEST,
) -> tuple[
    dict[tuple[str, int, str, str], float],
    dict[tuple[str, int, str], dict[str, Any]],
    list[dict[str, Any]],
    dict[str, str],
    list[dict[str, str]],
]:
    _require(len(comparison_paths) == 12, "Stage 3 requires exactly 12 Dense-reference comparison JSONs")
    prompts, manifest = _prompt_contract(prompt_csv, prompt_manifest)
    label_to_config, run_tags = _receipt_contract(receipt, prompt_csv, prompt_manifest)
    config_to_label = {config_id: label for label, config_id in label_to_config.items() if label != "dense"}
    scores: dict[tuple[str, int, str, str], float] = {}
    sidecars: dict[tuple[str, int, str], dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    file_keys: set[tuple[str, int]] = set()

    for path in comparison_paths:
        payload = _load_json(path, "media comparison")
        _require(payload.get("record_type") == "ovi_cfg_ablation_v2_media_comparison", f"invalid comparison type: {path}")
        pairs = payload.get("pairs")
        _require(isinstance(pairs, list) and payload.get("pair_count") == len(pairs) == 8, f"comparison must contain 8 pairs: {path}")
        labels = {config_to_label.get(str(pair.get("candidate_id", ""))) for pair in pairs if isinstance(pair, Mapping)}
        seeds = {pair.get("seed") for pair in pairs if isinstance(pair, Mapping)}
        _require(len(labels) == 1 and None not in labels, f"comparison candidate ID is not frozen: {path}")
        _require(len(seeds) == 1 and next(iter(seeds)) in STAGE_SEEDS["3"], f"comparison seed is invalid: {path}")
        label = str(next(iter(labels)))
        seed = int(next(iter(seeds)))
        _require((label, seed) not in file_keys, f"duplicate comparison file for {label}/seed{seed}")
        file_keys.add((label, seed))

        observed_prompts: set[int] = set()
        for pair in pairs:
            _require(isinstance(pair, Mapping), f"comparison pair is malformed: {path}")
            _require(pair.get("split") == "heldout", f"comparison is not held-out: {path}")
            _require(pair.get("candidate_id") == label_to_config[label], f"candidate ID changes within {path}")
            _require(pair.get("comparison_id") == f"{label}_vs_dense", f"comparison_id must be {label}_vs_dense in {path}")
            dense_path = Path(str(_nested(pair, "dense", "path"))).resolve()
            candidate_path = Path(str(_nested(pair, "candidate", "path"))).resolve()
            prompt_index = _prompt_index(candidate_path)
            _require(_prompt_index(dense_path) == prompt_index, f"Dense/candidate prompt mismatch: {path}")
            _require(candidate_path.parent.name == run_tags[(seed, label)], f"candidate run tag differs from frozen plan: {candidate_path}")
            _require(dense_path.parent.name == run_tags[(seed, "dense")], f"Dense run tag differs from frozen plan: {dense_path}")
            _require(prompt_index not in observed_prompts, f"duplicate prompt index in {path}: {prompt_index}")
            observed_prompts.add(prompt_index)
            _require(pair.get("prompt") == prompts[prompt_index], f"prompt text differs from authoritative held-out row: {path}")
            prompt_id = manifest[prompt_index]["prompt_id"]
            for metric, _higher, getter in METRICS:
                key = (label, seed, prompt_id, metric)
                _require(key not in scores, f"duplicate reconstruction score: {key}")
                scores[key] = _finite(getter(pair), f"{path} {prompt_id} {metric}")
            sidecars[(label, seed, prompt_id)] = _sidecar(
                candidate_path, seed, prompt_index
            )
        _require(observed_prompts == set(range(8)), f"comparison does not cover p000..p007: {path}")
        sources.append({"label": label, "seed": seed, **_binding(path)})

    expected_files = {(label, seed) for label in LABELS for seed in STAGE_SEEDS["3"]}
    _require(file_keys == expected_files, "comparison inputs do not cover 3 seeds x old/new x 12/14")
    return scores, sidecars, sorted(sources, key=lambda item: (item["label"], item["seed"])), label_to_config, manifest


def build_long_rows(
    scores: Mapping[tuple[str, int, str, str], float],
    *,
    label_to_config: Mapping[str, str],
    manifest: Sequence[Mapping[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    categories = {row["prompt_id"]: row["category"] for row in manifest}
    output: dict[str, list[dict[str, Any]]] = {}
    directions = {name: higher for name, higher, _getter in METRICS}
    for tier, (old_label, new_label) in TIERS.items():
        rows: list[dict[str, Any]] = []
        for seed in STAGE_SEEDS["3"]:
            for prompt_id in categories:
                for metric in directions:
                    candidate = scores[(new_label, seed, prompt_id, metric)]
                    comparator = scores[(old_label, seed, prompt_id, metric)]
                    rows.append(
                        {
                            "split": "heldout",
                            "tier": tier,
                            "prompt_id": prompt_id,
                            "category": categories[prompt_id],
                            "seed": seed,
                            "candidate_id": label_to_config[new_label],
                            "comparator_id": label_to_config[old_label],
                            "comparison_id": f"{new_label}_vs_{old_label}",
                            "metric": metric,
                            "higher_is_better": directions[metric],
                            "candidate_value": candidate,
                            "comparator_value": comparator,
                            "difference": candidate - comparator,
                        }
                    )
        output[tier] = rows
    return output


def workload_equivalence(
    sidecars: Mapping[tuple[str, int, str], Mapping[str, Any]],
    *,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    configurations = receipt["configurations"]
    output: dict[str, Any] = {}
    for tier, (old_label, new_label) in TIERS.items():
        old_config = configurations[old_label]
        new_config = configurations[new_label]
        expected_hits = old_config["cache_hits"]
        expected_calls = old_config["expected_video_self_attention_calls"]
        exact_compute = (
            new_config["cache_hits"] == expected_hits
            and new_config["expected_video_self_attention_calls"] == expected_calls
        )
        paired: list[dict[str, Any]] = []
        counters_passed = exact_compute
        for seed in STAGE_SEEDS["3"]:
            for prompt_index in range(1, 9):
                prompt_id = f"H{prompt_index:02d}"
                old = sidecars[(old_label, seed, prompt_id)]
                new = sidecars[(new_label, seed, prompt_id)]
                for record in (old, new):
                    dispatcher = record.get("video_self_attention_dispatcher")
                    counters_passed = counters_passed and (
                        record.get("cfg_cache_hits") == expected_hits
                        and isinstance(dispatcher, Mapping)
                        and dispatcher.get("calls_total") == expected_calls
                        and dispatcher.get("fallback_used") is False
                        and dispatcher.get("fallback_count") == 0
                    )
                old_seconds = _finite(old.get("denoise_seconds"), f"{old_label}/{seed}/{prompt_id} denoise")
                new_seconds = _finite(new.get("denoise_seconds"), f"{new_label}/{seed}/{prompt_id} denoise")
                _require(old_seconds > 0.0, "old-policy denoise time must be positive")
                paired.append(
                    {
                        "prompt_id": prompt_id,
                        "seed": seed,
                        "old_seconds": old_seconds,
                        "new_seconds": new_seconds,
                        "new_minus_old_percent": 100.0 * (new_seconds - old_seconds) / old_seconds,
                    }
                )
        median_percent = statistics.median(
            row["new_minus_old_percent"] for row in paired
        )
        output[tier] = {
            "cache_hits": expected_hits,
            "video_self_attention_calls": expected_calls,
            "sample_count": len(paired),
            "exact_workload_counters_passed": bool(counters_passed),
            "paired_median_new_minus_old_denoise_percent": median_percent,
            "paired_median_within_plus_minus_1_percent": abs(median_percent) <= 1.0,
            "pairs": paired,
        }
    return output


def summarize(
    *,
    frozen_receipt: Path,
    comparison_paths: Sequence[Path],
    prompt_csv: Path = DEFAULT_PROMPTS,
    prompt_manifest: Path = DEFAULT_PROMPT_MANIFEST,
    bootstrap_replicates: int = 5000,
    bootstrap_seed: int = 20260717,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], dict[str, Any]]:
    receipt = _load_json(frozen_receipt, "frozen receipt")
    scores, sidecars, sources, label_to_config, manifest = load_reconstruction_scores(
        comparison_paths,
        receipt=receipt,
        prompt_csv=prompt_csv,
        prompt_manifest=prompt_manifest,
    )
    long_rows = build_long_rows(scores, label_to_config=label_to_config, manifest=manifest)
    compare = _load_compare_module()
    analyses: dict[str, dict[str, Any]] = {}
    for tier, rows in long_rows.items():
        canonical_rows = []
        for row in rows:
            raw = float(row["difference"])
            canonical_rows.append(
                {
                    "split": "heldout",
                    "prompt": row["prompt_id"],
                    "category": row["category"],
                    "seed": row["seed"],
                    "metric": row["metric"],
                    "higher_is_better": row["higher_is_better"],
                    "candidate_value": row["candidate_value"],
                    "comparator_value": row["comparator_value"],
                    "raw_difference": raw,
                    "oriented_difference": raw if row["higher_is_better"] else -raw,
                }
            )
        analyses[tier] = compare.analyze_rows(
            canonical_rows,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        )

    equivalence = workload_equivalence(sidecars, receipt=receipt)
    machine_gates: dict[str, Any] = {}
    for tier in TIERS:
        primary = analyses[tier]["splits"]["heldout"]["metrics"]["video_ssim"]["overall"]
        machine_gates[tier] = {
            "primary_metric": "video_ssim",
            "positive_median_improvement": primary["median"] > 0.0,
            "pairwise_win_rate_at_least_70_percent": primary["win_rate"] >= 0.70,
            "primary_summary": primary,
            "workload_equivalence": equivalence[tier],
            "all_machine_gates_passed": (
                primary["median"] > 0.0
                and primary["win_rate"] >= 0.70
                and equivalence[tier]["exact_workload_counters_passed"]
                and equivalence[tier]["paired_median_within_plus_minus_1_percent"]
            ),
        }
    all_machine = all(item["all_machine_gates_passed"] for item in machine_gates.values())
    summary = {
        "schema_version": 1,
        "record_type": "ovi_cfg_ablation_v2_stage3_equivalence_summary",
        "protocol_id": PROTOCOL_ID,
        "status": (
            "machine_gates_passed_external_evaluations_pending"
            if all_machine
            else "machine_gate_failed_external_evaluations_pending"
        ),
        "frozen_receipt": _binding(frozen_receipt),
        "comparison_sources": sources,
        "split": "heldout",
        "cross_split_aggregation": "forbidden",
        "bootstrap": {
            "unit": "prompt",
            "seeds_retained_within_prompt": True,
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
        },
        "tiers": machine_gates,
        "pending_evaluations": {
            "asr": {"status": "pending"},
            "syncnet": {"status": "pending"},
            "human_blind_review": {
                "status": "pending",
                "minimum_independent_ratings_per_pair": 3,
            },
        },
        "final_acceptance": {
            "status": "pending_external_evaluations",
            "note": "machine reconstruction/equivalence gates cannot replace ASR, SyncNet, or blind human review",
        },
    }
    return long_rows, analyses, summary


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = (
        "split",
        "tier",
        "prompt_id",
        "category",
        "seed",
        "candidate_id",
        "comparator_id",
        "comparison_id",
        "metric",
        "higher_is_better",
        "candidate_value",
        "comparator_value",
        "difference",
    )
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparisons", nargs="+", type=Path)
    parser.add_argument("--frozen-receipt", type=Path, required=True)
    parser.add_argument("--prompt-csv", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--prompt-manifest", type=Path, default=DEFAULT_PROMPT_MANIFEST)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260717)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        _require(args.bootstrap_replicates > 0, "bootstrap replicates must be positive")
        _require(not args.output_dir.exists(), f"refusing to reuse output directory: {args.output_dir}")
        long_rows, analyses, summary = summarize(
            frozen_receipt=args.frozen_receipt,
            comparison_paths=args.comparisons,
            prompt_csv=args.prompt_csv,
            prompt_manifest=args.prompt_manifest,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        args.output_dir.mkdir(parents=True, exist_ok=False)
        combined = [row for tier in TIERS for row in long_rows[tier]]
        _write_csv(args.output_dir / "heldout_long.csv", combined)
        for tier in TIERS:
            _write_csv(args.output_dir / f"heldout_{tier}_long.csv", long_rows[tier])
            _write_json(
                args.output_dir / f"heldout_{tier}_clustered_analysis.json",
                analyses[tier],
            )
        _write_json(args.output_dir / "stage3_equivalence_summary.json", summary)
        print(
            json.dumps(
                {
                    "status": summary["status"],
                    "long_rows": len(combined),
                    "output_dir": str(args.output_dir.resolve()),
                    "report_heldout_analysis_args": [
                        str((args.output_dir / f"heldout_{tier}_clustered_analysis.json").resolve())
                        for tier in TIERS
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    except (SummaryError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
