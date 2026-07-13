#!/usr/bin/env python3
"""Fail unless every generated MP4 has valid Ovi video and non-silent audio."""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np


SPARGE_PROVENANCE = {
    "backend": "official_spargeattn",
    "repository": "https://github.com/thu-ml/SpargeAttn.git",
    "clone_url": "ssh://git@ssh.github.com:443/thu-ml/SpargeAttn.git",
    "pinned_commit": "ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a",
    "api": "spas_sage2_attn_meansim_topk_cuda",
    "tensor_layout": "NHD",
    "return_sparsity": False,
}


def run(command):
    return subprocess.check_output(command, stderr=subprocess.STDOUT)


def probe(path):
    payload = run([
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ])
    return json.loads(payload)


def decode_audio(path):
    raw = run([
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "f32le",
        "pipe:1",
    ])
    return np.frombuffer(raw, dtype="<f4")


def decode_video_gray(path):
    raw = run([
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-vf",
        "scale=64:64,format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ])
    return np.frombuffer(raw, dtype=np.uint8)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def validate_sparge_dispatcher(
    dispatcher,
    errors,
    *,
    expected_receipt=None,
    expected_settings=None,
    context="metrics",
):
    """Close the formal provenance loop for an official SpargeAttn record."""

    details = dispatcher.get("backend_details")
    if not isinstance(details, dict):
        errors.append(f"{context}: Sparge dispatcher is missing backend_details")
        return
    for field, expected in SPARGE_PROVENANCE.items():
        if details.get(field) != expected:
            errors.append(
                f"{context}: Sparge backend_details {field}="
                f"{details.get(field)!r} != {expected!r}"
            )
    if details.get("calls") != dispatcher.get("calls_total"):
        errors.append(
            f"{context}: Sparge backend calls={details.get('calls')} != "
            f"dispatcher calls_total={dispatcher.get('calls_total')}"
        )

    receipt = details.get("install_receipt")
    if not isinstance(receipt, dict):
        errors.append(f"{context}: Sparge backend install_receipt is missing")
    elif expected_receipt is not None and receipt != expected_receipt:
        errors.append(
            f"{context}: Sparge backend receipt differs from copied run receipt"
        )

    if expected_settings is not None:
        for field, expected in expected_settings.items():
            if details.get(field) != expected:
                errors.append(
                    f"{context}: Sparge setting {field}={details.get(field)!r} "
                    f"!= environment {expected!r}"
                )


def verify(path, require_metrics=True, expected_video_frames=121):
    info = probe(path)
    artifact_sha256 = sha256(path)
    videos = [stream for stream in info.get("streams", []) if stream.get("codec_type") == "video"]
    audios = [stream for stream in info.get("streams", []) if stream.get("codec_type") == "audio"]
    errors = []
    if len(videos) != 1:
        errors.append(f"expected exactly one video stream, found {len(videos)}")
    if len(audios) != 1:
        errors.append(f"expected exactly one audio stream, found {len(audios)}")

    video = videos[0] if videos else {}
    width = as_int(video.get("width"))
    height = as_int(video.get("height"))
    frames = as_int(video.get("nb_read_frames")) or as_int(video.get("nb_frames"))
    duration = float(info.get("format", {}).get("duration") or 0.0)
    if width is None or height is None or width <= 0 or height <= 0:
        errors.append(f"invalid video dimensions: {width}x{height}")
    elif width % 32 or height % 32:
        errors.append(f"video dimensions are not multiples of 32: {width}x{height}")
    if frames != expected_video_frames:
        errors.append(
            f"expected {expected_video_frames} video frames, found {frames}"
        )
    if not 4.5 <= duration <= 5.5:
        errors.append(f"expected about 5 seconds, found {duration:.6f}")

    samples = decode_audio(path) if audios else np.empty(0, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64))))) if samples.size else 0.0
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    active_ratio = float(np.mean(np.abs(samples) > 1e-3)) if samples.size else 0.0
    dbfs = float(20.0 * math.log10(max(rms, 1e-12)))
    if samples.size < 4 * 16000:
        errors.append(f"decoded audio is too short: {samples.size} samples")
    if not np.isfinite(samples).all():
        errors.append("decoded audio contains NaN or Inf")
    if not math.isfinite(rms) or rms <= 1e-3:
        errors.append(f"audio RMS is silent/invalid: {rms}")
    if not math.isfinite(peak) or peak <= 1e-2:
        errors.append(f"audio peak is silent/invalid: {peak}")
    if active_ratio <= 0.01:
        errors.append(f"audio active-sample ratio is too low: {active_ratio}")

    gray = decode_video_gray(path) if videos else np.empty(0, dtype=np.uint8)
    video_std = float(gray.std()) if gray.size else 0.0
    if gray.size == 0 or video_std <= 2.0:
        errors.append(f"decoded video is blank or nearly constant: std={video_std}")

    metrics_path = path.with_suffix(".metrics.json")
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else None
    if metrics is None and require_metrics:
        errors.append(f"missing metrics sidecar: {metrics_path}")
    elif metrics is not None:
        required_fields = (
            "status",
            "record_type",
            "denoise_seconds",
            "total_generation_seconds",
            "peak_memory_allocated_bytes",
            "peak_memory_reserved_bytes",
            "generated_video_shape",
            "generated_audio_shape",
            "actual_video_frame_height_width",
            "output_sha256",
            "save_video_seconds",
            "artifact_ready_seconds",
            "output_hash_seconds",
            "measurement_index",
            "benchmark_candidate",
            "attention_method",
            "use_cfg_cache",
            "cfg_cache_hits",
            "cfg_cache_refreshes",
            "cfg_negative_forwards",
            "expected_cfg_cache_metrics",
            "use_block_cache",
            "video_self_attention_dispatcher",
            "gpu_process_monitor",
        )
        missing = [field for field in required_fields if field not in metrics]
        if missing:
            errors.append(f"metrics sidecar missing required fields: {missing}")
        if metrics.get("status") != "ok" or metrics.get("record_type") != "measurement":
            errors.append(
                f"invalid metrics status/type: {metrics.get('status')}/{metrics.get('record_type')}"
            )
        if metrics.get("benchmark_valid") is not False:
            errors.append("per-artifact benchmark_valid must remain false until run verification")
        actual_hw = metrics.get("actual_video_frame_height_width")
        generated_shape = metrics.get("generated_video_shape")
        if actual_hw != [height, width]:
            errors.append(f"metrics actual size {actual_hw} != stream size {[height, width]}")
        if generated_shape and generated_shape[1:] != [frames, height, width]:
            errors.append(
                f"metrics generated shape {generated_shape} != stream shape "
                f"[channels,{frames},{height},{width}]"
            )
        if metrics.get("output_sha256") != artifact_sha256:
            errors.append(
                f"output SHA256 mismatch: metrics={metrics.get('output_sha256')} actual={artifact_sha256}"
            )
        if Path(metrics.get("output_path", "")).resolve() != path.resolve():
            errors.append(f"metrics output_path does not match artifact: {metrics.get('output_path')}")

        expected_cfg = metrics.get("expected_cfg_cache_metrics")
        if isinstance(expected_cfg, dict):
            for field in (
                "cfg_cache_hits",
                "cfg_cache_refreshes",
                "cfg_negative_forwards",
            ):
                if metrics.get(field) != expected_cfg.get(field):
                    errors.append(
                        f"{field}={metrics.get(field)} != expected "
                        f"{expected_cfg.get(field)}"
                    )
        elif expected_cfg is not None:
            errors.append("expected_cfg_cache_metrics must be a JSON object")

        block_cache_enabled = bool(metrics.get("use_block_cache"))
        block_metric_fields = (
            "block_cache_start_block",
            "block_cache_end_block",
            "block_cache_window_inclusive",
            "block_cache_policy",
            "block_cache_cosine_threshold",
            "block_cache_max_consecutive_reuses",
            "block_cache_hits",
            "block_cache_refreshes",
            "block_cache_saved_video_self_attention_calls",
            "block_cache_branch_metrics",
        )
        block_metrics_present = any(
            field in metrics for field in block_metric_fields
        )
        block_hits = 0
        block_refreshes = 0
        block_saved_calls = 0
        if block_cache_enabled or block_metrics_present:
            missing_block_fields = [
                field for field in block_metric_fields if field not in metrics
            ]
            if missing_block_fields:
                errors.append(
                    "block-cache metrics missing required fields: "
                    f"{missing_block_fields}"
                )
            block_hits = as_int(metrics.get("block_cache_hits"))
            block_refreshes = as_int(metrics.get("block_cache_refreshes"))
            block_saved_calls = as_int(
                metrics.get("block_cache_saved_video_self_attention_calls")
            )
            block_branches = metrics.get("block_cache_branch_metrics")
            if metrics.get("block_cache_policy") not in ("fixed", "cosine"):
                errors.append("invalid block_cache_policy")
            try:
                block_cosine_threshold = float(
                    metrics.get("block_cache_cosine_threshold")
                )
            except (TypeError, ValueError):
                block_cosine_threshold = float("nan")
            if (
                not math.isfinite(block_cosine_threshold)
                or not 0.0 <= block_cosine_threshold <= 1.0
            ):
                errors.append("invalid block_cache_cosine_threshold")
            if metrics.get("block_cache_max_consecutive_reuses") != 1:
                errors.append(
                    "block cache must cap consecutive reuses at exactly 1"
                )
            block_start = as_int(metrics.get("block_cache_start_block"))
            block_end = as_int(metrics.get("block_cache_end_block"))
            if (
                block_start is None
                or block_end is None
                or not 0 <= block_start <= block_end
            ):
                errors.append(
                    f"invalid block-cache window: {block_start}..{block_end}"
                )
            if metrics.get("block_cache_window_inclusive") is not True:
                errors.append("block-cache window must be recorded as inclusive")
            if not block_cache_enabled:
                if (block_hits, block_refreshes, block_saved_calls) != (0, 0, 0):
                    errors.append(
                        "disabled block cache recorded non-zero activity"
                    )
                if block_branches not in ({}, None):
                    errors.append(
                        "disabled block cache recorded branch payload metrics"
                    )
            elif isinstance(block_branches, dict):
                expected_branches = {"conditional", "unconditional"}
                if set(block_branches) != expected_branches:
                    errors.append(
                        f"block-cache branches {sorted(block_branches)} != "
                        f"{sorted(expected_branches)}"
                    )
                branch_hits = sum(
                    as_int(item.get("hits")) or 0
                    for item in block_branches.values()
                    if isinstance(item, dict)
                )
                branch_refreshes = sum(
                    as_int(item.get("refreshes")) or 0
                    for item in block_branches.values()
                    if isinstance(item, dict)
                )
                branch_saved_calls = sum(
                    as_int(item.get("saved_video_self_attention_calls")) or 0
                    for item in block_branches.values()
                    if isinstance(item, dict)
                )
                if branch_hits != block_hits:
                    errors.append(
                        f"block_cache_hits={block_hits} != branch sum "
                        f"{branch_hits}"
                    )
                if branch_refreshes != block_refreshes:
                    errors.append(
                        "block_cache_refreshes="
                        f"{block_refreshes} != branch sum {branch_refreshes}"
                    )
                if branch_saved_calls != block_saved_calls:
                    errors.append(
                        "block_cache_saved_video_self_attention_calls="
                        f"{block_saved_calls} != branch sum {branch_saved_calls}"
                    )
            else:
                errors.append("enabled block cache requires branch metrics")

        dispatcher = metrics.get("video_self_attention_dispatcher")
        if isinstance(dispatcher, dict):
            configured_method = metrics.get("attention_method")
            if dispatcher.get("configured_method") != configured_method:
                errors.append(
                    "dispatcher configured_method disagrees with attention_method"
                )
            if dispatcher.get("active_method") != configured_method:
                errors.append("dispatcher active_method disagrees with attention_method")
            if dispatcher.get("fallback_allowed") is not False:
                errors.append("dispatcher must not allow fallback")
            if dispatcher.get("fallback_used") is not False:
                errors.append("dispatcher unexpectedly used fallback")
            if dispatcher.get("fallback_count") != 0:
                errors.append("dispatcher fallback_count must be zero")
            if dispatcher.get("calls_total") != dispatcher.get("expected_calls"):
                errors.append(
                    f"dispatcher calls_total={dispatcher.get('calls_total')} != "
                    f"expected_calls={dispatcher.get('expected_calls')}"
                )
            expected_without_block_cache = dispatcher.get(
                "expected_calls_without_block_cache"
            )
            if expected_without_block_cache is not None:
                adjusted_expected = (
                    as_int(expected_without_block_cache) or 0
                ) - (block_saved_calls or 0)
                if dispatcher.get("expected_calls") != adjusted_expected:
                    errors.append(
                        "dispatcher expected_calls does not subtract the "
                        "recorded block-cache savings"
                    )
            if dispatcher.get("calls_match_expected") is not True:
                errors.append("dispatcher calls_match_expected must be true")
            errors_by_method = dispatcher.get("errors_by_method", {})
            if any(value for value in errors_by_method.values()):
                errors.append(
                    f"dispatcher recorded backend errors: {errors_by_method}"
                )
            if configured_method == "sparge":
                validate_sparge_dispatcher(dispatcher, errors)
        elif dispatcher is not None:
            errors.append("video_self_attention_dispatcher must be a JSON object")

        gpu_monitor = metrics.get("gpu_process_monitor")
        if gpu_monitor is not None and not isinstance(gpu_monitor, dict):
            errors.append("gpu_process_monitor must be a JSON object")

    return {
        "path": str(path.resolve()),
        "sha256": artifact_sha256,
        "status": "failed" if errors else "ok",
        "errors": errors,
        "video": {
            "codec": video.get("codec_name"),
            "width": width,
            "height": height,
            "frames": frames,
            "duration_seconds": duration,
            "decoded_pixel_std": video_std,
        },
        "audio": {
            "codec": audios[0].get("codec_name") if audios else None,
            "decoded_samples_16khz_mono": int(samples.size),
            "rms": rms,
            "peak": peak,
            "dbfs": dbfs,
            "active_sample_ratio_abs_gt_1e-3": active_ratio,
        },
    }


def verify_run_protocol(run_dir, reports):
    errors = []
    environment_path = run_dir / "environment.json"
    environment = json.loads(environment_path.read_text()) if environment_path.is_file() else {}
    attention_method = environment.get("attention_method")
    required_files = [
        "environment.json",
        "run_config.yaml",
        "preflight.json",
        "environment.freeze.txt",
        "checkpoint_manifest.json",
    ]
    if attention_method == "sparge":
        required_files.append("spargeattn-install.json")
    for filename in required_files:
        if not (run_dir / filename).is_file():
            errors.append(f"missing run evidence file: {filename}")

    candidate = bool(environment.get("benchmark_eligible"))
    expected_measurements = int(environment.get("expected_measurement_records", -1))
    expected_warmups = int(environment.get("expected_warmup_records", -1))
    measurement_runs = int(environment.get("measurement_runs", -1))
    per_repeat = int(environment.get("prompt_count", 0)) * int(
        environment.get("each_example_n_times", 0)
    )

    timings = read_jsonl(run_dir / "timings.jsonl")
    warmups = read_jsonl(run_dir / "warmup_timings.jsonl")
    if len(reports) != expected_measurements:
        errors.append(f"MP4 count {len(reports)} != expected {expected_measurements}")
    if len(timings) != expected_measurements:
        errors.append(f"timings count {len(timings)} != expected {expected_measurements}")
    if len(warmups) != expected_warmups:
        errors.append(f"warmup count {len(warmups)} != expected {expected_warmups}")

    expected_indices = {
        index for index in range(max(measurement_runs, 0))
        for _ in range(max(per_repeat, 0))
    }
    actual_indices = {item.get("measurement_index") for item in timings}
    if actual_indices != expected_indices:
        errors.append(
            f"measurement indices {sorted(str(x) for x in actual_indices)} "
            f"!= expected {sorted(expected_indices)}"
        )
    for index in range(max(measurement_runs, 0)):
        count = sum(item.get("measurement_index") == index for item in timings)
        if count != per_repeat:
            errors.append(f"measurement index {index} has {count} records, expected {per_repeat}")
    for item in timings:
        if item.get("status") != "ok" or item.get("record_type") != "measurement":
            errors.append("timings.jsonl contains a non-ok/non-measurement record")
            break
    for item in warmups:
        if item.get("status") != "ok" or item.get("record_type") != "warmup":
            errors.append("warmup_timings.jsonl contains an invalid warm-up record")
            break

    if candidate:
        for item in [*warmups, *timings]:
            monitor = item.get("gpu_process_monitor")
            if not isinstance(monitor, dict) or monitor.get(
                "valid_for_benchmark"
            ) is not True:
                errors.append(
                    "benchmark run has missing, unavailable, or contaminated "
                    "GPU process monitoring evidence"
                )
                break

    artifact_hashes = {report["sha256"] for report in reports}
    timing_hashes = {item.get("output_sha256") for item in timings}
    if artifact_hashes != timing_hashes:
        errors.append("timings.jsonl output hashes do not match the verified artifacts")

    preflight = {}
    preflight_path = run_dir / "preflight.json"
    if preflight_path.is_file():
        preflight = json.loads(preflight_path.read_text())
        if preflight.get("errors"):
            errors.append(f"preflight contains errors: {preflight['errors']}")

    if attention_method == "sparge":
        expected_protocol = {
            "sparge_topk": 0.5,
            "sparge_pvthreshd": 50.0,
            "sparge_smooth_k": True,
            "use_cfg_cache": False,
            "use_block_cache": False,
            "sp_size": 1,
        }
        if environment.get("run_kind") not in {
            "sparge_baseline",
            "sparge_diagnostic_smoke",
        }:
            errors.append(
                "Sparge run_kind is not an audited pure-Sparge protocol"
            )
        for field, expected in expected_protocol.items():
            if environment.get(field) != expected:
                errors.append(
                    f"Sparge protocol {field}={environment.get(field)!r} "
                    f"!= fixed value {expected!r}"
                )
        receipt_path = run_dir / "spargeattn-install.json"
        try:
            copied_receipt = json.loads(receipt_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            copied_receipt = None
            errors.append(f"invalid copied SpargeAttn receipt: {exc}")

        if not environment.get("spas_sage_attn"):
            errors.append("environment is missing spas_sage_attn package version")
        preflight_sparge = preflight.get("spargeattn")
        if not isinstance(preflight_sparge, dict):
            errors.append("Sparge run preflight is missing spargeattn evidence")
        else:
            if preflight_sparge.get("pinned_commit") != SPARGE_PROVENANCE["pinned_commit"]:
                errors.append("preflight SpargeAttn commit differs from formal pin")
            if preflight_sparge.get("api") != SPARGE_PROVENANCE["api"]:
                errors.append("preflight SpargeAttn API differs from formal pin")
            if preflight_sparge.get("installed_files_verified") is not True:
                errors.append("preflight did not verify installed SpargeAttn files")
            if preflight_sparge.get("install_receipt_contents") != copied_receipt:
                errors.append("preflight SpargeAttn receipt differs from copied receipt")
        receipt_microtest = (
            copied_receipt.get("microtest")
            if isinstance(copied_receipt, dict)
            else None
        )
        if not isinstance(receipt_microtest, dict) or receipt_microtest.get(
            "status"
        ) != "ok":
            errors.append("Sparge install receipt lacks a successful CUDA microtest")
        preflight_microtest = preflight.get("spargeattn_microtest")
        if not isinstance(preflight_microtest, dict) or preflight_microtest.get(
            "status"
        ) != "ok":
            errors.append("Sparge run preflight lacks a successful CUDA microtest")

        expected_settings = {
            "topk": environment.get("sparge_topk"),
            "pvthreshd": environment.get("sparge_pvthreshd"),
            "smooth_k": environment.get("sparge_smooth_k"),
        }
        for record_type, records in (("measurement", timings), ("warmup", warmups)):
            for index, item in enumerate(records):
                dispatcher = item.get("video_self_attention_dispatcher")
                if not isinstance(dispatcher, dict):
                    errors.append(
                        f"{record_type}[{index}] is missing video dispatcher evidence"
                    )
                    continue
                validate_sparge_dispatcher(
                    dispatcher,
                    errors,
                    expected_receipt=copied_receipt,
                    expected_settings=expected_settings,
                    context=f"{record_type}[{index}]",
                )

    evidence_hashes = environment.get("evidence_file_sha256", {})
    for filename, expected_hash in evidence_hashes.items():
        path = run_dir / filename
        actual_hash = sha256(path) if path.is_file() else None
        if not expected_hash or expected_hash != actual_hash:
            errors.append(
                f"evidence hash mismatch for {filename}: expected={expected_hash} actual={actual_hash}"
            )
    run_config_path = run_dir / "run_config.yaml"
    if run_config_path.is_file() and environment.get("run_config_sha256") != sha256(run_config_path):
        errors.append("run_config.yaml SHA256 does not match environment.json")

    if any(item.get("benchmark_candidate") != candidate for item in timings):
        errors.append("per-measurement benchmark_candidate disagrees with environment.json")
    benchmark_valid = bool(
        candidate
        and not errors
        and not environment.get("debug_forward")
        and not environment.get("git_dirty")
        and expected_warmups >= 1
        and measurement_runs >= 3
        and all(not report["errors"] for report in reports)
    )
    return {
        "status": "failed" if errors else "ok",
        "errors": errors,
        "expected_warmup_records": expected_warmups,
        "observed_warmup_records": len(warmups),
        "expected_measurement_records": expected_measurements,
        "observed_measurement_records": len(timings),
        "benchmark_candidate": candidate,
        "benchmark_valid": benchmark_valid,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="MP4 file or run directory")
    parser.add_argument(
        "--media-only",
        action="store_true",
        help="validate streams/content without FastA2V metrics or run protocol",
    )
    parser.add_argument(
        "--expected-video-frames",
        type=int,
        default=121,
        help="exact decoded frame count required (default: 121)",
    )
    args = parser.parse_args()

    if args.expected_video_frames < 1:
        parser.error("--expected-video-frames must be positive")

    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise SystemExit(f"required executable not found: {executable}")

    paths = [args.path] if args.path.is_file() else sorted(args.path.glob("*.mp4"))
    if not paths:
        raise SystemExit(f"no MP4 artifacts found under {args.path}")
    reports = [
        verify(
            path,
            require_metrics=not args.media_only,
            expected_video_frames=args.expected_video_frames,
        )
        for path in paths
    ]
    run_dir = args.path if args.path.is_dir() else args.path.parent
    protocol = (
        verify_run_protocol(run_dir, reports)
        if args.path.is_dir() and not args.media_only
        else None
    )
    failed = any(item["errors"] for item in reports) or (
        protocol is not None and protocol["errors"]
    )
    summary = {
        "status": "failed" if failed else "ok",
        "artifact_count": len(reports),
        "artifacts": reports,
        "protocol": protocol,
        "benchmark_valid": bool(protocol and protocol["benchmark_valid"]),
    }
    output_path = args.path.with_suffix(".verification.json") if args.path.is_file() else args.path / "verification.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
