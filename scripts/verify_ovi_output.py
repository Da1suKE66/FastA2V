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


def verify(path, require_metrics=True):
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
    if frames != 121:
        errors.append(f"expected 121 video frames, found {frames}")
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
    required_files = (
        "environment.json",
        "run_config.yaml",
        "preflight.json",
        "environment.freeze.txt",
        "checkpoint_manifest.json",
    )
    for filename in required_files:
        if not (run_dir / filename).is_file():
            errors.append(f"missing run evidence file: {filename}")

    environment_path = run_dir / "environment.json"
    environment = json.loads(environment_path.read_text()) if environment_path.is_file() else {}
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

    artifact_hashes = {report["sha256"] for report in reports}
    timing_hashes = {item.get("output_sha256") for item in timings}
    if artifact_hashes != timing_hashes:
        errors.append("timings.jsonl output hashes do not match the verified artifacts")

    preflight_path = run_dir / "preflight.json"
    if preflight_path.is_file():
        preflight = json.loads(preflight_path.read_text())
        if preflight.get("errors"):
            errors.append(f"preflight contains errors: {preflight['errors']}")

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

    candidate = bool(environment.get("benchmark_eligible"))
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
    args = parser.parse_args()

    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise SystemExit(f"required executable not found: {executable}")

    paths = [args.path] if args.path.is_file() else sorted(args.path.glob("*.mp4"))
    if not paths:
        raise SystemExit(f"no MP4 artifacts found under {args.path}")
    reports = [verify(path, require_metrics=not args.media_only) for path in paths]
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
