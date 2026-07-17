#!/usr/bin/env python3
"""Compare two generated audio-video files at the decoded-media level."""

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_audio(path, *, ffmpeg="ffmpeg"):
    process = subprocess.run(
        [
            str(ffmpeg),
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
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return np.frombuffer(process.stdout, dtype="<f4").astype(np.float64)


def probe_video(path, *, ffprobe="ffprobe"):
    process = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,nb_read_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    streams = json.loads(process.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"expected one video stream in {path}, found {len(streams)}")
    stream = streams[0]
    frames = stream.get("nb_read_frames") or stream.get("nb_frames")
    if frames in (None, "N/A"):
        raise RuntimeError(f"ffprobe did not report a frame count for {path}")
    return {
        "frames": int(frames),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "duration_seconds": float(stream.get("duration") or 0.0),
    }


def decode_tail_gray(path, frame_count, *, ffmpeg="ffmpeg"):
    start = max(frame_count - 2, 0)
    process = subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-vf",
            f"select=gte(n\\,{start}),scale=64:64,format=gray",
            "-vsync",
            "0",
            "-frames:v",
            "2",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    raw = np.frombuffer(process.stdout, dtype=np.uint8)
    frame_size = 64 * 64
    if raw.size != 2 * frame_size:
        raise RuntimeError(f"expected two decoded tail frames from {path}, got {raw.size} bytes")
    return raw.reshape(2, 64, 64).astype(np.float64)


def tail_psnr(path, frame_count, *, ffmpeg="ffmpeg"):
    frames = decode_tail_gray(path, frame_count, ffmpeg=ffmpeg)
    mse = float(np.mean(np.square(frames[0] - frames[1])))
    return math.inf if mse == 0.0 else float(10.0 * math.log10(255.0**2 / mse))


def ffmpeg_metric(
    reference,
    candidate,
    frame_count,
    filter_name,
    pattern,
    *,
    ffmpeg="ffmpeg",
):
    filter_graph = (
        f"[0:v:0]trim=end_frame={frame_count},setpts=PTS-STARTPTS[reference];"
        f"[1:v:0]trim=end_frame={frame_count},setpts=PTS-STARTPTS[candidate];"
        f"[reference][candidate]{filter_name}"
    )
    process = subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "info",
            "-i",
            str(reference),
            "-i",
            str(candidate),
            "-lavfi",
            filter_graph,
            "-f",
            "null",
            "-",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    matches = re.findall(pattern, process.stderr)
    if not matches:
        stderr_tail = process.stderr[-4000:]
        raise RuntimeError(
            f"could not parse {filter_name} from ffmpeg output "
            f"for reference={reference} candidate={candidate}; "
            f"pattern={pattern!r}; stderr_tail={stderr_tail!r}"
        )
    value = matches[-1]
    return math.inf if value.lower() == "inf" else float(value)


def json_number(value):
    return "inf" if math.isinf(value) else value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expect-equivalent", action="store_true")
    parser.add_argument(
        "--allow-reference-duplicate-tail",
        action="store_true",
        help="allow exactly one extra, near-duplicate tail frame in the reference",
    )
    args = parser.parse_args()

    reference_video = probe_video(args.reference)
    candidate_video = probe_video(args.candidate)
    compared_frames = min(reference_video["frames"], candidate_video["frames"])
    if compared_frames < 1:
        raise RuntimeError("one or both media files have no decoded video frames")

    reference_tail_psnr = None
    reference_tail_ignored = False
    if reference_video["frames"] == candidate_video["frames"]:
        frame_topology_equivalent = True
    elif (
        args.allow_reference_duplicate_tail
        and reference_video["frames"] == candidate_video["frames"] + 1
    ):
        reference_tail_psnr = tail_psnr(args.reference, reference_video["frames"])
        frame_topology_equivalent = bool(
            math.isinf(reference_tail_psnr) or reference_tail_psnr >= 60.0
        )
        reference_tail_ignored = frame_topology_equivalent
    else:
        frame_topology_equivalent = False

    video_psnr = ffmpeg_metric(
        args.reference,
        args.candidate,
        compared_frames,
        "psnr",
        r"average:([0-9.+-]+|inf)",
    )
    video_ssim = ffmpeg_metric(
        args.reference,
        args.candidate,
        compared_frames,
        "ssim",
        r"All:([0-9.+-]+|inf)",
    )

    reference_audio = decode_audio(args.reference)
    candidate_audio = decode_audio(args.candidate)
    reference_audio_samples = int(reference_audio.size)
    candidate_audio_samples = int(candidate_audio.size)
    sample_count = min(reference_audio.size, candidate_audio.size)
    if sample_count == 0:
        raise RuntimeError("one or both media files have no decoded audio samples")
    reference_audio = reference_audio[:sample_count]
    candidate_audio = candidate_audio[:sample_count]
    difference = reference_audio - candidate_audio
    audio_rmse = float(np.sqrt(np.mean(np.square(difference))))
    audio_max_abs = float(np.max(np.abs(difference)))
    reference_rms = float(np.sqrt(np.mean(np.square(reference_audio))))
    audio_snr_db = float(
        20.0 * math.log10(max(reference_rms, 1e-12) / max(audio_rmse, 1e-12))
    )
    if np.std(reference_audio) == 0 or np.std(candidate_audio) == 0:
        audio_correlation = 1.0 if np.array_equal(reference_audio, candidate_audio) else 0.0
    else:
        audio_correlation = float(np.corrcoef(reference_audio, candidate_audio)[0, 1])

    packaging_quirk = reference_tail_ignored
    min_video_psnr = 45.0 if packaging_quirk else 60.0
    min_video_ssim = 0.995 if packaging_quirk else 0.999
    audio_length_equivalent = abs(reference_audio_samples - candidate_audio_samples) <= 1024
    equivalent = bool(
        frame_topology_equivalent
        and (math.isinf(video_psnr) or video_psnr >= min_video_psnr)
        and video_ssim >= min_video_ssim
        and audio_correlation >= 0.999
        and audio_snr_db >= 60.0
        and audio_length_equivalent
    )
    report = {
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "reference_sha256": sha256(args.reference),
        "candidate_sha256": sha256(args.candidate),
        "container_sha256_equal": sha256(args.reference) == sha256(args.candidate),
        "reference_video": reference_video,
        "candidate_video": candidate_video,
        "compared_video_frames": compared_frames,
        "frame_topology_equivalent": frame_topology_equivalent,
        "reference_tail_ignored": reference_tail_ignored,
        "reference_tail_psnr_db": (
            json_number(reference_tail_psnr) if reference_tail_psnr is not None else None
        ),
        "video_psnr_db": json_number(video_psnr),
        "video_ssim": json_number(video_ssim),
        "minimum_video_psnr_db": min_video_psnr,
        "minimum_video_ssim": min_video_ssim,
        "reference_audio_samples": reference_audio_samples,
        "candidate_audio_samples": candidate_audio_samples,
        "audio_sample_count_compared": int(sample_count),
        "audio_length_equivalent": audio_length_equivalent,
        "audio_rmse": audio_rmse,
        "audio_max_abs_difference": audio_max_abs,
        "audio_snr_db": audio_snr_db,
        "audio_correlation": audio_correlation,
        "equivalent": equivalent,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 1 if args.expect_equivalent and not equivalent else 0


if __name__ == "__main__":
    raise SystemExit(main())
