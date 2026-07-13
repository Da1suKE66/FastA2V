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


def decode_audio(path):
    process = subprocess.run(
        [
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
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return np.frombuffer(process.stdout, dtype="<f4").astype(np.float64)


def ffmpeg_metric(reference, candidate, filter_name, pattern):
    process = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "info",
            "-i",
            str(reference),
            "-i",
            str(candidate),
            "-lavfi",
            f"[0:v:0][1:v:0]{filter_name}",
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
        raise RuntimeError(f"could not parse {filter_name} from ffmpeg output")
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
    args = parser.parse_args()

    video_psnr = ffmpeg_metric(
        args.reference,
        args.candidate,
        "psnr",
        r"average:([0-9.+-]+|inf)",
    )
    video_ssim = ffmpeg_metric(
        args.reference,
        args.candidate,
        "ssim",
        r"All:([0-9.+-]+|inf)",
    )

    reference_audio = decode_audio(args.reference)
    candidate_audio = decode_audio(args.candidate)
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

    equivalent = bool(
        (math.isinf(video_psnr) or video_psnr >= 60.0)
        and video_ssim >= 0.999
        and audio_correlation >= 0.999
        and audio_snr_db >= 60.0
    )
    report = {
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "reference_sha256": sha256(args.reference),
        "candidate_sha256": sha256(args.candidate),
        "container_sha256_equal": sha256(args.reference) == sha256(args.candidate),
        "video_psnr_db": json_number(video_psnr),
        "video_ssim": json_number(video_ssim),
        "audio_sample_count_compared": int(sample_count),
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
