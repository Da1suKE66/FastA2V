#!/usr/bin/env python3
"""Hash complete decoded Ovi video and audio streams.

The receipt binds the input container to an exact ffprobe observation and to
complete, streamed ffmpeg decodes.  Video is decoded as RGB24 and audio as
mono 16 kHz little-endian float32.  Raw streams are never retained on disk.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import BinaryIO


SCHEMA_VERSION = 1
CHUNK_BYTES = 1024 * 1024


class DecodedStreamError(RuntimeError):
    """Raised when probing or a complete decode cannot be proven."""


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def _tool_binding(name: str) -> dict[str, object]:
    selected = shutil.which(name)
    if selected is None:
        raise DecodedStreamError(f"required executable not found: {name}")
    path = Path(selected).resolve(strict=True)
    digest, size = _sha256_file(path)
    metadata = path.stat()
    return {
        "name": name,
        "path": str(path),
        "bytes": size,
        "sha256": digest,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
    }


def _run_probe(path: Path, ffprobe_path: str) -> dict[str, object]:
    command = [
        ffprobe_path,
        "-v",
        "error",
        "-count_frames",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stderr = completed.stderr.decode("utf-8", errors="replace").strip()
    if completed.returncode != 0:
        detail = stderr or "no stderr diagnostics"
        raise DecodedStreamError(
            f"ffprobe exited with status {completed.returncode}: {detail}"
        )
    if stderr:
        raise DecodedStreamError(
            f"ffprobe emitted error-level diagnostics: {stderr}"
        )
    try:
        payload = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DecodedStreamError(f"ffprobe returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise DecodedStreamError("ffprobe JSON root must be an object")
    return {
        "command": command,
        "stdout_bytes": len(completed.stdout),
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "payload": payload,
    }


def _stream_digest(
    stream: BinaryIO,
    *,
    collect_float32_stats: bool,
) -> dict[str, object]:
    digest = hashlib.sha256()
    total = 0
    pending = b""
    sample_count = 0
    square_sum = 0.0
    peak = 0.0
    nonfinite_samples = 0

    while True:
        chunk = stream.read(CHUNK_BYTES)
        if not chunk:
            break
        digest.update(chunk)
        total += len(chunk)
        if not collect_float32_stats:
            continue
        data = pending + chunk
        usable = len(data) - (len(data) % 4)
        pending = data[usable:]
        if not usable:
            continue
        values = array.array("f")
        values.frombytes(data[:usable])
        if sys.byteorder != "little":
            values.byteswap()
        for value in values:
            if not math.isfinite(value):
                nonfinite_samples += 1
                continue
            sample_count += 1
            square_sum += float(value) * float(value)
            peak = max(peak, abs(float(value)))

    if collect_float32_stats and pending:
        raise DecodedStreamError(
            f"decoded f32le audio has {len(pending)} trailing byte(s)"
        )
    result: dict[str, object] = {
        "bytes": total,
        "sha256": digest.hexdigest(),
    }
    if collect_float32_stats:
        if nonfinite_samples:
            raise DecodedStreamError(
                f"decoded audio contains {nonfinite_samples} non-finite sample(s)"
            )
        result.update(
            {
                "sample_count": sample_count,
                "rms": math.sqrt(square_sum / sample_count)
                if sample_count
                else 0.0,
                "peak": peak,
            }
        )
    return result


def _decode_stream(
    command: list[str],
    *,
    collect_float32_stats: bool = False,
) -> dict[str, object]:
    with tempfile.TemporaryFile() as stderr_handle:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=stderr_handle,
        )
        if process.stdout is None:
            process.kill()
            raise DecodedStreamError("ffmpeg stdout pipe was not created")
        try:
            receipt = _stream_digest(
                process.stdout,
                collect_float32_stats=collect_float32_stats,
            )
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            process.stdout.close()
        returncode = process.wait()
        stderr_handle.seek(0)
        stderr = stderr_handle.read().decode("utf-8", errors="replace").strip()
    if returncode != 0:
        detail = stderr or "no stderr diagnostics"
        raise DecodedStreamError(
            f"ffmpeg exited with status {returncode}: {detail}"
        )
    if stderr:
        raise DecodedStreamError(
            f"ffmpeg emitted error-level diagnostics: {stderr}"
        )
    receipt["command"] = command
    return receipt


def _json_int(value: object, context: str) -> int:
    if isinstance(value, bool):
        raise DecodedStreamError(f"{context} must be an integer")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise DecodedStreamError(f"{context} must be an integer") from exc
    if parsed < 1:
        raise DecodedStreamError(f"{context} must be positive")
    return parsed


def _only_stream(
    probe_payload: dict[str, object], stream_type: str
) -> dict[str, object]:
    streams = probe_payload.get("streams")
    if not isinstance(streams, list):
        raise DecodedStreamError("ffprobe streams must be an array")
    selected = [
        stream
        for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == stream_type
    ]
    if len(selected) != 1:
        raise DecodedStreamError(
            f"expected exactly one {stream_type} stream, found {len(selected)}"
        )
    return selected[0]


def hash_artifact(
    path: Path,
    *,
    ffmpeg_path: str,
    ffprobe_path: str,
) -> dict[str, object]:
    path = path.resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != ".mp4":
        raise DecodedStreamError(f"not an MP4 file: {path}")
    container_sha256, container_bytes = _sha256_file(path)
    probe = _run_probe(path, ffprobe_path)
    payload = probe["payload"]
    if not isinstance(payload, dict):
        raise DecodedStreamError("internal ffprobe payload type error")
    video_stream = _only_stream(payload, "video")
    audio_stream = _only_stream(payload, "audio")
    width = _json_int(video_stream.get("width"), "video width")
    height = _json_int(video_stream.get("height"), "video height")
    frames = _json_int(
        video_stream.get("nb_read_frames", video_stream.get("nb_frames")),
        "decoded frame count",
    )

    video_command = [
        ffmpeg_path,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    video_decode = _decode_stream(video_command)
    expected_video_bytes = frames * width * height * 3
    if video_decode["bytes"] != expected_video_bytes:
        raise DecodedStreamError(
            "RGB24 decode length mismatch: "
            f"expected {expected_video_bytes}, found {video_decode['bytes']}"
        )
    video_decode.update(
        {
            "pixel_format": "rgb24",
            "frames": frames,
            "width": width,
            "height": height,
        }
    )

    audio_command = [
        ffmpeg_path,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_f32le",
        "-f",
        "f32le",
        "pipe:1",
    ]
    audio_decode = _decode_stream(
        audio_command,
        collect_float32_stats=True,
    )
    if audio_decode["bytes"] == 0:
        raise DecodedStreamError("decoded audio stream is empty")
    audio_decode.update(
        {
            "sample_format": "f32le",
            "channels": 1,
            "sample_rate": 16000,
        }
    )

    return {
        "path": str(path),
        "container": {
            "bytes": container_bytes,
            "sha256": container_sha256,
        },
        "ffprobe": probe,
        "video": {
            "codec_name": video_stream.get("codec_name"),
            "source_pixel_format": video_stream.get("pix_fmt"),
            "decode": video_decode,
        },
        "audio": {
            "codec_name": audio_stream.get("codec_name"),
            "source_channels": audio_stream.get("channels"),
            "source_sample_rate": audio_stream.get("sample_rate"),
            "decode": audio_decode,
        },
    }


def discover_mp4s(path: Path) -> tuple[Path, list[Path], str]:
    selected = path.resolve(strict=True)
    if selected.is_file():
        if selected.suffix.lower() != ".mp4":
            raise DecodedStreamError(f"input file is not MP4: {selected}")
        return selected.parent, [selected], "file"
    if not selected.is_dir():
        raise DecodedStreamError(f"input is neither a file nor directory: {selected}")
    artifacts = sorted(
        item.resolve(strict=True)
        for item in selected.iterdir()
        if item.is_file() and item.suffix.lower() == ".mp4"
    )
    if not artifacts:
        raise DecodedStreamError(f"no MP4 artifacts found under {selected}")
    return selected, artifacts, "directory"


def build_report(path: Path) -> dict[str, object]:
    run_dir, artifacts, input_kind = discover_mp4s(path)
    ffmpeg = _tool_binding("ffmpeg")
    ffprobe = _tool_binding("ffprobe")
    reports = [
        hash_artifact(
            artifact,
            ffmpeg_path=str(ffmpeg["path"]),
            ffprobe_path=str(ffprobe["path"]),
        )
        for artifact in artifacts
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "input": {
            "kind": input_kind,
            "path": str(path.resolve(strict=True)),
            "run_dir": str(run_dir),
        },
        "tools": {"ffmpeg": ffmpeg, "ffprobe": ffprobe},
        "artifact_count": len(reports),
        "artifacts": reports,
    }


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="MP4 file or run directory")
    parser.add_argument(
        "--output",
        type=Path,
        help="receipt path (default: decoded_stream_hashes.json beside input)",
    )
    args = parser.parse_args()
    try:
        run_dir, _artifacts, input_kind = discover_mp4s(args.path)
        output = args.output
        if output is None:
            output = (
                args.path.with_suffix(".decoded_stream_hashes.json")
                if input_kind == "file"
                else run_dir / "decoded_stream_hashes.json"
            )
        report = build_report(args.path)
        _atomic_write_json(output, report)
    except (DecodedStreamError, OSError, subprocess.SubprocessError) as exc:
        print(f"decoded-stream hashing failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
