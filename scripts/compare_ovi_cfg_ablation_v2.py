#!/usr/bin/env python3
"""Lightweight Ovi CFG-cache quality comparison and paired analysis.

This tool is intentionally independent of the immutable run verifier.  It can
compare explicit MP4 files or matching MP4s in two run directories, and it can
summarize an explicitly paired development/held-out result table.  It never
manufactures ASR, SyncNet, or human-review scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_QUALITY_PROTOCOL = REPO_ROOT / "configs" / "quality_protocol.json"
SAMPLE_RATE = 16_000
DEFAULT_MAX_LAG_MS = 100.0
DEFAULT_ACTIVITY_THRESHOLD_DBFS = -45.0
KNOWN_DIRECTIONS = {
    "video_psnr_db": True,
    "video_ssim": True,
    "lpips_mean": False,
    "lpips_p95": False,
    "temporal_frame_difference_rmse": False,
    "audio_aligned_correlation": True,
    "audio_si_sdr_db": True,
    "audio_aligned_rmse": False,
    "audio_log_mel_l1_distance": False,
    "latent_relative_l2": False,
    "latent_cosine_similarity": True,
}


class ComparisonError(RuntimeError):
    """Fail-closed error for malformed media, dependencies, or analysis input."""


_MODULE_CACHE: dict[str, ModuleType] = {}


def _load_sibling_module(name: str) -> ModuleType:
    """Load one sibling script without requiring ``scripts`` to be a package."""

    if name in _MODULE_CACHE:
        return _MODULE_CACHE[name]
    path = SCRIPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_cfg_ablation_{name}", path)
    if spec is None or spec.loader is None:
        raise ComparisonError(f"cannot load helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _MODULE_CACHE[name] = module
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparisonError(message)


def _finite_float(value: Any, context: str) -> float:
    _require(
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_)),
        f"{context} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{context} must be finite")
    return result


def _parse_float(value: Any, context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ComparisonError(f"{context} must be numeric") from exc
    return _finite_float(parsed, context)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return "nan"
        if math.isinf(number):
            return "inf" if number > 0 else "-inf"
        return number
    return value


def _render_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        _json_safe(payload),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _write_or_print(payload: Mapping[str, Any], output: Path | None) -> None:
    rendered = _render_json(payload)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def decode_video_gray(
    path: Path,
    *,
    frames: int,
    width: int,
    height: int,
    ffmpeg: str | Path = "ffmpeg",
) -> np.ndarray:
    """Decode exactly ``frames`` full-resolution grayscale frames."""

    process = subprocess.run(
        [
            str(ffmpeg),
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-frames:v",
            str(frames),
            "-vsync",
            "0",
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
    expected = frames * width * height
    _require(
        raw.size == expected,
        f"expected {expected} decoded grayscale bytes from {path}, got {raw.size}",
    )
    return raw.reshape(frames, height, width)


def temporal_frame_difference_error(
    reference_frames: np.ndarray,
    candidate_frames: np.ndarray,
) -> float:
    """RMSE between successive-frame deltas, normalized to the [0, 1] range."""

    _require(
        reference_frames.shape == candidate_frames.shape,
        "temporal comparison requires identical decoded frame shapes",
    )
    _require(
        reference_frames.ndim >= 2 and reference_frames.shape[0] >= 2,
        "temporal comparison requires at least two frames",
    )
    reference_delta = np.diff(reference_frames.astype(np.float32), axis=0)
    candidate_delta = np.diff(candidate_frames.astype(np.float32), axis=0)
    difference = reference_delta - candidate_delta
    return float(np.sqrt(np.mean(np.square(difference))) / 255.0)


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    _require(left.size == right.size and left.size > 0, "correlation inputs differ")
    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= 1e-20:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def _segments_for_lag(
    reference: np.ndarray,
    candidate: np.ndarray,
    lag_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return overlap for a lag where positive means candidate is delayed."""

    if lag_samples >= 0:
        length = min(reference.size, candidate.size - lag_samples)
        if length <= 0:
            return reference[:0], candidate[:0]
        return reference[:length], candidate[lag_samples : lag_samples + length]
    reference_start = -lag_samples
    length = min(reference.size - reference_start, candidate.size)
    if length <= 0:
        return reference[:0], candidate[:0]
    return reference[reference_start : reference_start + length], candidate[:length]


def align_audio(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    max_lag_samples: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Find the best Pearson alignment within one fixed symmetric lag window."""

    _require(max_lag_samples >= 0, "maximum lag must be nonnegative")
    _require(reference.size > 0 and candidate.size > 0, "decoded audio is empty")
    minimum_overlap = max(32, min(reference.size, candidate.size) - max_lag_samples)
    candidates: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        left, right = _segments_for_lag(reference, candidate, lag)
        if left.size < minimum_overlap:
            continue
        candidates.append((_pearson(left, right), lag, left, right))
    _require(bool(candidates), "no audio overlap remains inside the lag window")
    # Prefer the strongest correlation; exact ties prefer the smallest shift,
    # then the negative shift for deterministic ordering.
    correlation, lag, left, right = max(
        candidates,
        key=lambda item: (item[0], -abs(item[1]), -item[1]),
    )
    return left, right, lag, correlation


def si_sdr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    _require(reference.size == candidate.size and reference.size > 0, "SI-SDR inputs differ")
    reference_energy = float(np.dot(reference, reference))
    _require(reference_energy > 1e-20, "SI-SDR reference is effectively silent")
    scale = float(np.dot(candidate, reference) / reference_energy)
    target = scale * reference
    noise = candidate - target
    target_energy = float(np.dot(target, target))
    noise_energy = float(np.dot(noise, noise))
    return float(10.0 * math.log10((target_energy + 1e-20) / (noise_energy + 1e-20)))


def _hz_to_mel(value: float | np.ndarray) -> float | np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(value) / 700.0)


def _mel_to_hz(value: float | np.ndarray) -> float | np.ndarray:
    return 700.0 * (np.power(10.0, np.asarray(value) / 2595.0) - 1.0)


def _mel_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for index in range(n_mels):
        left, center, right = (int(value) for value in bins[index : index + 3])
        center = max(center, left + 1)
        right = max(right, center + 1)
        right = min(right, n_fft // 2 + 1)
        center = min(center, right - 1)
        for frequency_bin in range(left, center):
            filters[index, frequency_bin] = (frequency_bin - left) / max(center - left, 1)
        for frequency_bin in range(center, right):
            filters[index, frequency_bin] = (right - frequency_bin) / max(right - center, 1)
    return filters


def _audio_frames(audio: np.ndarray, *, frame_length: int, hop_length: int) -> np.ndarray:
    _require(frame_length > 0 and hop_length > 0, "invalid audio frame geometry")
    if audio.size < frame_length:
        audio = np.pad(audio, (0, frame_length - audio.size))
    remainder = (audio.size - frame_length) % hop_length
    if remainder:
        audio = np.pad(audio, (0, hop_length - remainder))
    windows = np.lib.stride_tricks.sliding_window_view(audio, frame_length)
    return np.asarray(windows[::hop_length], dtype=np.float64)


def log_mel_spectrogram(
    audio: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = 512,
    frame_length: int = 400,
    hop_length: int = 160,
    n_mels: int = 64,
) -> np.ndarray:
    frames = _audio_frames(audio, frame_length=frame_length, hop_length=hop_length)
    window = np.hanning(frame_length)
    spectrum = np.fft.rfft(frames * window[None, :], n=n_fft, axis=1)
    power = np.square(np.abs(spectrum))
    filters = _mel_filterbank(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=20.0,
        fmax=min(7600.0, sample_rate / 2.0),
    )
    # ``einsum(..., optimize=False)`` avoids platform-BLAS floating-point
    # status leakage observed with NumPy 2.x matmul while computing the same
    # deterministic triangular-filter reduction.
    mel_power = np.einsum("tf,mf->tm", power, filters, optimize=False)
    return np.log(np.maximum(mel_power, 1e-10))


def log_mel_l1_distance(reference: np.ndarray, candidate: np.ndarray) -> float:
    left = log_mel_spectrogram(reference)
    right = log_mel_spectrogram(candidate)
    frame_count = min(left.shape[0], right.shape[0])
    _require(frame_count > 0, "log-mel comparison has no frames")
    return float(np.mean(np.abs(left[:frame_count] - right[:frame_count])))


def activity_metrics(
    audio: np.ndarray,
    *,
    threshold_dbfs: float = DEFAULT_ACTIVITY_THRESHOLD_DBFS,
    sample_rate: int = SAMPLE_RATE,
) -> dict[str, Any]:
    frame_length = max(1, int(round(0.020 * sample_rate)))
    frames = _audio_frames(audio, frame_length=frame_length, hop_length=frame_length)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    dbfs = 20.0 * np.log10(np.maximum(rms, 1e-12))
    active = dbfs > threshold_dbfs
    coverage = float(np.mean(active))
    return {
        "method": "20ms_frame_rms_dbfs",
        "threshold_dbfs": float(threshold_dbfs),
        "frame_count": int(active.size),
        "speech_activity_coverage": coverage,
        "silence_ratio": 1.0 - coverage,
    }


def aligned_audio_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    max_lag_samples: int,
    activity_threshold_dbfs: float,
) -> dict[str, Any]:
    left, right, lag, correlation = align_audio(
        reference,
        candidate,
        max_lag_samples=max_lag_samples,
    )
    rmse = float(np.sqrt(np.mean(np.square(left - right))))
    reference_activity = activity_metrics(left, threshold_dbfs=activity_threshold_dbfs)
    candidate_activity = activity_metrics(right, threshold_dbfs=activity_threshold_dbfs)
    return {
        "sample_rate_hz": SAMPLE_RATE,
        "max_lag_samples": int(max_lag_samples),
        "max_lag_ms": float(1000.0 * max_lag_samples / SAMPLE_RATE),
        "selected_lag_samples": int(lag),
        "selected_lag_ms": float(1000.0 * lag / SAMPLE_RATE),
        "lag_convention": "positive means candidate is delayed relative to dense",
        "aligned_sample_count": int(left.size),
        "aligned_correlation": float(correlation),
        "si_sdr_db": si_sdr_db(left, right),
        "aligned_rmse": rmse,
        "log_mel_l1_distance": log_mel_l1_distance(left, right),
        "dense_activity": reference_activity,
        "candidate_activity": candidate_activity,
        "speech_activity_coverage_difference": float(
            candidate_activity["speech_activity_coverage"]
            - reference_activity["speech_activity_coverage"]
        ),
        "silence_ratio_difference": float(
            candidate_activity["silence_ratio"] - reference_activity["silence_ratio"]
        ),
    }


def latent_similarity(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference = np.load(reference_path, allow_pickle=False)
    candidate = np.load(candidate_path, allow_pickle=False)
    _require(reference.shape == candidate.shape, "latent arrays have different shapes")
    _require(reference.size > 0, "latent arrays are empty")
    left = np.asarray(reference, dtype=np.float64).reshape(-1)
    right = np.asarray(candidate, dtype=np.float64).reshape(-1)
    _require(np.isfinite(left).all() and np.isfinite(right).all(), "latent arrays are non-finite")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    difference_norm = float(np.linalg.norm(right - left))
    cosine_denominator = left_norm * right_norm
    cosine = (
        float(np.dot(left, right) / cosine_denominator)
        if cosine_denominator > 1e-20
        else (1.0 if np.array_equal(left, right) else 0.0)
    )
    return {
        "status": "ok",
        "dense_path": str(reference_path.resolve()),
        "candidate_path": str(candidate_path.resolve()),
        "shape": list(reference.shape),
        "relative_l2": difference_norm / max(left_norm, 1e-20),
        "cosine_similarity": cosine,
    }


class PinnedLpipsEvaluator:
    """Delay every LPIPS and torch import until explicitly required."""

    def __init__(self, protocol_path: Path, receipt_path: Path | None):
        try:
            quality = _load_sibling_module("compare_ovi_quality")
            protocol, _ = quality.load_quality_protocol(protocol_path)
            receipt = quality.validate_lpips_receipt(
                protocol["lpips"], receipt_path=receipt_path
            )
            tool_receipt = quality.collect_media_tool_receipt()
            tool_paths = quality.validate_media_tool_receipt(tool_receipt)
            runner = quality.LpipsAlexCpu(protocol["lpips"], tool_paths)
        except Exception as exc:
            raise ComparisonError(
                f"required pinned LPIPS environment is unavailable or invalid: {exc}"
            ) from exc
        self._quality = quality
        self._protocol = protocol
        self._receipt = receipt
        self._runner = runner

    def __call__(self, reference: Path, candidate: Path) -> dict[str, Any]:
        values: list[float] = []
        original_model = self._runner.model

        class CaptureModel:
            def __call__(capture_self, *args: Any, **kwargs: Any) -> Any:
                result = original_model(*args, **kwargs)
                values.append(float(result.reshape(-1).mean().item()))
                return result

        self._runner.model = CaptureModel()
        try:
            mean, frame_count = self._runner(reference, candidate)
        finally:
            self._runner.model = original_model
        _require(
            len(values) == frame_count and frame_count > 0,
            "LPIPS did not produce one score per decoded frame",
        )
        # Revalidate after scoring so a dependency change cannot be hidden.
        post_receipt = self._quality.validate_lpips_receipt(
            self._protocol["lpips"],
            receipt_path=Path(self._receipt["receipt_path"]),
        )
        _require(post_receipt == self._receipt, "LPIPS receipt changed during scoring")
        return {
            "status": "ok",
            "implementation": "pinned lpips.LPIPS AlexNet CPU",
            "frame_count": int(frame_count),
            "mean": float(mean),
            "p95": float(np.percentile(np.asarray(values), 95.0)),
            "receipt_path": self._receipt["receipt_path"],
            "receipt_sha256": self._receipt["receipt_sha256"],
        }


def _pending_evaluations() -> dict[str, dict[str, str]]:
    return {
        "asr": {
            "status": "pending",
            "reason": "ASR WER/CER requires a separately pinned evaluator and transcript binding",
        },
        "syncnet": {
            "status": "pending",
            "reason": "lip-sync scoring requires a separately pinned SyncNet-style evaluator",
        },
        "human_blind_review": {
            "status": "pending",
            "reason": "independent blinded ratings must be supplied by reviewers",
        },
    }


def _automatic_latent(video: Path) -> Path | None:
    for path in (video.with_suffix(".latent.npy"), video.with_suffix(".npy")):
        if path.is_file():
            return path
    return None


def _resolve_latents(
    dense_video: Path,
    candidate_video: Path,
    dense_latent: Path | None,
    candidate_latent: Path | None,
) -> tuple[Path, Path] | None:
    _require(
        (dense_latent is None) == (candidate_latent is None),
        "dense and candidate latent paths must be supplied together",
    )
    if dense_latent is not None and candidate_latent is not None:
        _require(dense_latent.is_file(), f"dense latent does not exist: {dense_latent}")
        _require(
            candidate_latent.is_file(), f"candidate latent does not exist: {candidate_latent}"
        )
        return dense_latent, candidate_latent
    automatic_dense = _automatic_latent(dense_video)
    automatic_candidate = _automatic_latent(candidate_video)
    _require(
        (automatic_dense is None) == (automatic_candidate is None),
        "only one automatic latent sidecar exists; provide or remove the incomplete pair",
    )
    if automatic_dense is None or automatic_candidate is None:
        return None
    return automatic_dense, automatic_candidate


def compare_pair(
    dense_video: Path,
    candidate_video: Path,
    *,
    ffmpeg: str | Path = "ffmpeg",
    ffprobe: str | Path = "ffprobe",
    max_lag_ms: float = DEFAULT_MAX_LAG_MS,
    activity_threshold_dbfs: float = DEFAULT_ACTIVITY_THRESHOLD_DBFS,
    dense_latent: Path | None = None,
    candidate_latent: Path | None = None,
    lpips_evaluator: Callable[[Path, Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    _require(dense_video.is_file(), f"dense MP4 does not exist: {dense_video}")
    _require(candidate_video.is_file(), f"candidate MP4 does not exist: {candidate_video}")
    _require(max_lag_ms >= 0.0, "max lag must be nonnegative")
    media = _load_sibling_module("compare_media")
    dense_probe = media.probe_video(dense_video, ffprobe=ffprobe)
    candidate_probe = media.probe_video(candidate_video, ffprobe=ffprobe)
    for field in ("frames", "width", "height", "avg_frame_rate"):
        _require(
            dense_probe[field] == candidate_probe[field],
            f"candidate video {field} differs from dense",
        )
    frame_count = int(dense_probe["frames"])
    _require(frame_count >= 2, "video comparison requires at least two frames")
    psnr = media.ffmpeg_metric(
        dense_video,
        candidate_video,
        frame_count,
        "psnr",
        r"average:([0-9.+-]+|inf)",
        ffmpeg=ffmpeg,
    )
    ssim = media.ffmpeg_metric(
        dense_video,
        candidate_video,
        frame_count,
        "ssim",
        r"All:([0-9.+-]+|inf)",
        ffmpeg=ffmpeg,
    )
    dense_gray = decode_video_gray(
        dense_video,
        frames=frame_count,
        width=int(dense_probe["width"]),
        height=int(dense_probe["height"]),
        ffmpeg=ffmpeg,
    )
    candidate_gray = decode_video_gray(
        candidate_video,
        frames=frame_count,
        width=int(candidate_probe["width"]),
        height=int(candidate_probe["height"]),
        ffmpeg=ffmpeg,
    )
    dense_audio = media.decode_audio(dense_video, ffmpeg=ffmpeg)
    candidate_audio = media.decode_audio(candidate_video, ffmpeg=ffmpeg)
    max_lag_samples = int(round(max_lag_ms * SAMPLE_RATE / 1000.0))
    audio = aligned_audio_metrics(
        dense_audio,
        candidate_audio,
        max_lag_samples=max_lag_samples,
        activity_threshold_dbfs=activity_threshold_dbfs,
    )
    latent_paths = _resolve_latents(
        dense_video,
        candidate_video,
        dense_latent,
        candidate_latent,
    )
    latent = (
        latent_similarity(*latent_paths)
        if latent_paths is not None
        else {"status": "not_available"}
    )
    lpips = (
        dict(lpips_evaluator(dense_video, candidate_video))
        if lpips_evaluator is not None
        else {
            "status": "not_requested",
            "reason": "rerun with --require-lpips inside the pinned evaluator environment",
        }
    )
    return {
        "dense": {
            "path": str(dense_video.resolve()),
            "sha256": _sha256(dense_video),
            "video": dense_probe,
            "audio_samples_16khz_mono": int(dense_audio.size),
        },
        "candidate": {
            "path": str(candidate_video.resolve()),
            "sha256": _sha256(candidate_video),
            "video": candidate_probe,
            "audio_samples_16khz_mono": int(candidate_audio.size),
        },
        "metrics": {
            "video": {
                "compared_frames": frame_count,
                "psnr_db": psnr,
                "ssim": ssim,
                "temporal_frame_difference_rmse": temporal_frame_difference_error(
                    dense_gray, candidate_gray
                ),
                "temporal_metric_definition": (
                    "full-resolution decoded grayscale successive-frame delta RMSE / 255"
                ),
            },
            "audio": audio,
            "lpips": lpips,
            "latent": latent,
        },
        "pending_evaluations": _pending_evaluations(),
    }


def _sidecar_metadata(video: Path) -> dict[str, Any]:
    sidecar = video.with_suffix(".metrics.json")
    if not sidecar.is_file():
        return {}
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"cannot read metrics sidecar {sidecar}: {exc}") from exc
    return {
        key: payload.get(key)
        for key in ("prompt", "prompt_index", "seed", "sample_index", "measurement_index")
    }


def _resolve_video_pairs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    explicit = args.dense is not None or args.candidate is not None
    run_pair = args.dense_run is not None or args.candidate_run is not None
    _require(explicit != run_pair, "choose either explicit MP4s or one run-directory pair")
    if explicit:
        _require(
            args.dense is not None and args.candidate is not None,
            "--dense and --candidate are required together",
        )
        return [(args.dense, args.candidate)]
    _require(
        args.dense_run is not None and args.candidate_run is not None,
        "--dense-run and --candidate-run are required together",
    )
    _require(args.dense_run.is_dir(), f"dense run is not a directory: {args.dense_run}")
    _require(
        args.candidate_run.is_dir(), f"candidate run is not a directory: {args.candidate_run}"
    )
    dense_files = {
        str(path.relative_to(args.dense_run)): path
        for path in sorted(args.dense_run.rglob("*.mp4"))
    }
    candidate_files = {
        str(path.relative_to(args.candidate_run)): path
        for path in sorted(args.candidate_run.rglob("*.mp4"))
    }
    _require(bool(dense_files), f"dense run contains no MP4 files: {args.dense_run}")
    _require(
        set(dense_files) == set(candidate_files),
        "run directories must contain exactly the same relative MP4 paths",
    )
    return [(dense_files[key], candidate_files[key]) for key in sorted(dense_files)]


def _run_compare(args: argparse.Namespace) -> dict[str, Any]:
    pairs = _resolve_video_pairs(args)
    _require(
        len(pairs) == 1 or (args.dense_latent is None and args.candidate_latent is None),
        "explicit latent paths are only valid for a single MP4 pair",
    )
    lpips_evaluator = (
        PinnedLpipsEvaluator(args.quality_protocol, args.lpips_receipt)
        if args.require_lpips
        else None
    )
    records: list[dict[str, Any]] = []
    for index, (dense_video, candidate_video) in enumerate(pairs):
        metadata = _sidecar_metadata(candidate_video) or _sidecar_metadata(dense_video)
        prompt = args.prompt or metadata.get("prompt")
        prompt_id = args.prompt_id
        if prompt_id is None:
            prompt_index = metadata.get("prompt_index")
            prompt_id = (
                f"prompt-{int(prompt_index):03d}"
                if isinstance(prompt_index, int)
                else candidate_video.stem
            )
        if len(pairs) > 1 and args.prompt_id is not None:
            prompt_id = f"{args.prompt_id}:{index:03d}"
        seed = args.seed if args.seed is not None else metadata.get("seed")
        record = compare_pair(
            dense_video,
            candidate_video,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            max_lag_ms=args.max_audio_lag_ms,
            activity_threshold_dbfs=args.activity_threshold_dbfs,
            dense_latent=args.dense_latent,
            candidate_latent=args.candidate_latent,
            lpips_evaluator=lpips_evaluator,
        )
        record.update(
            {
                "split": args.split,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "category": args.category,
                "seed": seed,
                "candidate_id": args.candidate_id,
                "comparison_id": args.comparison_id,
            }
        )
        records.append(record)
    return {
        "schema_version": 1,
        "record_type": "ovi_cfg_ablation_v2_media_comparison",
        "pair_count": len(records),
        "require_lpips": bool(args.require_lpips),
        "pairs": records,
        "pending_evaluations": _pending_evaluations(),
    }


def _parse_bool(value: Any, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "higher"}:
            return True
        if normalized in {"false", "0", "no", "lower"}:
            return False
    raise ComparisonError(f"{context} must identify higher- or lower-is-better")


def _rows_from_payload(payload: Any, source: Path) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        _require(all(isinstance(item, Mapping) for item in payload), f"{source} rows are invalid")
        return list(payload)
    _require(isinstance(payload, Mapping), f"{source} JSON root must be an object or list")
    for key in ("records", "analysis_records"):
        rows = payload.get(key)
        if rows is not None:
            _require(isinstance(rows, list), f"{source} {key} must be a list")
            _require(all(isinstance(item, Mapping) for item in rows), f"{source} rows are invalid")
            return list(rows)
    # A single long-format record is also accepted.
    if "metric" in payload:
        return [payload]
    raise ComparisonError(
        f"{source} does not contain paired analysis rows; expected a list, records, or analysis_records"
    )


def load_analysis_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        _require(path.is_file(), f"analysis input does not exist: {path}")
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                source_rows: Iterable[Mapping[str, Any]] = list(csv.DictReader(handle))
        else:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ComparisonError(f"cannot parse analysis input {path}: {exc}") from exc
            source_rows = _rows_from_payload(payload, path)
        for row_index, source_row in enumerate(source_rows, start=1):
            row = dict(source_row)
            context = f"{path} row {row_index}"
            split = str(row.get("split", "")).strip().lower()
            _require(
                split in {"development", "heldout"},
                f"{context} split must be exactly development or heldout",
            )
            prompt = str(row.get("prompt_id") or row.get("prompt") or "").strip()
            category = str(row.get("category") or "").strip()
            metric = str(row.get("metric") or "").strip()
            seed_text = row.get("seed")
            _require(prompt != "", f"{context} is missing prompt_id/prompt")
            _require(category != "", f"{context} is missing category")
            _require(metric != "", f"{context} is missing metric")
            try:
                seed = int(seed_text)
            except (TypeError, ValueError) as exc:
                raise ComparisonError(f"{context} seed must be an integer") from exc
            if row.get("higher_is_better") not in (None, ""):
                higher_is_better = _parse_bool(
                    row["higher_is_better"], f"{context} higher_is_better"
                )
            else:
                _require(
                    metric in KNOWN_DIRECTIONS,
                    f"{context} metric direction is unknown; add higher_is_better",
                )
                higher_is_better = KNOWN_DIRECTIONS[metric]
            if row.get("candidate_value") not in (None, ""):
                candidate_value = _parse_float(
                    row["candidate_value"], f"{context} candidate_value"
                )
                comparator_value = _parse_float(
                    row.get("comparator_value"), f"{context} comparator_value"
                )
                raw_difference = candidate_value - comparator_value
            else:
                candidate_value = None
                comparator_value = None
                raw_difference = _parse_float(
                    row.get("difference"), f"{context} difference"
                )
            oriented = raw_difference if higher_is_better else -raw_difference
            rows.append(
                {
                    "split": split,
                    "prompt": prompt,
                    "category": category,
                    "seed": seed,
                    "metric": metric,
                    "higher_is_better": higher_is_better,
                    "candidate_value": candidate_value,
                    "comparator_value": comparator_value,
                    "raw_difference": raw_difference,
                    "oriented_difference": oriented,
                }
            )
    _require(bool(rows), "analysis inputs contain no rows")
    return rows


def _derived_seed(base_seed: int, *parts: str) -> int:
    material = "|".join([str(base_seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _cluster_bootstrap_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    statistic: Callable[[np.ndarray], float],
    replicates: int,
    seed: int,
) -> list[float]:
    _require(replicates > 0, "bootstrap replicates must be positive")
    by_prompt: dict[str, list[float]] = {}
    for row in rows:
        by_prompt.setdefault(str(row["prompt"]), []).append(
            float(row["oriented_difference"])
        )
    prompts = sorted(by_prompt)
    _require(bool(prompts), "bootstrap has no prompt clusters")
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, len(prompts), size=len(prompts))
        observations = np.concatenate(
            [np.asarray(by_prompt[prompts[item]], dtype=np.float64) for item in sampled]
        )
        values[index] = statistic(observations)
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    values = np.asarray([row["oriented_difference"] for row in rows], dtype=np.float64)
    return {
        "pair_count": int(values.size),
        "prompt_cluster_count": len({str(row["prompt"]) for row in rows}),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10.0)),
        "worst": float(np.min(values)),
        "win_rate": float(np.mean(values > 0.0)),
        "tie_rate": float(np.mean(values == 0.0)),
        "mean_cluster_bootstrap_ci95": _cluster_bootstrap_interval(
            rows,
            statistic=lambda array: float(np.mean(array)),
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "median_cluster_bootstrap_ci95": _cluster_bootstrap_interval(
            rows,
            statistic=lambda array: float(np.median(array)),
            replicates=bootstrap_replicates,
            seed=bootstrap_seed ^ 0x9E3779B97F4A7C15,
        ),
    }


def analyze_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    prompt_splits: dict[str, set[str]] = {}
    prompt_categories: dict[str, set[str]] = {}
    unique_keys: set[tuple[str, str, int, str]] = set()
    for row in rows:
        prompt = str(row["prompt"])
        prompt_splits.setdefault(prompt, set()).add(str(row["split"]))
        prompt_categories.setdefault(prompt, set()).add(str(row["category"]))
        key = (str(row["split"]), prompt, int(row["seed"]), str(row["metric"]))
        _require(key not in unique_keys, f"duplicate paired analysis row: {key}")
        unique_keys.add(key)
    overlap = sorted(prompt for prompt, splits in prompt_splits.items() if len(splits) > 1)
    _require(
        not overlap,
        "development and heldout prompt identities overlap: " + ", ".join(overlap),
    )
    inconsistent_categories = sorted(
        prompt for prompt, categories in prompt_categories.items() if len(categories) > 1
    )
    _require(
        not inconsistent_categories,
        "prompt category changes across rows: " + ", ".join(inconsistent_categories),
    )
    output_splits: dict[str, Any] = {}
    for split in ("development", "heldout"):
        split_rows = [row for row in rows if row["split"] == split]
        if not split_rows:
            output_splits[split] = {"status": "no_records", "metrics": {}}
            continue
        metric_output: dict[str, Any] = {}
        for metric in sorted({str(row["metric"]) for row in split_rows}):
            metric_rows = [row for row in split_rows if row["metric"] == metric]
            directions = {bool(row["higher_is_better"]) for row in metric_rows}
            _require(len(directions) == 1, f"metric {metric} changes direction")
            higher_is_better = directions.pop()
            categories: dict[str, Any] = {}
            for category in sorted({str(row["category"]) for row in metric_rows}):
                category_rows = [row for row in metric_rows if row["category"] == category]
                categories[category] = _summary(
                    category_rows,
                    bootstrap_replicates=bootstrap_replicates,
                    bootstrap_seed=_derived_seed(
                        bootstrap_seed, split, metric, category
                    ),
                )
            metric_output[metric] = {
                "higher_is_better": higher_is_better,
                "reported_quantity": (
                    "candidate_minus_comparator"
                    if higher_is_better
                    else "comparator_minus_candidate"
                ),
                "overall": _summary(
                    metric_rows,
                    bootstrap_replicates=bootstrap_replicates,
                    bootstrap_seed=_derived_seed(bootstrap_seed, split, metric, "overall"),
                ),
                "categories": categories,
            }
        output_splits[split] = {
            "status": "ok",
            "prompt_cluster_count": len({str(row["prompt"]) for row in split_rows}),
            "record_count": len(split_rows),
            "metrics": metric_output,
        }
    return {
        "schema_version": 1,
        "record_type": "ovi_cfg_ablation_v2_clustered_analysis",
        "bootstrap": {
            "unit": "prompt",
            "seeds_retained_within_prompt": True,
            "replicates": int(bootstrap_replicates),
            "fixed_seed": int(bootstrap_seed),
            "interval_percentiles": [2.5, 97.5],
        },
        "difference_orientation": "positive always favors candidate",
        "splits": output_splits,
        "cross_split_aggregation": "forbidden",
        "pending_evaluations": _pending_evaluations(),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lightweight Ovi CFG-cache media comparison and paired statistics"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser("compare", help="compare explicit MP4s or two run dirs")
    compare.add_argument("--dense", type=Path)
    compare.add_argument("--candidate", type=Path)
    compare.add_argument("--dense-run", type=Path)
    compare.add_argument("--candidate-run", type=Path)
    compare.add_argument("--dense-latent", type=Path)
    compare.add_argument("--candidate-latent", type=Path)
    compare.add_argument("--ffmpeg", default="ffmpeg")
    compare.add_argument("--ffprobe", default="ffprobe")
    compare.add_argument("--max-audio-lag-ms", type=float, default=DEFAULT_MAX_LAG_MS)
    compare.add_argument(
        "--activity-threshold-dbfs",
        type=float,
        default=DEFAULT_ACTIVITY_THRESHOLD_DBFS,
    )
    compare.add_argument("--require-lpips", action="store_true")
    compare.add_argument("--quality-protocol", type=Path, default=DEFAULT_QUALITY_PROTOCOL)
    compare.add_argument("--lpips-receipt", type=Path)
    compare.add_argument("--split", choices=("development", "heldout"))
    compare.add_argument("--prompt-id")
    compare.add_argument("--prompt")
    compare.add_argument("--category")
    compare.add_argument("--seed", type=int)
    compare.add_argument("--candidate-id")
    compare.add_argument("--comparison-id")
    compare.add_argument("--output", type=Path)

    analyze = subparsers.add_parser(
        "analyze",
        help=(
            "analyze long-format paired CSV/JSON rows with split,prompt,category,seed,"
            "metric and candidate/comparator values (or difference)"
        ),
    )
    analyze.add_argument("inputs", nargs="+", type=Path)
    analyze.add_argument("--bootstrap-replicates", type=int, default=5000)
    analyze.add_argument("--bootstrap-seed", type=int, default=20260717)
    analyze.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "compare":
            payload = _run_compare(args)
            _write_or_print(payload, args.output)
        elif args.command == "analyze":
            rows = load_analysis_rows(args.inputs)
            payload = analyze_rows(
                rows,
                bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_seed=args.bootstrap_seed,
            )
            _write_or_print(payload, args.output)
        else:
            raise AssertionError(f"unexpected command {args.command}")
    except (ComparisonError, OSError, subprocess.CalledProcessError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
