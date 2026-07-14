"""Lightweight experiment runner for the official LTX-2.3 pipelines.

This module deliberately keeps imports from ``torch``, ``ltx_core`` and
``ltx_pipelines`` inside runtime functions.  Argument parsing and result-file
tests therefore work before the separate LTX environment is installed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable


DISTILLED_STEPS = 11  # Official LTX-2.3 schedule: 8 stage-1 + 3 stage-2.
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 768
DEFAULT_NUM_FRAMES = 121
DEFAULT_FRAME_RATE = 24.0

RESULT_FIELDS = (
    "timestamp_utc",
    "model",
    "pipeline",
    "method",
    "prompt_id",
    "prompt",
    "seed",
    "steps",
    "height",
    "width",
    "num_frames",
    "frame_rate",
    "load_seconds",
    "pipeline_seconds",
    "stage_seconds",
    "denoise_seconds",
    "decode_encode_seconds",
    "total_seconds",
    "peak_memory_allocated_bytes",
    "peak_memory_reserved_bytes",
    "audio_rms",
    "audio_duration_seconds",
    "decoded_video_frames",
    "media_video_streams",
    "media_audio_streams",
    "media_video_duration_seconds",
    "media_audio_duration_seconds",
    "output_path",
    "status",
    "error",
    "attention_metrics",
)


@dataclass(frozen=True)
class PromptCase:
    prompt_id: str
    prompt: str


class TimedDiffusionStage:
    """Measure both whole-stage time and the inner denoising loop."""

    def __init__(self, stage: object, torch_module: object, default_loop: object) -> None:
        if not callable(default_loop):
            raise TypeError("default_loop must be callable")
        self._stage = stage
        self._torch = torch_module
        self._default_loop = default_loop
        self.stage_seconds = 0.0
        self.denoise_seconds = 0.0

    def reset(self) -> None:
        self.stage_seconds = 0.0
        self.denoise_seconds = 0.0

    def __getattr__(self, name: str) -> object:
        return getattr(self._stage, name)

    def __call__(self, *args: object, **kwargs: object) -> object:
        stage_kwargs = dict(kwargs)
        loop = stage_kwargs.get("loop") or self._default_loop

        def timed_loop(*loop_args: object, **loop_kwargs: object) -> object:
            _synchronize_cuda(self._torch)
            loop_start = time.perf_counter()
            try:
                return loop(*loop_args, **loop_kwargs)  # type: ignore[operator]
            finally:
                _synchronize_cuda(self._torch)
                self.denoise_seconds += time.perf_counter() - loop_start

        stage_kwargs["loop"] = timed_loop
        _synchronize_cuda(self._torch)
        start = time.perf_counter()
        try:
            return self._stage(*args, **stage_kwargs)  # type: ignore[operator]
        finally:
            _synchronize_cuda(self._torch)
            self.stage_seconds += time.perf_counter() - start


def _synchronize_cuda(torch_module: object) -> None:
    cuda = getattr(torch_module, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    synchronize = getattr(cuda, "synchronize", None)
    if callable(is_available) and is_available() and callable(synchronize):
        synchronize()


def _safe_prompt_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-.")
    if not normalized:
        raise ValueError(f"invalid empty prompt id after normalization: {value!r}")
    return normalized[:80]


def load_prompt_cases(
    *,
    prompt: str | None,
    prompt_id: str,
    prompts_csv: str | Path | None,
) -> list[PromptCase]:
    """Load one inline prompt or a small prompt CSV.

    The CSV accepts ``prompt`` or ``text_prompt`` plus ``prompt_id`` or ``id``.
    """

    if (prompt is None) == (prompts_csv is None):
        raise ValueError("provide exactly one of --prompt or --prompts-csv")
    if prompt is not None:
        text = prompt.strip()
        if not text:
            raise ValueError("--prompt must not be empty")
        return [PromptCase(_safe_prompt_id(prompt_id), text)]

    path = Path(prompts_csv)  # type: ignore[arg-type]
    if not path.is_file():
        raise FileNotFoundError(f"prompt CSV not found: {path}")
    cases: list[PromptCase] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        id_field = "prompt_id" if "prompt_id" in fields else "id" if "id" in fields else None
        prompt_field = "prompt" if "prompt" in fields else "text_prompt" if "text_prompt" in fields else None
        if prompt_field is None or id_field is None:
            raise ValueError(
                "prompt CSV requires prompt (or text_prompt) and prompt_id (or id) columns"
            )
        for line_number, row in enumerate(reader, start=2):
            case_id = _safe_prompt_id(row.get(id_field, ""))
            text = row.get(prompt_field, "").strip()
            if not text:
                raise ValueError(f"empty prompt at {path}:{line_number}")
            if case_id in seen:
                raise ValueError(f"duplicate prompt id {case_id!r} at {path}:{line_number}")
            seen.add(case_id)
            cases.append(PromptCase(case_id, text))
    if not cases:
        raise ValueError(f"prompt CSV has no data rows: {path}")
    return cases


def append_result(path: str | Path, record: dict[str, Any]) -> None:
    """Append one lightweight result row to JSONL or CSV."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix.lower()
    if suffix == ".jsonl":
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return
    if suffix == ".csv":
        exists = target.exists() and target.stat().st_size > 0
        row = dict(record)
        metrics = row.get("attention_metrics")
        if not isinstance(metrics, str):
            row["attention_metrics"] = json.dumps(metrics, ensure_ascii=False, sort_keys=True)
        with target.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow({field: row.get(field) for field in RESULT_FIELDS})
        return
    raise ValueError("--results must end in .jsonl or .csv")


def _validate_runtime_paths(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    gemma_root = Path(args.gemma_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"LTX checkpoint not found: {checkpoint}")
    if not gemma_root.is_dir():
        raise FileNotFoundError(
            "Gemma root not found; accept the official Google model terms and "
            f"download it first: {gemma_root}"
        )
    if args.pipeline == "distilled":
        if not args.spatial_upsampler:
            raise ValueError("--spatial-upsampler is required for the distilled pipeline")
        if not Path(args.spatial_upsampler).is_file():
            raise FileNotFoundError(f"spatial upsampler not found: {args.spatial_upsampler}")
    if args.height <= 0 or args.width <= 0:
        raise ValueError("height and width must be positive")
    if args.num_frames <= 0 or (args.num_frames - 1) % 8:
        raise ValueError("num-frames must equal 8*K+1")
    if args.frame_rate <= 0:
        raise ValueError("frame-rate must be positive")
    if args.steps <= 0:
        raise ValueError("steps must be positive")


def _load_official_runtime(args: argparse.Namespace) -> tuple[object, object, int, str]:
    """Instantiate one official pipeline and return it with torch and metadata."""

    import torch
    from ltx_pipelines.utils.types import OffloadMode

    offload_mode = OffloadMode(args.offload)
    if args.pipeline == "distilled":
        from ltx_pipelines.distilled import DistilledPipeline

        pipeline = DistilledPipeline(
            distilled_checkpoint_path=args.checkpoint,
            gemma_root=args.gemma_root,
            spatial_upsampler_path=args.spatial_upsampler,
            loras=(),
            offload_mode=offload_mode,
        )
        return pipeline, torch, DISTILLED_STEPS, "ltx-2.3-distilled"

    from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline

    pipeline = TI2VidOneStagePipeline(
        checkpoint_path=args.checkpoint,
        gemma_root=args.gemma_root,
        loras=(),
        offload_mode=offload_mode,
    )
    return pipeline, torch, args.steps, "ltx-2.3-dev"


def _install_attention_backend(
    args: argparse.Namespace, pipeline: object
) -> tuple[object | None, int | None]:
    if args.method == "dense":
        return None, None

    try:
        from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
    except ImportError as exc:
        raise RuntimeError(
            "SpargeAttn is not installed in the LTX environment; run "
            "scripts/install_ltx2_sparge_attn.sh"
        ) from exc

    from ltx2.video_attention import (
        SpargeVideoSelfAttentionBackend,
        with_ltx2_video_self_attention,
    )

    backend = SpargeVideoSelfAttentionBackend(
        kernel=spas_sage2_attn_meansim_topk_cuda,
        topk=args.topk,
        pvthreshd=args.pvthreshd,
        fallback_to_dense=args.allow_dense_fallback,
    )
    stage = getattr(pipeline, "stage", None)
    if stage is None:
        raise RuntimeError("official LTX pipeline no longer exposes stage")
    builder = getattr(stage, "_transformer_builder", None)
    model_config = getattr(builder, "model_config", None)
    if not callable(model_config):
        raise RuntimeError("official LTX stage no longer exposes transformer model_config")
    try:
        expected_blocks = int(model_config()["transformer"]["num_layers"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("could not resolve official LTX transformer block count") from exc
    if expected_blocks <= 0:
        raise RuntimeError(f"invalid official LTX transformer block count: {expected_blocks}")
    pipeline.stage = with_ltx2_video_self_attention(stage, backend)  # type: ignore[attr-defined]
    return backend, expected_blocks


def validate_sparse_metrics(metrics: dict[str, Any], expected_blocks: int) -> None:
    """Reject a nominal Sparge run unless sparse attention actually covered every block."""

    if metrics.get("errors") != 0:
        raise RuntimeError(f"Sparge backend reported errors: {metrics.get('errors')}")
    if not isinstance(metrics.get("sparse_calls"), int) or metrics["sparse_calls"] <= 0:
        raise RuntimeError("Sparge backend made no sparse calls")
    if metrics.get("fallback_count") != 0:
        raise RuntimeError(
            "Sparge benchmark used dense fallback and is therefore invalid: "
            f"{metrics.get('fallback_reasons')}"
        )
    calls_by_block = metrics.get("calls_by_block")
    if not isinstance(calls_by_block, dict):
        raise RuntimeError("Sparge metrics lack calls_by_block")
    try:
        covered = {
            int(block): int(count)
            for block, count in calls_by_block.items()
            if block != "unbound" and int(count) > 0
        }
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Sparge calls_by_block is malformed") from exc
    expected = set(range(expected_blocks))
    if set(covered) != expected:
        missing = sorted(expected - set(covered))
        unexpected = sorted(set(covered) - expected)
        raise RuntimeError(
            "Sparge did not cover every video self-attention block: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _reset_cuda_metrics(torch_module: object) -> None:
    cuda = getattr(torch_module, "cuda")
    if cuda.is_available():
        cuda.synchronize()
        cuda.reset_peak_memory_stats()


def _peak_cuda_metrics(torch_module: object) -> tuple[int | None, int | None]:
    cuda = getattr(torch_module, "cuda")
    if not cuda.is_available():
        return None, None
    cuda.synchronize()
    return int(cuda.max_memory_allocated()), int(cuda.max_memory_reserved())


def _invoke_official_pipeline(
    args: argparse.Namespace,
    pipeline: object,
    torch_module: object,
    case: PromptCase,
) -> tuple[object, object, float, int]:
    """Run the official pipeline, leaving lazy video decode to the encoder."""

    _synchronize_cuda(torch_module)
    pipeline_start = time.perf_counter()
    if args.pipeline == "distilled":
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number

        tiling_config = TilingConfig.default()
        video, audio = pipeline(  # type: ignore[operator]
            prompt=case.prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            images=[],
            tiling_config=tiling_config,
            enhance_prompt=False,
        )
        chunks = get_video_chunks_number(args.num_frames, tiling_config)
    else:
        from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, LTX_2_3_PARAMS

        negative_prompt = args.negative_prompt or DEFAULT_NEGATIVE_PROMPT
        video, audio = pipeline(  # type: ignore[operator]
            prompt=case.prompt,
            negative_prompt=negative_prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            num_inference_steps=args.steps,
            video_guider_params=LTX_2_3_PARAMS.video_guider_params,
            audio_guider_params=LTX_2_3_PARAMS.audio_guider_params,
            images=[],
            enhance_prompt=False,
            max_batch_size=args.max_batch_size,
        )
        chunks = 1
    _synchronize_cuda(torch_module)
    pipeline_seconds = time.perf_counter() - pipeline_start
    return video, audio, pipeline_seconds, chunks


def validate_audio_output(
    audio: object,
    torch_module: object,
    *,
    num_frames: int,
    frame_rate: float,
) -> dict[str, float]:
    waveform = getattr(audio, "waveform", None)
    sampling_rate = getattr(audio, "sampling_rate", None)
    if waveform is None or not isinstance(sampling_rate, int) or sampling_rate <= 0:
        raise RuntimeError("official LTX pipeline did not return decoded audio")
    shape = tuple(getattr(waveform, "shape", ()))
    if len(shape) != 2 or 2 not in shape or waveform.numel() <= 0:
        raise RuntimeError(f"decoded audio must be non-empty stereo; got shape={shape}")
    if not bool(torch_module.isfinite(waveform).all().item()):
        raise RuntimeError("decoded audio contains NaN or Inf")
    rms = float(waveform.to(dtype=torch_module.float32).square().mean().sqrt().item())
    if rms <= 1e-6:
        raise RuntimeError(f"decoded audio is effectively silent: rms={rms:.3e}")
    samples = max(int(dimension) for dimension in shape)
    audio_duration = samples / sampling_rate
    expected_duration = num_frames / frame_rate
    tolerance = max(0.25, 2.0 / frame_rate)
    if abs(audio_duration - expected_duration) > tolerance:
        raise RuntimeError(
            "decoded audio duration differs from requested video duration: "
            f"audio={audio_duration:.3f}s, requested={expected_duration:.3f}s"
        )
    return {
        "audio_rms": rms,
        "audio_duration_seconds": audio_duration,
    }


def _validated_video_chunks(
    video: object,
    torch_module: object,
    stats: dict[str, int],
) -> Iterable[object]:
    chunks = iter((video,)) if isinstance(video, torch_module.Tensor) else iter(video)  # type: ignore[arg-type]
    for chunk in chunks:
        shape = tuple(getattr(chunk, "shape", ()))
        if len(shape) != 4 or int(shape[0]) <= 0:
            raise RuntimeError(f"decoded video chunk must be [F,H,W,C]; got shape={shape}")
        if not bool(torch_module.isfinite(chunk).all().item()):
            raise RuntimeError("decoded video contains NaN or Inf")
        stats["chunks"] += 1
        stats["frames"] += int(shape[0])
        yield chunk


def close_video_output(video: object | None) -> None:
    """Close an official lazy decoder iterator after success or failure."""

    if video is None:
        return
    close = getattr(video, "close", None)
    if callable(close):
        close()


def inspect_encoded_media(output_path: Path, frame_rate: float) -> dict[str, Any]:
    import av

    def duration(stream: object) -> float | None:
        value = getattr(stream, "duration", None)
        time_base = getattr(stream, "time_base", None)
        if value is None or time_base is None:
            return None
        return float(value * time_base)

    with av.open(str(output_path), mode="r") as container:
        video_streams = list(container.streams.video)
        audio_streams = list(container.streams.audio)
        video_duration = duration(video_streams[0]) if video_streams else None
        audio_duration = duration(audio_streams[0]) if audio_streams else None
    if len(video_streams) != 1 or len(audio_streams) != 1:
        raise RuntimeError(
            "encoded MP4 must contain exactly one video and one audio stream: "
            f"video={len(video_streams)}, audio={len(audio_streams)}"
        )
    if video_duration is not None and audio_duration is not None:
        if abs(video_duration - audio_duration) > max(0.25, 2.0 / frame_rate):
            raise RuntimeError(
                "encoded video/audio durations differ: "
                f"video={video_duration:.3f}s, audio={audio_duration:.3f}s"
            )
    return {
        "media_video_streams": len(video_streams),
        "media_audio_streams": len(audio_streams),
        "media_video_duration_seconds": video_duration,
        "media_audio_duration_seconds": audio_duration,
    }


def _decode_encode_and_validate(
    args: argparse.Namespace,
    video: object,
    audio: object,
    torch_module: object,
    output_path: Path,
    chunks: int,
) -> tuple[float, dict[str, Any]]:
    from ltx_pipelines.utils.media_io import encode_video

    video_stats = {"chunks": 0, "frames": 0}
    decode_encode_start = time.perf_counter()
    encode_video(
        video=_validated_video_chunks(video, torch_module, video_stats),
        fps=args.frame_rate,
        audio=audio,
        output_path=str(output_path),
        video_chunks_number=chunks,
    )
    decode_encode_seconds = time.perf_counter() - decode_encode_start
    if video_stats["frames"] != args.num_frames:
        raise RuntimeError(
            "decoded video frame count differs from request: "
            f"decoded={video_stats['frames']}, requested={args.num_frames}"
        )
    media = inspect_encoded_media(output_path, args.frame_rate)
    media["decoded_video_frames"] = video_stats["frames"]
    return decode_encode_seconds, media


def _output_path(args: argparse.Namespace, case: PromptCase, steps: int) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_tag = args.pipeline.replace("-", "_")
    return output_dir / (
        f"{case.prompt_id}_{pipeline_tag}_{args.method}_seed{args.seed}_steps{steps}.mp4"
    )


def _base_record(
    args: argparse.Namespace,
    case: PromptCase,
    *,
    model: str,
    steps: int,
    load_seconds: float,
    output_path: Path,
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "pipeline": args.pipeline,
        "method": args.method,
        "prompt_id": case.prompt_id,
        "prompt": case.prompt,
        "seed": args.seed,
        "steps": steps,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "frame_rate": args.frame_rate,
        "load_seconds": round(load_seconds, 6),
        "pipeline_seconds": None,
        "stage_seconds": None,
        "denoise_seconds": None,
        "decode_encode_seconds": None,
        "total_seconds": None,
        "peak_memory_allocated_bytes": None,
        "peak_memory_reserved_bytes": None,
        "audio_rms": None,
        "audio_duration_seconds": None,
        "decoded_video_frames": None,
        "media_video_streams": None,
        "media_audio_streams": None,
        "media_video_duration_seconds": None,
        "media_audio_duration_seconds": None,
        "output_path": str(output_path),
        "status": "failed",
        "error": None,
        "attention_metrics": None,
    }


def run(args: argparse.Namespace) -> int:
    cases = load_prompt_cases(
        prompt=args.prompt,
        prompt_id=args.prompt_id,
        prompts_csv=args.prompts_csv,
    )
    results_path = Path(args.results) if args.results else Path(args.output_dir) / "results.jsonl"
    _validate_runtime_paths(args)

    load_start = time.perf_counter()
    try:
        pipeline, torch_module, steps, model = _load_official_runtime(args)
        backend, expected_blocks = _install_attention_backend(args, pipeline)
        from ltx_pipelines.utils.samplers import euler_denoising_loop
        from ltx_pipelines.utils.helpers import cleanup_memory

        timer = TimedDiffusionStage(  # type: ignore[attr-defined]
            pipeline.stage, torch_module, euler_denoising_loop
        )
        pipeline.stage = timer  # type: ignore[attr-defined]
        load_seconds = time.perf_counter() - load_start
    except Exception as exc:
        load_seconds = time.perf_counter() - load_start
        for case in cases:
            steps = DISTILLED_STEPS if args.pipeline == "distilled" else args.steps
            model = "ltx-2.3-distilled" if args.pipeline == "distilled" else "ltx-2.3-dev"
            record = _base_record(
                args,
                case,
                model=model,
                steps=steps,
                load_seconds=load_seconds,
                output_path=_output_path(args, case, steps),
            )
            record["error"] = f"{type(exc).__name__}: {exc}"
            append_result(results_path, record)
        raise

    failures = 0
    for case in cases:
        video = None
        audio = None
        output_path = _output_path(args, case, steps)
        record = _base_record(
            args,
            case,
            model=model,
            steps=steps,
            load_seconds=load_seconds,
            output_path=output_path,
        )
        try:
            timer.reset()
            if backend is not None:
                backend.reset_metrics()
            if output_path.exists():
                raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
            _reset_cuda_metrics(torch_module)
            with torch_module.inference_mode():
                video, audio, pipeline_seconds, chunks = _invoke_official_pipeline(
                    args, pipeline, torch_module, case
                )
                record["pipeline_seconds"] = round(pipeline_seconds, 6)
                record["stage_seconds"] = round(timer.stage_seconds, 6)
                record["denoise_seconds"] = round(timer.denoise_seconds, 6)
                attention_metrics = (
                    backend.metrics()
                    if backend is not None
                    else {"method": "dense", "backend": "official_dense"}
                )
                if backend is not None:
                    assert expected_blocks is not None
                    validate_sparse_metrics(attention_metrics, expected_blocks)
                audio_metrics = validate_audio_output(
                    audio,
                    torch_module,
                    num_frames=args.num_frames,
                    frame_rate=args.frame_rate,
                )
                record.update(audio_metrics)
                decode_encode_seconds, media_metrics = _decode_encode_and_validate(
                    args,
                    video,
                    audio,
                    torch_module,
                    output_path,
                    chunks,
                )
            peak_allocated, peak_reserved = _peak_cuda_metrics(torch_module)
            if not output_path.is_file() or output_path.stat().st_size <= 0:
                raise RuntimeError(f"official encoder did not produce a non-empty MP4: {output_path}")
            record.update(
                {
                    "pipeline_seconds": round(pipeline_seconds, 6),
                    "stage_seconds": round(timer.stage_seconds, 6),
                    "denoise_seconds": round(timer.denoise_seconds, 6),
                    "decode_encode_seconds": round(decode_encode_seconds, 6),
                    "total_seconds": round(pipeline_seconds + decode_encode_seconds, 6),
                    "peak_memory_allocated_bytes": peak_allocated,
                    "peak_memory_reserved_bytes": peak_reserved,
                    "status": "ok",
                    "attention_metrics": attention_metrics,
                    **audio_metrics,
                    **media_metrics,
                }
            )
        except Exception as exc:
            failures += 1
            record["stage_seconds"] = round(timer.stage_seconds, 6)
            record["denoise_seconds"] = round(timer.denoise_seconds, 6)
            record["attention_metrics"] = (
                backend.metrics() if backend is not None else {"method": "dense", "backend": "official_dense"}
            )
            record["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            cleanup_needed = record["status"] != "ok"
            release_errors: list[str] = []
            try:
                close_video_output(video)
            except Exception as exc:
                cleanup_needed = True
                release_errors.append(f"video close failed: {type(exc).__name__}: {exc}")
            video = None
            audio = None
            if cleanup_needed:
                try:
                    cleanup_memory()
                except Exception as exc:
                    release_errors.append(f"memory cleanup failed: {type(exc).__name__}: {exc}")
            if release_errors:
                if record["status"] == "ok":
                    failures += 1
                    record["status"] = "failed"
                suffix = "; ".join(release_errors)
                record["error"] = f"{record['error']}; {suffix}" if record["error"] else suffix
        append_result(results_path, record)
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)

    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run official LTX-2.3 dense or video-self-attention Sparge experiments."
    )
    parser.add_argument("--pipeline", choices=("distilled", "one-stage"), default="distilled")
    parser.add_argument("--method", choices=("dense", "sparge"), default="dense")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--spatial-upsampler")
    parser.add_argument("--gemma-root", required=True)
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompts-csv")
    parser.add_argument("--prompt-id", default="smoke")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--results", help="Append results to .jsonl or .csv (default: OUTPUT_DIR/results.jsonl)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--frame-rate", type=float, default=DEFAULT_FRAME_RATE)
    parser.add_argument("--steps", type=int, default=30, help="One-stage denoising steps; distilled uses fixed 11")
    parser.add_argument("--negative-prompt")
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--offload", choices=("none", "cpu", "disk"), default="none")
    parser.add_argument("--topk", type=float, default=0.5)
    parser.add_argument("--pvthreshd", type=float, default=50.0)
    parser.add_argument(
        "--allow-dense-fallback",
        action="store_true",
        help="Explicitly allow counted dense fallback for unsupported Sparge input shapes",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        return run(args)
    except Exception as exc:
        print(f"LTX experiment failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
