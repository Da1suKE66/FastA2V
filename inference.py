import os
import sys
import hashlib
import importlib.metadata
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from ovi.utils.io_utils import save_video
from ovi.utils.processing_utils import format_prompt_for_filename, validate_and_process_user_prompt
from ovi.utils.utils import get_arguments
from ovi.distributed_comms.util import get_world_size, get_local_rank, get_global_rank
from ovi.distributed_comms.parallel_states import initialize_sequence_parallel_state, get_sequence_parallel_state, nccl_info
from ovi.ovi_fusion_engine import OviFusionEngine
from ovi.gpu_process_monitor import GpuProcessMonitor


def _command_output(command):
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _collect_environment(config, config_file, engine_load_seconds, prompt_count):
    driver_version = _command_output([
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    ])
    git_commit = _command_output(["git", "rev-parse", "HEAD"])
    git_status = _command_output(["git", "status", "--porcelain"])
    evidence_files = {}
    output_dir = os.path.abspath(config.get("output_dir"))
    evidence_filenames = [
        "preflight.json",
        "environment.freeze.txt",
        "checkpoint_manifest.json",
    ]
    if config.get("attention_method", "dense") == "sparge":
        evidence_filenames.append("spargeattn-install.json")
    for filename in evidence_filenames:
        path = os.path.join(output_dir, filename)
        evidence_files[filename] = _sha256(path) if os.path.isfile(path) else None

    measurement_runs = int(config.get("measurement_runs", 1))
    each_example_n_times = int(config.get("each_example_n_times", 1))
    return {
        "config_file": os.path.abspath(config_file),
        "git_commit": git_commit,
        "git_dirty": bool(git_status),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "flash_attn": _package_version("flash-attn"),
        "spas_sage_attn": _package_version("spas_sage_attn"),
        "transformers": _package_version("transformers"),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "driver_version": driver_version.splitlines()[0] if driver_version else None,
        "engine_load_seconds": engine_load_seconds,
        "model_name": config.get("model_name"),
        "sp_size": int(config.get("sp_size", 1)),
        "attention_method": config.get("attention_method", "dense"),
        "sparge_topk": float(config.get("sparge_topk", 0.5)),
        "sparge_pvthreshd": float(config.get("sparge_pvthreshd", 50)),
        "sparge_smooth_k": bool(config.get("sparge_smooth_k", True)),
        "use_cfg_cache": bool(config.get("use_cfg_cache", False)),
        "cfg_cache_start_step": int(config.get("cfg_cache_start_step", 10)),
        "cfg_cache_end_step": int(config.get("cfg_cache_end_step", 39)),
        "cfg_cache_window_inclusive": True,
        "cfg_cache_refresh_interval": int(
            config.get("cfg_cache_refresh_interval", 5)
        ),
        "use_block_cache": bool(config.get("use_block_cache", False)),
        "block_cache_start_block": int(
            config.get("block_cache_start_block", 10)
        ),
        "block_cache_end_block": int(
            config.get("block_cache_end_block", 19)
        ),
        "block_cache_window_inclusive": True,
        "block_cache_policy": str(
            config.get("block_cache_policy", "fixed")
        ).lower(),
        "block_cache_cosine_threshold": float(
            config.get("block_cache_cosine_threshold", 0.95)
        ),
        "block_cache_max_consecutive_reuses": int(
            config.get("block_cache_max_consecutive_reuses", 1)
        ),
        "debug_forward": bool(config.get("debug_forward", False)),
        "run_kind": config.get("run_kind", "unspecified"),
        "benchmark_eligible": bool(config.get("benchmark_eligible", False)),
        "gpu_process_monitor_interval_seconds": float(
            config.get("gpu_process_monitor_interval_seconds", 5.0)
        ),
        "warmup_runs": int(config.get("warmup_runs", 0)),
        "measurement_runs": int(config.get("measurement_runs", 1)),
        "each_example_n_times": each_example_n_times,
        "prompt_count": prompt_count,
        "expected_warmup_records": int(config.get("warmup_runs", 0)) if prompt_count else 0,
        "expected_measurement_records": measurement_runs * each_example_n_times * prompt_count,
        "run_id": os.path.basename(os.path.abspath(config.get("output_dir"))),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_home": os.environ.get("CUDA_HOME"),
        "hf_endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        "evidence_file_sha256": evidence_files,
    }


def _append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _prepare_output_dir(config):
    output_dir = os.path.abspath(config.get("output_dir", "./outputs"))
    run_kind = str(config.get("run_kind", "unspecified"))
    if run_kind != "unspecified":
        selected_dir = os.environ.get("FASTA2V_RUN_DIR")
        if not selected_dir or os.path.abspath(selected_dir) != output_dir:
            raise RuntimeError(
                "FASTA2V_RUN_DIR must select the resolved output_dir for a "
                f"{run_kind!r} run; use the repository run script."
            )
        allowed_pre_run_files = {
            "checkpoint_manifest.json",
            "environment.freeze.txt",
            "preflight.json",
            "stdout.log",
        }
        if config.get("attention_method", "dense") == "sparge":
            allowed_pre_run_files.add("spargeattn-install.json")
        unexpected = sorted(
            name for name in os.listdir(output_dir)
            if name not in allowed_pre_run_files
        )
        if unexpected:
            raise RuntimeError(
                f"run directory is not fresh: {output_dir}; unexpected={unexpected}"
            )
    else:
        os.makedirs(output_dir, exist_ok=True)
    return output_dir



def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def main(config, args): 

    world_size = get_world_size()
    global_rank = get_global_rank()
    local_rank = get_local_rank()
    device = local_rank
    torch.cuda.set_device(local_rank)
    sp_size = config.get("sp_size", 1)
    assert sp_size <= world_size and world_size % sp_size == 0, "sp_size must be less than or equal to world_size and world_size must be divisible by sp_size."

    _init_logging(global_rank)

    if world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=global_rank,
            world_size=world_size)
    else:
        assert sp_size == 1, f"When world_size is 1, sp_size must also be 1, but got {sp_size}."
        ## TODO: assert not sharding t5 etc...


    initialize_sequence_parallel_state(sp_size)
    logging.info(f"Using SP: {get_sequence_parallel_state()}, SP_SIZE: {sp_size}")
    
    args.local_rank = local_rank
    args.device = device
    target_dtype = torch.bfloat16

    # validate inputs before loading model to not waste time if input is not valid
    text_prompt = config.get("text_prompt")
    image_path = config.get("image_path", None)
    assert config.get("mode") in ["t2v", "i2v", "t2i2v"], f"Invalid mode {config.get('mode')}, must be one of ['t2v', 'i2v', 't2i2v']"
    text_prompts, image_paths = validate_and_process_user_prompt(text_prompt, image_path, mode=config.get("mode"))
    if config.get("mode") != "i2v":
        logging.info(f"mode: {config.get('mode')}, setting all image_paths to None")
        image_paths = [None] * len(text_prompts)
    else:
        assert all(p is not None and os.path.isfile(p) for p in image_paths), f"In i2v mode, all image paths must be provided.{image_paths}"

    output_dir = _prepare_output_dir(config)

    logging.info("Loading OVI Fusion Engine...")
    torch.cuda.synchronize(device)
    engine_load_started = time.perf_counter()
    ovi_engine = OviFusionEngine(config=config, device=device, target_dtype=target_dtype)
    torch.cuda.synchronize(device)
    engine_load_seconds = time.perf_counter() - engine_load_started
    logging.info("OVI Fusion Engine loaded in %.3f seconds!", engine_load_seconds)
    
    run_config_path = os.path.join(output_dir, "run_config.yaml")
    OmegaConf.save(config=config, f=run_config_path, resolve=True)
    _write_json(
        os.path.join(output_dir, "environment.json"),
        {
            **_collect_environment(
                config,
                args.config_file,
                engine_load_seconds,
                prompt_count=len(text_prompts),
            ),
            "run_config_sha256": _sha256(run_config_path),
        },
    )

    # Load CSV data
    all_eval_data = list(zip(text_prompts, image_paths))

    # Get SP configuration
    use_sp = get_sequence_parallel_state()
    if use_sp:
        sp_size = nccl_info.sp_size
        sp_rank = nccl_info.rank_within_group
        sp_group_id = global_rank // sp_size
        num_sp_groups = world_size // sp_size
    else:
        # No SP: treat each GPU as its own group
        sp_size = 1
        sp_rank = 0
        sp_group_id = global_rank
        num_sp_groups = world_size

    # Data distribution - by SP groups
    total_files = len(all_eval_data)

    require_sample_padding = False
    
    if total_files == 0:
        logging.error(f"ERROR: No evaluation files found")
        this_rank_eval_data = []
    else:
        # Pad to match number of SP groups
        remainder = total_files % num_sp_groups
        if require_sample_padding and remainder != 0:
            pad_count = num_sp_groups - remainder
            all_eval_data += [all_eval_data[0]] * pad_count
        
        # Distribute across SP groups
        this_rank_eval_data = all_eval_data[sp_group_id :: num_sp_groups]

    video_frame_height_width = config.get("video_frame_height_width", None)
    seed = int(config.get("seed", 100))
    generation_kwargs = {
        "video_frame_height_width": video_frame_height_width,
        "solver_name": config.get("solver_name", "unipc"),
        "sample_steps": int(config.get("sample_steps", 50)),
        "shift": float(config.get("shift", 5.0)),
        "video_guidance_scale": float(config.get("video_guidance_scale", 4.0)),
        "audio_guidance_scale": float(config.get("audio_guidance_scale", 3.0)),
        "slg_layer": int(config.get("slg_layer", 11)),
        "video_negative_prompt": config.get("video_negative_prompt", ""),
        "audio_negative_prompt": config.get("audio_negative_prompt", ""),
    }
    run_id = os.path.basename(os.path.abspath(output_dir))
    gpu_monitor_interval = float(
        config.get("gpu_process_monitor_interval_seconds", 5.0)
    )

    def run_one(text_prompt, image_path, sample_seed):
        monitor = GpuProcessMonitor(
            device_index=device,
            interval_seconds=gpu_monitor_interval,
        )
        try:
            with monitor:
                return ovi_engine.generate(
                    text_prompt=text_prompt,
                    image_path=image_path,
                    seed=sample_seed,
                    **generation_kwargs,
                )
        finally:
            ovi_engine.last_run_metrics["gpu_process_monitor"] = monitor.summary()

    def record_failure(phase, text_prompt, sample_seed, repeat_index):
        failure = dict(ovi_engine.last_run_metrics)
        failure.update({
            "phase": phase,
            "prompt": text_prompt,
            "seed": sample_seed,
            "repeat_index": repeat_index,
            "rank": global_rank,
            "run_id": run_id,
        })
        if sp_rank == 0:
            failure_path = os.path.join(
                output_dir,
                f"failed_{phase}_{time.time_ns()}.metrics.json",
            )
            _write_json(failure_path, failure)
            _append_jsonl(os.path.join(output_dir, "failures.jsonl"), failure)

    warmup_runs = int(config.get("warmup_runs", 0))
    measurement_runs = int(config.get("measurement_runs", 1))
    if warmup_runs < 0 or measurement_runs < 1:
        raise ValueError("warmup_runs must be >= 0 and measurement_runs must be >= 1")

    if warmup_runs and this_rank_eval_data:
        warmup_prompt, warmup_image = this_rank_eval_data[0]
        for warmup_index in range(warmup_runs):
            logging.info("Warm-up run %s/%s (excluded from benchmark)", warmup_index + 1, warmup_runs)
            warmup_result = run_one(warmup_prompt, warmup_image, seed)
            if warmup_result is None:
                record_failure("warmup", warmup_prompt, seed, warmup_index)
                raise RuntimeError(
                    f"Ovi warm-up failed: {ovi_engine.last_run_metrics.get('error', 'unknown error')}"
                )
            warmup_metrics = dict(ovi_engine.last_run_metrics)
            warmup_metrics.update({
                "record_type": "warmup",
                "benchmark_valid": False,
                "warmup_index": warmup_index,
                "prompt": warmup_prompt,
                "seed": seed,
                "rank": global_rank,
                "run_id": run_id,
            })
            if sp_rank == 0:
                _append_jsonl(os.path.join(output_dir, "warmup_timings.jsonl"), warmup_metrics)
            del warmup_result

    for measurement_index in range(measurement_runs):
        for prompt_index, (text_prompt, image_path) in tqdm(
            enumerate(this_rank_eval_data),
            total=len(this_rank_eval_data),
            desc=f"measurement {measurement_index + 1}/{measurement_runs}",
        ):
            for sample_index in range(int(config.get("each_example_n_times", 1))):
                sample_seed = seed + sample_index
                sample_started = time.perf_counter()
                generation_result = run_one(text_prompt, image_path, sample_seed)
                if generation_result is None:
                    record_failure("measurement", text_prompt, sample_seed, measurement_index)
                    raise RuntimeError(
                        f"Ovi generation failed: {ovi_engine.last_run_metrics.get('error', 'unknown error')}"
                    )
                generated_video, generated_audio, generated_image = generation_result

                if sp_rank == 0:
                    metrics = dict(ovi_engine.last_run_metrics)
                    actual_hw = metrics.get(
                        "actual_video_frame_height_width",
                        list(generated_video.shape[-2:]),
                    )
                    formatted_prompt = format_prompt_for_filename(text_prompt)
                    output_path = os.path.join(
                        output_dir,
                        f"p{prompt_index:03d}_{formatted_prompt}_actual{'x'.join(map(str, actual_hw))}_"
                        f"seed{sample_seed}_rank{global_rank}_rep{measurement_index:02d}.mp4",
                    )
                    save_started = time.perf_counter()
                    save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)
                    if generated_image is not None:
                        generated_image.save(output_path.replace('.mp4', '.png'))
                    save_seconds = time.perf_counter() - save_started
                    artifact_ready_seconds = time.perf_counter() - sample_started

                    hash_started = time.perf_counter()
                    output_sha256 = _sha256(output_path)
                    hash_seconds = time.perf_counter() - hash_started
                    metrics.update({
                        "record_type": "measurement",
                        "output_path": os.path.abspath(output_path),
                        "output_sha256": output_sha256,
                        "prompt": text_prompt,
                        "seed": sample_seed,
                        "rank": global_rank,
                        "run_id": run_id,
                        "prompt_index": prompt_index,
                        "sample_index": sample_index,
                        "measurement_index": measurement_index,
                        "save_video_seconds": save_seconds,
                        "artifact_ready_seconds": artifact_ready_seconds,
                        "output_hash_seconds": hash_seconds,
                    })
                    metrics_path = output_path.replace(".mp4", ".metrics.json")
                    _write_json(metrics_path, metrics)
                    _append_jsonl(os.path.join(output_dir, "timings.jsonl"), metrics)
        


if __name__ == "__main__":
    args = get_arguments()
    config = OmegaConf.load(args.config_file)
    main(config=config,args=args)
