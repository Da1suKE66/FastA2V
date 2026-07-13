import os
import sys
import hashlib
import importlib.metadata
import json
import logging
import subprocess
import time
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from ovi.utils.io_utils import save_video
from ovi.utils.processing_utils import format_prompt_for_filename, validate_and_process_user_prompt
from ovi.utils.utils import get_arguments
from ovi.distributed_comms.util import get_world_size, get_local_rank, get_global_rank
from ovi.distributed_comms.parallel_states import initialize_sequence_parallel_state, get_sequence_parallel_state, nccl_info
from ovi.ovi_fusion_engine import OviFusionEngine


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


def _collect_environment(config, config_file, engine_load_seconds):
    driver_version = _command_output([
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    ])
    git_commit = _command_output(["git", "rev-parse", "HEAD"])
    git_status = _command_output(["git", "status", "--porcelain"])
    return {
        "config_file": os.path.abspath(config_file),
        "git_commit": git_commit,
        "git_dirty": bool(git_status),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "flash_attn": _package_version("flash-attn"),
        "transformers": _package_version("transformers"),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "driver_version": driver_version.splitlines()[0] if driver_version else None,
        "engine_load_seconds": engine_load_seconds,
        "model_name": config.get("model_name"),
        "attention_method": config.get("attention_method", "dense"),
        "use_cfg_cache": bool(config.get("use_cfg_cache", False)),
        "use_block_cache": bool(config.get("use_block_cache", False)),
    }



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

    logging.info("Loading OVI Fusion Engine...")
    torch.cuda.synchronize(device)
    engine_load_started = time.perf_counter()
    ovi_engine = OviFusionEngine(config=config, device=device, target_dtype=target_dtype)
    torch.cuda.synchronize(device)
    engine_load_seconds = time.perf_counter() - engine_load_started
    logging.info("OVI Fusion Engine loaded in %.3f seconds!", engine_load_seconds)
    
    output_dir = config.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)
    OmegaConf.save(config=config, f=os.path.join(output_dir, "run_config.yaml"))
    _write_json(
        os.path.join(output_dir, "environment.json"),
        _collect_environment(config, args.config_file, engine_load_seconds),
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

    for _, (text_prompt, image_path) in tqdm(enumerate(this_rank_eval_data)):
        video_frame_height_width = config.get("video_frame_height_width", None)
        seed = config.get("seed", 100)
        solver_name = config.get("solver_name", "unipc")
        sample_steps = config.get("sample_steps", 50)
        shift = config.get("shift", 5.0)
        video_guidance_scale = config.get("video_guidance_scale", 4.0)
        audio_guidance_scale = config.get("audio_guidance_scale", 3.0)
        slg_layer = config.get("slg_layer", 11)
        video_negative_prompt = config.get("video_negative_prompt", "")
        audio_negative_prompt = config.get("audio_negative_prompt", "")
        for idx in range(config.get("each_example_n_times", 1)):
            generation_result = ovi_engine.generate(text_prompt=text_prompt,
                                                     image_path=image_path,
                                                     video_frame_height_width=video_frame_height_width,
                                                     seed=seed+idx,
                                                     solver_name=solver_name,
                                                     sample_steps=sample_steps,
                                                     shift=shift,
                                                     video_guidance_scale=video_guidance_scale,
                                                     audio_guidance_scale=audio_guidance_scale,
                                                     slg_layer=slg_layer,
                                                     video_negative_prompt=video_negative_prompt,
                                                     audio_negative_prompt=audio_negative_prompt)
            if generation_result is None:
                raise RuntimeError(
                    f"Ovi generation failed: {ovi_engine.last_run_metrics.get('error', 'unknown error')}"
                )
            generated_video, generated_audio, generated_image = generation_result
            
            if sp_rank == 0:
                formatted_prompt = format_prompt_for_filename(text_prompt)
                output_path = os.path.join(output_dir, f"{formatted_prompt}_{'x'.join(map(str, video_frame_height_width))}_{seed+idx}_{global_rank}.mp4")
                save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)
                if generated_image is not None:
                    generated_image.save(output_path.replace('.mp4', '.png'))

                metrics = dict(ovi_engine.last_run_metrics)
                metrics.update({
                    "output_path": os.path.abspath(output_path),
                    "output_sha256": _sha256(output_path),
                    "prompt": text_prompt,
                    "seed": seed + idx,
                    "rank": global_rank,
                })
                metrics_path = output_path.replace(".mp4", ".metrics.json")
                _write_json(metrics_path, metrics)
                with open(os.path.join(output_dir, "timings.jsonl"), "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n")
        


if __name__ == "__main__":
    args = get_arguments()
    config = OmegaConf.load(args.config_file)
    main(config=config,args=args)
