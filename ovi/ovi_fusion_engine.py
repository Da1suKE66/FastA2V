import os
import sys
import uuid
import cv2
import glob
import torch
import logging
import time
from textwrap import indent
import torch.nn as nn
from diffusers import FluxPipeline
from tqdm import tqdm
from ovi.distributed_comms.parallel_states import get_sequence_parallel_state, nccl_info
from ovi.utils.model_loading_utils import init_fusion_score_model_ovi, init_text_model, init_mmaudio_vae, init_wan_vae_2_2, load_fusion_checkpoint
from ovi.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from diffusers import FlowMatchEulerDiscreteScheduler
from ovi.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
import traceback
from omegaconf import OmegaConf
from ovi.utils.processing_utils import clean_text, preprocess_image_tensor, snap_hw_to_multiple_of_32, scale_hw_to_area_divisible
import re
from optimum.quanto import freeze, qint8, quantize
from ovi.cfg_cache import (
    CfgNegativeCache,
    expected_cfg_cache_metrics,
    validate_cfg_cache_config,
)
from ovi.block_cache import FusionBlockCache, validate_block_cache_config
from ovi.modules.video_attention_dispatcher import (
    VideoSelfAttentionDispatcher,
    expected_video_self_attention_calls,
)

DEFAULT_CONFIG = OmegaConf.load('ovi/configs/inference/inference_fusion.yaml')

NAME_TO_MODEL_SPECS_MAP = {
    "720x720_5s": {
        "path": "model.safetensors",
        "video_latent_length": 31,
        "audio_latent_length": 157,
        "video_area": 720 * 720,
        "formatter": lambda text: re.sub(r"Audio:\s*(.*)", r"<AUDCAP>\1<ENDAUDCAP>", text, flags=re.S)
    },
    "960x960_5s": {
        "path": "model_960x960.safetensors",
        "video_latent_length": 31,
        "audio_latent_length": 157,
        "video_area": 960 * 960,
        "formatter": lambda text: re.sub(r"<AUDCAP>(.*?)<ENDAUDCAP>", r"Audio: \1", text, flags=re.S)
    }, 
    "960x960_10s": {
        "path": "model_960x960_10s.safetensors",
        "video_latent_length": 61,
        "audio_latent_length": 314,
        "video_area": 960 * 960,
        "formatter": lambda text: re.sub(r"<AUDCAP>(.*?)<ENDAUDCAP>", r"Audio: \1", text, flags=re.S)
    }
}


class OviFusionEngine:
    def __init__(self, config=DEFAULT_CONFIG, device=0, target_dtype=torch.bfloat16):
        # Load fusion model
        self.device = device
        self.target_dtype = target_dtype
        self.attention_method = str(config.get("attention_method", "dense")).lower()
        if self.attention_method not in {"dense", "sparge", "radial", "svg"}:
            raise ValueError(
                f"Unsupported attention_method={self.attention_method!r}; "
                "expected one of dense, sparge, radial, svg."
            )
        self.video_self_attention_dispatcher = VideoSelfAttentionDispatcher(
            self.attention_method
        )
        self.use_cfg_cache = bool(config.get("use_cfg_cache", False))
        self.use_block_cache = bool(config.get("use_block_cache", False))
        self.cfg_cache_start_step = int(config.get("cfg_cache_start_step", 10))
        self.cfg_cache_end_step = int(config.get("cfg_cache_end_step", 39))
        self.cfg_cache_refresh_interval = int(
            config.get("cfg_cache_refresh_interval", 5)
        )
        self.block_cache_start_block = int(
            config.get("block_cache_start_block", 10)
        )
        self.block_cache_end_block = int(
            config.get("block_cache_end_block", 19)
        )
        self.block_cache_policy = str(
            config.get("block_cache_policy", "fixed")
        ).lower()
        self.block_cache_cosine_threshold = float(
            config.get("block_cache_cosine_threshold", 0.95)
        )
        self.block_cache_max_consecutive_reuses = int(
            config.get("block_cache_max_consecutive_reuses", 1)
        )
        if self.use_cfg_cache:
            validate_cfg_cache_config(
                self.cfg_cache_start_step,
                self.cfg_cache_end_step,
                self.cfg_cache_refresh_interval,
            )
        if self.use_block_cache:
            validate_block_cache_config(
                self.block_cache_start_block,
                self.block_cache_end_block,
                self.block_cache_policy,
                self.block_cache_cosine_threshold,
                self.block_cache_max_consecutive_reuses,
            )
            if int(config.get("sp_size", 1)) != 1:
                raise NotImplementedError(
                    "The first block-cache implementation is limited to "
                    "sp_size=1 so every reuse decision has one owner."
                )
        self.debug_forward = bool(config.get("debug_forward", False))
        self.debug_forward_step = int(config.get("debug_forward_step", 0))
        self.run_kind = str(config.get("run_kind", "unspecified"))
        self.benchmark_eligible = bool(config.get("benchmark_eligible", False))
        self.last_run_metrics = {}
        meta_init = True
        self.cpu_offload = config.get("cpu_offload", False) or config.get("mode") == "t2i2v"
        if self.cpu_offload:
            logging.info("CPU offloading is enabled. Initializing all models aside from VAEs on CPU")

        model, video_config, audio_config = init_fusion_score_model_ovi(rank=device, meta_init=meta_init)
        if self.use_block_cache:
            validate_block_cache_config(
                self.block_cache_start_block,
                self.block_cache_end_block,
                self.block_cache_policy,
                self.block_cache_cosine_threshold,
                self.block_cache_max_consecutive_reuses,
                num_blocks=model.num_blocks,
            )
        model.set_video_self_attention_dispatcher(
            self.video_self_attention_dispatcher
        )

        fp8 = config.get("fp8", False)
        int8 = config.get("qint8", False)
        if fp8:
            assert not config.get("mode") == "t2i2v", "Image generation with FluxPipeline is not supported with fp8 quantization. This is because if you are unable to run the bf16 model, you likely cannot run image gen model"

        if not meta_init:
            if not fp8:
                model = model.to(dtype=target_dtype)
            model = (
                model.to(device=device if not self.cpu_offload else "cpu")
                .eval()
            )

        # Load VAEs
        vae_model_video = init_wan_vae_2_2(config.ckpt_dir, rank=device)
        vae_model_video.model.requires_grad_(False).eval()
        vae_model_video.model = vae_model_video.model.bfloat16()
        self.vae_model_video = vae_model_video

        vae_model_audio = init_mmaudio_vae(config.ckpt_dir, rank=device)
        vae_model_audio.requires_grad_(False).eval()
        self.vae_model_audio = vae_model_audio.bfloat16()

        # Load T5 text model
        self.text_model = init_text_model(config.ckpt_dir, rank=device, cpu_offload=self.cpu_offload)
        if config.get("shard_text_model", False):
            raise NotImplementedError("Sharding text model is not implemented yet.")
        if self.cpu_offload:
            self.offload_to_cpu(self.text_model.model)

        # Find fusion ckpt in the same dir used by other components
        model_name = config.get("model_name", "960x960_5s")
        self.model_name = model_name
        assert model_name in NAME_TO_MODEL_SPECS_MAP, f"Model name {model_name} not found in predefined model name to path map."
        model_specs = NAME_TO_MODEL_SPECS_MAP[model_name]
        basename = model_specs["path"]
        if fp8:
            assert model_name == "720x720_5s", "FP8 quantization is only supported for 720x720_5s model currently."
            basename = "model_fp8_e4m3fn.safetensors"
        
        checkpoint_path = os.path.join(
            config.ckpt_dir,
            "Ovi",
            basename,
        )

        if not os.path.exists(checkpoint_path):
            raise RuntimeError(f"REQUIRED fusion checkpoint not found in {config.ckpt_dir}, please download...")

        load_fusion_checkpoint(model, checkpoint_path=checkpoint_path, from_meta=meta_init)

        if meta_init:
            if not fp8:
                model = model.to(dtype=target_dtype)
            model = model.to(device=device if not self.cpu_offload else "cpu").eval()
            model.set_rope_params()
        self.model = model
        if int8:
            quantize(self.model, qint8)
            freeze(self.model)

        ## Load t2i as part of pipeline
        self.image_model = None
        
        if config.get("mode") == "t2i2v":
            logging.info(f"Loading Flux Krea for first frame generation...")
            self.image_model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16)
            self.image_model.enable_model_cpu_offload(gpu_id=self.device) #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU VRAM

        # Fixed attributes, non-configurable
        self.audio_latent_channel = audio_config.get("in_dim")
        self.video_latent_channel = video_config.get("in_dim")
        self.video_latent_length = model_specs["video_latent_length"]
        self.audio_latent_length = model_specs["audio_latent_length"]
        self.text_formatter = model_specs["formatter"]
        self.target_area = model_specs["video_area"]


        logging.info(f"OVI Fusion Engine initialized, cpu_offload={self.cpu_offload}. GPU VRAM allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GB, reserved: {torch.cuda.memory_reserved(device)/1e9:.2f} GB")

    @torch.inference_mode()
    def generate(self,
                    text_prompt, 
                    image_path=None,
                    video_frame_height_width=None,
                    seed=100,
                    solver_name="unipc",
                    sample_steps=50,
                    shift=5.0,
                    video_guidance_scale=5.0,
                    audio_guidance_scale=4.0,
                    slg_layer=9,
                    video_negative_prompt="",
                    audio_negative_prompt=""
                ):

        original_text_prompt = text_prompt
        self.video_self_attention_dispatcher.reset_metrics()
        expected_cfg_metrics = (
            expected_cfg_cache_metrics(
                sample_steps,
                self.cfg_cache_start_step,
                self.cfg_cache_end_step,
                self.cfg_cache_refresh_interval,
            )
            if self.use_cfg_cache
            else {
                "cfg_cache_hits": 0,
                "cfg_cache_refreshes": 0,
                "cfg_negative_forwards": int(sample_steps),
            }
        )
        expected_attention_calls = expected_video_self_attention_calls(
            sample_steps=int(sample_steps),
            num_blocks=int(self.model.num_blocks),
            slg_layer=int(slg_layer),
            negative_forward_count=expected_cfg_metrics[
                "cfg_negative_forwards"
            ],
        )
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        generation_started = time.perf_counter()
        # The mutable cache is scoped to this generation.  It is never stored on
        # the engine/model and is cleared in ``finally`` on success or failure.
        cfg_cache_state = (
            CfgNegativeCache(
                self.cfg_cache_start_step,
                self.cfg_cache_end_step,
                self.cfg_cache_refresh_interval,
            )
            if self.use_cfg_cache
            else None
        )
        # Allocate block-cache payload state only after entering the protected
        # try/finally below.  Configuration lives on the engine; mutable cached
        # tensors never do.
        block_cache_state = None
        dense_negative_forwards = 0
        self.last_run_metrics = {
            "status": "running",
            "model_name": self.model_name,
            "attention_method": self.attention_method,
            "use_cfg_cache": self.use_cfg_cache,
            "cfg_cache_start_step": self.cfg_cache_start_step,
            "cfg_cache_end_step": self.cfg_cache_end_step,
            "cfg_cache_window_inclusive": True,
            "cfg_cache_refresh_interval": self.cfg_cache_refresh_interval,
            "cfg_cache_hits": 0,
            "cfg_cache_refreshes": 0,
            "cfg_negative_forwards": 0,
            "expected_cfg_cache_metrics": expected_cfg_metrics,
            "use_block_cache": self.use_block_cache,
            "block_cache_start_block": self.block_cache_start_block,
            "block_cache_end_block": self.block_cache_end_block,
            "block_cache_window_inclusive": True,
            "block_cache_policy": self.block_cache_policy,
            "block_cache_cosine_threshold": (
                self.block_cache_cosine_threshold
            ),
            "block_cache_max_consecutive_reuses": (
                self.block_cache_max_consecutive_reuses
            ),
            "block_cache_hits": 0,
            "block_cache_refreshes": 0,
            "block_cache_saved_video_self_attention_calls": 0,
            "block_cache_branch_metrics": {},
            "debug_forward": self.debug_forward,
            "run_kind": self.run_kind,
            "benchmark_candidate": self.benchmark_eligible and not self.debug_forward,
            # Only the post-run verifier can certify benchmark validity after
            # checking warm-up/measurement cardinality and every artifact.
            "benchmark_valid": False,
            "sample_steps": int(sample_steps),
            "seed": int(seed),
            "original_text_prompt": original_text_prompt,
            "requested_video_frame_height_width": (
                list(video_frame_height_width)
                if video_frame_height_width is not None
                else None
            ),
            "video_self_attention_dispatcher": {
                **self.video_self_attention_dispatcher.metrics(),
                "expected_calls": expected_attention_calls,
            },
        }

        params = {
            "Text Prompt": text_prompt,
            "Image Path": image_path if image_path else "None (T2V mode)",
            "Frame Height Width": video_frame_height_width,
            "Seed": seed,
            "Solver": solver_name,
            "Sample Steps": sample_steps,
            "Shift": shift,
            "Video Guidance Scale": video_guidance_scale,
            "Audio Guidance Scale": audio_guidance_scale,
            "SLG Layer": slg_layer,
            "Video Negative Prompt": video_negative_prompt,
            "Audio Negative Prompt": audio_negative_prompt,
        }

        pretty = "\n".join(f"{k:>24}: {v}" for k, v in params.items())
        logging.info("\n========== Generation Parameters ==========\n"
                    f"{pretty}\n"
                    "==========================================")
        try:
            if self.use_block_cache:
                block_cache_state = FusionBlockCache(
                    self.block_cache_start_block,
                    self.block_cache_end_block,
                    self.block_cache_policy,
                    self.block_cache_cosine_threshold,
                    self.block_cache_max_consecutive_reuses,
                    num_blocks=self.model.num_blocks,
                )
            scheduler_video, timesteps_video = self.get_scheduler_time_steps(
                sampling_steps=sample_steps,
                device=self.device,
                solver_name=solver_name,
                shift=shift
            )
            scheduler_audio, timesteps_audio = self.get_scheduler_time_steps(
                sampling_steps=sample_steps,
                device=self.device,
                solver_name=solver_name,
                shift=shift
            )

            is_t2v = image_path is None
            is_i2v = not is_t2v

            first_frame = None
            image = None

            # text and image checks
            formatted_text_prompt = self.text_formatter(text_prompt)
            self.last_run_metrics["formatted_text_prompt"] = formatted_text_prompt
            if formatted_text_prompt != text_prompt:
                logging.info(f"Wrong audio description format detected! Please use <AUDCAP>...<ENDAUDCAP> tags for 720x720_5s model and Audio: ... for 960x960 models.\n \
                             Original prompt: {text_prompt}\nFormatted prompt: {formatted_text_prompt}")
                text_prompt = formatted_text_prompt

            if is_i2v and not self.image_model:
                # Load first frame from path
                first_frame = preprocess_image_tensor(image_path, self.device, self.target_dtype, resize_total_area=self.target_area)
            else:
                assert video_frame_height_width is not None, f"If mode=t2v or t2i2v, video_frame_height_width must be provided."

                # input resolution should be at least 0.9x of video area of model spec
                input_area = video_frame_height_width[0] * video_frame_height_width[1]
                if input_area < 0.9 * self.target_area or input_area > 1.1 * self.target_area:
                    logging.warning(f"[Detected model: {self.model_name}] Input video frame area {input_area} is more than 10% smaller or larger than model's target area {self.target_area}. This may lead to suboptimal results, please refer to readme for best resolutions or use the right model. DEFAULTING TO MODEL'S TARGET AREA while preserving given aspect ratio.")

                video_h, video_w = video_frame_height_width
                video_h, video_w = snap_hw_to_multiple_of_32(video_h, video_w, area = self.target_area)
                video_latent_h, video_latent_w = video_h // 16, video_w // 16
                self.last_run_metrics.update({
                    "actual_video_frame_height_width": [video_h, video_w],
                    "video_latent_height_width": [video_latent_h, video_latent_w],
                })
                if self.image_model is not None:
                    # this already means t2v mode with image model
                    image_h, image_w = scale_hw_to_area_divisible(video_h, video_w, area = 1024 * 1024)
                    image = self.image_model(
                        clean_text(text_prompt),
                        height=image_h,
                        width=image_w,
                        guidance_scale=4.5,
                        generator=torch.Generator().manual_seed(seed)
                    ).images[0]
                    first_frame = preprocess_image_tensor(image, self.device, self.target_dtype, resize_total_area=self.target_area)
                    is_i2v = True
                else:
                    print(f"Pure T2V mode: calculated video latent size: {video_latent_h} x {video_latent_w}")

            
            if self.cpu_offload:
                self.text_model.model = self.text_model.model.to(self.device)
            text_embeddings = self.text_model([text_prompt, video_negative_prompt, audio_negative_prompt], self.text_model.device)
            text_embeddings = [emb.to(self.target_dtype).to(self.device) for emb in text_embeddings]

            if self.cpu_offload:
                self.offload_to_cpu(self.text_model.model)

            # Split embeddings
            text_embeddings_audio_pos = text_embeddings[0]
            text_embeddings_video_pos = text_embeddings[0] 

            text_embeddings_video_neg = text_embeddings[1]
            text_embeddings_audio_neg = text_embeddings[2]

            if is_i2v:
                if self.cpu_offload:
                    self.vae_model_video.model = self.vae_model_video.model.to(
                        self.device
                    )
                with torch.no_grad():
                    latents_images = self.vae_model_video.wrapped_encode(first_frame[:, :, None]).to(self.target_dtype).squeeze(0) # c 1 h w 
                latents_images = latents_images.to(self.target_dtype)
                video_latent_h, video_latent_w = latents_images.shape[2], latents_images.shape[3]
                if self.cpu_offload:
                    self.offload_to_cpu(self.vae_model_video.model)

            video_noise = torch.randn((self.video_latent_channel, self.video_latent_length, video_latent_h, video_latent_w), device=self.device, dtype=self.target_dtype, generator=torch.Generator(device=self.device).manual_seed(seed))  # c, f, h, w
            audio_noise = torch.randn((self.audio_latent_length, self.audio_latent_channel), device=self.device, dtype=self.target_dtype, generator=torch.Generator(device=self.device).manual_seed(seed))  # 1, l c -> l, c
            
            # Calculate sequence lengths from actual latents
            max_seq_len_audio = audio_noise.shape[0]  # L dimension from latents_audios shape [1, L, D]
            _patch_size_h, _patch_size_w = self.model.video_model.patch_size[1], self.model.video_model.patch_size[2]
            max_seq_len_video = video_noise.shape[1] * video_noise.shape[2] * video_noise.shape[3] // (_patch_size_h*_patch_size_w) # f * h * w from [1, c, f, h, w]
            self.last_run_metrics.update({
                "audio_sequence_length": int(max_seq_len_audio),
                "video_sequence_length": int(max_seq_len_video),
            })
            
            # Sampling loop
            if self.cpu_offload:
                self.offload_to_cpu(self.vae_model_video.model)
                self.offload_to_cpu(self.vae_model_audio)
                self.model = self.model.to(self.device)
            with torch.amp.autocast('cuda', enabled=self.target_dtype != torch.float32, dtype=self.target_dtype):
                torch.cuda.synchronize(self.device)
                denoise_started = time.perf_counter()
                for i, (t_v, t_a) in tqdm(enumerate(zip(timesteps_video, timesteps_audio))):
                    timestep_input = torch.full((1,), t_v, device=self.device)
                    debug_this_step = self.debug_forward and i == self.debug_forward_step
                    # Avoid a GPU-to-CPU synchronization on every timed step.
                    timestep_value = (
                        float(t_v.detach().cpu().item()) if debug_this_step else None
                    )

                    if is_i2v:
                        video_noise[:, :1] = latents_images

                    # Positive (conditional) forward pass
                    if debug_this_step:
                        logging.info(
                            "[debug_forward] step=%s timestep=%s conditional_forward=start",
                            i,
                            timestep_value,
                        )
                    pos_forward_args = {
                        'audio_context': [text_embeddings_audio_pos],
                        'vid_context': [text_embeddings_video_pos],
                        'vid_seq_len': max_seq_len_video,
                        'audio_seq_len': max_seq_len_audio,
                        'first_frame_is_clean': is_i2v
                    }
                    if block_cache_state is not None:
                        pos_forward_args.update({
                            'block_cache_state': block_cache_state,
                            'block_cache_context': {
                                'step': i,
                                'branch': 'conditional',
                            },
                        })

                    pred_vid_pos, pred_audio_pos = self.model(
                        vid=[video_noise],
                        audio=[audio_noise],
                        t=timestep_input,
                        debug_context={
                            "enabled": debug_this_step,
                            "step": i,
                            "timestep": timestep_value,
                            "branch": "conditional",
                        },
                        **pos_forward_args
                    )
                    if debug_this_step:
                        logging.info(
                            "[debug_forward] step=%s timestep=%s conditional_forward=end "
                            "video_prediction=%s audio_prediction=%s",
                            i,
                            timestep_value,
                            tuple(pred_vid_pos[0].shape),
                            tuple(pred_audio_pos[0].shape),
                        )
                    
                    # Negative (unconditional) forward pass.  With CFG cache
                    # enabled, video/audio predictions are reused only as one
                    # atomic pair inside the inclusive configured window.
                    cache_action = (
                        cfg_cache_state.action(i)
                        if cfg_cache_state is not None
                        else "disabled"
                    )
                    if debug_this_step and cache_action != "hit":
                        logging.info(
                            "[debug_forward] step=%s timestep=%s "
                            "unconditional_forward=start cfg_cache_action=%s",
                            i,
                            timestep_value,
                            cache_action,
                        )
                    neg_forward_args = {
                        'audio_context': [text_embeddings_audio_neg],
                        'vid_context': [text_embeddings_video_neg],
                        'vid_seq_len': max_seq_len_video,
                        'audio_seq_len': max_seq_len_audio,
                        'first_frame_is_clean': is_i2v,
                        'slg_layer': slg_layer
                    }
                    if block_cache_state is not None:
                        neg_forward_args.update({
                            'block_cache_state': block_cache_state,
                            'block_cache_context': {
                                'step': i,
                                'branch': 'unconditional',
                            },
                        })

                    negative_debug_context = {
                        "enabled": debug_this_step,
                        "step": i,
                        "timestep": timestep_value,
                        "branch": "unconditional",
                    }
                    if cfg_cache_state is None:
                        # Preserve the official dense CFG path when caching is
                        # disabled: one negative model forward on every step.
                        dense_negative_forwards += 1
                        pred_vid_neg, pred_audio_neg = self.model(
                            vid=[video_noise],
                            audio=[audio_noise],
                            t=timestep_input,
                            debug_context=negative_debug_context,
                            **neg_forward_args
                        )
                    else:
                        def negative_forward():
                            return self.model(
                                vid=[video_noise],
                                audio=[audio_noise],
                                t=timestep_input,
                                debug_context=negative_debug_context,
                                **neg_forward_args
                            )

                        (
                            (pred_vid_neg, pred_audio_neg),
                            cache_action,
                        ) = cfg_cache_state.resolve(i, negative_forward)
                    if debug_this_step:
                        if cache_action == "hit":
                            logging.info(
                                "[debug_forward] step=%s timestep=%s "
                                "unconditional_forward=cfg_cache_hit "
                                "video_prediction=%s audio_prediction=%s",
                                i,
                                timestep_value,
                                tuple(pred_vid_neg[0].shape),
                                tuple(pred_audio_neg[0].shape),
                            )
                        else:
                            logging.info(
                                "[debug_forward] step=%s timestep=%s "
                                "unconditional_forward=end cfg_cache_action=%s "
                                "video_prediction=%s audio_prediction=%s",
                                i,
                                timestep_value,
                                cache_action,
                                tuple(pred_vid_neg[0].shape),
                                tuple(pred_audio_neg[0].shape),
                            )

                    # Apply classifier-free guidance
                    pred_video_guided = pred_vid_neg[0] + video_guidance_scale * (pred_vid_pos[0] - pred_vid_neg[0])
                    pred_audio_guided = pred_audio_neg[0] + audio_guidance_scale * (pred_audio_pos[0] - pred_audio_neg[0])

                    # Update noise using scheduler
                    video_noise = scheduler_video.step(
                        pred_video_guided.unsqueeze(0), t_v, video_noise.unsqueeze(0), return_dict=False
                    )[0].squeeze(0)

                    audio_noise = scheduler_audio.step(
                        pred_audio_guided.unsqueeze(0), t_a, audio_noise.unsqueeze(0), return_dict=False
                    )[0].squeeze(0)

                    if debug_this_step:
                        logging.info(
                            "[debug_forward] step=%s timestep=%s scheduler_update=end "
                            "video_latent=%s audio_latent=%s",
                            i,
                            timestep_value,
                            tuple(video_noise.shape),
                            tuple(audio_noise.shape),
                        )

                torch.cuda.synchronize(self.device)
                denoise_seconds = time.perf_counter() - denoise_started

                if self.cpu_offload:
                    self.offload_to_cpu(self.model)
                    self.vae_model_video.model = self.vae_model_video.model.to(
                        self.device
                    )
                    self.vae_model_audio = self.vae_model_audio.to(self.device)

                if is_i2v:
                    video_noise[:, :1] = latents_images

                # Decode audio
                audio_latents_for_vae = audio_noise.unsqueeze(0).transpose(1, 2)  # 1, c, l
                generated_audio = self.vae_model_audio.wrapped_decode(audio_latents_for_vae)
                if not torch.isfinite(generated_audio).all():
                    raise FloatingPointError("audio decoder returned NaN or Inf")
                generated_audio = generated_audio.squeeze().cpu().float().numpy()
                
                # Decode video  
                video_latents_for_vae = video_noise.unsqueeze(0)  # 1, c, f, h, w
                generated_video = self.vae_model_video.wrapped_decode(video_latents_for_vae)
                if not torch.isfinite(generated_video).all():
                    raise FloatingPointError("video decoder returned NaN or Inf")
                generated_video = generated_video.squeeze(0).cpu().float().numpy()  # c, f, h, w
                if self.cpu_offload:
                    self.offload_to_cpu(self.vae_model_video.model)
                    self.offload_to_cpu(self.vae_model_audio)

            torch.cuda.synchronize(self.device)
            total_generation_seconds = time.perf_counter() - generation_started
            self.last_run_metrics.update({
                "status": "ok",
                "denoise_seconds": denoise_seconds,
                "total_generation_seconds": total_generation_seconds,
                "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
                "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
                "generated_video_shape": list(generated_video.shape),
                "generated_audio_shape": list(generated_audio.shape),
                "video_self_attention_dispatcher": {
                    **self.video_self_attention_dispatcher.metrics(),
                    "expected_calls": expected_attention_calls,
                },
            })
            logging.info(
                "Generation metrics: total=%.3fs denoise=%.3fs peak_allocated=%.3fGiB",
                total_generation_seconds,
                denoise_seconds,
                self.last_run_metrics["peak_memory_allocated_bytes"] / (1024 ** 3),
            )
            return generated_video, generated_audio, image


        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            self.last_run_metrics.update({
                "status": "failed",
                "error": repr(e),
                "total_generation_seconds": time.perf_counter() - generation_started,
                "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)) if torch.cuda.is_available() else 0,
                "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)) if torch.cuda.is_available() else 0,
                "video_self_attention_dispatcher": {
                    **self.video_self_attention_dispatcher.metrics(),
                    "expected_calls": expected_attention_calls,
                },
            })
            logging.error(traceback.format_exc())
            return None
        finally:
            if cfg_cache_state is None:
                self.last_run_metrics["cfg_negative_forwards"] = (
                    dense_negative_forwards
                )
            else:
                self.last_run_metrics.update(cfg_cache_state.metrics())
                cfg_cache_state.clear()
            if block_cache_state is None:
                block_cache_metrics = {
                    "block_cache_hits": 0,
                    "block_cache_refreshes": 0,
                    "block_cache_saved_video_self_attention_calls": 0,
                    "block_cache_branch_metrics": {},
                }
            else:
                block_cache_metrics = block_cache_state.metrics()
                block_cache_state.clear()
            self.last_run_metrics.update(block_cache_metrics)
            block_adjusted_expected_attention_calls = (
                expected_attention_calls
                - block_cache_metrics[
                    "block_cache_saved_video_self_attention_calls"
                ]
            )
            dispatcher_metrics = self.video_self_attention_dispatcher.metrics()
            self.last_run_metrics["video_self_attention_dispatcher"] = {
                **dispatcher_metrics,
                "expected_calls_without_block_cache": expected_attention_calls,
                "expected_calls": block_adjusted_expected_attention_calls,
                "calls_match_expected": (
                    dispatcher_metrics["calls_total"]
                    == block_adjusted_expected_attention_calls
                ),
            }
            
    def offload_to_cpu(self, model):
        model = model.cpu()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        return model

    def get_scheduler_time_steps(self, sampling_steps, solver_name='unipc', device=0, shift=5.0):
        torch.manual_seed(4)

        if solver_name == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=device, shift=shift)
            timesteps = sample_scheduler.timesteps

        elif solver_name == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift=shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=device,
                sigmas=sampling_sigmas)
            
        elif solver_name == 'euler':
            sample_scheduler = FlowMatchEulerDiscreteScheduler(
                shift=shift
            )
            timesteps, sampling_steps = retrieve_timesteps(
                sample_scheduler,
                sampling_steps,
                device=device,
            )
        
        else:
            raise NotImplementedError("Unsupported solver.")
        
        return sample_scheduler, timesteps
