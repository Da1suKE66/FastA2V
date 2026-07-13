
import logging

import torch
import torch.nn as nn
from ovi.modules.model import WanLayerNorm, WanModel, WanRMSNorm, gradient_checkpointing, rope_apply
from ovi.modules.attention import flash_attention
from ovi.modules.video_attention_dispatcher import VideoSelfAttentionDispatcher
from ovi.distributed_comms.communications import all_gather, all_to_all_4D
from ovi.distributed_comms.parallel_states import nccl_info, get_sequence_parallel_state

class FusionModel(nn.Module):
    def __init__(self, video_config=None, audio_config=None):
        super().__init__()
        self.video_self_attention_dispatcher = VideoSelfAttentionDispatcher("dense")
        has_video = True 
        has_audio = True
        if video_config is not None:
            self.video_model = WanModel(**video_config)
        else:
            has_video = False
            self.video_model = None
            print("Warning: No video model is provided!")
        
        if audio_config is not None:
            self.audio_model = WanModel(**audio_config)
        else:
            has_audio = False
            self.audio_model = None
            print("Warning: No audio model is provided!")

        if has_video and has_audio:
            assert len(self.video_model.blocks) == len(self.audio_model.blocks)
            self.num_blocks = len(self.video_model.blocks)

            self.use_sp = get_sequence_parallel_state()
            if self.use_sp:
                self.sp_size = nccl_info.sp_size
                self.sp_rank = nccl_info.rank_within_group
            self.inject_cross_attention_kv_projections()

        self.init_weights()

    def set_video_self_attention_dispatcher(self, dispatcher):
        if not isinstance(dispatcher, VideoSelfAttentionDispatcher):
            raise TypeError(
                "dispatcher must be a VideoSelfAttentionDispatcher instance"
            )
        self.video_self_attention_dispatcher = dispatcher
        
    def inject_cross_attention_kv_projections(self):
        for vid_block in self.video_model.blocks:
            vid_block.cross_attn.k_fusion = nn.Linear(vid_block.dim, vid_block.dim)
            vid_block.cross_attn.v_fusion = nn.Linear(vid_block.dim, vid_block.dim)
            vid_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(vid_block.dim, elementwise_affine=True)
            vid_block.cross_attn.norm_k_fusion = WanRMSNorm(vid_block.dim, eps=1e-6) if vid_block.qk_norm else nn.Identity()

        
        for audio_block in self.audio_model.blocks:
            audio_block.cross_attn.k_fusion = nn.Linear(audio_block.dim, audio_block.dim)
            audio_block.cross_attn.v_fusion = nn.Linear(audio_block.dim, audio_block.dim)
            audio_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(audio_block.dim, elementwise_affine=True)
            audio_block.cross_attn.norm_k_fusion = WanRMSNorm(audio_block.dim, eps=1e-6) if audio_block.qk_norm else nn.Identity()


    def merge_kwargs(self, vid_kwargs, audio_kwargs):
        """
        keys in each kwarg:
        e
        seq_lens
        grid_sizes
        freqs
        context
        context_lens
        """
        merged_kwargs = {}
        for key in vid_kwargs:
            merged_kwargs[f"vid_{key}"] = vid_kwargs[key]
        for key in audio_kwargs:
            merged_kwargs[f"audio_{key}"] = audio_kwargs[key]
        return merged_kwargs

    def single_fusion_cross_attention_forward(self,
                                            cross_attn_block,
                                            src_seq,
                                            src_grid_sizes,
                                            src_freqs,
                                            target_seq,
                                            target_seq_lens,
                                            target_grid_sizes,
                                            target_freqs,
                                            context,
                                            context_lens
                                            ):
        b, n, d = src_seq.size(0), cross_attn_block.num_heads, cross_attn_block.head_dim
        if hasattr(cross_attn_block, "k_img"):
            ## means is i2v block
            q, k, v, k_img, v_img = cross_attn_block.qkv_fn(src_seq, context)
        else:
            ## means is t2v block
            q, k, v = cross_attn_block.qkv_fn(src_seq, context)
            k_img = v_img = None

        
        if self.use_sp:
            q = all_to_all_4D(q, scatter_dim=2, gather_dim=1)
            k = torch.chunk(k, self.sp_size, dim=2)[self.sp_rank]
            v = torch.chunk(v, self.sp_size, dim=2)[self.sp_rank]
            if k_img is not None:
                k_img = torch.chunk(k_img, self.sp_size, dim=2)[self.sp_rank]
            if v_img is not None:
                v_img = torch.chunk(v_img, self.sp_size, dim=2)[self.sp_rank]
            
        x = flash_attention(q, k, v, k_lens=context_lens)

        if k_img is not None:
            img_x = flash_attention(q, k_img, v_img, k_lens=None)
            x = x + img_x

        is_vid = src_grid_sizes.shape[1] > 1
        # compute target attention
        target_seq = cross_attn_block.pre_attn_norm_fusion(target_seq)
        k_target = cross_attn_block.norm_k_fusion(cross_attn_block.k_fusion(target_seq)).view(b, -1, n, d)
        v_target = cross_attn_block.v_fusion(target_seq).view(b, -1, n, d)
        if self.use_sp: 
            k_target = all_to_all_4D(k_target, scatter_dim=2, gather_dim=1) # [B, L, H/P, C/H]
            v_target = all_to_all_4D(v_target, scatter_dim=2, gather_dim=1) # [B, L, H/P, C/H]
        
        q = rope_apply(q, src_grid_sizes, src_freqs)
        k_target = rope_apply(k_target, target_grid_sizes, target_freqs)
        
        target_x = flash_attention(q, k_target, v_target, k_lens=target_seq_lens)
        
        x = x + target_x
        if self.use_sp:
            x = all_to_all_4D(x, scatter_dim=1, gather_dim=2) # [B, L/P, H, C/H]
        
        x = x.flatten(2) # [B, L/P, C]

        x = cross_attn_block.o(x)
        return x

    def single_fusion_cross_attention_ffn_forward(self,
                                            attn_block,
                                            src_seq,
                                            src_grid_sizes,
                                            src_freqs,
                                            target_seq,
                                            target_seq_lens,
                                            target_grid_sizes,
                                            target_freqs,
                                            context,
                                            context_lens,
                                            src_e):
        
        src_seq = src_seq + self.single_fusion_cross_attention_forward(attn_block.cross_attn,
                                                                       attn_block.norm3(src_seq),
                                                                       src_grid_sizes=src_grid_sizes,
                                                                       src_freqs=src_freqs,
                                                                       target_seq=target_seq,
                                                                       target_seq_lens=target_seq_lens,
                                                                       target_grid_sizes=target_grid_sizes,
                                                                       target_freqs=target_freqs,
                                                                       context=context,
                                                                       context_lens=context_lens
                                                                       )
        y = attn_block.ffn(attn_block.norm2(src_seq).bfloat16() * (1 + src_e[4].squeeze(2)) + src_e[3].squeeze(2))
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            src_seq = src_seq + y * src_e[5].squeeze(2)
        return src_seq
        
    def single_fusion_block_forward(self,
                                    vid_block,
                                    audio_block,
                                    vid,
                                    audio,
                                    vid_e,
                                    vid_seq_lens,
                                    vid_grid_sizes,
                                    vid_freqs,
                                    vid_context,
                                    vid_context_lens,
                                    audio_e,
                                    audio_seq_lens,
                                    audio_grid_sizes,
                                    audio_freqs,
                                    audio_context,
                                    audio_context_lens,
                                    block_index=None,
                                    debug_context=None,
                                    ):
        ## audio modulation
        assert audio_e.dtype == torch.bfloat16
        assert len(audio_e.shape) == 4 and audio_e.size(2) == 6 and audio_e.shape[1] == audio.shape[1], f"{audio_e.shape}, {audio.shape}"
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            audio_e = audio_block.modulation(audio_e).chunk(6, dim=2)
        assert audio_e[0].dtype == torch.bfloat16

        # audio self-attention
        audio_y = audio_block.self_attn(
            audio_block.norm1(audio).bfloat16() * (1 + audio_e[1].squeeze(2)) + audio_e[0].squeeze(2), audio_seq_lens, audio_grid_sizes,
            audio_freqs)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            audio = audio + audio_y * audio_e[2].squeeze(2)

        ## video modulation
        assert len(vid_e.shape) == 4 and vid_e.size(2) == 6 and vid_e.shape[1] == vid.shape[1], f"{vid_e.shape}, {vid.shape}"
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            vid_e = vid_block.modulation(vid_e).chunk(6, dim=2)

        # video self-attention
        vid_self_attn_input = (
            vid_block.norm1(vid).bfloat16()
            * (1 + vid_e[1].squeeze(2))
            + vid_e[0].squeeze(2)
        )
        debug_enabled = bool(debug_context and debug_context.get("enabled"))
        if debug_enabled:
            logging.info(
                "[debug_forward] step=%s timestep=%s branch=%s block=%s "
                "video_self_attn input=%s seq_lens=%s grid_sizes=%s",
                debug_context.get("step"),
                debug_context.get("timestep"),
                debug_context.get("branch"),
                block_index,
                tuple(vid_self_attn_input.shape),
                tuple(vid_seq_lens.shape),
                vid_grid_sizes.detach().cpu().tolist(),
            )

        vid_y = self.video_self_attention_dispatcher(
            vid_block.self_attn,
            vid_self_attn_input,
            vid_seq_lens,
            vid_grid_sizes,
            vid_freqs,
            block_index=block_index,
            debug_context=debug_context,
        )

        if debug_enabled:
            logging.info(
                "[debug_forward] step=%s branch=%s block=%s "
                "video_self_attn output=%s",
                debug_context.get("step"),
                debug_context.get("branch"),
                block_index,
                tuple(vid_y.shape),
            )

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            vid = vid + vid_y * vid_e[2].squeeze(2)

        og_audio = audio

        # audio cross-attention
        audio = self.single_fusion_cross_attention_ffn_forward(
            audio_block,
            audio,
            audio_grid_sizes,
            audio_freqs,
            vid,
            vid_seq_lens,
            vid_grid_sizes,
            vid_freqs,
            audio_context,
            audio_context_lens,
            audio_e
        )

        assert not torch.equal(og_audio, audio), "Audio should be changed after cross-attention!"

        # video cross-attention
        vid = self.single_fusion_cross_attention_ffn_forward(
            vid_block,
            vid,
            vid_grid_sizes,
            vid_freqs,
            og_audio,
            audio_seq_lens,
            audio_grid_sizes,
            audio_freqs,
            vid_context,
            vid_context_lens,
            vid_e
        )

        return vid, audio

    def forward(
        self,
        vid,
        audio,
        t,
        vid_context,
        audio_context,
        vid_seq_len,
        audio_seq_len,
        clip_fea=None,
        clip_fea_audio=None,
        y=None,
        first_frame_is_clean=False,
        slg_layer=False,
        debug_context=None,
        block_cache_state=None,
        block_cache_context=None,
    ):  

        assert clip_fea is None 
        assert y is None

        if vid is None or all([x is None for x in vid]):
            assert vid_context is None
            assert vid_seq_len is None
            assert self.audio_model is not None

            return None, self.audio_model(x=audio, t=t, context=audio_context, seq_len=audio_seq_len, clip_fea=clip_fea_audio, y=None)
        
        if audio is None or all([x is None for x in audio]):
            assert clip_fea_audio is None
            assert audio_context is None
            assert audio_seq_len is None
            assert self.video_model is not None

            return self.video_model(x=vid, t=t, context=vid_context, seq_len=vid_seq_len, clip_fea=clip_fea, y=y, first_frame_is_clean=first_frame_is_clean), None
        
        vid, vid_e, vid_kwargs = self.video_model.prepare_transformer_block_kwargs(
            x=vid, t=t, context=vid_context, seq_len=vid_seq_len, clip_fea=clip_fea, y=y, first_frame_is_clean=first_frame_is_clean
        )

        audio, audio_e, audio_kwargs = self.audio_model.prepare_transformer_block_kwargs(
            x=audio, t=t, context=audio_context, seq_len=audio_seq_len, clip_fea=clip_fea_audio, y=None, first_frame_is_clean=False
        )

        kwargs = self.merge_kwargs(vid_kwargs, audio_kwargs)

        if block_cache_state is None:
            # Keep the official dense loop untouched when block caching is
            # disabled.  In particular, no branch bookkeeping or cache lookup
            # is performed on this path.
            for i in range(self.num_blocks):
                """
                1 fusion block refers to 1 audio block with 1 video block.
                """
                if slg_layer > 0 and i == slg_layer:
                    if debug_context and debug_context.get("enabled"):
                        logging.info(
                            "[debug_forward] step=%s branch=%s block=%s skipped_by_slg=true",
                            debug_context.get("step"),
                            debug_context.get("branch"),
                            i,
                        )
                    continue
                vid_block = self.video_model.blocks[i]
                audio_block = self.audio_model.blocks[i]
                vid, audio = gradient_checkpointing(
                        enabled=(self.training and self.gradient_checkpointing),
                        module=self.single_fusion_block_forward,
                        vid_block=vid_block,
                        audio_block=audio_block,
                        vid=vid,
                        audio=audio,
                        block_index=i,
                        debug_context=debug_context,
                        **kwargs
                    )
        else:
            if not isinstance(block_cache_context, dict):
                raise ValueError(
                    "block_cache_context with step and branch is required "
                    "when block caching is enabled"
                )
            if "step" not in block_cache_context or "branch" not in block_cache_context:
                raise ValueError(
                    "block_cache_context must contain step and branch"
                )
            cache_start = int(block_cache_state.start_block)
            cache_end = int(block_cache_state.end_block)
            if not 0 <= cache_start <= cache_end < self.num_blocks:
                raise ValueError(
                    f"invalid fusion block-cache window "
                    f"{cache_start}..{cache_end} for {self.num_blocks} blocks"
                )
            cache_step = int(block_cache_context["step"])
            cache_branch = str(block_cache_context["branch"])
            normalized_slg_layer = int(slg_layer) if slg_layer else 0
            skipped_blocks = (
                (normalized_slg_layer,)
                if 0 < normalized_slg_layer < self.num_blocks
                else ()
            )
            slg_signature = (
                "slg_layer",
                normalized_slg_layer,
                "num_blocks",
                self.num_blocks,
            )

            def run_block(block_index, block_vid, block_audio):
                if slg_layer > 0 and block_index == slg_layer:
                    if debug_context and debug_context.get("enabled"):
                        logging.info(
                            "[debug_forward] step=%s branch=%s block=%s skipped_by_slg=true",
                            debug_context.get("step"),
                            debug_context.get("branch"),
                            block_index,
                        )
                    return block_vid, block_audio
                vid_block = self.video_model.blocks[block_index]
                audio_block = self.audio_model.blocks[block_index]
                return gradient_checkpointing(
                    enabled=(self.training and self.gradient_checkpointing),
                    module=self.single_fusion_block_forward,
                    vid_block=vid_block,
                    audio_block=audio_block,
                    vid=block_vid,
                    audio=block_audio,
                    block_index=block_index,
                    debug_context=debug_context,
                    **kwargs
                )

            block_index = 0
            while block_index < self.num_blocks:
                if block_index != cache_start:
                    vid, audio = run_block(block_index, vid, audio)
                    block_index += 1
                    continue

                cache_input_pair = (vid, audio)

                def compute_window():
                    window_vid, window_audio = cache_input_pair
                    for window_block_index in range(cache_start, cache_end + 1):
                        window_vid, window_audio = run_block(
                            window_block_index, window_vid, window_audio
                        )
                    return window_vid, window_audio

                (vid, audio), cache_action = block_cache_state.resolve(
                    step=cache_step,
                    branch=cache_branch,
                    input_pair=cache_input_pair,
                    slg_signature=slg_signature,
                    skipped_blocks=skipped_blocks,
                    compute_window=compute_window,
                )
                if debug_context and debug_context.get("enabled"):
                    logging.info(
                        "[debug_forward] step=%s branch=%s "
                        "block_cache_window=%s..%s action=%s "
                        "slg_skipped_blocks=%s",
                        cache_step,
                        cache_branch,
                        cache_start,
                        cache_end,
                        cache_action,
                        skipped_blocks,
                    )
                block_index = cache_end + 1

        vid = self.video_model.post_transformer_block_out(vid, vid_kwargs['grid_sizes'], vid_e)
        audio = self.audio_model.post_transformer_block_out(audio, audio_kwargs['grid_sizes'], audio_e)

        return vid, audio

    def init_weights(self):
        if self.audio_model is not None:
            self.audio_model.init_weights()

        if self.video_model is not None:
            self.video_model.init_weights()

        for name, mod in self.video_model.named_modules():
            if "fusion" in name and isinstance(mod, nn.Linear):
                with torch.no_grad():
                    mod.weight.div_(10.0)

    
    def set_rope_params(self):
        self.video_model.set_rope_params()
        self.audio_model.set_rope_params()
