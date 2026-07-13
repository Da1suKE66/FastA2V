# FastA2V experiment scope

FastA2V starts directly from the official
[`character-ai/Ovi`](https://github.com/character-ai/Ovi) repository at commit
`5b69b25a4b3115216e9ea53a37a04410be6ad39a`. No existing Ovi, SVG, or cache
fork is used as the implementation baseline.

The historical results in `音视频推理加速 (1).pdf` pages 45-48 are reference
targets only. They do not prove that this repository already contains an Ovi
sparse-attention or cache implementation.

The first reproducible baseline is fixed to:

- Ovi `720x720_5s`
- T2V at `720 x 720`
- BF16, single A100 GPU, `sp_size=1`, no CPU offload
- UniPC, shift `5.0`, 50 denoising steps
- video guidance `4.0`, audio guidance `3.0`, SLG layer `11`
- seed `103` and `prompts/ovi_smoke.csv`
- dense video and audio attention

`720x720_5s` is the official checkpoint/model name. Ovi's official helper
requires dimensions divisible by 32 and snaps a requested square `720 x 720`
frame to an actual `704 x 704` output (`round(720 / 32) * 32`). FastA2V records
both requested and actual dimensions and uses the actual dimensions in artifact
filenames; it does not relabel a 704-pixel artifact as literal 720 pixels.

The formal baseline performs one full warm-up generation in the same process,
excludes it from measurements, then records three repeated measurements with
the same prompt, seed, shape, and loaded engine. Every invocation receives a
unique run directory so failed reruns cannot leave old videos looking current.
The official unmodified source is also kept as a detached worktree at the same
base commit. `scripts/run_ovi_official_reference.sh` runs the matching 20-step
smoke configuration, after which `scripts/compare_media.py` performs decoded
video PSNR/SSIM and audio waveform comparisons against the instrumented dense
run.

Acceleration work is restricted to Python/PyTorch integration. Official
third-party kernels may be installed and called through their public APIs, but
FastA2V does not implement or modify CUDA or Triton kernels and does not train
or distill models.

The supported configuration surface is:

```text
attention_method = dense | sparge | radial | svg
use_cfg_cache = true | false
use_block_cache = true | false
```

The first CFG-cache implementation uses an inclusive step window configured by
`cfg_cache_start_step`, `cfg_cache_end_step`, and
`cfg_cache_refresh_interval`. Refreshes are anchored at the start step and cache
the joint video/audio negative prediction as one pair. Outside the window the
official negative forward runs every step; an interval of `1` is therefore
schedule-equivalent to dense CFG. Each generation records `cfg_cache_hits`,
`cfg_cache_refreshes`, and `cfg_negative_forwards` and clears its local cache in
a `finally` block.

The first block-cache implementation covers the inclusive fusion-block window
`10..19` and caches its complete `(video, audio)` output pair. Conditional and
unconditional payloads are physically separate and local to one generation.
The fixed policy is deliberately limited to `compute -> reuse -> compute`
(`block_cache_max_consecutive_reuses=1`). A denoising-step gap, video/audio
shape, dtype, or device change, or a changed SLG signature forces a refresh;
this also prevents an unconditional hit after CFG cache skipped intervening
negative forwards. The optional cosine policy additionally requires
`min(video_cosine, audio_cosine)` to meet its threshold. Run
`scripts/run_ovi_block_cache_smoke.sh` before the formal
`scripts/run_ovi_block_cache_baseline.sh` protocol.

All acceleration options default to the official dense path. Sparse attention
will only replace video self-attention; audio self-attention, text
cross-attention, and bidirectional audio-video cross-attention remain dense.

The first sparse adapter pins official SpargeAttn commit
`ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a` and calls its public
`spas_sage2_attn_meansim_topk_cuda` API. See
[`docs/SPARGEATTN.md`](SPARGEATTN.md) for installation, input constraints, and
the exact Ovi ownership boundary. This repository contains no copied or custom
SpargeAttn CUDA/Triton source.

The Radial adapter separately pins official `mit-han-lab/radial-attention`
commit `72788d4f0a6d202f1ec5f1c98a6e4c8b2e34fdbc`. It calls the upstream
`gen_log_mask_shrinked` function from an audited mask-only derived Python copy
whose only patch makes unrelated optional imports lazy, then executes the mask
through public FlashInfer APIs. Pure Radial runs keep CFG and block cache off.
See [`docs/RADIAL_ATTENTION.md`](RADIAL_ATTENTION.md) for the 14,976-token
sparse prefix, 28-token dense/LSE-merged tail, empty-row repair audit, and
guarded runtime-validation boundary.
