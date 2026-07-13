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

All acceleration options default to the official dense path. Sparse attention
will only replace video self-attention; audio self-attention, text
cross-attention, and bidirectional audio-video cross-attention remain dense.
