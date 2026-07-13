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

