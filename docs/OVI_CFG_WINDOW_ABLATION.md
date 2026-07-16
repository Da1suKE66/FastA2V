# Ovi CFG-cache window-position ablation

## Question

The current 30-step Ovi results show that CFG cache is active and saves nearly
constant time per skipped negative forward. The remaining question is whether
quality loss is controlled only by the number of skipped forwards or is also
sensitive to where those skips occur in the denoising trajectory.

This diagnostic holds the cache workload constant and changes only its window:

| Inclusive window | Refresh steps | Cache hits | Negative forwards | Status |
|---|---|---:|---:|---|
| 6-23 | 6, 11, 16, 21 | 14 | 16 | Existing |
| 9-26 | 9, 14, 19, 24 | 14 | 16 | Existing |
| 12-29 | 12, 17, 22, 27 | 14 | 16 | New ablation |

All three use the same model, prompt, seed, 30-step Euler schedule, guidance,
and dense video attention. With 18 eligible steps and refresh interval 5, each
window has four refreshes and 14 cache hits. Any material quality difference is
therefore attributable to timestep placement rather than cache-hit count.

## Existing evidence

| Window | Generation | Denoise | PSNR vs dense | SSIM vs dense |
|---|---:|---:|---:|---:|
| Dense | 112.3040 s | 98.4911 s | Reference | Reference |
| 6-23/r5 | 89.7044 s | 75.8774 s | 25.4365 dB | 0.8805 |
| 9-26/r5 | 89.6852 s | 75.8509 s | 28.4593 dB | 0.9184 |

The two cached runs have essentially identical latency but substantially
different reconstruction metrics. This already points to timestep sensitivity,
not cache misses or a failed implementation.

## Run policy

Run only on an idle physical GPU 0:

```bash
cd /workspace/liluchen/FastA2V
FASTA2V_RUN_TAG=20260717-late-window-r5 \
  bash scripts/run_ovi_cfg_window_ablation.sh
```

The runner uses the fail-closed GPU check and performs only media validation;
it intentionally does not build formal audit evidence. Stop after this point if
12-29 does not improve on the 9-26 quality result. Do not resume broad Sparge,
Radial, or block-cache sweeps from this diagnostic.

The timing record additionally separates pre-denoise, denoise, audio decode,
and video decode. No profiler or per-step synchronization is added.
