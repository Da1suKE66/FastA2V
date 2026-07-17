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

| Window | Generation | Denoise | PSNR vs dense | SSIM vs dense | Audio corr. |
|---|---:|---:|---:|---:|---:|
| Dense | 112.3040 s | 98.4911 s | Reference | Reference | Reference |
| 6-23/r5 | 89.7044 s | 75.8774 s | 25.4365 dB | 0.8805 | 0.1041 |
| 9-26/r5 | 89.6852 s | 75.8509 s | 28.4593 dB | 0.9184 | 0.2315 |
| 12-29/r5 | 90.4780 s | 76.4946 s | 31.0744 dB | 0.9362 | 0.4333 |

The two cached runs have essentially identical latency but substantially
different reconstruction metrics. This already points to timestep sensitivity,
not cache misses or a failed implementation.

The execution counts also show that the implementation is realizing the
available compute saving. Dense executes 1,770 video self-attention calls;
either r5 window executes 1,364. The call ratio is `1364/1770 = 0.7706`, while
the measured denoise ratio is `75.8509/98.4911 = 0.7701`. The difference is
under 0.1 percentage point. Non-denoise time remains about 13.8 seconds in both
runs. The speed ceiling is therefore dominated by coverage and fixed work, not
cache misses or dispatch fallback.

The 12-29 point was run on a separate idle A100-SXM4-80GB because the original
host was occupied by unrelated work. Its exact workload counts are directly
comparable, while its absolute latency is engineering evidence rather than a
paired benchmark. It is only 0.9% slower than 9-26 despite the host change and
executes the same 1,364 attention calls.

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

## Result

The single planned 12-29/r5 run completed successfully at commit
`4c77d1e8a954c805e26ce35a8f97befbec447960`. Media-only verification passed:
the output has 121 decoded 704x704 H.264 frames and a non-silent AAC stream.
The GPU monitor found no contention. Runtime counters matched the analytical
expectation exactly: 14 cache hits, four refreshes, 16 negative forwards, and
1,364/1,364 expected video self-attention calls with no fallback.

Moving the equal-compute cache window later improves all three reconstruction
signals relative to 9-26: +2.6150 dB PSNR, +0.0178 SSIM, and +0.2018 audio
correlation. Across 6-23, 9-26, and 12-29, quality improves monotonically as the
window moves later while the work count stays fixed. For this prompt and seed,
stale negative predictions in earlier, higher-noise steps are therefore more
damaging than stale predictions near the end of denoising.

The measured generation phases were:

| Phase | Seconds | Share of generation |
|---|---:|---:|
| Pre-denoise | 0.1407 | 0.16% |
| Denoise | 76.4946 | 84.55% |
| Audio decode | 0.0597 | 0.07% |
| Video decode | 13.7830 | 15.23% |
| Total generation | 90.4780 | 100% |
| Artifact ready | 93.0872 | - |

Video decode accounts for 98.6% of the 13.9833 seconds outside denoising in
the measured generation interval. Saving and muxing adds another 2.5329
seconds before the artifact is ready. This identifies video decoding, followed
by packaging, as the next fixed-latency target; another attention-only change
cannot remove that ceiling.

This remains a one-prompt, one-seed engineering ablation, not a formal quality
benchmark. The broad window sweep stops here. Use 12-29/r5 as the candidate for
a later held-out evaluation rather than continuing to tune on this sample.
Machine-readable evidence is recorded in
`docs/results/ovi_cfg_window_ablation_20260717.json`.
