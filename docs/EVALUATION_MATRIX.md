# Ovi evaluation matrix

`configs/ovi_eval_matrix.json` is the fixed comparison manifest for the Ovi
`720x720_5s` work. It contains seven required slots and one optional
block-cache slot. A--F are implemented; G (best sparse + CFG) and H (best
sparse + block cache) require an explicit winner selection and run mapping, so
no result is inferred for them.

The formal protocol uses `prompts/ovi_formal8.csv`, three seeds per prompt, and
three measurement indices. Each mapped method must therefore contain exactly
72 measured artifacts plus one excluded warm-up. The older six-prompt,
one-seed matrix remains development evidence and must not be labeled formal.

Build a CSV by selecting every run explicitly:

```bash
python scripts/build_ovi_eval_csv.py \
  --output /cache/liluchen/FastA2V/results/ovi_eval.csv \
  dense=/cache/liluchen/FastA2V/runs/ovi_720ckpt_dense_50step/RUN_ID \
  dense_cfg_cache=/cache/liluchen/FastA2V/runs/ovi_720ckpt_cfg_cache_50step/RUN_ID \
  sparge_topk75=/cache/liluchen/FastA2V/runs/ovi_720ckpt_sparge_topk75_50step/RUN_ID \
  sparge_topk50=/cache/liluchen/FastA2V/runs/ovi_720ckpt_sparge_50step/RUN_ID \
  radial_conservative=/cache/liluchen/FastA2V/runs/ovi_720ckpt_radial_conservative_50step/RUN_ID \
  radial_aggressive=/cache/liluchen/FastA2V/runs/ovi_720ckpt_radial_aggressive_50step/RUN_ID
```

There is intentionally no runs-root argument and no "latest" lookup.  An
unmapped slot stays in the CSV with `status=pending` and blank numeric fields.
A slot whose implementation is still `pending` rejects a supplied run path.

For a mapped run, the builder independently requires:

- both top-level and protocol `benchmark_valid=true` from `verification.json`;
- a clean full Git commit and the fixed formal configuration;
- exactly three unique measurement indices, eight prompts, three seeds, and
  exactly one excluded warm-up;
- finite, positive generation/denoise/memory evidence and complete save/hash
  timings;
- exact equality between every `timings.jsonl` record and its per-artifact
  `.metrics.json` source record;
- uncontended, stable single-process GPU monitor evidence for every repeat;
- the on-disk MP4 SHA256 to match both `timings.jsonl` and the verifier report;
- the checkpoint manifest hash to match `environment.json`.

All mapped methods must also have the same commit, checkpoint-file
fingerprint, GPU identity, prompt, seed, requested/actual tensor shapes, and
step count.  This means results from different code revisions are not silently
mixed; the older method must be rerun at the comparison commit.

For downstream audit, the CSV records SHA256 values for `verification.json`,
`timings.jsonl`, every metrics sidecar, the checkpoint manifest, and every MP4.

The CSV reports medians across the complete 72-artifact formal identity set for
denoising, total generation, artifact readiness, allocated GiB, and reserved
GiB. Speedup is the Dense
median divided by each method median and stays blank until an explicit valid
Dense run is provided.  Quality and manual-review cells remain blank—not
numeric zero—and the overall row status remains `pending` even when
`timing_status=valid`.  A later quality workflow must fill those fields under a
separately reviewed protocol; this builder does not calculate LPIPS or invent a
human judgment.

For G/H selection, the sparse candidates are C--F only. A candidate is eligible
only when its timing row is valid, its complete quality receipt is valid, and
all 72 hash-bound manual rows are `pass` (a `fail` or `uncertain` is not an
acceptance). Among eligible candidates, choose the smallest median total
generation time; an exact tie is broken by canonical `method_id`. The selected
method and all input hashes must be recorded before either G or H is run.
