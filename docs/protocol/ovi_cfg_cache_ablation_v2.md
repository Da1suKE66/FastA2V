# Ovi CFG-cache ablation v2

## 1. Goal

Determine, under a frozen 30-step Ovi Euler configuration, how CFG-cache quality loss depends on:

1. **timestep position** of skipped negative forwards;
2. **number of skipped negative forwards**;
3. **maximum cache age / refresh interval**;
4. prompt and seed;
5. whether a new late-window policy improves quality at **exactly the same compute count** as the current policy.

The experiment is split into a development ablation and a held-out confirmation. Development results may be used to choose candidates. Held-out results may not be used to tune or add new candidates.

## 2. Immediate correction to the current evidence

The existing 6–23/r5 and 9–26/r5 points were produced at commit `24bfbaac798f0e85617210f239c3d725b12fea95`; the new 12–29/r5 point was reported at commit `4c77d1e8a954c805e26ce35a8f97befbec447960` and on another A100 host.

Before combining these points into one causal position curve, rerun at least Dense, 9–26/r5, and 12–29/r5 on the **same commit, same checkpoint, same software environment, and same physical GPU**. A code diff claimed to be “instrumentation only” is not sufficient by itself. Reproduction of the 9–26 output or decoded-output hash is the required bridge.

The current three-window result remains useful engineering evidence, but the new protocol below is the first point at which “placement causes the difference” can be stated cleanly.

## 3. Frozen generation constants

The following fields must remain identical in every run unless the matrix explicitly changes them:

| Field | Frozen value |
|---|---|
| Model | Ovi 720×720 5 s checkpoint |
| Actual decoded output | 121 frames, 704×704, H.264 + non-silent AAC |
| Sampling | 30 steps, Euler, shift 5.0 |
| Parallelism | single GPU, SP=1, batch=1 |
| Guidance | audio 3.0, video 4.0, SLG layer 11 |
| Attention | dense video attention |
| Precision/offload | fp8=false, qint8=false, cpu_offload=false |
| Negative prompts | unchanged from the current baseline |
| Block/Radial/Sparge | disabled |
| Profiling | disabled in all quality and benchmark runs |

Use zero-based, inclusive timestep indices in filenames and reports. Do not use percentage names such as `30_90` as the primary identifier because they obscure rounding and inclusivity.

## 4. Analytical workload checks

For an inclusive cache window of length `L` and refresh interval `r`:

```text
refreshes        = ceil(L / r)
cache_hits       = L - refreshes
negative_forwards = 30 - cache_hits
attention_calls  = 1770 - 29 * cache_hits
```

Every measured record must match these values exactly. A count mismatch, dispatcher fallback, GPU contention sample, missing output stream, or prompt/seed mismatch invalidates that record.

## 5. Stage 0 — comparability, null control, and determinism

Run one development prompt (`客厅看球`) with seed 103 in this order on the same idle physical GPU:

1. Dense reference, `D0`;
2. 12–29/r1 null-cache control;
3. 9–26/r5 same-compute anchor;
4. 12–29/r5 candidate, repetition 1;
5. 12–29/r5 candidate, repetition 2;
6. Dense reference, `D1`.

### Expected counts

| Config | Hits | Refreshes | Negative forwards | Attention calls |
|---|---:|---:|---:|---:|
| Dense | 0 | 0 | 30 | 1770 |
| 12–29/r1 | 0 | 18 | 30 | 1770 |
| 9–26/r5 | 14 | 4 | 16 | 1364 |
| 12–29/r5 | 14 | 4 | 16 | 1364 |

### Mandatory gates

Stop the experiment if any gate fails:

- `D0` and `D1` must reproduce the same uncompressed decoded output hash, or the same final latent hash when latent export is enabled.
- The two 12–29/r5 repetitions must reproduce exactly.
- 12–29/r1 must match Dense. This tests the cache code path with zero reuse; a difference means the cache branch changes semantics even without a cache hit.
- The new 9–26/r5 output must match the old 9–26/r5 output. If it does not, do not mix old and new quality measurements; rerun all points used in the position curve.
- Equal-hit configurations must have identical runtime counters. Absolute latency may differ slightly, but a difference larger than 1% in paired denoise time should be investigated before proceeding.

When encoded MP4 bytes are not deterministic, compare final latents, pre-encode RGB tensors, decoded raw-video hashes, and decoded PCM hashes in that order. Do not use container-file SHA alone as the determinism criterion.

## 6. Stage 1 — clean timestep-position map

The current long windows overlap heavily. Replace that diagnostic with six non-overlapping five-step bins. Each bin performs one refresh followed by four cache hits, so every candidate has the same hit count, the same cache-age pattern `[1, 2, 3, 4]`, and the same theoretical compute.

| Window | Refresh step | Cache-hit steps | Hits | Negative forwards | Attention calls |
|---|---:|---|---:|---:|---:|
| 0–4/r5 | 0 | 1,2,3,4 | 4 | 26 | 1654 |
| 5–9/r5 | 5 | 6,7,8,9 | 4 | 26 | 1654 |
| 10–14/r5 | 10 | 11,12,13,14 | 4 | 26 | 1654 |
| 15–19/r5 | 15 | 16,17,18,19 | 4 | 26 | 1654 |
| 20–24/r5 | 20 | 21,22,23,24 | 4 | 26 | 1654 |
| 25–29/r5 | 25 | 26,27,28,29 | 4 | 26 | 1654 |

### Development sample

Use three already-seen prompts only, because this stage is allowed to influence policy selection:

- 客厅看球 — stable close-up speech;
- 黄昏跑步 — body motion, breathing, outdoor background;
- 吉他录音室 — speech plus a controlled musical transient.

Use seeds `103` and `211`. Generate Dense references for every prompt–seed pair at the frozen commit. This produces six independent prompt–seed units for each bin.

### Primary analysis

For each prompt–seed unit, compute candidate-vs-Dense drift for every bin. Plot drift against the mean cache-hit timestep. Report all six paired curves rather than only an aggregate median.

The position claim is supported only when the late half (15–29) is less damaging than the early half (0–14) in at least five of the six prompt–seed units for the primary video metric, with no systematic reversal in task audio metrics.

A non-monotonic local region is not a failure. It means the policy should be based on a sensitivity mask or multiple windows rather than a single “later is always safer” rule.

## 7. Stage 2 — late-window policy and cache-age ablation

After Stage 1, run the following policy cells. First run all cells on one development prompt and seed. Remove obviously dominated or failed cells, then advance at most three cells to all five existing development prompts with seeds 103 and 211.

### 7.1 Fixed late window: practical speed/quality curve

| Config | Hits | Max cache age | Negative forwards | Attention calls | Purpose |
|---|---:|---:|---:|---:|---|
| 12–29/r2 | 9 | 1 | 21 | 1509 | safe endpoint |
| 12–29/r3 | 12 | 2 | 18 | 1422 | conservative late candidate |
| 12–29/r4 | 13 | 3 | 17 | 1393 | intermediate point |
| 12–29/r5 | 14 | 4 | 16 | 1364 | aggressive late candidate |

This curve intentionally varies hit count and cache age together because that is the actual deployable refresh-policy knob.

### 7.2 Equal-compute comparisons

These comparisons hold hit count and attention-call count fixed.

#### Twelve-hit tier

| Config | Hits | Calls | Interpretation |
|---|---:|---:|---|
| 12–29/r3 | 12 | 1422 | wider window, cache age ≤2 |
| 15–29/r5 | 12 | 1422 | later/narrower window, cache age ≤4 |

This pair tests whether excluding steps 12–14 compensates for allowing older reuse later in the trajectory.

#### Fourteen-hit tier

| Config | Hits | Calls | Interpretation |
|---|---:|---:|---|
| 9–26/r5 | 14 | 1364 | current anchor |
| 12–29/r5 | 14 | 1364 | pure late shift at the same age pattern |
| 14–29/r8 | 14 | 1364 | later window, cache age ≤7 |
| 15–29/r15 | 14 | 1364 | one-refresh stress test, cache age ≤14 |

The last two cells are falsification tests for the claim that only hit count and placement matter. Because their windows are later, the position-only hypothesis predicts non-worse quality. If they degrade, cache age is independently important. Stop the r15 cell after the first sample if it produces an obvious semantic, speech, synchronization, or severe reconstruction failure.

### Candidate-freezing rule

Freeze exactly two candidates before opening held-out results:

- one **12-hit conservative candidate**, chosen between 12–29/r3 and 15–29/r5;
- one **14-hit aggressive candidate**, normally 12–29/r5 unless another 14-hit cell is clearly superior without a task-metric regression.

Optionally retain 12–29/r4 only when it lies on the development Pareto frontier and there is a real product need for the 13-hit tier. Do not advance more cells merely because their differences are small.

## 8. Stage 3 — held-out confirmation

Use the supplied eight-prompt held-out CSV. These prompts cover stable speech, object interaction, body and camera motion, two-speaker turn-taking, fine hand motion, non-speech environmental audio, and music-only audio. Do not modify the file after the first held-out run.

Use new seeds `503`, `887`, and `1291`. Neither prompts nor seeds may have appeared in candidate selection.

Run these five frozen configurations:

1. Dense;
2. current conservative 6–23/r3;
3. frozen new 12-hit candidate;
4. current aggressive 9–26/r5;
5. frozen new 14-hit candidate.

This yields two primary same-compute comparisons:

- new 12-hit candidate vs 6–23/r3;
- new 14-hit candidate vs 9–26/r5.

Balance configuration order across the three seed blocks. A simple order is:

```text
seed 503:  Dense, old-12, new-12, old-14, new-14
seed 887:  new-12, old-14, new-14, Dense, old-12
seed 1291: new-14, Dense, old-12, new-12, old-14
```

Do not add configurations or alter thresholds after inspecting held-out outputs.

## 9. Quality measurements

### 9.1 Mechanistic/reference-similarity metrics

Compute from uncompressed or decoded-aligned media:

- video PSNR;
- video SSIM;
- frame-average and p95 LPIPS when the pinned environment supports it;
- temporal frame-difference error or temporal LPIPS;
- final-latent relative L2 and cosine similarity when latent export is enabled.

PSNR/SSIM measure drift from Dense, not absolute generative quality. They must not be the only deployment gate.

### 9.2 Audio and task metrics

Raw zero-lag waveform correlation is not a sufficient primary audio metric. Before waveform comparison, align candidate and Dense within a small fixed lag window and record the chosen lag. Report:

- aligned waveform correlation;
- SI-SDR or aligned waveform error;
- log-mel distance;
- ASR WER/CER against the text inside `<S>…<E>`;
- speech activity coverage and silence ratio;
- lip-sync score when a pinned SyncNet-style evaluator is available.

For non-speech prompts, replace ASR with event timing and log-mel/audio-envelope similarity.

### 9.3 Blind human review

For the held-out set, randomize and blind candidate identity. Use at least three independent ratings per pair for:

- speech intelligibility;
- lip synchronization;
- visual artifacts and temporal stability;
- prompt adherence;
- overall preference.

Reviewers should compare the two equal-compute candidates against each other and may inspect Dense separately. They should not be shown latency or method names.

## 10. Statistical analysis

The primary unit is the **prompt**, with seeds treated as repeated observations within a prompt. Do not treat all 24 prompt–seed outputs as fully independent.

For each same-compute comparison:

1. compute paired differences for every prompt–seed unit;
2. report median, mean, 10th percentile, worst case, and pairwise win rate;
3. construct a 95% cluster-bootstrap interval by resampling prompts and retaining all seeds within each sampled prompt;
4. report per-category results in addition to the global aggregate.

Predeclare one primary reconstruction metric, preferably SSIM until a pinned perceptual metric is available. Treat PSNR, LPIPS, temporal metrics, and audio metrics as secondary but mandatory diagnostics. Do not select whichever metric happens to favor the candidate after the run.

### Recommended acceptance gates

Replace these provisional values with product tolerances when available:

- exact workload counters for every sample;
- equal-hit candidates have paired median denoise-time difference within ±1%;
- new candidate has positive median paired improvement in the primary reconstruction metric;
- at least 70% pairwise wins on the primary metric;
- no material regression in ASR, lip-sync, or human failure rate;
- no new severe failure in the worst held-out prompt–seed unit.

A candidate that improves median PSNR/SSIM but degrades speech, lip-sync, or the tail failure rate is rejected.

## 11. Stage 4 — formal latency benchmark

Quality generation and latency benchmarking are separate experiments.

Benchmark only:

- Dense;
- frozen 12-hit candidate;
- frozen 14-hit candidate.

Use an immutable audited run kind, a clean commit or recorded patch hash, fixed prompt file, and the same physical GPU. Use at least three warmups and five measurements per workload. Repeat in balanced configuration order across blocks rather than running every Dense sample first and every candidate sample later.

Report separately:

- pre-denoise;
- denoise;
- audio decode;
- video decode;
- total warm-service generation;
- save/mux;
- artifact-ready latency;
- cold model-load latency.

No profiler, debug logging, per-step synchronization, or extra tensor export is allowed in this benchmark. Run profiling only in a separate non-benchmark process.

## 12. Evidence and fail-closed requirements

Every final run directory must contain or bind:

- `preflight.json`;
- `environment.freeze.txt`;
- `checkpoint_manifest.json`;
- `pre_run_gpu.json`;
- frozen YAML and prompt CSV hashes;
- git commit plus clean-tree status, or a complete patch hash;
- measurement and warmup record counts;
- runtime counters and dispatcher/fallback status;
- GPU identity, process samples, clocks, power, and temperature where available;
- media validation and decoded stream hashes.

“Package validation passed” and “media validation passed” are not substitutes for protocol validity.

## 13. Stop rules

Stop a cell immediately when:

- expected and observed cache counts differ;
- any video-attention fallback occurs;
- GPU contention is detected;
- output is missing, silent when speech is required, or malformed;
- the r1 null control differs from Dense;
- deterministic repetitions differ without an understood source;
- a cache-age stress cell shows an obvious severe failure;
- a cell is strictly dominated at the same hit count and attention-call count.

Do not resume Sparge, Radial, or block-cache sweeps from this protocol. The only objective is to identify a reliable CFG-cache schedule and quantify its generalization.

## 14. Expected final conclusion formats

The experiment should end in one of four conclusions:

1. **Late placement generalizes:** both 12-hit and 14-hit late candidates improve held-out quality at unchanged compute.
2. **Late placement helps only at one tier:** adopt the improved tier and keep the other current policy.
3. **Cache age dominates:** use shorter refresh intervals or an explicit sensitivity-aware schedule even in late steps.
4. **Prompt-dependent sensitivity:** abandon one global contiguous window and implement an adaptive or multi-window schedule, then evaluate it as a new frozen method.

The report must include all samples, including failures, and must keep development and held-out tables separate.
