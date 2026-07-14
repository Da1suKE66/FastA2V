# Radial Attention integration status

This integration starts from official Ovi and the official
`mit-han-lab/radial-attention` repository. It contains no FastA2V CUDA or
Triton kernel and does not copy the upstream mask algorithm.

## Fixed provenance

- Repository: `https://github.com/mit-han-lab/radial-attention.git`
- SSH clone path: `ssh://git@ssh.github.com:443/mit-han-lab/radial-attention.git`
- Commit: `72788d4f0a6d202f1ec5f1c98a6e4c8b2e34fdbc`
- Official mask API: `gen_log_mask_shrinked`
- Fixed FlashInfer candidate: `flashinfer-python==0.2.5+cu124torch2.6` from
  the CUDA 12.4, PyTorch 2.6 wheel index
- Fixed Linux wheel SHA256:
  `43d767b912c0c43a04be99595e0123eab9385fc72530a2874b5fb08e3145c0be`

The upstream checkout under `/cache/liluchen/FastA2V/sources` must remain
pristine. The installer creates a separate derived copy and applies
`third_party/radial-attention-optional-imports.patch`. That mask-only derived
module imports Matplotlib and SageAttention-family backends lazily, only if
their plotting or Sage execution branch is explicitly called; source, patch,
and derived-file SHA256 values are fixed in
`ovi/radial_evidence.py` and in the install receipt.

The installer caches the exact wheel under
`/cache/liluchen/FastA2V/wheels`, resumes interrupted downloads, and verifies
its fixed byte count and SHA256 before installation. The receipt also binds
the wheel, resolved FlashInfer module path, and every installed package file.
Native `.so` files must exist, have fixed SHA256 values, and pass `ldd` with no
unresolved library. ASLR addresses are normalized while resolved library paths
remain part of the `ldd` fingerprint. The immutable manifest is copied to each
run as `radial-flashinfer-manifest.json` and rechecked by the final verifier.

## Exact Ovi execution protocol

Only Ovi `720x720_5s`, BF16, batch 1, `sp_size=1`, and video grid
`[31, 22, 22]` are accepted. Ovi QKV projection and normalization, Ovi RoPE,
and Ovi output projection are reused directly.

The 15,004 video tokens are not padded:

1. The official mask generator produces the 117-by-117 block mask for the
   14,976-token complete-block prefix.
2. The official generator leaves rows 22, 56, and 90 empty for both fixed
   profiles. Those three query block rows are made fully dense and the raw and
   repaired masks are checked against fixed hashes and true-block counts.
3. Prefix queries against prefix keys use the public
   `flashinfer.BlockSparseAttentionWrapper` API with `return_lse=True`.
4. Prefix queries against the 28 tail keys use dense FlashInfer with
   `return_lse=True`; the two prefix states are combined with
   `flashinfer.merge_state`.
5. The 28 tail queries attend densely to all 15,004 keys.

Plans are cached by immutable shape/device/dtype/profile keys. Per-generation
metrics reset without discarding the plan cache. Unsupported shapes, grids,
dtypes, dependencies, sequence parallelism, or API results raise immediately;
there is no dense fallback and a failed run cannot be reported as Radial.

Fixed profiles:

| Profile | decay factor | raw blocks | repaired blocks |
|---|---:|---:|---:|
| conservative | 4.0 | 5,338 | 5,689 |
| aggressive | 1.0 | 4,159 | 4,510 |

CFG cache and block cache are disabled in every pure-Radial smoke/formal
configuration so timing and quality results have one acceleration owner.

## Commands and evidence boundary

Install the pinned source and fixed FlashInfer candidate:

```bash
bash scripts/install_radial_attention.sh
```

Then, only when the runner's physical-GPU-0 idle guard succeeds, run one of:

```bash
bash scripts/run_ovi_radial_conservative_smoke.sh
bash scripts/run_ovi_radial_aggressive_smoke.sh
bash scripts/run_ovi_radial_conservative_baseline.sh
bash scripts/run_ovi_radial_aggressive_baseline.sh
```

Each Radial runner resolves physical GPU 0 by UUID and exports that UUID as
`CUDA_VISIBLE_DEVICES` before creating the idle evidence. Thus logical CUDA 0
cannot silently map to another physical device.

Each run copies the install receipt, FlashInfer manifest, pristine source
module, derived source module, and optional-imports patch into its fresh run
directory. Preflight and
the final verifier bind those files, the fixed run protocol, dispatcher call
counts, mask audit, tail strategy, GPU identity, and generated media.

After the runner proves physical GPU 0 is idle, preflight repeatedly launches
the same exact-shape conservative backend on BF16 A100 tensors. It requires
finite output, the fixed mask audit, one planned cache entry, the prefix/tail
merge strategy, allocator evidence covering the live QKV tensors, and a CUDA
UUID matching the physical-GPU-0 idle record before Ovi checkpoints load.

Every GPU identity/process query retains its fixed command, trusted executable
fingerprint, raw stdout/stderr, exit code, timestamps, byte counts, and hashes.
The final verifier reparses those raw receipts. Samples after CUDA context
creation must remain a stable singleton, include complete queries inside the
exact backend window, continue through final synchronization, and have no gap
larger than the fixed bound.

Pre-run GPU evidence and generation GPU-monitor summaries use evidence schema
version 2. Version 1 formal-run artifacts are intentionally rejected rather
than inferred or rewritten: their missing schema guarantees cannot be migrated
after collection, so every affected method must be rerun in a fresh directory.

There are two explicitly different process-evidence outcomes:

- `direct_c_observed`: `nvidia-smi pmon` reports the sampled host PID as a
  direct `C` client inside the exact backend window.
- `pmon_reported_all_idle_during_audited_window`: the trusted `pmon` stream is
  syntactically valid but reports only strict idle rows while independent
  `query-compute-apps` receipts show one stable process. This is a degraded
  host-observability result. It proves only a sampled temporal association
  after the idle guard; host-PID ownership, MPS absence, and continuous
  exclusivity remain unknown and are never claimed.

Linux PID namespaces are checked through `/proc/self/status` `NSpid`. The
direct mode requires the outer PID to match. The degraded association is
allowed only for a single-level namespace when `/proc/<sampled-host-pid>` is
absent with `ENOENT`; permission or other lookup failures are rejected.

The observed all-idle `pmon` behavior is therefore preserved as evidence of a
host/container observability limitation, not rewritten as a direct-compute or
MPS-free success. A formal Radial result still requires the exact wrapper,
dispatcher, output, media, and final verifier checks to pass.
