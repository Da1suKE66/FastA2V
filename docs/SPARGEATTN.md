# Official SpargeAttn adapter

FastA2V pins the official
[`thu-ml/SpargeAttn`](https://github.com/thu-ml/SpargeAttn) source to commit
[`ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a`](https://github.com/thu-ml/SpargeAttn/commit/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a).
The machine-readable pin is
`third_party/SpargeAttn.commit`.

The pinned upstream source recommends the public
`spas_sage2_attn_meansim_topk_cuda` API. Its `tensor_layout="NHD"` mode accepts
the Ovi video Q/K/V layout `[batch, sequence, heads, head_dim]`; the baseline
shape is `[1, 15004, 24, 128]`. FastA2V passes `return_sparsity=False` for timed
runs because the optional sparsity return performs a host-visible scalar read
in the upstream implementation.

## Installation on `lsh-stable30138`

Set up the Ovi environment first, then build the pinned official extension:

```bash
cd /workspace/liluchen/FastA2V
bash scripts/setup_ovi_env.sh
bash scripts/install_sparge_attn.sh
```

The installer clones the official source through `ssh.github.com:443` with
`/home/ma-user/.ssh/id_ed25519_github`; the canonical HTTPS repository identity
is still recorded in the receipt. It keeps the checkout, build cache,
environment, and installation receipt below `/cache/liluchen/FastA2V`. It sets
`TORCH_CUDA_ARCH_LIST=8.0` by default for the A100 and invokes the upstream
`setup.py` through `pip`; FastA2V does not copy, modify, or implement any CUDA
or Triton kernel. A successful build writes
`/cache/liluchen/FastA2V/spargeattn-install.json` and preserves the complete
build log. Before the build and its real CUDA microtest, the installer requires
physical GPU 0 to be idle and records its UUID and process list. Sparse
inference refuses to
start if that receipt does not identify the pinned repository, commit, and API,
or if the installed `core.py`, `_qattn*.so`, and `_fused*.so` fingerprints have
changed. The installed `core.py` must also be byte-identical to the pinned
source checkout, the package must resolve from the fixed Ovi environment, and
the copied build log must match its receipt hash. The formal run verifier
cross-checks the copied receipt against
preflight evidence and every warm-up/measurement backend record. Installation
also launches the pinned public API at `topk=0.5` and `topk=1.0` on a real BF16
NHD `[1,132,24,128]` tensor. The non-block-aligned sequence length exercises a
tail path like Ovi's 15004 tokens; the test checks finite outputs and requires
a broad full-mask cosine agreement with SDPA. All comparison numbers must be
finite. Preflight repeats that microtest before loading the Ovi model and binds
it to the runner's idle-GPU UUID evidence.

The upstream package requires CUDA 12 or newer and supports head dimensions 64
and 128. The first GPU validation must therefore check the actual compiler,
PyTorch CUDA ABI, A100 load, and output before treating a run as valid; the
local tests only prove Python routing and API ownership.

## Adapter boundary

`ovi/modules/sparge_attention_backend.py` is registered only when
`attention_method: "sparge"`. It reuses, in order:

1. `vid_block.self_attn.qkv_fn`, including Ovi Q/K normalization;
2. Ovi sequence-parallel collectives when enabled;
3. Ovi `rope_apply` for Q and K;
4. the pinned official SpargeAttn public kernel in NHD layout;
5. Ovi's original `vid_block.self_attn.o` output projection.

The adapter rejects padded sequence lengths because the selected official
top-k API has no per-sample length argument. It also rejects CPU inputs,
unsupported head dimensions, non-global Ovi attention windows, incompatible
API returns, and missing build receipts. None of those failures can route to
dense attention. Audio
self-attention, text cross-attention, and audio-video cross-attention do not
pass through this dispatcher.

The audited experiment has two isolated keep-ratio protocols. Both keep
`sparge_pvthreshd: 50`, `sparge_smooth_k: true`, CFG cache disabled, block
cache disabled, and `sp_size: 1`:

| Keep ratio | Diagnostic smoke (20 steps) | Formal benchmark (50 steps) |
| --- | --- | --- |
| `topk=0.50` | `bash scripts/run_ovi_sparge_smoke.sh` | `bash scripts/run_ovi_sparge_baseline.sh` |
| `topk=0.75` | `bash scripts/run_ovi_sparge_topk75_smoke.sh` | `bash scripts/run_ovi_sparge_topk75_baseline.sh` |

The `topk=0.75` runners write below run parents containing
`sparge_topk75`; they cannot reuse or overwrite either `topk=0.50` run parent.
The verifier binds each `run_kind` to its exact keep ratio, step count,
warm-up/measurement count, benchmark eligibility, and debug mode. Changing a
config after the run therefore invalidates the protocol instead of relabeling
the result. These settings are integration and comparison points, not a
quality-equivalence claim. Run the dense and both Sparge smoke tests with the
same prompt, seed, dimensions, solver, and step count before formal benchmarks.

For this pinned upstream commit, `sparge_smooth_k` must remain `true`: its
public implementation defines an internal key mean only on that branch but
uses the value unconditionally. FastA2V rejects `false` before model loading
instead of allowing an `UnboundLocalError` during the first attention call.
