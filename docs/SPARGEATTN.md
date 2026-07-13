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

The installer keeps the checkout, build cache, environment, and installation
receipt below `/cache/liluchen/FastA2V`. It sets
`TORCH_CUDA_ARCH_LIST=8.0` by default for the A100 and invokes the upstream
`setup.py` through `pip`; FastA2V does not copy, modify, or implement any CUDA
or Triton kernel. A successful build writes
`/cache/liluchen/FastA2V/spargeattn-install.json`. Sparse inference refuses to
start if that receipt does not identify the pinned repository, commit, and API,
or if the installed `core.py`, `_qattn*.so`, and `_fused*.so` fingerprints have
changed. The formal run verifier cross-checks the copied receipt against
preflight evidence and every warm-up/measurement backend record.

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

The first smoke configuration uses `sparge_topk: 0.5`, the upstream default
`sparge_pvthreshd: 50`, and `sparge_smooth_k: true`. Those settings are an
integration starting point, not a quality-equivalence claim. Run the dense and
Sparge smoke tests with the same prompt, seed, dimensions, solver, and step
count before a formal benchmark.

For this pinned upstream commit, `sparge_smooth_k` must remain `true`: its
public implementation defines an internal key mean only on that branch but
uses the value unconditionally. FastA2V rejects `false` before model loading
instead of allowing an `UnboundLocalError` during the first attention call.
