# Audited Ovi quality protocol

`scripts/compare_ovi_quality.py` compares one explicitly selected formal Dense
run with one explicitly selected formal candidate run. It does not scan for a
latest run, assign a sparse-quality pass threshold, or fill a human-review
field.

## Fixed comparison contract

- Both directories must already pass the formal `build_ovi_eval_csv.py`
  validator: clean commit, immutable 50-step protocol, one excluded warm-up,
  three valid measurements, no GPU contention, checkpoint fingerprint, media
  hash, metrics sidecar, and verifier agreement.
- Measurements are paired only as Dense index `0/1/2` to candidate index
  `0/1/2`. Prompt index, sample index, prompt, seed, requested and actual shape,
  generated tensor shapes, sample steps, commit, and checkpoint fingerprint
  must match. The generating GPU identity must match as well.
- Both MP4 files are SHA256-checked before and after every metric call.
- PSNR and SSIM reuse the decoded-video FFmpeg logic in
  `scripts/compare_media.py`. Audio reuses its mono 16 kHz float32 decode and
  RMSE, maximum absolute difference, SNR, and correlation formulas. The formal
  quality protocol requires exact frame topology and exact decoded audio sample
  count; the special official-reference duplicate-tail exception is not used.
- LPIPS is fixed to `lpips.LPIPS(net="alex", version="0.1")`, RGB24 frames,
  input range `[-1,1]`, batch size one, one Torch compute/inter-op thread,
  deterministic algorithms, disabled MKLDNN, and CPU. A missing dependency, wrong
  module path/version, absent or changed weight, missing source, or non-finite
  result stops the comparison. No LPIPS result has been precomputed or claimed
  by this scaffold.

The exact machine-readable contract is `configs/quality_protocol.json`.

## Environment and dependency receipt

The installer is one-shot and refuses an existing eval environment, wheelhouse,
or receipt. This prevents a stale `.pth`, `sitecustomize`, wheel, or bytecode
file from entering the trust chain. The first run is explicitly a resolver
bootstrap:

```bash
QUALITY_INSTALL_MODE=bootstrap bash scripts/install_ovi_quality_env.sh
```

It writes only below `/cache/liluchen/FastA2V`, installs CPU builds in
`/cache/liluchen/FastA2V/envs/eval`, caches the official torchvision AlexNet
weight below `/cache/liluchen/FastA2V/checkpoints/eval/torch`, and produces:

```text
/cache/liluchen/FastA2V/checkpoints/eval/lpips_alex_v0.1_receipt.json
```

The receipt records the complete resolved environment, including pip,
setuptools, all Torch transitive dependencies, and the seven direct metric
dependencies. Every distribution is bound to a version, original source URL,
full archive SHA256, retained wheel, and installed wheel `RECORD`; direct
dependencies additionally bind the exact import module path and source-file
hash. It separately records the LPIPS AlexNet linear calibration weight and
torchvision AlexNet backbone with absolute path, byte count, full SHA256, and
source. Installation uses wheels only, disables bytecode compilation, and
rejects symlinks and any distribution not present in the resolver reports. It
sets `umask 027` before creating the fixed environment so every directory is
deterministically owner/group-readable but never world-accessible.

Every pip resolver invocation combines `--isolated` with
`PIP_CONFIG_FILE=/dev/null`: the first disables ambient `PIP_*` options and
user configuration, while the second disables global, virtual-environment,
and explicit configuration files. This prevents an inherited extra index,
find-links directory, constraint, or credential-bearing index URL from joining
the fixed resolver inputs. Because isolated pip ignores `PIP_CACHE_DIR`, the
installer passes the fixed `/cache/liluchen/FastA2V/cache/pip-eval` directory
with `--cache-dir` on every invocation. It also disables interactive input and
pip's version check. Every pip invocation fixes the HTTPS read timeout at 300
seconds and the connection retry budget at 10 so a slow official archive host
can be tolerated without changing the fixed resolver inputs. The dual
isolation above remains the source boundary. Bootstrap uses only its explicitly
written indexes. Pinned materialization adds no index and may fetch only the
reviewed hash-qualified direct URLs; the subsequent install reads only its
fresh retained wheelhouse.

After bootstrap installation, the exact archive URLs already recorded in the
three pip reports are materialized into the retained wheelhouse by isolated pip
with `--no-index --no-deps`. This second pass reuses the fixed pip cache, allows
bounded HTTP range resumption, and passes each report hash to pip as a direct
URL SHA256 fragment so bytes are authenticated before wheel metadata parsing.
It requires the wheelhouse filename set to match the reports exactly, rejects
every non-regular or symlink top-level entry, and verifies every full archive
SHA256 again. It does not run a second dependency resolution and cannot add an
index or an unreported wheel.

The canonical CPU resolver entry remains
`https://download.pytorch.org/whl/cpu`, while PyTorch currently serves wheel
bytes from either `download.pytorch.org` or its official
`download-r2.pytorch.org` CDN host. Both hosts map to that one canonical
`source_index`. Dependency URLs are accepted only as HTTPS on the default port
or explicit port 443, without credentials, query, or fragment, and below the
exact `/whl/cpu/` path boundary. PyPI wheels remain restricted to the exact
`files.pythonhosted.org` host below `/packages/` under the canonical
`https://pypi.org/simple` source index. Lookalike hosts and path prefixes are
rejected before any retained-wheel download.

The first installer run is deliberately a bootstrap, not a trust decision. It
also emits:

```text
/cache/liluchen/FastA2V/checkpoints/eval/quality_dependency_lock_candidate.json
```

Before any LPIPS score is allowed, independently verify the candidate's full
`trusted_environment_packages` payload (distribution, version, original URL,
and SHA256 for every wheel), its canonical
`trusted_environment_lock_sha256`, the seven duplicated direct-wheel hashes,
and both weight hashes. Copy that payload and the matching hashes into
`configs/quality_protocol.json`, change `trusted_lock_status` to `pinned`, and
commit the protocol. Partial locks, bootstrap `null` values, a lone digest
without its reconstructible package payload, a self-signed receipt, or an
uncommitted protocol fail before model construction.

The checked-in lock was promoted from the independently audited v8 bootstrap.
Its canonical 17-package lock is
`d6c856a60a3241efe1907351541e1d0843d057240276960fe27f32e45750279c`;
the audited dependency receipt is
`4d15e1fcfb0074133c43763ed565e04f7eee75c3a6b24afd339cc041dc0d39ed`.
The isolated read-only auditor used for that decision has SHA256
`2a680b577be3735d1bf4c6300d7ad2914143741067fcb8faf1db635b9b8ce932`.

After promotion, move the bootstrap artifacts aside and reproduce the fixed
environment from the reviewed lock:

```bash
QUALITY_INSTALL_MODE=pinned bash scripts/install_ovi_quality_env.sh
```

Pinned mode materializes only the exact reviewed direct URLs with isolated pip,
the same fixed cache, `--no-index`, and `--no-deps`; it verifies every full
hash, then asks pip to install the retained wheels with a hash-required
requirements file. The comparator then checks the exact installed
distribution set, every retained wheel and `RECORD`, all files in
site-packages (including rejection of unowned bytecode, other unowned files,
and symlinks), the fixed direct module paths/versions, and both weights before
and after scoring. The formal final-table merger additionally requires the raw
dependency receipt to bind exactly the fixed
`quality-pinned-pip-report.json`; the three bootstrap reports are promotion
evidence only and cannot be used for formal final publication.

Validate the reproduced environment without computing a metric:

```bash
/cache/liluchen/FastA2V/envs/eval/bin/python -I -S -B \
  scripts/compare_ovi_quality.py validate-receipt
```

The installer is not part of Ovi inference setup and does not run on import.

## Compare two formal runs

Use the fixed evaluation Python, and name the candidate matrix method
explicitly:

```bash
/cache/liluchen/FastA2V/envs/eval/bin/python -I -S -B \
  scripts/compare_ovi_quality.py compare \
  --dense-run /cache/liluchen/FastA2V/runs/ovi/dense/EXACT_RUN_ID \
  --candidate-run /cache/liluchen/FastA2V/runs/ovi/sparge_topk50/EXACT_RUN_ID \
  --candidate-method-id sparge_topk50 \
  --output-dir /cache/liluchen/FastA2V/runs/quality/EXACT_COMPARISON_ID
```

No output is written until every formal identity pair and all post-metric
hashes succeed. For the fixed formal8 protocol this means 72 pairs. The
directory then contains 72 identity sidecars plus one median:

```text
measurement_00_prompt_000_sample_000.quality.json
measurement_00_prompt_000_sample_001.quality.json
...
measurement_02_prompt_007_sample_002.quality.json
median.quality.json
```

`-I -S -B` is mandatory: user site and environment variables are ignored,
automatic site processing is disabled until the complete tree passes its
pre-import audit. Python never reads generated or unowned bytecode and never
writes bytecode. A bytecode member already present in an exact retained wheel
is accepted only when both wheel and installed `RECORD` bind its size and full
SHA256; NumPy 1.26.4 contains one such reviewed member.
Each pair sidecar binds both artifacts and both run identities, including
commit, checkpoint hashes, prompt, seed, shapes, steps, acceleration environment,
`environment.json`, dependency receipt, evaluator commit/matrix/script hashes,
and absolute FFmpeg/FFprobe paths and hashes. Every paired MP4, its metrics
sidecar, both run evidence sets, evaluator sources, tools, dependencies, and
weights are checked again after all metrics and immediately before the
exclusive atomic sidecar write. `median.quality.json` records each pair-sidecar
hash and the metric medians over all 72 identities. Exact-match PSNR is
represented explicitly as the string `"inf"`; JSON NaN and numeric placeholders
are rejected.

## Manual synchronization review

`eval/manual_sync_reviews.csv` is deliberately header-only. A human reviewer
may copy it, add either zero rows or all 72 formal identity rows, and enter
`pass`, `fail`, or `uncertain`. The reviewer, UTC timestamp, and rating are
mandatory human-authored fields. Every row must contain the exact Dense and
candidate artifact SHA256 for its measurement/prompt/seed identity.

Validate the completed CSV against the persisted median sidecar:

```bash
/cache/liluchen/FastA2V/envs/eval/bin/python -I -S -B \
  scripts/compare_ovi_quality.py validate-manual \
  --quality-report /cache/liluchen/FastA2V/runs/quality/EXACT_COMPARISON_ID/median.quality.json \
  --manual-reviews /path/to/human_completed_manual_sync_reviews.csv
```

The validator also re-hashes each pair sidecar referenced by the median file.
For a complete three-row human CSV it exclusively creates
`manual-review.validation.json` beside the median. That receipt binds the
median SHA256, CSV SHA256, protocol SHA256, and all three Dense/candidate hashes.
An empty template remains pending and creates no review receipt. The validator
never edits the CSV and never converts a blank template into a judgment.
