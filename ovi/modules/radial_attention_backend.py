"""Thin Ovi adapter for pinned official Radial Attention mask generation.

FastA2V does not implement a sparse CUDA/Triton kernel or a replacement mask
algorithm.  It calls ``gen_log_mask_shrinked`` from the audited upstream source
and executes the resulting block mask through FlashInfer's public APIs.  Ovi's
own QKV projections/norms, RoPE, and output projection remain authoritative.

The official generator floors a 15,004-token sequence to 117 complete
128-token blocks.  This adapter therefore treats 14,976 tokens as a sparse
prefix and handles the 28-token tail exactly: prefix-query/tail-key attention is
dense and LSE-merged with the sparse prefix result, while tail queries attend
dense to all keys.  No padding and no dense fallback are permitted.
"""

from dataclasses import dataclass
import hashlib
from importlib import import_module, metadata, util
import json
import os
from pathlib import Path
import subprocess
import sys

from ovi.radial_evidence import (
    FLASHINFER_DISTRIBUTION,
    FLASHINFER_REQUIRED_APIS,
    FLASHINFER_VERSION,
    RADIAL_BLOCK_SIZE,
    RADIAL_COMMIT,
    RADIAL_DERIVED_MODULE_SHA256,
    RADIAL_EMPTY_ROWS,
    RADIAL_GRID,
    RADIAL_HEAD_DIM,
    RADIAL_HEADS,
    RADIAL_MASK_API,
    RADIAL_MODEL_TYPE,
    RADIAL_OPTIONAL_IMPORTS_PATCH_SHA256,
    RADIAL_PREFIX_SEQUENCE,
    RADIAL_PROFILE_AUDITS,
    RADIAL_REPOSITORY,
    RADIAL_SEQUENCE,
    RADIAL_SOURCE_MODULE_SHA256,
    RADIAL_TAIL_SEQUENCE,
    flashinfer_manifest_evidence_errors,
    normalize_ldd_output,
    radial_profile,
    radial_receipt_evidence_errors,
)


class RadialAttentionDependencyError(RuntimeError):
    """Raised before model loading when pinned dependencies are unavailable."""


class RadialAttentionInputError(ValueError):
    """Raised when an Ovi input cannot be represented by the fixed protocol."""


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_fingerprint(metadata_value, label):
    if not isinstance(metadata_value, dict):
        raise RadialAttentionDependencyError(f"Radial receipt lacks {label}")
    path = Path(metadata_value.get("path", "")).resolve()
    if not path.is_file():
        raise RadialAttentionDependencyError(f"Radial {label} is missing: {path}")
    actual_bytes = path.stat().st_size
    actual_sha256 = _sha256_file(path)
    if (
        metadata_value.get("bytes") != actual_bytes
        or metadata_value.get("sha256") != actual_sha256
    ):
        raise RadialAttentionDependencyError(
            f"Radial {label} fingerprint mismatch: bytes={actual_bytes} "
            f"sha256={actual_sha256}"
        )
    return path


def _verify_installed_flashinfer_files(receipt):
    package_root = Path(receipt["installed_flashinfer_package_root"]).resolve()
    if not package_root.is_dir():
        raise RadialAttentionDependencyError(
            f"Installed FlashInfer package root is missing: {package_root}"
        )
    module_path = _verify_fingerprint(
        receipt["flashinfer_module"], "FlashInfer module"
    )
    if module_path != package_root / "__init__.py":
        raise RadialAttentionDependencyError(
            "FlashInfer module fingerprint is outside the installed package root"
        )
    expected_files = set(receipt["installed_flashinfer_files"])
    actual_files = {
        str(path.relative_to(package_root))
        for path in package_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    mismatches = []
    if actual_files != expected_files:
        mismatches.append(
            "installed file set differs: "
            f"missing={sorted(expected_files - actual_files)} "
            f"unexpected={sorted(actual_files - expected_files)}"
        )
    ldd_env = os.environ.copy()
    search_paths = [package_root.parent / "torch" / "lib"]
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        search_paths.append(Path(cuda_home) / "lib64")
    existing_ld_path = ldd_env.get("LD_LIBRARY_PATH")
    if existing_ld_path:
        search_paths.append(existing_ld_path)
    ldd_env["LD_LIBRARY_PATH"] = ":".join(str(path) for path in search_paths)
    for relative_name, expected in sorted(
        receipt["installed_flashinfer_files"].items()
    ):
        path = (package_root / relative_name).resolve()
        try:
            path.relative_to(package_root)
        except ValueError:
            mismatches.append(f"path escapes package root: {relative_name}")
            continue
        if not path.is_file():
            mismatches.append(f"missing installed file: {relative_name}")
            continue
        if (
            path.stat().st_size != expected.get("bytes")
            or _sha256_file(path) != expected.get("sha256")
        ):
            mismatches.append(f"fingerprint mismatch: {relative_name}")
            continue
        if path.suffix == ".so":
            try:
                ldd_output = subprocess.check_output(
                    ["ldd", str(path)],
                    text=True,
                    stderr=subprocess.STDOUT,
                    env=ldd_env,
                )
            except (OSError, subprocess.CalledProcessError) as exc:
                mismatches.append(f"ldd failed: {relative_name}: {exc}")
                continue
            actual_ldd_sha256 = hashlib.sha256(
                normalize_ldd_output(ldd_output).encode("utf-8")
            ).hexdigest()
            if "not found" in ldd_output:
                mismatches.append(
                    f"ldd has unresolved libraries: {relative_name}"
                )
            if actual_ldd_sha256 != expected.get("ldd_sha256"):
                mismatches.append(f"ldd fingerprint mismatch: {relative_name}")
    if mismatches:
        raise RadialAttentionDependencyError(
            f"Installed FlashInfer files differ from receipt: {mismatches}"
        )
    return package_root


def _verify_flashinfer_manifest(receipt):
    manifest_path = _verify_fingerprint(
        receipt["flashinfer_manifest"], "FlashInfer manifest"
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RadialAttentionDependencyError(
            f"Invalid FlashInfer manifest at {manifest_path}: {exc}"
        ) from exc
    errors = flashinfer_manifest_evidence_errors(manifest, receipt)
    if errors:
        raise RadialAttentionDependencyError(
            f"FlashInfer manifest differs from install receipt: {errors}"
        )
    return manifest_path, manifest


def verify_radial_install_receipt(receipt_path=None):
    """Verify pinned source, derived source, patch, and CPU mask evidence."""

    cache_root = os.environ.get("FASTA2V_CACHE_ROOT", "/cache/liluchen/FastA2V")
    if receipt_path is None:
        receipt_path = Path(cache_root) / "radialattn-install.json"
    else:
        receipt_path = Path(receipt_path)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RadialAttentionDependencyError(
            f"Radial install receipt not found at {receipt_path}; run "
            "'bash scripts/install_radial_attention.sh'."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RadialAttentionDependencyError(
            f"Invalid Radial install receipt at {receipt_path}: {exc}"
        ) from exc

    errors = radial_receipt_evidence_errors(
        receipt, expected_cache_root=cache_root
    )
    if errors:
        raise RadialAttentionDependencyError(
            "Radial receipt differs from the audited official dependency: "
            f"{errors}"
        )

    source_path = _verify_fingerprint(receipt["source_module"], "source module")
    derived_path = _verify_fingerprint(
        receipt["derived_module"], "derived module"
    )
    patch_path = _verify_fingerprint(
        receipt["optional_imports_patch"], "optional-imports patch"
    )
    expected_paths = {
        "source": Path(receipt["source_dir"]) / "radial_attn" / "attn_mask.py",
        "derived": Path(receipt["derived_dir"])
        / "radial_attn"
        / "attn_mask.py",
    }
    if source_path != expected_paths["source"].resolve():
        raise RadialAttentionDependencyError(
            "Radial source module is outside the pinned pristine checkout"
        )
    if derived_path != expected_paths["derived"].resolve():
        raise RadialAttentionDependencyError(
            "Radial derived module is outside the fixed derived directory"
        )
    repo_root = Path(__file__).resolve().parents[2]
    expected_patch_path = (
        repo_root / "third_party" / "radial-attention-optional-imports.patch"
    ).resolve()
    if patch_path != expected_patch_path:
        raise RadialAttentionDependencyError(
            "Radial receipt points to an unexpected optional-imports patch"
        )
    fixed_hashes = {
        source_path: RADIAL_SOURCE_MODULE_SHA256,
        derived_path: RADIAL_DERIVED_MODULE_SHA256,
        patch_path: RADIAL_OPTIONAL_IMPORTS_PATCH_SHA256,
    }
    for path, expected_digest in fixed_hashes.items():
        if _sha256_file(path) != expected_digest:
            raise RadialAttentionDependencyError(
                f"Audited Radial file changed after receipt creation: {path}"
            )
    _verify_installed_flashinfer_files(receipt)
    _verify_flashinfer_manifest(receipt)
    return receipt_path.resolve(), receipt


def load_official_radial_mask_module(derived_module_path):
    """Load the patched upstream module from its receipt-bound exact path."""

    path = Path(derived_module_path).resolve()
    if _sha256_file(path) != RADIAL_DERIVED_MODULE_SHA256:
        raise RadialAttentionDependencyError(
            "Derived upstream Radial module does not match the audited hash"
        )
    module_name = "fasta2v_pinned_radial_attn_mask"
    spec = util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RadialAttentionDependencyError(
            f"Cannot load pinned Radial source module: {path}"
        )
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise RadialAttentionDependencyError(
            "Pinned Radial module import failed; optional dependencies may be "
            f"incompatible: {type(exc).__name__}: {exc}"
        ) from exc
    required = (RADIAL_MASK_API, "get_indptr_from_mask", "get_indices_from_mask")
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise RadialAttentionDependencyError(
            f"Pinned Radial module lacks required public helpers: {missing}"
        )
    return module


def load_flashinfer_api(expected_package_root=None):
    """Load exactly the candidate distribution named by the install receipt."""

    try:
        installed_version = metadata.version(FLASHINFER_DISTRIBUTION)
    except metadata.PackageNotFoundError as exc:
        raise RadialAttentionDependencyError(
            f"{FLASHINFER_DISTRIBUTION}=={FLASHINFER_VERSION} is not installed"
        ) from exc
    if installed_version != FLASHINFER_VERSION:
        raise RadialAttentionDependencyError(
            f"{FLASHINFER_DISTRIBUTION} version {installed_version!r} != "
            f"fixed candidate {FLASHINFER_VERSION!r}"
        )
    try:
        module = import_module("flashinfer")
    except Exception as exc:
        raise RadialAttentionDependencyError(
            f"flashinfer import failed: {type(exc).__name__}: {exc}"
        ) from exc
    if expected_package_root is not None:
        loaded_file = getattr(module, "__file__", None)
        loaded_root = Path(loaded_file).resolve().parent if loaded_file else None
        expected_root = Path(expected_package_root).resolve()
        if loaded_root != expected_root:
            raise RadialAttentionDependencyError(
                "Python imported flashinfer from a different location than "
                f"the receipt: loaded={loaded_root}, expected={expected_root}"
            )
    missing = [
        name
        for name in FLASHINFER_REQUIRED_APIS
        if not callable(getattr(module, name, None))
    ]
    if missing:
        raise RadialAttentionDependencyError(
            f"fixed FlashInfer candidate lacks official Radial APIs: {missing}"
        )
    return module


def summarize_bool_rows(rows):
    """Hash a block mask as canonical row-major bytes without NumPy."""

    normalized = []
    width = None
    for row in rows:
        values = tuple(bool(value) for value in row)
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise RadialAttentionInputError("Radial block mask rows are ragged")
        normalized.append(values)
    if not normalized or width != len(normalized):
        raise RadialAttentionInputError(
            "Radial block mask must be a non-empty square matrix"
        )
    raw = bytes(value for row in normalized for value in row)
    return {
        "shape": [len(normalized), width],
        "true_blocks": sum(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "empty_rows": [
            index for index, row in enumerate(normalized) if not any(row)
        ],
    }


def _mask_rows(mask):
    detached = mask.detach() if callable(getattr(mask, "detach", None)) else mask
    cpu = detached.cpu() if callable(getattr(detached, "cpu", None)) else detached
    tolist = getattr(cpu, "tolist", None)
    if not callable(tolist):
        raise RadialAttentionInputError(
            "official Radial mask must expose CPU tolist() for audit"
        )
    return tolist()


def audit_and_repair_radial_mask(mask, profile):
    """Repair only empty block rows and bind both masks to fixed hashes."""

    expected = radial_profile(profile)
    raw = summarize_bool_rows(_mask_rows(mask))
    expected_size = RADIAL_PREFIX_SEQUENCE // RADIAL_BLOCK_SIZE
    if raw["shape"] != [expected_size, expected_size]:
        raise RadialAttentionInputError(
            f"official Radial mask shape {raw['shape']} != "
            f"{[expected_size, expected_size]}"
        )
    for field, expected_field in (
        ("true_blocks", "raw_true_blocks"),
        ("sha256", "raw_sha256"),
        ("empty_rows", "empty_rows"),
    ):
        if raw[field] != expected[expected_field]:
            raise RadialAttentionInputError(
                f"official Radial {profile} raw mask {field}={raw[field]!r} "
                f"!= audited {expected[expected_field]!r}"
            )
    repaired = mask.clone()
    for row in expected["empty_rows"]:
        repaired[row, :] = True
    repaired_summary = summarize_bool_rows(_mask_rows(repaired))
    if repaired_summary["empty_rows"]:
        raise RadialAttentionInputError(
            "Radial empty-row repair did not produce a valid block row"
        )
    if (
        repaired_summary["true_blocks"] != expected["repaired_true_blocks"]
        or repaired_summary["sha256"] != expected["repaired_sha256"]
    ):
        raise RadialAttentionInputError(
            f"Radial {profile} repaired mask differs from audited evidence"
        )
    return repaired, dict(expected)


def _tensor_shape(value, label):
    try:
        return tuple(int(dimension) for dimension in value.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RadialAttentionInputError(
            f"{label} must expose a concrete tensor shape"
        ) from exc


def _host_values(value, label):
    detached = value.detach() if callable(getattr(value, "detach", None)) else value
    cpu = detached.cpu() if callable(getattr(detached, "cpu", None)) else detached
    tolist = getattr(cpu, "tolist", None)
    if not callable(tolist):
        raise RadialAttentionInputError(f"{label} must expose tolist()")
    result = tolist()
    return result if isinstance(result, list) else [result]


@dataclass(frozen=True)
class _RadialPlan:
    wrapper: object
    key: tuple
    audit: tuple


class RadialVideoSelfAttentionBackend:
    """Strict FlashInfer execution for the exact Ovi 720p/5s video shape."""

    def __init__(
        self,
        *,
        torch_module,
        flashinfer_module,
        mask_generator,
        get_indptr_from_mask,
        get_indices_from_mask,
        rope_apply_fn,
        profile,
        install_receipt=None,
    ):
        dependencies = {
            "mask_generator": mask_generator,
            "get_indptr_from_mask": get_indptr_from_mask,
            "get_indices_from_mask": get_indices_from_mask,
            "rope_apply_fn": rope_apply_fn,
        }
        missing = [name for name, value in dependencies.items() if not callable(value)]
        if missing:
            raise TypeError(f"Radial backend dependencies are not callable: {missing}")
        self._torch = torch_module
        self._flashinfer = flashinfer_module
        self._mask_generator = mask_generator
        self._get_indptr = get_indptr_from_mask
        self._get_indices = get_indices_from_mask
        self._rope_apply = rope_apply_fn
        self.profile_name = str(profile).strip().lower()
        self.profile = radial_profile(self.profile_name)
        self.install_receipt = dict(install_receipt or {})
        self._plan_cache = {}
        self.reset_metrics()

    def reset_metrics(self):
        """Reset generation counters while retaining immutable keyed plans."""

        self._calls = 0
        self._plan_cache_hits = 0
        self._plan_cache_misses = 0
        self._last_shape = None
        self._last_grid = None
        self._last_device = None
        self._last_dtype = None
        self._last_mask_audit = None

    def metrics(self):
        last_mask_audit = dict(self._last_mask_audit or ())
        if "empty_rows" in last_mask_audit:
            last_mask_audit["empty_rows"] = list(
                last_mask_audit["empty_rows"]
            )
        return {
            "backend": "official_radial_attention_flashinfer",
            "repository": RADIAL_REPOSITORY,
            "pinned_commit": RADIAL_COMMIT,
            "mask_api": RADIAL_MASK_API,
            "profile": self.profile_name,
            "decay_factor": self.profile["decay_factor"],
            "model_type": RADIAL_MODEL_TYPE,
            "block_size": RADIAL_BLOCK_SIZE,
            "sequence": RADIAL_SEQUENCE,
            "prefix_sequence": RADIAL_PREFIX_SEQUENCE,
            "tail_sequence": RADIAL_TAIL_SEQUENCE,
            "tail_strategy": "dense_lse_merge_no_padding",
            "empty_row_policy": "dense_row",
            "empty_rows": list(RADIAL_EMPTY_ROWS),
            "fallback_allowed": False,
            "calls": self._calls,
            "plan_cache_entries": len(self._plan_cache),
            "plan_cache_hits": self._plan_cache_hits,
            "plan_cache_misses": self._plan_cache_misses,
            "last_shape": self._last_shape,
            "last_grid": self._last_grid,
            "last_device": self._last_device,
            "last_dtype": self._last_dtype,
            "last_mask_audit": last_mask_audit,
            "install_receipt": dict(self.install_receipt),
        }

    def _validate_inputs(self, q, k, v, seq_lens, grid_sizes):
        shapes = {
            name: _tensor_shape(value, name)
            for name, value in (("q", q), ("k", k), ("v", v))
        }
        expected_shape = (
            1,
            RADIAL_SEQUENCE,
            RADIAL_HEADS,
            RADIAL_HEAD_DIM,
        )
        if any(shape != expected_shape for shape in shapes.values()):
            raise RadialAttentionInputError(
                "fixed Ovi Radial protocol requires q/k/v shape "
                f"{expected_shape}, got {shapes}"
            )
        device_values = {str(getattr(value, "device", None)) for value in (q, k, v)}
        if len(device_values) != 1:
            raise RadialAttentionInputError(
                f"Radial q/k/v devices differ: {sorted(device_values)}"
            )
        device_type = getattr(getattr(q, "device", None), "type", None)
        if device_type != "cuda":
            raise RadialAttentionInputError(
                "FlashInfer Radial execution requires CUDA q/k/v tensors"
            )
        dtype_values = {str(getattr(value, "dtype", None)) for value in (q, k, v)}
        if dtype_values != {"torch.bfloat16"}:
            raise RadialAttentionInputError(
                f"fixed Radial protocol requires BF16 q/k/v, got {dtype_values}"
            )
        # Ovi constructs these tiny tensors on CPU for every model forward, so
        # checking their values on every block does not synchronize CUDA.  Do
        # not cache only by QKV shape/device: a changed grid with the same token
        # count would otherwise reuse the fixed mask while applying different
        # RoPE coordinates.
        lengths = [int(item) for item in _host_values(seq_lens, "seq_lens")]
        if lengths != [RADIAL_SEQUENCE]:
            raise RadialAttentionInputError(
                "fixed Radial protocol does not support padding or variable "
                f"lengths: seq_lens={lengths}"
            )
        grid = _host_values(grid_sizes, "grid_sizes")
        if grid != [list(RADIAL_GRID)]:
            raise RadialAttentionInputError(
                f"fixed Radial protocol requires grid {[list(RADIAL_GRID)]}, "
                f"got {grid}"
            )
        self._last_shape = list(expected_shape)
        self._last_grid = list(grid[0])
        self._last_device = next(iter(device_values))
        self._last_dtype = next(iter(dtype_values))
        return expected_shape

    def _get_plan(self, q):
        key = (
            tuple(int(value) for value in q.shape),
            str(q.device),
            str(q.dtype),
            self.profile_name,
            RADIAL_BLOCK_SIZE,
            RADIAL_MODEL_TYPE,
        )
        cached = self._plan_cache.get(key)
        if cached is not None:
            self._plan_cache_hits += 1
            self._last_mask_audit = cached.audit
            return cached.wrapper

        self._plan_cache_misses += 1
        mask = self._mask_generator(
            q,
            RADIAL_SEQUENCE,
            RADIAL_SEQUENCE,
            RADIAL_GRID[0],
            block_size=RADIAL_BLOCK_SIZE,
            sparse_type="radial",
            decay_factor=self.profile["decay_factor"],
            model_type=RADIAL_MODEL_TYPE,
        )
        repaired_mask, audit = audit_and_repair_radial_mask(
            mask, self.profile_name
        )
        indptr = self._get_indptr(repaired_mask, q)
        indices = self._get_indices(repaired_mask, q)
        workspace = self._torch.empty(
            128 * 1024 * 1024,
            device=q.device,
            dtype=self._torch.uint8,
        )
        wrapper = self._flashinfer.BlockSparseAttentionWrapper(
            workspace,
            backend="fa2",
        )
        wrapper.plan(
            indptr=indptr,
            indices=indices,
            M=RADIAL_PREFIX_SEQUENCE,
            N=RADIAL_PREFIX_SEQUENCE,
            R=RADIAL_BLOCK_SIZE,
            C=RADIAL_BLOCK_SIZE,
            num_qo_heads=RADIAL_HEADS,
            num_kv_heads=RADIAL_HEADS,
            head_dim=RADIAL_HEAD_DIM,
            q_data_type=q.dtype,
            kv_data_type=q.dtype,
        )
        frozen_audit = tuple(
            sorted(
                (
                    name,
                    tuple(value) if isinstance(value, list) else value,
                )
                for name, value in audit.items()
            )
        )
        self._plan_cache[key] = _RadialPlan(
            wrapper=wrapper,
            key=key,
            audit=frozen_audit,
        )
        self._last_mask_audit = frozen_audit
        return wrapper

    def __call__(
        self,
        original_attention,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        *,
        block_index=None,
        debug_context=None,
    ):
        del block_index, debug_context
        if bool(getattr(original_attention, "use_sp", False)):
            raise RadialAttentionInputError(
                "fixed Radial protocol is limited to sp_size=1"
            )
        if tuple(getattr(original_attention, "window_size", ())) != (-1, -1):
            raise RadialAttentionInputError(
                "fixed Radial adapter supports only Ovi global attention"
            )
        qkv_fn = getattr(original_attention, "qkv_fn", None)
        output_projection = getattr(original_attention, "o", None)
        if not callable(qkv_fn) or not callable(output_projection):
            raise TypeError(
                "Radial Ovi adapter requires original qkv_fn and output projection o"
            )

        q, k, v = qkv_fn(x)
        self._validate_inputs(q, k, v, seq_lens, grid_sizes)
        q = self._rope_apply(q, grid_sizes, freqs)
        k = self._rope_apply(k, grid_sizes, freqs)
        v_shape = _tensor_shape(v, "v")
        if (
            _tensor_shape(q, "RoPE q") != v_shape
            or _tensor_shape(k, "RoPE k") != v_shape
        ):
            raise RadialAttentionInputError("Ovi RoPE changed the fixed q/k shape")
        if (
            str(getattr(q, "device", None)) != self._last_device
            or str(getattr(k, "device", None)) != self._last_device
            or str(getattr(q, "dtype", None)) != self._last_dtype
            or str(getattr(k, "dtype", None)) != self._last_dtype
        ):
            raise RadialAttentionInputError(
                "Ovi RoPE changed q/k BF16 dtype or CUDA device"
            )

        wrapper = self._get_plan(q)
        q_shd, k_shd, v_shd = q[0], k[0], v[0]
        sparse_result = wrapper.run(
            q_shd[:RADIAL_PREFIX_SEQUENCE],
            k_shd[:RADIAL_PREFIX_SEQUENCE],
            v_shd[:RADIAL_PREFIX_SEQUENCE],
            return_lse=True,
        )
        if not isinstance(sparse_result, tuple) or len(sparse_result) != 2:
            raise RuntimeError(
                "FlashInfer block-sparse run did not return (output, lse)"
            )
        prefix_sparse, prefix_sparse_lse = sparse_result
        tail_result = self._flashinfer.single_prefill_with_kv_cache(
            q=q_shd[:RADIAL_PREFIX_SEQUENCE],
            k=k_shd[RADIAL_PREFIX_SEQUENCE:],
            v=v_shd[RADIAL_PREFIX_SEQUENCE:],
            causal=False,
            return_lse=True,
        )
        if not isinstance(tail_result, tuple) or len(tail_result) != 2:
            raise RuntimeError(
                "FlashInfer prefix-query/tail-key call did not return (output, lse)"
            )
        prefix_tail, prefix_tail_lse = tail_result
        merged = self._flashinfer.merge_state(
            v_a=prefix_sparse,
            s_a=prefix_sparse_lse,
            v_b=prefix_tail,
            s_b=prefix_tail_lse,
        )
        if not isinstance(merged, tuple) or len(merged) != 2:
            raise RuntimeError("FlashInfer merge_state did not return (output, lse)")
        prefix_output, _prefix_lse = merged
        tail_output = self._flashinfer.single_prefill_with_kv_cache(
            q=q_shd[RADIAL_PREFIX_SEQUENCE:],
            k=k_shd,
            v=v_shd,
            causal=False,
            return_lse=False,
        )
        output = self._torch.cat((prefix_output, tail_output), dim=0).unsqueeze(0)
        expected_output_shape = (
            1,
            RADIAL_SEQUENCE,
            RADIAL_HEADS,
            RADIAL_HEAD_DIM,
        )
        if _tensor_shape(output, "Radial output") != expected_output_shape:
            raise RuntimeError(
                f"Radial output shape {_tensor_shape(output, 'Radial output')} "
                f"!= {expected_output_shape}"
            )
        if (
            str(getattr(output, "device", None)) != self._last_device
            or str(getattr(output, "dtype", None)) != self._last_dtype
        ):
            raise RuntimeError("Radial output changed BF16 dtype or CUDA device")
        self._calls += 1
        return output_projection(output.flatten(2))


def build_radial_video_backend(config):
    """Build the strict backend before Ovi checkpoint allocation."""

    if bool(config.get("use_cfg_cache", False)) or bool(
        config.get("use_block_cache", False)
    ):
        raise RadialAttentionInputError(
            "audited Radial baselines require CFG cache and block cache disabled"
        )
    if int(config.get("sp_size", 1)) != 1:
        raise RadialAttentionInputError("audited Radial baselines require sp_size=1")
    profile_name = str(config.get("radial_profile", "conservative")).lower()
    profile = radial_profile(profile_name)
    configured_decay = float(
        config.get("radial_decay_factor", profile["decay_factor"])
    )
    if configured_decay != profile["decay_factor"]:
        raise RadialAttentionInputError(
            f"radial_profile={profile_name!r} requires decay_factor="
            f"{profile['decay_factor']}, got {configured_decay}"
        )
    if int(config.get("radial_block_size", RADIAL_BLOCK_SIZE)) != RADIAL_BLOCK_SIZE:
        raise RadialAttentionInputError("audited Radial block_size is exactly 128")
    if str(config.get("radial_model_type", RADIAL_MODEL_TYPE)) != RADIAL_MODEL_TYPE:
        raise RadialAttentionInputError("audited Radial model_type is exactly 'wan'")
    receipt_path, receipt = verify_radial_install_receipt(
        config.get("radial_install_receipt", None)
    )
    flashinfer = load_flashinfer_api(
        receipt["installed_flashinfer_package_root"]
    )
    module = load_official_radial_mask_module(receipt["derived_module"]["path"])
    import torch
    from ovi.modules.model import rope_apply

    receipt_summary = {
        "path": str(receipt_path),
        "commit": receipt["commit"],
        "derived_module_sha256": receipt["derived_module"]["sha256"],
        "flashinfer_version": receipt["flashinfer_version"],
    }
    return RadialVideoSelfAttentionBackend(
        torch_module=torch,
        flashinfer_module=flashinfer,
        mask_generator=getattr(module, RADIAL_MASK_API),
        get_indptr_from_mask=module.get_indptr_from_mask,
        get_indices_from_mask=module.get_indices_from_mask,
        rope_apply_fn=rope_apply,
        profile=profile_name,
        install_receipt=receipt_summary,
    )
