"""Official SpargeAttn adapter for Ovi video self-attention.

The adapter deliberately owns no CUDA or Triton implementation.  It calls the
public ``spas_sage2_attn_meansim_topk_cuda`` API from the pinned upstream
SpargeAttn package and reconstructs the surrounding Ovi ``WanSelfAttention``
forward from the original module's own QKV function, RoPE function, sequence
parallel collectives, and output projection.

Only ``FusionModel.video_self_attention_dispatcher`` can invoke this backend;
audio self-attention and all cross-attention call sites remain unchanged.
"""

from importlib import import_module
import hashlib
import inspect
import json
import os
from pathlib import Path


SPARGEATTN_REPOSITORY = "https://github.com/thu-ml/SpargeAttn.git"
SPARGEATTN_COMMIT = "ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a"
SPARGEATTN_API = "spas_sage2_attn_meansim_topk_cuda"


class SpargeAttentionDependencyError(RuntimeError):
    """Raised when the pinned official SpargeAttn dependency is unavailable."""


class SpargeAttentionInputError(ValueError):
    """Raised when an input cannot be represented by the official kernel API."""


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_installed_file_fingerprints(receipt):
    package_root_value = receipt.get("installed_package_root")
    installed_files = receipt.get("installed_files")
    if not isinstance(package_root_value, str) or not package_root_value:
        raise SpargeAttentionDependencyError(
            "SpargeAttn receipt is missing installed_package_root"
        )
    if not isinstance(installed_files, dict) or not installed_files:
        raise SpargeAttentionDependencyError(
            "SpargeAttn receipt is missing installed file fingerprints"
        )

    package_root = Path(package_root_value).resolve()
    required_kinds = {
        "core.py": any(Path(name).name == "core.py" for name in installed_files),
        "_qattn*.so": any(
            Path(name).name.startswith("_qattn") and Path(name).suffix == ".so"
            for name in installed_files
        ),
        "_fused*.so": any(
            Path(name).name.startswith("_fused") and Path(name).suffix == ".so"
            for name in installed_files
        ),
    }
    missing_kinds = sorted(
        name for name, present in required_kinds.items() if not present
    )
    if missing_kinds:
        raise SpargeAttentionDependencyError(
            "SpargeAttn receipt lacks required installed artifacts: "
            f"{missing_kinds}"
        )

    mismatches = []
    for relative_name, expected in sorted(installed_files.items()):
        if not isinstance(relative_name, str) or not isinstance(expected, dict):
            mismatches.append(f"invalid receipt entry {relative_name!r}")
            continue
        path = (package_root / relative_name).resolve()
        try:
            path.relative_to(package_root)
        except ValueError:
            mismatches.append(f"path escapes package root: {relative_name}")
            continue
        if not path.is_file():
            mismatches.append(f"missing installed file: {relative_name}")
            continue
        actual_bytes = path.stat().st_size
        actual_sha256 = _sha256_file(path)
        if (
            expected.get("bytes") != actual_bytes
            or expected.get("sha256") != actual_sha256
        ):
            mismatches.append(
                f"fingerprint mismatch: {relative_name} "
                f"bytes={actual_bytes} sha256={actual_sha256}"
            )
    if mismatches:
        raise SpargeAttentionDependencyError(
            "Installed SpargeAttn files differ from the pinned install "
            f"receipt: {mismatches}"
        )
    return package_root


def verify_sparge_install_receipt(receipt_path=None):
    """Verify that the installed extension was built from the pinned commit."""

    if receipt_path is None:
        cache_root = os.environ.get(
            "FASTA2V_CACHE_ROOT", "/cache/liluchen/FastA2V"
        )
        receipt_path = Path(cache_root) / "spargeattn-install.json"
    else:
        receipt_path = Path(receipt_path)

    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpargeAttentionDependencyError(
            f"SpargeAttn install receipt not found at {receipt_path}; run "
            "'bash scripts/install_sparge_attn.sh' before sparse inference."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise SpargeAttentionDependencyError(
            f"Invalid SpargeAttn install receipt at {receipt_path}: {exc}"
        ) from exc

    expected = {
        "repository": SPARGEATTN_REPOSITORY,
        "commit": SPARGEATTN_COMMIT,
        "api": SPARGEATTN_API,
    }
    mismatches = {
        key: {"expected": value, "actual": receipt.get(key)}
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if mismatches:
        raise SpargeAttentionDependencyError(
            "SpargeAttn install receipt does not match the pinned official "
            f"dependency: {mismatches}. Re-run "
            "'bash scripts/install_sparge_attn.sh'."
        )
    _verify_installed_file_fingerprints(receipt)
    return receipt_path, receipt


def load_official_sparge_kernel(expected_package_root=None):
    """Load and validate the public kernel entrypoint without a dense fallback."""

    try:
        module = import_module("spas_sage_attn")
    except Exception as exc:
        raise SpargeAttentionDependencyError(
            "attention_method='sparge' requires the official SpargeAttn "
            f"package at commit {SPARGEATTN_COMMIT}; run "
            "'bash scripts/install_sparge_attn.sh'. Import failed with: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if expected_package_root is not None:
        loaded_file = getattr(module, "__file__", None)
        loaded_root = Path(loaded_file).resolve().parent if loaded_file else None
        expected_root = Path(expected_package_root).resolve()
        if loaded_root != expected_root:
            raise SpargeAttentionDependencyError(
                "Python imported spas_sage_attn from a different location "
                f"than the pinned receipt: loaded={loaded_root}, "
                f"expected={expected_root}"
            )

    kernel = getattr(module, SPARGEATTN_API, None)
    if not callable(kernel):
        raise SpargeAttentionDependencyError(
            "Installed spas_sage_attn does not expose the required public API "
            f"{SPARGEATTN_API!r}. Reinstall official SpargeAttn at commit "
            f"{SPARGEATTN_COMMIT}."
        )

    try:
        parameters = inspect.signature(kernel).parameters
    except (TypeError, ValueError):
        parameters = None
    if parameters is not None:
        required_keywords = {
            "dropout_p",
            "topk",
            "is_causal",
            "pvthreshd",
            "smooth_k",
            "tensor_layout",
            "return_sparsity",
        }
        missing = sorted(required_keywords - set(parameters))
        if missing:
            raise SpargeAttentionDependencyError(
                f"Installed {SPARGEATTN_API} has an incompatible signature; "
                f"missing keyword parameters {missing}. Expected official "
                f"commit {SPARGEATTN_COMMIT}."
            )
    return kernel


def _tensor_shape(tensor, name):
    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise SpargeAttentionInputError(f"{name} must expose a tensor shape")
    try:
        return tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError) as exc:
        raise SpargeAttentionInputError(
            f"{name} has a non-concrete shape: {shape!r}"
        ) from exc


def _sequence_lengths(seq_lens):
    value = seq_lens
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if not callable(tolist):
        raise SpargeAttentionInputError(
            "seq_lens must expose tolist(); SpargeAttn cannot safely ignore "
            "unknown padding"
        )
    lengths = tolist()
    if not isinstance(lengths, list):
        lengths = [lengths]
    return [int(length) for length in lengths]


class SpargeVideoSelfAttentionBackend:
    """Callable backend for Ovi's NHD ``[B, S, H, D]`` video attention.

    ``kernel``, ``rope_apply_fn``, and ``all_to_all_4d_fn`` are injected so the
    Python ownership and argument contract can be tested without importing
    PyTorch or loading a GPU extension.  Production construction is provided by
    :func:`build_sparge_video_backend`.
    """

    def __init__(
        self,
        *,
        kernel,
        rope_apply_fn,
        all_to_all_4d_fn=None,
        topk=0.5,
        pvthreshd=50,
        smooth_k=True,
        install_receipt=None,
    ):
        if not callable(kernel):
            raise TypeError("kernel must be callable")
        if not callable(rope_apply_fn):
            raise TypeError("rope_apply_fn must be callable")
        if all_to_all_4d_fn is not None and not callable(all_to_all_4d_fn):
            raise TypeError("all_to_all_4d_fn must be callable or None")
        if isinstance(topk, bool) or not isinstance(topk, (int, float)):
            raise TypeError("sparge_topk must be a number in (0, 1]")
        if not 0.0 < float(topk) <= 1.0:
            raise ValueError(f"sparge_topk must be in (0, 1], got {topk!r}")
        if isinstance(pvthreshd, bool) or not isinstance(pvthreshd, (int, float)):
            raise TypeError("sparge_pvthreshd must be numeric")
        if float(pvthreshd) <= 0:
            raise ValueError(
                f"sparge_pvthreshd must be positive, got {pvthreshd!r}"
            )
        # The pinned upstream core defines ``km`` only inside ``if smooth_k``
        # and then uses it unconditionally.  Treat false as an incompatible
        # configuration rather than allowing a delayed UnboundLocalError.
        if smooth_k is not True:
            raise ValueError(
                "sparge_smooth_k must be true for pinned official SpargeAttn "
                f"commit {SPARGEATTN_COMMIT}"
            )

        self._kernel = kernel
        self._rope_apply = rope_apply_fn
        self._all_to_all_4d = all_to_all_4d_fn
        self.topk = float(topk)
        self.pvthreshd = float(pvthreshd)
        self.smooth_k = True
        self.install_receipt = dict(install_receipt or {})
        self.reset_metrics()

    def reset_metrics(self):
        """Start a new generation and invalidate its full-length proof."""

        self._validated_full_length_signatures = set()
        self._calls = 0
        self._last_nhd_shape = None

    def metrics(self):
        return {
            "backend": "official_spargeattn",
            "repository": SPARGEATTN_REPOSITORY,
            "pinned_commit": SPARGEATTN_COMMIT,
            "api": SPARGEATTN_API,
            "tensor_layout": "NHD",
            "topk": self.topk,
            "pvthreshd": self.pvthreshd,
            "smooth_k": self.smooth_k,
            "return_sparsity": False,
            "install_receipt": dict(self.install_receipt),
            "calls": self._calls,
            "last_nhd_shape": self._last_nhd_shape,
        }

    def _validate_qkv(self, q, k, v, seq_lens):
        shapes = {
            "q": _tensor_shape(q, "q"),
            "k": _tensor_shape(k, "k"),
            "v": _tensor_shape(v, "v"),
        }
        if any(len(shape) != 4 for shape in shapes.values()):
            raise SpargeAttentionInputError(
                "official SpargeAttn NHD inputs must have shape [B, S, H, D]; "
                f"got {shapes}"
            )
        if len(set(shapes.values())) != 1:
            raise SpargeAttentionInputError(
                f"Ovi video q/k/v shapes must match for self-attention: {shapes}"
            )

        shape = shapes["q"]
        batch, sequence, _heads, head_dim = shape
        if sequence < 128:
            raise SpargeAttentionInputError(
                f"official SpargeAttn requires sequence length >= 128, got {sequence}"
            )
        if head_dim not in (64, 128):
            raise SpargeAttentionInputError(
                "official SpargeAttn supports head dimensions 64 or 128; "
                f"got {head_dim}"
            )

        devices = {
            str(getattr(tensor, "device", None)) for tensor in (q, k, v)
        }
        if len(devices) != 1:
            raise SpargeAttentionInputError(
                f"q/k/v must be on the same device, got {sorted(devices)}"
            )
        device_type = getattr(getattr(q, "device", None), "type", None)
        if device_type != "cuda":
            raise SpargeAttentionInputError(
                "official SpargeAttn is a CUDA backend; q/k/v must be CUDA tensors"
            )

        # The public top-k API does not accept per-sample lengths or a padding
        # mask.  Validate once per shape/device signature rather than forcing a
        # host synchronization in every transformer block.
        signature = (shape, next(iter(devices)))
        if signature not in self._validated_full_length_signatures:
            lengths = _sequence_lengths(seq_lens)
            if len(lengths) != batch or any(length != sequence for length in lengths):
                raise SpargeAttentionInputError(
                    "official SpargeAttn top-k API cannot represent padded Ovi "
                    f"video attention: seq_lens={lengths}, NHD sequence={sequence}"
                )
            self._validated_full_length_signatures.add(signature)

        self._last_nhd_shape = list(shape)
        return shape

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
        qkv_fn = getattr(original_attention, "qkv_fn", None)
        output_projection = getattr(original_attention, "o", None)
        if not callable(qkv_fn) or not callable(output_projection):
            raise TypeError(
                "Sparge Ovi adapter requires the original attention object's "
                "qkv_fn and output projection o"
            )
        window_size = getattr(original_attention, "window_size", None)
        try:
            window_size = tuple(window_size)
        except TypeError as exc:
            raise SpargeAttentionInputError(
                "Ovi video attention must expose its window_size"
            ) from exc
        if window_size != (-1, -1):
            raise SpargeAttentionInputError(
                "official SpargeAttn top-k adapter currently preserves only "
                f"Ovi global attention window (-1, -1), got {window_size}"
            )

        # WanSelfAttention.qkv_fn owns Q/K normalization.  Reusing it avoids
        # duplicating or drifting from Ovi's trained projections and norms.
        q, k, v = qkv_fn(x)
        use_sp = bool(getattr(original_attention, "use_sp", False))
        if use_sp:
            if self._all_to_all_4d is None:
                raise SpargeAttentionInputError(
                    "Ovi sequence parallelism is enabled but no official Ovi "
                    "all_to_all_4D function was supplied"
                )
            q = self._all_to_all_4d(q, scatter_dim=2, gather_dim=1)
            k = self._all_to_all_4d(k, scatter_dim=2, gather_dim=1)
            v = self._all_to_all_4d(v, scatter_dim=2, gather_dim=1)

        nhd_shape = self._validate_qkv(q, k, v, seq_lens)
        q = self._rope_apply(q, grid_sizes, freqs)
        k = self._rope_apply(k, grid_sizes, freqs)

        # return_sparsity=True calls .item() in the official implementation and
        # introduces a device synchronization.  Formal timed inference always
        # requests only the attention output.
        attn_output = self._kernel(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
            topk=self.topk,
            pvthreshd=self.pvthreshd,
            smooth_k=self.smooth_k,
            tensor_layout="NHD",
            return_sparsity=False,
        )
        if isinstance(attn_output, tuple):
            raise RuntimeError(
                "official SpargeAttn returned a tuple despite "
                "return_sparsity=False; refusing an incompatible API"
            )
        if _tensor_shape(attn_output, "SpargeAttn output") != nhd_shape:
            raise RuntimeError(
                "official SpargeAttn output shape differs from its NHD input: "
                f"output={_tensor_shape(attn_output, 'SpargeAttn output')} "
                f"input={nhd_shape}"
            )

        if use_sp:
            attn_output = self._all_to_all_4d(
                attn_output, scatter_dim=1, gather_dim=2
            )

        input_shape = _tensor_shape(x, "Ovi video attention input")
        output_shape = _tensor_shape(attn_output, "SpargeAttn gathered output")
        if (
            len(input_shape) != 3
            or len(output_shape) != 4
            or output_shape[:2] != input_shape[:2]
            or output_shape[2] * output_shape[3] != input_shape[2]
        ):
            raise RuntimeError(
                "SpargeAttn output cannot be restored to the Ovi hidden shape: "
                f"attention={output_shape}, hidden={input_shape}"
            )

        self._calls += 1
        return output_projection(attn_output.flatten(2))


def build_sparge_video_backend(config):
    """Build the production adapter and fail before model loading if missing."""

    _receipt_path, receipt = verify_sparge_install_receipt(
        config.get("sparge_install_receipt", None)
    )
    kernel = load_official_sparge_kernel(receipt["installed_package_root"])
    from ovi.distributed_comms.communications import all_to_all_4D
    from ovi.modules.model import rope_apply

    return SpargeVideoSelfAttentionBackend(
        kernel=kernel,
        rope_apply_fn=rope_apply,
        all_to_all_4d_fn=all_to_all_4D,
        topk=float(config.get("sparge_topk", 0.5)),
        pvthreshd=float(config.get("sparge_pvthreshd", 50)),
        smooth_k=config.get("sparge_smooth_k", True),
        install_receipt=receipt,
    )
